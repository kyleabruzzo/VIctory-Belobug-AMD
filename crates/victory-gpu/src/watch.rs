//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! The title shows one physical page at several addresses. A write of its
//! own through any of them is seen: the runtime tells us the views it maps
//! ([`title_memory_view`]), guarding a block makes every other view of its
//! page read-only too, and a fault at one of them opens them all. Without
//! this a copy taken at one address stayed as it was for the rest of the
//! run when the title wrote through another (parts of moving vehicles left
//! behind and stretched across the screen, 2026-11-25). The idea is
//! victory-fork #115's (Spexivus, joroyt) as carried by victory-amd; the
//! code here is written anew. `VICTORY_VIEW_GUARD=0` turns it off.
//!
//! shortcut: the runtime's own writes into physical pages through a view of
//! its own (xmem's `write_physical`, DirectStorage's page destinations) are
//! told by address ([`title_memory_written`]), not through this. A page
//! shown at more than [`MANY_VIEWS`] addresses is guarded one view at a
//! time. A system call writing into a guarded block fails (none is known
//! to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-64-MiB-region epoch table. Incremented for only the region(s) touched
/// whenever a guarded block is opened (VEH guard fault or `writing`),
/// remapped (`title_memory_remapped`), or rewritten (`title_memory_written`).
/// While a span's region epoch stays unchanged, any range already guarded and
/// copied at that epoch cannot have been modified by the title — and
/// `copy_back` on one buffer no longer invalidates clean spans in other regions.
const REGION_SHIFT: usize = 26; // 64 MiB per bucket
const REGION_BUCKETS: usize = 1024;
static REGION_EPOCHS: [AtomicU64; REGION_BUCKETS] =
    [const { AtomicU64::new(1) }; REGION_BUCKETS];

#[inline]
fn bump_region_epochs(start: usize, end: usize) {
    if start >= end {
        return;
    }
    let first = start >> REGION_SHIFT;
    let last = (end - 1) >> REGION_SHIFT;
    let count = last - first + 1;
    if count >= REGION_BUCKETS {
        for slot in &REGION_EPOCHS {
            slot.fetch_add(1, Ordering::Release);
        }
    } else {
        for r in first..=last {
            REGION_EPOCHS[r & (REGION_BUCKETS - 1)].fetch_add(1, Ordering::Release);
        }
    }
}

#[inline]
pub(crate) fn region_epoch(start: usize, end: usize) -> u64 {
    if start >= end {
        return 0;
    }
    let first = start >> REGION_SHIFT;
    let last = (end - 1) >> REGION_SHIFT;
    let count = last - first + 1;
    let mut hash = 0u64;
    if count >= REGION_BUCKETS {
        for slot in &REGION_EPOCHS {
            hash = hash
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(slot.load(Ordering::Acquire));
        }
    } else {
        for r in first..=last {
            hash = hash
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(REGION_EPOCHS[r & (REGION_BUCKETS - 1)].load(Ordering::Acquire));
        }
    }
    hash
}

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

// Reset and publication must be indivisible to another query touching any of
// the same console pages. Disjoint queries can enter the kernel in parallel.
// One stripe covers 4 MiB; indices repeat every 256 MiB (a collision only
// serializes more work). Multi-stripe requests always lock in index order.
const QUERY_STRIPES: usize = u64::BITS as usize;
const PAGES_PER_STRIPE: usize = (4 << 20) / PAGE;
static QUERIES: [Mutex<()>; QUERY_STRIPES] = [const { Mutex::new(()) }; QUERY_STRIPES];
static ASKED: AtomicU64 = AtomicU64::new(0);
static WRITTEN: std::sync::RwLock<BTreeMap<usize, u64>> = std::sync::RwLock::new(BTreeMap::new());

fn stripe_mask(first: usize, last: usize) -> u64 {
    if first >= last {
        return 0;
    }
    let begin = first / PAGES_PER_STRIPE;
    let count = (last - 1) / PAGES_PER_STRIPE - begin + 1;
    if count >= QUERY_STRIPES {
        return u64::MAX;
    }
    ((1u64 << count) - 1).rotate_left((begin % QUERY_STRIPES) as u32)
}

fn while_checking<T>(first: usize, last: usize, check: impl FnOnce() -> T) -> T {
    let mut mask = stripe_mask(first, last);
    if mask.is_power_of_two() {
        let _guard = QUERIES[mask.trailing_zeros() as usize].lock().unwrap_or_else(|e| e.into_inner());
        return check();
    }
    let mut held = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        let index = mask.trailing_zeros() as usize;
        held.push(QUERIES[index].lock().unwrap_or_else(|e| e.into_inner()));
        mask &= mask - 1;
    }
    check()
}

/// Several dirty kernel pages can describe one console page. Only adjacent
/// duplicates are discarded, so the kernel's answer may be in any order.
fn dirty_console_pages(addresses: &[usize]) -> impl Iterator<Item = usize> + '_ {
    let mut previous = None;
    addresses.iter().map(|&at| at / PAGE).filter(move |&page| {
        previous.replace(page) != Some(page)
    })
}

type History = (u64, Vec<(usize, u64)>);

fn publish(first: usize, last: usize, pages: impl IntoIterator<Item = usize>) -> History {
    let mut pages = pages.into_iter().peekable();
    let snapshot = |written: &BTreeMap<usize, u64>| {
        written.range(first..last).map(|(&p, &w)| (p, w)).collect()
    };
    // Allocate the epoch with the journal locked. A concurrent runtime write
    // published before this snapshot therefore cannot receive a later epoch.
    if pages.peek().is_none() {
        let history = WRITTEN.read().unwrap_or_else(|e| e.into_inner());
        return (ASKED.fetch_add(1, Ordering::Relaxed) + 1, snapshot(&history));
    }
    let mut history = WRITTEN.write().unwrap_or_else(|e| e.into_inner());
    let now = ASKED.fetch_add(1, Ordering::Relaxed) + 1;
    for page in pages {
        history.insert(page, now);
    }
    (now, snapshot(&history))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order.
/// `None` if the memory is not write-watched (nor, with `guarded`, a view).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<History> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    thread_local! {
        static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    while_checking(first, last, || FOUND.with_borrow_mut(|found| {
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        if found.len() < count {
            found.resize(count, 0);
        }
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: the thread's scratch holds at least `count` answer entries.
        let failed = unsafe {
            GetWriteWatch(WRITE_WATCH_FLAG_RESET, (first * PAGE) as *const _, (last - first) * PAGE,
                found.as_mut_ptr().cast(), &mut count, &mut granularity)
        } != 0;
        let mapped = if failed {
            if !guarded {
                return None;
            }
            count = 0;
            guard(first * PAGE, last * PAGE)?
        } else {
            Vec::new()
        };
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        Some(publish(first, last, dirty_console_pages(&found[..count]).chain(mapped)))
    }))
}

/// Same as [`writes`] for memory known to be guarded mapped view memory.
pub(crate) fn writes_guarded(start: usize, end: usize) -> Option<History> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    while_checking(first, last, || {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let guarded = guard(first * PAGE, last * PAGE)?;
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        Some(publish(first, last, guarded))
    })
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Block numbers as bits, in chunks of 4096 blocks (256 MiB of address
/// space). The guards' checks look at every block of a range under the
/// guards' lock, which every copy's check waits for: a bit is several times
/// quicker to look at than a tree's node.
struct BlockSet {
    chunks: std::collections::HashMap<usize, Box<[u64; 64]>, std::hash::BuildHasherDefault<ChunkHash>>,
}

/// Chunk numbers are spread enough by a multiplication.
#[derive(Default)]
struct ChunkHash(u64);

impl std::hash::Hasher for ChunkHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

impl BlockSet {
    const SHIFT: usize = 12;

    const fn new() -> Self {
        Self { chunks: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()) }
    }

    fn at(block: usize) -> (usize, usize, u64) {
        let bit = block & ((1 << Self::SHIFT) - 1);
        (block >> Self::SHIFT, bit / 64, 1u64 << (bit % 64))
    }

    fn contains(&self, block: &usize) -> bool {
        let (chunk, word, bit) = Self::at(*block);
        self.chunks.get(&chunk).is_some_and(|c| c[word] & bit != 0)
    }

    /// Whether it was not in the set.
    fn insert(&mut self, block: usize) -> bool {
        let (chunk, word, bit) = Self::at(block);
        let c = self.chunks.entry(chunk).or_insert_with(|| Box::new([0; 64]));
        let new = c[word] & bit == 0;
        c[word] |= bit;
        new
    }

    /// Whether it was in the set.
    fn remove(&mut self, block: &usize) -> bool {
        let (chunk, word, bit) = Self::at(*block);
        let Some(c) = self.chunks.get_mut(&chunk) else { return false };
        let had = c[word] & bit != 0;
        c[word] &= !bit;
        had
    }

    /// The blocks of `range` in the set, in order.
    fn range(&self, range: std::ops::Range<usize>) -> std::vec::IntoIter<usize> {
        let mut found = Vec::new();
        if range.start < range.end {
            let (first, last) = (range.start >> Self::SHIFT, (range.end - 1) >> Self::SHIFT);
            let mut chunks: Vec<usize> = if last - first < self.chunks.len() {
                (first..=last).filter(|c| self.chunks.contains_key(c)).collect()
            } else {
                let mut all: Vec<usize> =
                    self.chunks.keys().copied().filter(|c| (first..=last).contains(c)).collect();
                all.sort_unstable();
                all
            };
            chunks.dedup();
            for chunk in chunks {
                let bits = &self.chunks[&chunk];
                for (w, &word) in bits.iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        let block = (chunk << Self::SHIFT) + w * 64 + word.trailing_zeros() as usize;
                        word &= word - 1;
                        if range.contains(&block) {
                            found.push(block);
                        }
                    }
                }
            }
        }
        found.into_iter()
    }
}

impl Extend<usize> for BlockSet {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, blocks: I) {
        for block in blocks {
            self.insert(block);
        }
    }
}

struct Guards {
    /// Blocks we made read-only.
    protected: BlockSet,
    /// Of those, the ones not written since a copy took them.
    clean: BlockSet,
    /// Unmapped blocks (`VirtualProtect` failed) within guarded ranges;
    /// cleared by [`title_memory_remapped`] when the runtime maps them.
    holes: BlockSet,
    /// Blocks whose page was guarded together with its other views (those
    /// views' blocks are in it too): a write to one opens them all.
    aliased: BTreeSet<usize>,
    /// Blocks the kernel is about to write for the runtime ([`writing`]),
    /// from, to, and until when they are left writable: a read may be
    /// asynchronous and land after the call that asked for it returned, and
    /// the kernel does not fault into our handler, it fails. Until then
    /// they are not made read-only, and every check reports them written.
    pins: Vec<(usize, usize, std::time::Instant)>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BlockSet::new(),
    clean: BlockSet::new(),
    holes: BlockSet::new(),
    aliased: BTreeSet::new(),
    pins: Vec::new(),
});

/// How long blocks stay writable after [`writing`].
const PIN: std::time::Duration = std::time::Duration::from_secs(2);

impl Guards {
    /// Whether `block` is left writable for a kernel write.
    fn pinned(&self, block: usize) -> bool {
        self.pins.iter().any(|&(from, to, _)| from <= block && block < to)
    }
}

/// Views of one page past which its views are not guarded together (the
/// page the title shows wherever it has nothing has thousands).
pub(crate) const MANY_VIEWS: usize = 64;

/// The views of physical pages the runtime mapped: by address (length,
/// section, offset in it), and the addresses each page of a section is
/// shown at.
#[derive(Default)]
struct Views {
    by_addr: BTreeMap<usize, (usize, usize, u64)>,
    pages: std::collections::HashMap<(usize, u64), Vec<usize>>,
}

static VIEWS: Mutex<Option<Views>> = Mutex::new(None);

/// The longest stretch of views around `[at, end)` that follow one another
/// in the address space and in one section alike, within `reach` bytes of
/// `at` rounded down to it: its start, its end, the section and the offset
/// in it. `None` if the range is not in one such stretch.
pub(crate) fn stretch(at: usize, end: usize, reach: usize) -> Option<(usize, usize, usize, u64)> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let views = views.as_ref()?;
    let (&first, &(n, section, offset)) = views.by_addr.range(..=at).next_back()?;
    if first + n <= at {
        return None;
    }
    let (low, high) = (at - at % reach, (at - at % reach).checked_add(reach)?);
    if end > high {
        return None;
    }
    // Up from the view holding `at`.
    let (mut stop, mut next) = (first + n, offset + n as u64);
    for (&a, &(n, s, o)) in views.by_addr.range(first + 1..high) {
        if a != stop || s != section || o != next {
            break;
        }
        (stop, next) = (a + n, o + n as u64);
    }
    if stop < end {
        return None;
    }
    // And down from it.
    let (mut start, mut from) = (first, offset);
    for (&a, &(n, s, o)) in views.by_addr.range(low..first).rev() {
        if a + n != start || s != section || o + n as u64 != from {
            break;
        }
        (start, from) = (a, o);
    }
    let start_in = start.max(low);
    Some((start_in, stop.min(high), section, from + (start_in - start) as u64))
}

/// Diagnostics: how `[at, at + len)` is made of views of physical pages:
/// the views, their sections, the longest view, and the runs of views that
/// follow one another in their section too (one run: the range is one
/// stretch of a section).
pub(crate) fn layout(at: usize, len: usize) -> (usize, usize, usize, usize) {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(views) = views.as_ref() else { return (0, 0, 0, 0) };
    let end = at.saturating_add(len);
    let first = views
        .by_addr
        .range(..=at)
        .next_back()
        .filter(|(a, (n, _, _))| *a + n > at)
        .map(|(a, _)| *a)
        .unwrap_or(at);
    let (mut count, mut longest, mut runs) = (0, 0, 0);
    let mut sections = std::collections::BTreeSet::new();
    let mut before: Option<(usize, usize, u64)> = None;
    for (&a, &(n, section, offset)) in views.by_addr.range(first..end) {
        count += 1;
        longest = longest.max(n);
        sections.insert(section);
        let follows = before.is_some_and(|(pa, ps, po)| pa == a && ps == section && po == offset);
        if !follows {
            runs += 1;
        }
        before = Some((a + n, section, offset + n as u64));
    }
    (count, sections.len(), longest, runs)
}

/// Whether all the views of a guarded page are guarded together.
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

impl Views {
    /// Forgets the views overlapping `[at, at + len)`, whole.
    fn forget(&mut self, at: usize, len: usize) {
        let end = at.saturating_add(len);
        let first = self
            .by_addr
            .range(..=at)
            .next_back()
            .filter(|(a, (n, _, _))| *a + n > at)
            .map(|(a, _)| *a);
        let hit: Vec<usize> = first
            .into_iter()
            .chain(self.by_addr.range(at.saturating_add(1)..end).map(|(a, _)| *a))
            .collect();
        for a in hit {
            let Some((n, section, offset)) = self.by_addr.remove(&a) else {
                continue;
            };
            for k in 0..n / BLOCK {
                let key = (section, offset / BLOCK as u64 + k as u64);
                if let Some(shown) = self.pages.get_mut(&key) {
                    shown.retain(|&x| x != a + k * BLOCK);
                    if shown.is_empty() {
                        self.pages.remove(&key);
                    }
                }
            }
        }
    }

    /// Every address the page shown at `at` (a block's start) is shown at,
    /// `at` included; empty if `at` is in no view we were told of.
    fn shown(&self, at: usize) -> &[usize] {
        let Some((&base, &(n, section, offset))) = self.by_addr.range(..=at).next_back() else {
            return &[];
        };
        if at >= base + n {
            return &[];
        }
        let key = (section, offset / BLOCK as u64 + ((at - base) / BLOCK) as u64);
        self.pages.get(&key).map_or(&[], Vec::as_slice)
    }
}

/// Every address the page of the block at `at` is shown at (see
/// [`Views::shown`]).
fn aliases(at: usize) -> Vec<usize> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    views.as_ref().map_or(Vec::new(), |v| v.shown(at - at % BLOCK).to_vec())
}

/// The runtime mapped a view at `[at, at + len)` of the bytes from `offset`
/// on of the physical-page section `section`. The view is writable, so the
/// same pages guarded at another view are opened: their copies check them
/// again, which also guards the new view with them.
pub fn title_memory_view(at: usize, len: usize, section: usize, offset: u64) {
    if !view_guard() || len == 0 || !at.is_multiple_of(BLOCK) || !offset.is_multiple_of(BLOCK as u64) {
        return;
    }
    let mut others = Vec::new();
    {
        let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
        let views = views.get_or_insert_with(Views::default);
        views.forget(at, len);
        views.by_addr.insert(at, (len, section, offset));
        for k in 0..len / BLOCK {
            let shown = views.pages.entry((section, offset / BLOCK as u64 + k as u64)).or_default();
            if shown.len() < MANY_VIEWS {
                others.extend_from_slice(shown);
            }
            shown.push(at + k * BLOCK);
        }
    }
    if !others.is_empty() {
        let mut g = guards();
        for a in others {
            open_here(&mut g, a, a + BLOCK);
        }
    }
}

/// The runtime unmapped the views in `[at, at + len)`.
pub fn title_memory_view_gone(at: usize, len: usize) {
    if let Some(views) = VIEWS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        views.forget(at, len);
    }
}

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Fast path: if we already guard or recorded a hole at `first`, we already
    // know this is a mapped view and can skip `QueryVirtualMemoryInformation`.
    {
        let mut g = guards();
        if g.protected.contains(&first) || g.holes.contains(&first) {
            let mut written = Vec::new();
            let mut run: Option<usize> = None;
            for block in first..=last {
                let open = block < last && !g.clean.contains(&block) && !g.holes.contains(&block);
                if open {
                    if g.protected.contains(&block) {
                        g.clean.insert(block);
                        written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                    } else {
                        run.get_or_insert(block);
                        continue;
                    }
                }
                if let Some(from) = run.take() {
                    protect(&mut g, from, block, &mut written);
                }
            }
            return Some(written);
        }
    }
    // Slow path (first encounter): `start` is itself a mapped view, or if
    // `start` is an unmapped hole (common in 2 GiB sparse shadows), jump past
    // leading holes via `VirtualQuery` to find the first committed page.
    let probe = if mapped(start) {
        start
    } else {
        first_mapped(start, end)?
    };
    let check_each = probe != start;
    install();
    let mut g = guards();
    if !g.pins.is_empty() {
        let now = std::time::Instant::now();
        g.pins.retain(|&(_, _, until)| until > now);
    }
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    for block in first..=last {
        let open = block < last
            && !g.clean.contains(&block)
            && !g.holes.contains(&block)
            && (!check_each || mapped(block * BLOCK));
        if open {
            if g.pinned(block) {
                // The kernel may be writing it: left writable, reported
                // written at every check until the pin ends.
                written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                // No range holding it is clean meanwhile (`region_epoch`).
                bump_region_epochs(block * BLOCK, (block + 1) * BLOCK);
            } else if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
                written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block, &mut written);
        }
    }
    Some(written)
}

/// Scans `[start, end)` jumping over unmapped holes; returns the first
/// committed address if it is a mapped view, or `None` if `[start, end)` has
/// no committed mapped view (or starts with committed non-view memory).
fn first_mapped(start: usize, end: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut at = start;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries our own address space without dereferencing `at`.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.RegionSize == 0 {
            return None;
        }
        if info.State == MEM_COMMIT {
            return (info.Type == MEM_MAPPED).then_some(at);
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at {
            return None;
        }
        at = next;
    }
    None
}

/// Whether the current allocation at `at` is a mapped view. Asking only for
/// the allocation type avoids enumerating pages with identical protection,
/// which can span an entire large view after its blocks were guarded.
fn mapped(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_MAPPED, MEMORY_BASIC_INFORMATION, MemoryRegionInfo, QueryVirtualMemoryInformation,
        VirtualQuery, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: both queries describe an address without dereferencing it. The
    // output buffers match their information classes.
    let queried = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    };
    if queried != 0 {
        // These are the Windows SDK's named bitfields in declaration order.
        // windows-sys exposes only the union's raw bits.
        const PRIVATE: u32 = 1 << 0;
        const MAPPED_DATA_FILE: u32 = 1 << 1;
        const MAPPED_IMAGE: u32 = 1 << 2;
        const MAPPED_PAGE_FILE: u32 = 1 << 3;
        const MAPPED_PHYSICAL: u32 = 1 << 4;
        // SAFETY: this union was filled by the successful query above.
        match unsafe { region.Anonymous.Flags } {
            MAPPED_DATA_FILE | MAPPED_PAGE_FILE | MAPPED_PHYSICAL => return true,
            PRIVATE | MAPPED_IMAGE => return false,
            _ => {}
        }
    }
    // Unknown flag combinations and failed allocation queries keep the
    // VirtualQuery type check, including for a free address.
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: the output buffer has the required layout and size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.Type == MEM_MAPPED
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, recording unmapped blocks in `g.holes`
/// so subsequent checks neither retry `VirtualProtect` nor fault copying them.
fn protect(g: &mut Guards, from: usize, to: usize, written: &mut Vec<usize>) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, written: &mut Vec<usize>, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
        written.extend(from * BLOCK / PAGE..to * BLOCK / PAGE);
    };
    if set(from, to) {
        note(g, written, from, to);
    } else {
        for block in from..to {
            if set(block, block + 1) {
                note(g, written, block, block + 1);
            } else {
                g.holes.insert(block);
            }
        }
    }
    if !view_guard() {
        return;
    }
    // The same pages at their other views: read-only too, so that a write
    // through one of those faults. They are not noted clean: the copies
    // taken at those views have not taken the pages, and their next check
    // must report them (a block read-only and not clean is reported).
    // One look at the views for the whole run, and no list made for a page
    // with one view (most of them): this runs under the guards' lock, which
    // every copy's check waits for (6 % of the submitting thread's time when
    // it asked block by block, run p1).
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(views) = views.as_ref() else { return };
    for block in from..to {
        let shown = views.shown(block * BLOCK);
        if shown.len() < 2 || shown.len() > MANY_VIEWS || !g.protected.contains(&block) {
            continue;
        }
        for &at in shown {
            let alias = at / BLOCK;
            g.aliased.insert(alias);
            if alias != block && !g.protected.contains(&alias) && !g.pinned(alias) {
                let mut old = 0;
                // SAFETY: title view memory; only its protection changes.
                if unsafe { VirtualProtect(at as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
                    g.protected.insert(alias);
                }
            }
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same pages at their
/// other views are opened too: every guard over them must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for at in aliases(block * BLOCK) {
            if at / BLOCK != block {
                open_here(g, at, at + BLOCK);
            }
            g.aliased.remove(&(at / BLOCK));
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .collect();
    if blocks.is_empty() {
        return false;
    }
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    // Runs of blocks next to each other in one call (fewer kernel
    // transitions and TLB shootdowns). One call cannot span two views: the
    // title's page mappings are one view a page, and a run over several
    // failed whole, leaving blocks read-only that were no longer noted
    // guarded (a kernel read into them then failed and the title stopped).
    // So a run that fails is given back block by block.
    let give = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: blocks we made read-only; only their protection changes.
        let whole =
            unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READWRITE, &mut old) } != 0;
        if !whole && to - from > 1 {
            for block in from..to {
                // SAFETY: as above, one block.
                unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
            }
        }
    };
    let mut run_start = blocks[0];
    let mut run_end = run_start + 1;
    for &block in &blocks[1..] {
        if block == run_end {
            run_end = block + 1;
        } else {
            give(run_start, run_end);
            run_start = block;
            run_end = block + 1;
        }
    }
    give(run_start, run_end);
    bump_region_epochs(start, end);
    true
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / BLOCK, (at + len).div_ceil(BLOCK));
    // Textures placed where the write lands, at any view of its pages.
    let mut stale = vec![(at, at + len)];
    {
        let mut g = guards();
        let until = std::time::Instant::now() + PIN;
        g.pins.push((first, last, until));
        open(&mut g, at, at + len);
        if view_guard() {
            // The same pages at their other views: the write lands in them
            // too, so their copies must take them again, now and when it
            // has landed.
            for block in first..last {
                let shown = aliases(block * BLOCK);
                if shown.len() < 2 || shown.len() > MANY_VIEWS {
                    continue;
                }
                for alias in shown {
                    if alias / BLOCK != block {
                        g.pins.push((alias / BLOCK, alias / BLOCK + 1, until));
                        open_here(&mut g, alias, alias + BLOCK);
                        stale.push((alias, alias + BLOCK));
                    }
                }
            }
        }
    }
    // Uploaded again at their next use: what the GPU wrote into a texture's
    // memory through another view of it (a buffer) and we copy back reached
    // the texture only when its turn to be looked at again came, if ever
    // (tattoos and skin of noise on people: right with every texture looked
    // at at every use, VICTORY_TEXTURES=always).
    for (from, to) in stale {
        crate::resource::textures_rewritten(from as u64, to as u64);
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded, and any placed
/// textures backed by that range are evicted.
pub fn title_memory_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    {
        let mut g = guards();
        let blocks: Vec<usize> = g
            .protected
            .range(at / BLOCK..end.div_ceil(BLOCK))
            .collect();
        let holes: Vec<usize> = g
            .holes
            .range(at / BLOCK..end.div_ceil(BLOCK))
            .collect();
        let changed = !holes.is_empty() || !blocks.is_empty();
        for block in holes {
            g.holes.remove(&block);
        }
        for block in blocks {
            g.protected.remove(&block);
            g.clean.remove(&block);
            // A view mapped a moment ago may already have been made
            // read-only as another view of a guarded page (between the
            // runtime's `title_memory_view` and this call): it is given
            // back, or the title's next write there would fault with no
            // guard to answer for it.
            let mut old = 0;
            // SAFETY: only the protection of title view memory changes; the
            // call fails harmlessly on memory that is no longer a view.
            unsafe {
                windows_sys::Win32::System::Memory::VirtualProtect(
                    (block * BLOCK) as *const _,
                    BLOCK,
                    windows_sys::Win32::System::Memory::PAGE_READWRITE,
                    &mut old,
                )
            };
        }
        let aliased: Vec<usize> = g.aliased.range(at / BLOCK..end.div_ceil(BLOCK)).copied().collect();
        for block in aliased {
            g.aliased.remove(&block);
        }
        if changed {
            bump_region_epochs(at, end);
        }
    }
    crate::shadow::remapped(at as u64, end as u64);
    crate::resource::evict_placed_textures(at as u64, end as u64);
    crate::descriptor::host::bindless_memory_changed(at as u64, end as u64);
}

/// The title committed memory `[at, at + len)` it had reserved.
pub fn title_memory_committed(at: usize, len: usize) {
    if len != 0 {
        crate::shadow::filled(at as u64, (at + len) as u64);
    }
}

/// The runtime wrote new contents into title memory `[at, at + len)` (a
/// streamed file arrived): textures placed there are uploaded again at their
/// next use, not when their turn to be looked at comes.
pub fn title_memory_written(at: usize, len: usize) {
    if len != 0 {
        bump_region_epochs(at, at.saturating_add(len));
        // The runtime writes through a view of its own, which neither the
        // kernel's write watch nor our guards see: the pages are noted
        // written here, or host copies of buffers made before the file
        // arrived kept what the memory held then for good (meshes read as
        // zeros: walls, floors and bodies missing, others each launch).
        {
            let mut history = WRITTEN.write().unwrap_or_else(|e| e.into_inner());
            let now = ASKED.fetch_add(1, Ordering::Relaxed) + 1;
            for page in at / PAGE..(at + len).div_ceil(PAGE) {
                history.insert(page, now);
            }
        }
        crate::resource::textures_rewritten(at as u64, (at + len) as u64);
    }
}

/// Opens a guarded block at `at` (or returns `true` if another thread already
/// made it read-write). Called both by our own VEH handler and by
/// `host::safe_copy`'s VEH handler on write faults.
pub(crate) fn try_open_guarded(at: usize) -> bool {
    let mut g = guards();
    if open(&mut g, at, at + 1) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardFaults, 1);
        return true;
    }
    if writable(at) {
        return true;
    }
    // Read-only, in a view we were told of, and not noted guarded: only we
    // make title views read-only, so it is ours to give back.
    if view_guard() && !aliases(at).is_empty() {
        let mut old = 0;
        let block = at - at % BLOCK;
        // SAFETY: only the protection of title view memory changes.
        let opened = unsafe {
            windows_sys::Win32::System::Memory::VirtualProtect(
                block as *const _,
                BLOCK,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
                &mut old,
            )
        } != 0;
        if opened {
            bump_region_epochs(block, block + BLOCK);
            tracing::warn!(at = format_args!("{at:#x}"), "a read-only view block that no guard knew was opened");
        }
        return opened;
    }
    false
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        if try_open_guarded(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_console_pages_fold_kernel_pages_without_losing_reordered_answers() {
        let consecutive: Vec<_> = (0..2 * PAGE).step_by(KERNEL_PAGE).collect();
        assert_eq!(dirty_console_pages(&consecutive).collect::<Vec<_>>(), [0, 1]);
        let reordered = [2 * PAGE, PAGE, PAGE + KERNEL_PAGE, 2 * PAGE + KERNEL_PAGE, 0];
        assert_eq!(dirty_console_pages(&reordered).collect::<Vec<_>>(), [2, 1, 2, 0]);
        assert_eq!(dirty_console_pages(&[]).next(), None);
    }

    /// The bit set answers as a tree does, ranges across chunks included.
    #[test]
    fn block_set_matches_a_tree() {
        let mut set = BlockSet::new();
        let mut tree = BTreeSet::new();
        let mut x = 12345u64;
        for _ in 0..20_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let block = ((x >> 33) % 30_000) as usize + if x & 1 == 0 { 0 } else { 1 << 30 };
            if x & 6 == 0 {
                assert_eq!(set.remove(&block), tree.remove(&block));
            } else {
                assert_eq!(set.insert(block), tree.insert(block));
            }
            assert_eq!(set.contains(&block), tree.contains(&block));
        }
        set.extend(5000..9000);
        tree.extend(5000..9000);
        for r in [0..30_000, 4095..4097, 8191..12_289, (1 << 30)..(1 << 30) + 30_000, 0..usize::MAX / 2] {
            let ours: Vec<usize> = set.range(r.clone()).collect();
            let theirs: Vec<usize> = tree.range(r).copied().collect();
            assert_eq!(ours, theirs);
        }
    }
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn overlapping_pages_share_a_gate_across_stripe_wraps() {
        let wrap = QUERY_STRIPES * PAGES_PER_STRIPE;
        assert_eq!(stripe_mask(10, 10), 0);
        assert_eq!(stripe_mask(0, 1) & stripe_mask(PAGES_PER_STRIPE, PAGES_PER_STRIPE + 1), 0);
        assert_eq!(stripe_mask(0, wrap * 2), u64::MAX);
        for boundary in [PAGES_PER_STRIPE, wrap, wrap + PAGES_PER_STRIPE] {
            let crossing = stripe_mask(boundary - 1, boundary + 1);
            assert_eq!(crossing.count_ones(), 2);
            for page in [boundary - 1, boundary] {
                assert_ne!(crossing & stripe_mask(page, page + 1), 0);
            }
        }
    }

    #[test]
    fn overlapping_native_resets_publish_the_same_write_to_all_askers() {
        use std::sync::Barrier;
        // SAFETY: two pages of write-watch memory owned by this test.
        let ptr = unsafe { VirtualAlloc(std::ptr::null_mut(), 2 * PAGE,
            MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(ptr, 0);
        let (previous, _) = writes(ptr, ptr + 2 * PAGE, false).unwrap();
        // SAFETY: the test owns the committed allocation.
        unsafe { ((ptr + PAGE + 7) as *mut u8).write(1) };
        let gate = Barrier::new(2);
        let generation = |from, end| {
            gate.wait();
            let (_, changes) = writes(from, end, false).unwrap();
            changes.into_iter().find(|(page, _)| *page == ptr / PAGE + 1).unwrap().1
        };
        let (one, two) = std::thread::scope(|s| {
            let w = s.spawn(|| generation(ptr, ptr + 2 * PAGE));
            let other = generation(ptr + PAGE + 1, ptr + 2 * PAGE);
            (w.join().unwrap(), other)
        });
        assert_eq!(one, two);
        assert!(one > previous);
        // SAFETY: all workers are joined, and this is our allocation.
        unsafe { VirtualFree(ptr as *mut _, 0, MEM_RELEASE) };
    }

    /// Two views of the same two physical pages, told as the runtime tells
    /// them: the two addresses.
    fn two_views() -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: a pagefile section of two blocks, mapped read-write twice.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            assert!(one != 0 && two != 0);
            title_memory_view(one, 2 * BLOCK, section as usize, 0);
            title_memory_view(two, 2 * BLOCK, section as usize, 0);
            (one, two)
        }
    }

    fn drop_views(views: &[usize]) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, 2 * BLOCK);
            title_memory_view_gone(view, 2 * BLOCK);
            // SAFETY: our view.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        }
    }

    /// The title writes the pages of a guarded copy through another view of
    /// them: the guarded view's next check reports the page written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        let (guarded, other) = two_views();
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view; the handler opens it on the fault.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(Vec::new()));
        drop_views(&[guarded, other]);
    }

    /// Views that follow one another in memory and in their section make a
    /// stretch; a gap in either ends it.
    #[test]
    fn a_stretch_is_views_that_follow_in_memory_and_in_the_section() {
        let base = 0x7a00_0000_0000usize;
        let (section, other) = (0x51, 0x52);
        for k in 0..3 {
            title_memory_view(base + k * BLOCK, BLOCK, section, (k * BLOCK) as u64);
        }
        // Next in memory, not in the section.
        title_memory_view(base + 3 * BLOCK, BLOCK, section, (9 * BLOCK) as u64);
        // Next in the section's offsets, of another section.
        title_memory_view(base + 4 * BLOCK, BLOCK, other, (10 * BLOCK) as u64);
        let reach = 1 << 30;
        assert_eq!(
            stretch(base + BLOCK + 8, base + BLOCK + 16, reach),
            Some((base, base + 3 * BLOCK, section, 0))
        );
        assert_eq!(
            stretch(base + 3 * BLOCK, base + 3 * BLOCK + 1, reach),
            Some((base + 3 * BLOCK, base + 4 * BLOCK, section, (9 * BLOCK) as u64))
        );
        // Across the gap: no stretch holds it.
        assert_eq!(stretch(base + 2 * BLOCK, base + 3 * BLOCK + 1, reach), None);
        // Outside every view.
        assert_eq!(stretch(base + 6 * BLOCK, base + 6 * BLOCK + 1, reach), None);
        title_memory_view_gone(base, 5 * BLOCK);
        assert_eq!(stretch(base, base + 1, reach), None);
    }

    /// Both views have copies: a write through either is reported at both.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        let (one, two) = two_views();
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + 2 * BLOCK), Some(vec![pages(one, 0), pages(one, 1)]));
        // The other view was made read-only with the first, not noted clean.
        assert_eq!(guard(two, two + 2 * BLOCK), Some(vec![pages(two, 0), pages(two, 1)]));
        assert_eq!(guard(one, one + 2 * BLOCK), Some(Vec::new()));
        assert_eq!(guard(two, two + 2 * BLOCK), Some(Vec::new()));
        // SAFETY: the first view; the handler opens it on the fault.
        unsafe { ((one + 16) as *mut u8).write(7) };
        assert_eq!(guard(two, two + 2 * BLOCK), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + 2 * BLOCK), Some(vec![pages(one, 0)]));
        // SAFETY: the same byte through the other view.
        assert_eq!(unsafe { ((two + 16) as *const u8).read() }, 7);
        drop_views(&[one, two]);
    }

    /// A view mapped beside a guarded one is writable: the guarded view
    /// checks its pages again.
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let (section, guarded) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (section, MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize)
        };
        title_memory_view(guarded, 2 * BLOCK, section as usize, 0);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(Vec::new()));
        // SAFETY: a second view of the second page only.
        let other = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, BLOCK as u32, BLOCK).Value as usize };
        assert_ne!(other, 0);
        title_memory_view(other, BLOCK, section as usize, BLOCK as u64);
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(1)]));
        title_memory_remapped(other, BLOCK);
        title_memory_view_gone(other, BLOCK);
        // SAFETY: our view.
        unsafe {
            windows_sys::Win32::System::Memory::UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: other as *mut _ },
            )
        };
        drop_views(&[guarded]);
    }

    #[test]
    fn allocation_query_recognizes_file_views_and_rejects_nonviews() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
        };
        let file = tempfile::tempfile().unwrap();
        file.set_len(PAGE as u64).unwrap();
        // SAFETY: a valid, open file and a read-only mapping of its contents.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: an open section containing at least one page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_READ, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        assert!(mapped(view.Value as usize + 13));
        // SAFETY: our successful mapping and its section handle.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(section);
        }
        assert!(!mapped((&raw const file) as usize)); // private stack allocation
        assert!(!mapped(mapped as *const () as usize)); // executable image
        assert!(!mapped(0)); // no allocation to query
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }
}
