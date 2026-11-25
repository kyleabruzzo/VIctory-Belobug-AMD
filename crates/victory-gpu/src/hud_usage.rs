//! Once-per-second readings for this process. No counter query runs on the
//! presentation thread. GPU percentage is the busiest WDDM engine on the
//! D3D12 device's adapter, measured from its scheduled running time.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows_sys::Wdk::Graphics::Direct3D::*;
use windows_sys::Win32::Foundation::{FILETIME, LUID};
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::Threading::{
    ALL_PROCESSOR_GROUPS, GetActiveProcessorCount, GetCurrentProcess, GetProcessTimes,
};

#[derive(Clone, Copy, Default)]
pub(crate) struct Usage {
    pub cpu: Option<f64>,
    pub gpu: Option<f64>,
    pub ram_bytes: Option<usize>,
}

pub(crate) struct Sampler(Arc<Mutex<(Instant, Usage)>>);

impl Sampler {
    pub(crate) fn start(adapter: windows::Win32::Foundation::LUID) -> Self {
        let latest = Arc::new(Mutex::new((Instant::now(), Usage::default())));
        let output = Arc::downgrade(&latest);
        let luid = LUID {
            LowPart: adapter.LowPart,
            HighPart: adapter.HighPart,
        };
        let _ = std::thread::Builder::new()
            .name("victory-hud-usage".into())
            .spawn(move || {
                // SAFETY: queries the number of logical processors in all groups.
                let processors = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) };
                let nodes = engine_count(luid).unwrap_or_default();
                let mut previous = (Instant::now(), cpu_ticks(), engine_ticks(luid, nodes));
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    let Some(output) = output.upgrade() else {
                        break;
                    };
                    let now = Instant::now();
                    let elapsed = now.duration_since(previous.0);
                    let cpu = cpu_ticks();
                    let gpu = engine_ticks(luid, nodes);
                    let usage = Usage {
                        cpu: previous
                            .1
                            .zip(cpu)
                            .and_then(|(a, b)| percent(a, b, elapsed, processors)),
                        gpu: busiest(&previous.2, &gpu, elapsed),
                        ram_bytes: working_set(),
                    };
                    previous = (now, cpu, gpu);
                    *output.lock().unwrap_or_else(|e| e.into_inner()) = (now, usage);
                }
            });
        Self(latest)
    }

    pub(crate) fn latest(&self) -> Usage {
        self.0
            .try_lock()
            .ok()
            .filter(|s| s.0.elapsed() < Duration::from_secs(3))
            .map_or_else(Usage::default, |s| s.1)
    }
}

fn cpu_ticks() -> Option<u64> {
    let [mut creation, mut exit, mut kernel, mut user] = [FILETIME::default(); 4];
    // SAFETY: current process and four writable FILETIMEs.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    let ticks = |t: FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
    (ok != 0).then(|| ticks(kernel) + ticks(user))
}

fn working_set() -> Option<usize> {
    let bytes = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let mut info = PROCESS_MEMORY_COUNTERS {
        cb: bytes,
        ..Default::default()
    };
    // SAFETY: current process and a properly sized writable structure.
    (unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut info, bytes) } != 0)
        .then_some(info.WorkingSetSize)
}

fn engine_count(adapter: LUID) -> Option<u32> {
    let mut query = D3DKMT_QUERYSTATISTICS {
        Type: D3DKMT_QUERYSTATISTICS_ADAPTER,
        AdapterLuid: adapter,
        ..Default::default()
    };
    // SAFETY: the query writes the result union for the requested adapter type.
    unsafe {
        (D3DKMTQueryStatistics(&raw mut query) >= 0)
            .then(|| query.QueryResult.AdapterInformation.NodeCount)
    }
}

fn engine_ticks(adapter: LUID, count: u32) -> Vec<Option<u64>> {
    (0..count)
        .map(|node| {
            // SAFETY: current-process handle; a successful process-node query writes its RunningTime in 100 ns units.
            unsafe {
                let mut query = D3DKMT_QUERYSTATISTICS {
                    Type: D3DKMT_QUERYSTATISTICS_PROCESS_NODE,
                    AdapterLuid: adapter,
                    hProcess: GetCurrentProcess(),
                    ..Default::default()
                };
                query.Anonymous.QueryProcessNode =
                    D3DKMT_QUERYSTATISTICS_QUERY_NODE { NodeId: node };
                (D3DKMTQueryStatistics(&raw mut query) >= 0)
                    .then(|| {
                        u64::try_from(query.QueryResult.ProcessNodeInformation.RunningTime).ok()
                    })
                    .flatten()
            }
        })
        .collect()
}

fn percent(start: u64, end: u64, elapsed: Duration, capacity: u32) -> Option<f64> {
    if capacity == 0 || elapsed.is_zero() {
        return None;
    }
    Some(
        (end.checked_sub(start)? as f64
            / (100_000.0 * elapsed.as_secs_f64() * f64::from(capacity)))
        .min(100.0),
    )
}

fn busiest(before: &[Option<u64>], after: &[Option<u64>], elapsed: Duration) -> Option<f64> {
    if before.is_empty() || before.len() != after.len() {
        return None;
    }
    // If an engine could not be measured, the busiest engine is unknown.
    before
        .iter()
        .zip(after)
        .try_fold(0.0_f64, |maximum, (&a, &b)| {
            Some(maximum.max(percent(a?, b?, elapsed, 1)?))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hud_rates_respect_cpu_capacity_overlapping_engines_and_missing_data() {
        let elapsed = Duration::from_secs(2);
        assert_eq!(percent(10, 20_000_010, elapsed, 8), Some(12.5));
        assert_eq!(
            busiest(&[Some(0); 2], &[Some(12_000_000), Some(8_000_000)], elapsed),
            Some(60.0)
        );
        assert_eq!(busiest(&[Some(2)], &[Some(1)], elapsed), None);
        assert_eq!(busiest(&[None], &[Some(7)], elapsed), None);
        assert_eq!(busiest(&[], &[], elapsed), None);
    }
}
