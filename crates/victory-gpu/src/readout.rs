//! Performance readout drawn directly into the back buffer. The independent
//! VICTORY_HUD switch overrides the historic VICTORY_FPS=0 display setting.

use std::sync::OnceLock;

use crate::hud_usage::Usage;
use windows::Win32::Foundation::RECT;

pub(crate) const BRAND: &str = "alwayskyleab";

/// Whether the readout is drawn.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        enabled(
            std::env::var("VICTORY_HUD").ok().as_deref(),
            std::env::var("VICTORY_FPS").ok().as_deref(),
        )
    })
}

fn enabled(hud: Option<&str>, fps: Option<&str>) -> bool {
    match hud {
        Some("1") => true,
        Some("0") => false,
        _ => fps != Some("0"),
    }
}

/// Prepared only when the measurements or swap-chain size change.
pub(crate) struct Panel {
    pub extent: (u32, u32),
    pub background: RECT,
    pub brand: Vec<RECT>,
    pub text: Vec<RECT>,
}

impl Panel {
    pub(crate) fn new(
        extent: (u32, u32),
        frame: Option<(f64, f64, f64)>,
        usage: Usage,
    ) -> Option<Self> {
        let percent =
            |value: Option<f64>| value.map_or_else(|| "N/A".to_owned(), |n| format!("{n:.0}%"));
        let memory = usage.ram_bytes.map_or_else(
            || "N/A".to_owned(),
            |n| format!("{:.1} GiB", n as f64 / (1_u64 << 30) as f64),
        );
        let (timing, draws) = frame.map_or_else(
            || ("N/A FPS   N/A MS".to_owned(), "N/A".to_owned()),
            |(fps, ms, draws)| (format!("{fps:.1} FPS   {ms:.1} MS"), format!("{draws:.0}")),
        );
        let lines = [
            timing,
            format!("GPU {}   CPU {}", percent(usage.gpu), percent(usage.cpu)),
            format!("RAM {memory}   DRAWS {draws}"),
        ];
        let columns = lines.iter().map(|s| s.len()).max()?.max(24) as i32;
        let width = columns * 6 + 5;
        let height = 42;
        let dot = ((extent.1 / 360).max(1) as i32)
            .min(extent.0 as i32 / (width + 8))
            .min(extent.1 as i32 / (height + 8));
        if dot < 1 {
            return None;
        }
        let left = 4 * dot;
        let top = 4 * dot;
        let background = RECT {
            left,
            top,
            right: left + width * dot,
            bottom: top + height * dot,
        };
        let origin = (left + dot, top + dot);
        let brand = rects(BRAND, origin, dot);
        let mut text = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            text.extend(rects(
                line,
                (origin.0, origin.1 + (12 + i as i32 * 9) * dot),
                dot,
            ));
        }
        Some(Self {
            extent,
            background,
            brand,
            text,
        })
    }
}

/// A glyph's seven rows, top first, of five dots each (bit 4 is the left).
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
        '1' => [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
        '2' => [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f],
        '3' => [0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e],
        '4' => [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
        '5' => [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
        '6' => [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
        '7' => [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
        '9' => [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c],
        'F' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
        'P' => [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
        'S' => [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e],
        'M' => [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11],
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'G' => [14, 17, 16, 23, 17, 17, 14],
        'N' => [17, 25, 25, 21, 19, 19, 17],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'W' => [17, 17, 17, 21, 21, 27, 17],
        'a' => [0, 0, 14, 1, 15, 17, 15],
        'b' => [16, 16, 30, 17, 17, 17, 30],
        'e' => [0, 0, 14, 17, 31, 16, 14],
        'i' => [4, 0, 12, 4, 4, 4, 14],
        'k' => [16, 16, 18, 20, 24, 20, 18],
        'l' => [12, 4, 4, 4, 4, 4, 14],
        's' => [0, 0, 15, 16, 14, 1, 30],
        'w' => [0, 0, 17, 17, 21, 21, 10],
        'y' => [0, 0, 17, 17, 15, 1, 14],
        '%' => [25, 26, 2, 4, 8, 11, 19],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        _ => [0; 7],
    }
}

/// The rectangles of `text` with its top left margin at `at` and dots of
/// `dot` pixels: one for each run of dots in a row.
pub(crate) fn rects(text: &str, at: (i32, i32), dot: i32) -> Vec<RECT> {
    let mut all = Vec::new();
    for (i, c) in text.chars().enumerate() {
        let left = at.0 + (1 + i as i32 * 6) * dot;
        for (row, bits) in glyph(c).into_iter().enumerate() {
            let top = at.1 + (1 + row as i32) * dot;
            let mut x = 0;
            while x < 5 {
                if bits & (0x10 >> x) == 0 {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < 5 && bits & (0x10 >> x) != 0 {
                    x += 1;
                }
                all.push(RECT {
                    left: left + start * dot,
                    top,
                    right: left + x * dot,
                    bottom: top + dot,
                });
            }
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_of_dots_is_one_rectangle() {
        // The top row of "F" is five dots.
        let all = rects("F", (10, 20), 4);
        assert_eq!(
            all[0],
            RECT {
                left: 14,
                top: 24,
                right: 34,
                bottom: 28
            }
        );
        assert!(rects(" ", (0, 0), 4).is_empty());
    }

    #[test]
    fn hud_fits_1080p_and_draws_all_labels_and_missing_values() {
        for ch in "alwayskyleab0123456789.FPSMGPUCRAiBDWN/%".chars() {
            assert!(glyph(ch).iter().any(|row| *row != 0), "{ch}");
        }
        for extent in [(1920, 1080), (640, 360), (320, 240)] {
            let p = Panel::new(extent, Some((0.1, 11446.2, 65535.0)), Usage::default()).unwrap();
            assert!(
                p.background.right <= extent.0 as i32 && p.background.bottom <= extent.1 as i32
            );
            assert!(
                p.text
                    .iter()
                    .chain(&p.brand)
                    .all(|r| r.left >= p.background.left
                        && r.right <= p.background.right
                        && r.top >= p.background.top
                        && r.bottom <= p.background.bottom)
            );
        }
        assert!(Panel::new((8, 8), None, Usage::default()).is_none());
        assert!(!enabled(None, Some("0")));
        assert!(enabled(Some("1"), Some("0")));
        assert!(!enabled(Some("0"), Some("30")));
    }
}
