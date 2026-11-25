# Lucia's prison cell on a Radeon RX 9070 XT

On 25 November 2026, `alwayskyleab` tested the Belobug-based runtime at revision `bc5f578` with a Ryzen 9 9900X, Radeon RX 9070 XT, 32 GB of system RAM, and Windows 10 build 19045. The test used the Series X profile, a 1920×1080 client window, the same prison checkpoint and a warmed shader cache. The streamed texture cap was 256 pixels for both captures. CPU-side changes in this revision preserve deferred texture checks, select sparse buffer tracking by range and combine repeated write notifications for the same console page.

| Run | Displayed frames | Duration | Displayed FPS | 95th-percentile frame interval |
| --- | ---: | ---: | ---: | ---: |
| First visit | 222 | 53.98 s | 4.11 | 527.7 ms |
| Separate launch | 215 | 54.02 s | 3.98 | 527.8 ms |

Both captures cover the rendered cell with Lucia visible. The fastest ten-second interval in the first trace was 5.5 FPS and the slowest was 2.4 FPS. Other scenes, long-session stability, and ray-traced lighting need their own measurements. The performance target is 15 displayed FPS.

The CSV files contain selected PresentMon 2.6.0 timing columns for the game's DXGI swapchain. Display timestamps are the sum of `TimeInMs` and `MsUntilDisplayed`. The first interval uses the preceding display change reported by PresentMon. The mean rate is the number of displayed intervals divided by their total duration.

[First visit](cell-first.csv) · [Separate launch](cell-repeat.csv)
