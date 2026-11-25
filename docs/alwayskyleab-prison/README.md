# Lucia's prison cell on a Radeon RX 9070 XT

On 25 November 2026, `alwayskyleab` tested the Belobug-based runtime at revision `bc5f578` with a Ryzen 9 9900X, Radeon RX 9070 XT, 32 GB of system RAM, and Windows 10 build 19045. The test used the Series X profile, a 1920×1080 client window, the same prison checkpoint and a warmed shader cache. The streamed texture cap was 256 pixels for both captures. CPU-side changes in this revision preserve deferred texture checks, select sparse buffer tracking by range and combine repeated write notifications for the same console page.

| Run | Displayed frames | Duration | Displayed FPS | 95th-percentile frame interval |
| --- | ---: | ---: | ---: | ---: |
| First visit | 222 | 53.98 s | 4.11 | 527.7 ms |
| Separate launch | 215 | 54.02 s | 3.98 | 527.8 ms |

Both captures cover the rendered cell with Lucia visible. The fastest ten-second interval in the first trace was 5.5 FPS and the slowest was 2.4 FPS. Other scenes, long-session stability, and ray-traced lighting need their own measurements. The performance target is 15 displayed FPS.

The CSV files contain selected PresentMon 2.6.0 timing columns for the game's DXGI swapchain. Display timestamps are the sum of `TimeInMs` and `MsUntilDisplayed`. The first interval uses the preceding display change reported by PresentMon. The mean rate is the number of displayed intervals divided by their total duration.

[First visit](cell-first.csv) · [Separate launch](cell-repeat.csv)

## Branded renderer follow-up

Revision `80ac578` drew its performance panel directly into the captured game image. The prison yard and cell rendered on the same machine. With the 256-pixel cap removed, a fresh shader cache initially left the mission black while the renderer built hundreds of pipelines. The picture returned during that launch. After the runtime's pipeline queue and external compiler had finished, the cell displayed 202 frames in 54.17 seconds, averaging **3.73 FPS**. Its 95th-percentile frame interval was 548.8 ms. The texture settings and renderer revision both changed compared with the earlier runs. [Branded renderer frame timings](cell-branded.csv)

Revision `21786e4` retained the same shader translator and used adaptive texture limits. The reload reused the warmed shader cache and 2,427 early pipeline hints. The yard and cell rendered again. In the cell, 225 frames reached the display in 54.15 seconds, averaging **4.16 FPS**, with a 507.1 ms 95th-percentile interval. Ten-second intervals ranged from 3.5 to 4.9 FPS. The game still presented a 3840×2160 surface in the 1920×1080 window when the private settings copy requested 1080p. This single reload changed cache warmth and texture policy together. [Adaptive renderer frame timings](cell-adaptive.csv)
