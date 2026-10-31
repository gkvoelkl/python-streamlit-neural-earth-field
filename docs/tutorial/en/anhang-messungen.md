# Storing geodata in neural networks
# Appendix · Measurements

*[Deutsche Fassung](../de/anhang-messungen.md) · Translated from the German original.*

**Contents:**

* All presets of the app and the networks from chapters 3 to 5 compared, measured under the same conditions
* What float16 costs for each network
* The measurements from the individual chapters in one place, with their settings

## How the measurements were made

To make the numbers comparable, all networks of the main table ran under the same conditions:

| | |
|---|---|
| **Computer** | Apple M4 with 32 GB memory, macOS 27.0.1; computed on the graphics chip (*Device* `auto` = MPS) |
| **Software** | Python 3.14.6, PyTorch 2.14.0 |
| **Data** | ETOPO 2022 (60 arc-seconds, ice surface), averaged to the standard grid 1080 × 2160 = 2 332 800 elevation values, raw size 4.4 MB (2 bytes per value) |
| **Training** | Default settings of the sidebar: 4 000 steps, 32 768 cells per step, learning rate 0.003 (rising slowly over the first 100 steps, then gradually falling to 2 % of it), *Area-weighted sampling* switched on |
| **Runs** | three runs per network with the seeds 0, 1 and 2 for the starting weights and the selection of cells; only the linear model with *L* = 40 once, because one run takes 5 minutes |
| **Evaluation** | after training on the **whole** grid, not just on a sample |
| **File size** | as in chapter 1: weights, normalisation, seed and context field where applicable, without the bookkeeping of the `.pt` file format (around 4 KB, chapter 10) |

The networks from chapter 3 have longitude and latitude as input and no encoding; the network from chapter 4 is the preset *Plain MLP (fails on purpose)* with (x, y, z). All other rows are the app's presets, unchanged.

## Main table

For the weighted RMSE the table shows the mean of the three runs, in brackets the smallest and the largest value. All other error measures and the duration are means. The error measures are explained in chapter 2, PSNR in chapter 9.

| Network | Chapter | Weights | File | Rate | Weighted RMSE | RMSE | MAE | Largest error | PSNR | R² | Duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 layers (longitude, latitude) | 3 | 3 | 20 bytes | 233 280 | 2 386 m (2 386–2 386) | 2 771 m | 2 254 m | 8 593 m | 15.4 dB | −0.092 | 5 s |
| 1 layer (longitude, latitude) | 3 | 513 | 2.0 KB | 2 265 | 1 425 m (1 390–1 457) | 1 491 m | 1 159 m | 8 307 m | 20.8 dB | 0.684 | 15 s |
| 4 layers (longitude, latitude) | 2, 3 | 50 049 | 196 KB | 23 | 706 m (685–731) | 648 m | 426 m | 7 977 m | 28.0 dB | 0.940 | 55 s |
| *Plain MLP (fails on purpose)*, (x, y, z) | 4 | 50 177 | 196 KB | 23 | 593 m (578–604) | 568 m | 373 m | 7 885 m | 29.2 dB | 0.954 | 58 s |
| *Fourier MLP, very small* | 5 | 10 945 | 43 KB | 107 | 372 m (368–378) | 358 m | 228 m | 5 070 m | 33.2 dB | 0.982 | 29 s |
| *Fourier MLP (recommended)* | 5 | 82 561 | 323 KB | 14 | 245 m (243–248) | 232 m | 141 m | 4 387 m | 37.0 dB | 0.992 | 74 s |
| *Fourier MLP, large* | 5 | 394 753 | 1 542 KB | 3.0 | 185 m (183–187) | 174 m | 102 m | 3 829 m | 39.4 dB | 0.996 | 275 s |
| *Spherical harmonic expansion (L = 40, linear)* | 6 | 1 682 | 6.6 KB | 693 | 631 m | 603 m | 405 m | 7 344 m | 28.7 dB | 0.948 | 305 s |
| *Spherical harmonics (L = 16)* | 6 | 86 785 | 339 KB | 13 | 241 m (240–242) | 227 m | 131 m | 4 332 m | 37.1 dB | 0.993 | 116 s |
| *SIREN (sine activation)* | 7 | 78 081 | 305 KB | 15 | 233 m (215–246) | 223 m | 148 m | 4 110 m | 37.3 dB | 0.993 | 112 s |
| *Fourier MLP + context field* | 8 | 82 689 | 355 KB | 13 | 186 m (185–187) | 179 m | 111 m | 3 701 m | 39.2 dB | 0.995 | 85 s |

## What the numbers show

**The path through the chapters pays off.** From chapter 3 to chapter 5 the weighted RMSE drops from 2 386 m via 706 m and 593 m to 245 m. The spherical coordinates from chapter 4 bring a good 100 m for the same network size, the Fourier features from chapter 5 almost another 350 m – with a somewhat larger file.

**The simplest model is worse than nothing.** Its R² is below zero. That means: a map that shows only the mean elevation everywhere would be closer to the original than the inclined plane. The reason is probably the weighting: the network learns with area-weighted sampling (chapter 2), but R² counts every cell equally – including the many small cells at the poles, where the plane is particularly far off.

**The context field is the largest single gain.** At 355 KB, *Fourier MLP + context field* is as accurate as *Fourier MLP, large* at 1 542 KB (186 m against 185 m) – at just under a quarter of the file size. This confirms the comparison from chapter 8.

**Networks of similar size are close together.** *Fourier MLP (recommended)*, *Spherical harmonics (L = 16)* and *SIREN* are between 305 KB and 339 KB in size and lie between 233 m and 245 m. The SIREN network is the most accurate on average but varies the most: its three runs are 31 m apart, those of the other two only 2 to 5 m.

**The linear model with L = 40 is tiny.** At 6.6 KB it reaches a rate of 693 and is more accurate than the network with 4 layers from chapter 3, which is 30 times as large. But the spherical harmonics up to degree 40 contain no wave shorter than about 1 000 km (chapter 6); it cannot get any finer.

**The runs are well reproducible.** For most networks the three runs are only a few metres apart. Differences of less than 10 m between two networks should therefore not be overrated if they come from a single run.

## float16

The same networks, rounded to float16 after training (chapter 10). Means of the three runs:

| Network | File float16 | Rate float16 | Weighted RMSE float32 | Weighted RMSE float16 |
|---|---:|---:|---:|---:|
| 0 layers (longitude, latitude) | 14 bytes | 333 257 | 2 386.4 m | 2 386.4 m |
| 1 layer (longitude, latitude) | 1.0 KB | 4 512 | 1 425.1 m | 1 425.1 m |
| 4 layers (longitude, latitude) | 98 KB | 47 | 706.0 m | 706.1 m |
| *Plain MLP (fails on purpose)* | 98 KB | 47 | 593.1 m | 593.1 m |
| *Fourier MLP, very small* | 21 KB | 213 | 371.9 m | 371.9 m |
| *Fourier MLP (recommended)* | 161 KB | 28 | 244.9 m | 244.9 m |
| *Fourier MLP, large* | 771 KB | 5.9 | 184.7 m | 184.7 m |
| *Spherical harmonic expansion (L = 40, linear)* | 3.3 KB | 1 384 | 630.9 m | 630.9 m |
| *Spherical harmonics (L = 16)* | 170 KB | 27 | 240.7 m | 240.7 m |
| *SIREN (sine activation)* | 153 KB | 30 | 233.3 m | 233.7 m |
| *Fourier MLP + context field* | 193 KB | 24 | 186.2 m | 186.2 m |

For no network does the weighted RMSE change by more than half a metre. Only for the SIREN network is there any difference at all, as described in chapter 10. For the file size, mean, standard deviation and seed are still counted at 4 bytes each; when rounding for the error measurement, they were also rounded to float16.

## Measurements from the chapters

Some chapters contain their own measurements for a particular question. They were computed with different settings from the main table, mostly in a single run:

| Chapter | Question | Settings |
|---|---|---|
| [5](05-fourier-features.md) | *spectral bias* on the Alpine profile: simple network against Fourier features after 50, 500 and 3 000 steps | only the profile at 46.4° north, not the whole Earth |
| [7](07-siren.md) | SIREN network with and without the initialisation after Sitzmann | only every second point of the Alpine profile trained, the others as a check |
| [8](08-kontextfeld.md) | context field at the same file size; resolution of the context field | 1080 × 2160, 4 000 steps; networks without context field a little wider for the same file size |
| [9](09-rate-und-fehler.md) | rate-distortion curve | 1080 × 2160, nine network widths from 16 to 256, 4 layers, σ = 8, 2 000 and 600 steps; error on 200 000 cells |
| [10](10-abfragen-und-exportieren.md) | float16 on the whole grid; queries at individual places; querying more finely against the grid 2160 × 4320 | 1080 × 2160, 4 000 steps |
| [11](11-einordnung-und-ausblick.md) | land or sea: share of the Earth's surface on the correct side of the 0 m line | network from chapter 10 |
| [12](12-kueste-als-abstandsfeld.md) | coast from the elevation network against the distance network; limiting the distance; Fourier MLP, SIREN and context field: correct side, coastal cells, coast offset | 1080 × 2160, 4 000 steps, distance to the Natural Earth coastline 1:50 million |

The numbers in the chapters therefore differ from the main table by a few metres. For example, the standard example in chapter 10 has a weighted RMSE of 245 m, *Fourier MLP + context field* 191 m instead of 186 m; chapter 8 gives 245 m and 185 m. Such differences are within the variation from run to run.

### Finer grid 2160 × 4320

From development there is a measurement on the app's finest grid, 2160 × 4320 (17.8 MB raw size). It was trained for only 1 500 steps, also on an Apple chip, once per network. The networks are not yet fully trained:

| Network | File | Rate | Weighted RMSE | PSNR |
|---|---:|---:|---:|---:|
| Fourier MLP, σ = 8 | 322 KB | 57 | 311 m | 35.3 dB |
| + context field 90 × 180 | 355 KB | 51 | 257 m | 36.9 dB |
| + context field 180 × 360 | 450 KB | 41 | 233 m | 37.8 dB |
| SIREN network | 305 KB | 60 | 258 m | 36.8 dB |

The rates are four times as high as on the standard grid, because the grid is four times as large while the network stays the same. The error is larger, because the finer grid contains more detail. In this short measurement the SIREN network is about level with the context field 90 × 180; on the standard grid with 4 000 steps the context field is clearly better (186 m against 233 m).

## Measuring it yourself

All networks of the main table can be recomputed in the app: choose the preset, leave *Steps* at 4 000 and click *Train*. The error measures then appear in the *Training* tab, file size and rate in the *Idea & Budget* tab. On a computer without a graphics chip a run takes considerably longer; for a first impression the grid 360 × 720 with 1 000 steps is enough. The online version of the app is not suitable for re-measuring: it computes with at most 540 × 1080 cells and shorter training.

A single run in the app will not give exactly the same numbers as here, because the starting weights depend on chance. But it should lie close to the given range.

---

[Overview](README.md) · Back: [Chapter 12 · The coast as a distance field](12-kueste-als-abstandsfeld.md)

*This tutorial was created by Gerhard Völkl using Claude Code (Anthropic). It is meant to help you learn more about storing geodata in neural networks. Suggestions for improvement are welcome at gkvoelkl@nelson-games.de.*
