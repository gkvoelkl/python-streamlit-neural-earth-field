# Storing geodata in neural networks
# Chapter 16 · Gaussian splatting

*[Deutsche Fassung](../de/16-gaussian-splatting.md) · Translated from the German original.*

![In a workshop in the evening, Chester, wearing a blue cap, sprays a jet of yellow oval blobs of paint onto a canvas with an airbrush. The canvas shows an old town painted from above: a wide blue river, a stone arch bridge, red roofs and two church towers. Next to Chester stands Aiko, the glowing neural network, spreading its arms. On the right, on a table, a translucent model of a Gothic cathedral with two towers glows.](../img/kapitel-16-chester.jpg)

**Contents:**

* Another way to store geodata: not a network, but many small Gaussians
* The aerial photo from chapter 14 as a sum of Gaussians – compared with the network and with JPEG
* The buildings from chapter 15 as Gaussians in space
* What the well-known 3D Gaussian splatting does in addition


## Starting point

Since chapter 1, everything has been in a neural network. Every place, every pixel, every voxel passes through the same weights. When training changes a weight, the result changes a little everywhere.

In chapter 14 this showed a limit: the network draws the aerial photo softly. Many small things that occur in only one place – a car, a roof edge, a tree – the network has to describe with weights that are needed for all other pixels at the same time.

This chapter tries the opposite: you store the image as many small building blocks, and each block is responsible only for its surroundings. The blocks are **Gaussians**. The method is called **Gaussian splatting**. It became known in 2023 through Kerbl and colleagues, who use it to display whole scenes from photos in real time [1]. Here, first, the simplest form: an image in the plane [2].

## What a Gaussian is

A Gaussian is a soft blob: strongest in the middle, evenly weaker towards the outside, without a hard edge [3]. In the plane it has the shape of an ellipse. It is described by **eight numbers**:

| Numbers | Meaning |
|---:|---|
| 2 | the centre: easting and northing |
| 2 | two widths: how far the blob reaches in one direction and in the other |
| 1 | a rotation: which way the ellipse points |
| 3 | a colour: red, green and blue |

The image is the **sum** of all Gaussians. Each Gaussian gives its colour to the pixels near it, fully in the middle, less and less further out. Where Gaussians overlap, the colours add up. Applying blobs like this is called *splatting*.

![Three panels, each with one glowing Gaussian on a black background, the centre as a yellow dot and one width as a dashed ellipse: a round, large green Gaussian, a long, thin red Gaussian and a long blue Gaussian turned by 40°. Next to them their sum: where the Gaussians overlap, the colours mix into yellow and turquoise. Below, a box: one Gaussian = 8 numbers, centre (2), two widths (2), rotation (1), colour red, green, blue (3).](../img/en/splat-glocken.svg)

The difference from the neural network lies in who is responsible for what. A Gaussian only acts on a small window of pixels around its centre. When training changes a Gaussian, something changes only there. And a Gaussian can find its own shape: long and thin at an edge, round and large on an even area.

## How the Gaussians learn

Training works as in chapter 2, only the weights are now the numbers of the Gaussians:

1. **Start:** The Gaussians lie on a regular grid over the tile, each with the colour of the pixel below its centre.
2. **One step:** All Gaussians are painted onto the image, and the result is compared with the aerial photo. Then all numbers are improved a little: centres move, widths grow or shrink, ellipses turn, colours change.
3. **Repeat** until the image fits.

![Three 100 m × 100 m details at the Stone Bridge side by side. On the left the start: the Gaussians lie as equal yellow circles on a regular grid, the image below is blurred and checkered. In the middle after 1 000 steps: the circles have moved, many have been stretched and turned into ellipses; along the bright bridge on the right and along the riverside path at the bottom they lie close together and thin in the direction of the edge, and the image below shows meadow, trees, water and bridge. On the right the original as the goal of training.](../img/en/splat-training.svg)

Unlike with the networks, each step compares the **whole** image, not just a sample of pixels. That is possible because each Gaussian only has to compute its small window – up to three times its width. To keep the window small, a Gaussian may become at most a good half as wide (0.6 times) as the spacing of the Gaussians at the start.

To keep the comparison with chapter 14 fair, there are three presets with the same file size as the three networks. As there, each number is stored in float16, that is with 2 bytes:

| Preset | Gaussians | File | Network from chapter 14 |
|---|---:|---:|---:|
| *Small* | 10 201 | 159 KB | 162 KB |
| *Medium* | 41 209 | 644 KB | 644 KB |
| *Large* | 164 025 | 2.5 MB | 2.5 MB |

## Result: the aerial photo

Measured with the same tile as in chapter 14, 1 250 × 1 250 pixels, on an Apple M4 (MPS), one run each. The PSNR of the whole image is given; the values for the network come from chapter 14.

| Preset | Gaussians, 1 000 steps | Gaussians, 3 000 steps | Network, 1 000 steps | Network, 3 000 steps | JPEG of the same size |
|---|---:|---:|---:|---:|---:|
| *Small* | **22.6 dB** | **23.4 dB** | 21.5 dB | 23.0 dB | 27.9 dB |
| *Medium* | **25.9 dB** | **27.4 dB** | 23.7 dB | 26.1 dB | 36.3 dB |
| *Large* | **30.9 dB** | **33.3 dB** | 24.6 dB | 27.9 dB | 38.6 dB (only 829 KB) |

![Chart: PSNR in dB against file size, logarithmic from 50 KB to beyond 4.5 MB. The grey JPEG line rises from 22 dB at 53 KB to 38.6 dB at 829 KB. At 160 KB, 644 KB and 2.5 MB there are two pairs of marks each for 1 000 and 3 000 steps: open circles for the network, filled diamonds for the Gaussians. The diamonds always lie above the circles, most clearly at 2.5 MB with 30.9 and 33.3 dB against 24.6 and 27.9 dB. All marks lie below the JPEG line.](../img/en/splat-messung.svg)

Three things stand out:

- **The Gaussians beat the network at every size.** With *Small* the lead is small, a good 1 dB. The more Gaussians, the larger it becomes: with *Large* it is 6 dB after 1 000 steps – the error is only half as large. Many small building blocks can describe many small things where they are; a network has to tune all its weights at once for that.
- **JPEG stays ahead.** Even *Large* is still almost 8 dB behind a JPEG a third of its size. JPEG has been tuned to photos for more than 30 years and additionally packs its numbers losslessly. The Gaussians, by contrast, simply store each number with 2 bytes. With exactly such Gaussians, Zhang and colleagues come close to the network methods from research when they round and pack the numbers cleverly [2].
- **Training takes more time.** 1 000 steps take 6 to 8 minutes for all three presets – about the same, because every step paints the whole image. The *Small* network needs 45 seconds for that. Only with *Large* are both roughly equally fast.

### What it looks like

Here too, the detail at the Stone Bridge shows the difference:

![Four details side by side, each with the Stone Bridge as a vertical bright strip on the right, piers in the water, trees and a meadow at the top left. On the left the original, sharp. Next to it the Small network with 21.5 dB: softly blurred, the piers as round bumps. Then the Small Gaussians with 22.6 dB: also soft, but the bridge, the riverside path and the paths on the meadow are sharper, and the trees consist of blobs. On the right the JPEG with 27.9 dB: sharp, but broken into small blocks.](../img/en/splat-bruecke.svg)

The Gaussians make a third kind of error, next to the blur of the network and the blocks of JPEG: the image looks **blotchy**. Even areas such as the water of the Danube are captured well. Edges, by contrast, consist of many oval blobs side by side. The bridge and the riverside paths are sharper than with the network, because there the Gaussians lie closer together and stretch in the direction of the edge.

The app shows this directly: for a detail it draws each Gaussian as an ellipse. The Gaussians never become really long and thin – their width is limited so that the window stays small.

![The detail at the Stone Bridge, strongly enlarged and softly blurred, with each Gaussian drawn on top as a thin yellow ellipse. On the water and the meadow, almost round ellipses lie evenly side by side. On the right along the bright bridge and at the bottom along the riverside path, the ellipses are slightly stretched in the direction of the edge and lie closer together.](../img/app-splat-ellipsen.png)

## Buildings as Gaussians in space

The same idea works in three dimensions. Instead of colours, the Gaussians now add up **how much building** there is at a place. Where the sum is greater than ½, a voxel counts as inside, otherwise as outside. The surface of the buildings is where the sum is exactly ½ – like the 0 line of the distance in chapter 15.

A Gaussian in space again has **eight numbers**: the centre (three numbers), three widths, a turn about the vertical axis and a strength. One turn is enough, because the walls in the LoD2 city model are always vertical. The three presets again have exactly the file size of the networks from chapter 15: 10 320, 41 120 and 164 224 Gaussians.

One difference from the network has to be stated openly: the Gaussians **start inside the buildings**, at randomly drawn voxels. The network in chapter 15, by contrast, started without any knowledge of the buildings. This start is common in Gaussian splatting – the real 3D method begins with a point cloud computed from the photos [1]. But it explains part of the lead.

Measured with the 1 m grid on an Apple M4 (MPS), one run each (seed 0). The values for the network come from chapter 15 (limit ±5 m, mean of three runs). "Start" is the IoU of the Gaussians before the first training step.

| Preset | File | Gaussians: start | Gaussians: 300 steps | Offset median / 90 % | Network: 1 000 steps | Network: 3 000 steps |
|---|---:|---:|---:|---:|---:|---:|
| *Small* | 161 KB | 0.71 | **0.93** | 0.33 m / 0.53 m | 0.40 | 0.70 |
| *Medium* | 643 KB | 0.80 | **0.97** | 0.15 m / 0.50 m | 0.89 | 0.95 |
| *Large* | 2.5 MB | 0.87 | **1.00** | 0.09 m / 0.28 m | 0.93 | 0.98 |

*Medium* with 1 000 instead of 300 steps reaches an IoU of 0.99 and an offset of 0.11 m / 0.42 m. Computing time for 300 steps: *Small* a good 7 minutes, *Medium* just under 10, *Large* 13 minutes; *Medium* with 1 000 steps 34 minutes.

Three things stand out:

- **The Gaussians capture the buildings much better than the network.** Even *Small*, at 0.93 after 300 steps, lies above the *Medium* network, four times its size, after 1 000 steps (0.89). *Large* gets practically every voxel right (IoU 0.997). For *Medium* and *Large* the offset is below half a metre and thus at the limit of what the 1 m grid can measure at all (chapter 15).
- **A good part of that is the start.** Even before the first step, the Gaussians reach an IoU of 0.71 to 0.87 – more than the *Small* network after 6 000 steps (0.79). The Gaussians are told at the start where buildings are; the network has to find out by itself. Training still brings a lot afterwards: for *Medium* the IoU rises from 0.80 to 0.97.
- **Buildings suit the Gaussians.** Houses are blocks with vertical walls. A stretched Gaussian that turns about the vertical axis fits well into a house wall; many of them fill a block. A network has to assemble the same edges from waves that span the whole section.

For comparison with chapter 15, the same viewpoint close to the cathedral, *Medium* with 300 steps (IoU 0.97):

![At the top the numbers Volume overlap (IoU) 0.97, Surface offset (median) 0.2 m, Surface offset (90 %) 0.5 m and Gaussians vs CityGML 642.5 KB · 12 : 1. Below, two 3D views of the same houses around the cathedral from the same viewpoint. On the left the city model with smooth walls, the two cathedral towers and a thin ridge turret. On the right the Gaussians: hardly distinguishable from the city model, the edges only slightly softer; a short piece of the ridge turret is left, and a tiny dot floats above one tower tip.](../img/app-splat-gebaeude.png)

Compared with the network in chapter 15, the walls are straight, the edges hardly rounded, and the cathedral towers keep their shape.

## What real 3D Gaussian splatting does in addition

The method of Kerbl and colleagues starts from photos, not from a finished grid [1]. This leads to three differences from this chapter:

- **The Gaussians are seen from the side.** For each photo it is computed how the Gaussians look from the camera. For that they are sorted by distance, and nearer ones partly hide farther ones. In this chapter a simple sum is enough, because the aerial photo comes from above and the buildings are given as a voxel grid.
- **The colour depends on the viewing direction.** A roof shines from one direction and not from another. For this, each Gaussian stores its colour as a sum of spherical harmonics – the same waves on the sphere as in chapter 6.
- **There are millions of Gaussians.** Kerbl and colleagues work with 1 to 5 million Gaussians per scene. During training, Gaussians are split where details are missing and removed where they are superfluous.

For whole cities from aerial photos, this is active research. The main difficulty is memory: a city needs so many Gaussians that they have to be divided into blocks [4].

## What it costs

| | Size |
|---|---:|
| Aerial photo, uncompressed (1 250 × 1 250 pixels) | 4.5 MB |
| Gaussians *Small* / *Medium* / *Large* | 159 KB / 644 KB / 2.5 MB |
| JPEG with the PSNR of the *Medium* Gaussians (27.4 dB) | under 160 KB |
| CityGML of the 226 buildings (chapter 15) | 7.3 MB |
| Gaussians for the buildings *Small* / *Medium* / *Large* | 161 KB / 643 KB / 2.5 MB |

As compression for aerial photos, the Gaussians in this simple form pay off as little as the network: JPEG reaches the same quality at a quarter of the size. What the Gaussians offer is something else. They are fast to draw, because each only needs its small window – Zhang and colleagues draw an image more than a thousand times per second [2]. And you can look at, move or delete each Gaussian individually. In a network there is no weight that is "the bridge".

## Try it out

The **Gaussian Splatting** tab needs the data from the *Orthophoto* and *Buildings 3D* tabs and a lot of computing time. It therefore only runs in the local version of the app, not online.

### Aerial photo

At the top, choose **Aerial photo (2D)**.

| Setting | Value | Meaning |
|---|---|---|
| *Pixels* | 1250 × 1250 · 0.8 m | resolution, as in chapter 14 |
| *Gaussians* | Small · 10,201 | one of the three presets from the table above |
| *Steps* | 1000 | number of training steps |
| *Device* | auto | graphics chip, if available |

![The Gaussian Splatting tab: at the top the choice Store with Aerial photo (2D) and Buildings (3D), below it a short text and the settings Pixels 1250 × 1250 · 0.8 m, Gaussians Small · 10,201, Steps 1000 and Device auto. Below them the line 1,562,500 pixels × 3 colours = 4.5 MB uncompressed · 10,201 Gaussians × 8 numbers = 159.4 KB in float16 (about the size of the Small network in the Orthophoto tab) and the attribution, below that the Train button.](../img/app-splat-einstellungen.png)

Click **Train**. After about 6 minutes, the PSNR of the Gaussians, the file size and the PSNR of a JPEG of the same size appear. If you have trained the network of the same size in the *Orthophoto* tab, its PSNR is shown as well. Below come the original, the Gaussians and the JPEG side by side, then a detail with the error image and at the very bottom the Gaussians as ellipses.

![The tab after training: at the top PSNR of the Gaussians 22.6 dB, File size (float16) 159.4 KB, Compression ratio 29 : 1 and JPEG of the same size (quality 26) 27.9 dB. Below, the whole tile three times side by side: Original (DOP40), Gaussians and JPEG, 158.1 KB. In the Gaussians the roofs of the old town are soft and blotchy, while the river, bridges and islands are easy to make out.](../img/app-splat-ergebnis.png)

### Buildings

At the top, choose **Buildings (3D)**, then *Voxel size* 1 m, *Gaussians* Medium and *Steps* 300, and click **Train**. After just under 10 minutes, training is finished. The app then shows the IoU and the surface offset, and the buildings as a 3D view next to the city model.

## Ideas to explore further

- In the detail with the ellipses, compare the cathedral, the bridge and the town hall. Where do the Gaussians become long and thin, where round?
- Train *Small* with 500 × 500 pixels. The Gaussians are then just as large relative to the image, but there are fewer details. How close do they get to JPEG?
- Train the *Large* network in the *Orthophoto* tab and the *Large* Gaussians in the *Gaussian Splatting* tab. Which errors do both make in the same place?
- Look in the detail with the ellipses at how far the Gaussians have moved from their starting grid during training. Do they gather where there are many details?
- How would you store an elevation model of the whole Earth with Gaussians? What happens at the seam at ±180° (chapter 4)?

## Further reading

- [1] Kerbl et al.: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079), ACM Transactions on Graphics 2023 – scenes from photos as millions of 3D Gaussians, displayed in real time ([project page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/))
- [2] Zhang et al.: [GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](https://arxiv.org/abs/2403.08551), ECCV 2024 – images as a sum of 2D Gaussians with 8 numbers per Gaussian
- [3] Wikipedia: [Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function) – the bell curve and its width
- [4] HUG: [Hierarchical Urban Gaussian Splatting with Block-Based Reconstruction for Large-Scale Aerial Scenes](https://arxiv.org/abs/2504.16606), 2025 – whole urban areas from aerial photos
- Dupont et al.: [COIN: COmpression with Implicit Neural representations](https://arxiv.org/abs/2103.03123), 2021 – the counterpart with a network, as in chapter 14

---

[Overview](README.md) · Back: [Chapter 15 · Buildings as a 3D field](15-gebaeude-als-3d-feld.md) · Next: [Appendix · Measurements](anhang-messungen.md)

*This tutorial was created by Gerhard Völkl using Claude Code (Anthropic). It is meant to help you learn more about storing geodata in neural networks. Suggestions for improvement are welcome at gkvoelkl@nelson-games.de.*
