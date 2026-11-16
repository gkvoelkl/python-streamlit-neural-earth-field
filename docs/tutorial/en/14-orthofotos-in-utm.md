# Storing geodata in neural networks
# Chapter 14 · Orthophotos in UTM

*[Deutsche Fassung](../de/14-orthofotos-in-utm.md) · Translated from the German original.*

![Chester, wearing a blue cap, and Aiko, the glowing neural network, float in the basket of a colourfully striped hot-air balloon above an old town with red roofs. Chester holds a large camera, Aiko a square photo of river and houses. In the background a Gothic cathedral with two towers and an old stone bridge across a wide river; a delicate, glowing square grid lies over the landscape.](../img/kapitel-14-chester.jpg)

**Contents:**

* An aerial photo as a function: X and Y values as input, red, green and blue as the result
* Why the Earth is flat for one kilometre, and what UTM coordinates are
* How well does a neural network store a photo?


## Starting point

Chapters 1 to 13 were always about the whole Earth: a grid of longitudes and latitudes, a sphere, cells several kilometres across. Many geodata look quite different. They cover only a small area, but in great detail. The best-known example is **aerial photos**.

This chapter stores an aerial photo of one square kilometre in a neural network. Three things change compared with the previous chapters:

1. **Plane instead of sphere.** The network gets coordinates in metres on a plane, not a point on a sphere.
2. **Three outputs instead of one.** Instead of one elevation, three colours come out: red, green and blue.
3. **A completely different picture.** A photo of a town is far more detailed than an elevation model of the Earth.

## The orthophoto

When an aerial photo is taken, it is distorted – just like any other photo. At the edge you see houses a little from the side, and a slope appears compressed or stretched depending on the viewing angle. An **orthophoto** is an aerial photo that has been rectified with a terrain model so that it fits a map true to scale [1]. Each pixel then has a fixed position in metres. With it, you can measure the distance between two points.

![Two sections through the same terrain with a red house on the left and a hill on the right; blue points lie on the terrain at equal distances, below them a strip "image on the map". On the left the aerial photo: all rays run from the points to the camera at the top centre. Extended down to the plane, high points move outwards in the image – the points on the hill lie further apart in the strip, and the corners of the roof slide away from the house. On the right the orthophoto: every point goes straight down, the points in the strip are equally spaced, the roof exactly above the house.](../img/en/ortho-entzerrung.svg)

The Bavarian surveying administration has provided its orthophotos free of charge since 2023, under the CC BY 4.0 licence [2, 3]. This chapter uses **DOP40** (digital orthophoto, one pixel corresponds to 40 × 40 cm on the ground):

| Property | Value |
|---|---|
| Tile | 1 km × 1 km, old town of Regensburg |
| Pixels | 2 500 × 2 500, each 40 cm × 40 cm |
| Colours | red, green, blue, 1 byte each (0 to 255) |
| Capture | flight on 11 May 2025 |
| File | GeoTIFF, about 18 MB |

The tile contains the cathedral, the Stone Bridge, the Danube with two islands and hundreds of roofs.

![Aerial photo of the tile: at the top the houses of Stadtamhof, in the middle the Danube with two islands, crossed from top to bottom by the long Stone Bridge, on the right a second, narrow bridge. At the bottom the densely built old town with red roofs, narrow lanes and the cathedral. On the right, on the south bank, a large white building.](../img/ortho-kachel.jpg)

*Data: Bayerische Vermessungsverwaltung – www.geodaten.bayern.de, CC BY 4.0*

For training, the app scales the tile down to 1 250 × 1 250 pixels; each 2 × 2 pixels are averaged. A pixel then corresponds to 80 × 80 cm.

This makes the image smaller and training faster, but roofs, trees and cars are still easy to recognise. Uncompressed, that is 1 250 · 1 250 · 3 = 4 687 500 bytes, or **4.5 MB**.

## Plane instead of sphere

### Why the sphere does not matter here

Chapter 4 explained why the Earth must be a sphere for a network: longitudes and latitudes are distorted at the pole, and at the date line longitudes jump from +180° to −180°. For one kilometre, all of this is irrelevant. Over a distance of 500 m, the sphere deviates from a plane by only about **2 cm** – less than a tenth of a pixel. For such a small area, the Earth is practically flat.

### UTM: coordinates in metres

For maps in metres there are map projections. In Germany the official system is **ETRS89 / UTM** [4]. UTM divides the Earth into strips 6° wide, the *zones*. Each zone has its own central meridian, a north–south line in the middle. A place is described by two numbers in metres:

- The **easting** (E) is the distance from the central meridian. So that it is never negative, the central meridian counts as 500 000 m.
- The **northing** (N) is the distance from the equator.

Most of Bavaria lies in zone 32 with the central meridian at 9° east. Regensburg actually already lies in zone 33, but the surveying administration computes all of Bavaria in zone 32 [5]. The tile starts at the bottom left at **E = 726 000 m** and **N = 5 434 000 m**: 226 km east of the central meridian and 5 434 km north of the equator. That is also where the tile's file name comes from, `32726_5434`: zone 32, easting 726 km, northing 5 434 km.

![On the left a map of Central Europe with the North Sea and the Baltic at the top and the Adriatic at the bottom. Two vertical strips are coloured: zone 32 from 6° to 12° east in blue with the dashed central meridian at 9°, zone 33 from 12° to 18° east in orange with the central meridian at 15°. Regensburg, a red dot, lies just east of 12°, that is already in the orange strip. On the right the tile as an aerial photo, with the bottom left corner E 726 000 m, N 5 434 000 m and the top right corner E 727 000 m, N 5 435 000 m; below it tile 32726_5434: 1 000 m × 1 000 m, zone 32, easting 726 km, northing 5 434 km.](../img/en/ortho-utm.svg)

Even here it does not work entirely without distortion: a plane can never represent the sphere exactly. In Bavaria, distances on the UTM map are up to 12 cm per 100 m longer or shorter than in reality [5]. Within a one-kilometre tile, however, this factor is practically constant. The image is only very slightly stretched, evenly, and that does not matter for the network.

### What the neural network gets

The network gets the easting and the northing, both converted to −1 to 1: −1 at the left or bottom edge of the tile, +1 at the right or top edge. To do so, the app first subtracts the bottom left corner.

This step is more important than it looks. A network computes in float32, and float32 stores only about 7 digits. A northing such as 5 434 000.4 m, however, has 8. In float32, northings of this size can only be represented in steps of **50 cm** – coarser than a pixel. Only once the corner has been subtracted do small numbers remain that float32 stores accurately enough. The corner itself belongs to the network's file: two numbers of 8 bytes each.

Spherical coordinates (chapter 4) and spherical harmonics (chapter 6) are thus dropped. The Fourier features from chapter 5 remain: they work just the same in the plane, even more simply.

## Three colours instead of one elevation

The network has the same shape as the Fourier MLP of the earlier chapters. Only the last layer has three neurons instead of one: for red, green and blue. A function called *sigmoid* squeezes each output into the range 0 to 1; multiplied by 255, that gives the colour values. So the network can never output a colour below 0 or above 255.

The app has three presets:

| Preset | Frequencies | Layers × neurons | Weights | File (float16) | Ratio |
|---|---:|---|---:|---:|---:|
| *Small* | 128 | 4 × 128 | 82 819 | 162 KB | 28 |
| *Medium* | 256 | 4 × 256 | 329 475 | 644 KB | 7.1 |
| *Large* | 256 | 5 × 512 | 1 314 819 | 2.5 MB | 1.8 |

The file is calculated with float16, that is 2 bytes per weight (chapter 10), plus the corner of the tile and the seed of the Fourier features. *Small* is almost exactly as large as the standard example from chapter 1 with 82 561 weights and has the same ratio in float16: 28.

For the Fourier features, **σ = 20**. In the plane this means: a typical wave fits about 20 times along the edge of the tile, so it is about 50 m long. Because the frequencies are drawn at random, much shorter ones are included too.

Training works as in chapter 2: in each step the app draws 65 536 pixels at random and improves the weights so that the colours fit better. All pixels count equally – on a flat tile they are, after all, all the same size.

## Measures for images

### PSNR

For images, **PSNR** is the usual measure (chapter 9). The range here is 255, the range of a colour value. It is measured over all pixels and all three colours. For orientation:

| PSNR | mean error | Impression |
|---:|---:|---|
| 20 dB | about 26 levels | clearly blurred, colours distorted |
| 30 dB | about 8 levels | looks like the original at first glance |
| 40 dB | about 3 levels | hardly distinguishable from the original |

As in chapter 9: +6 dB means the error is half as large.

### The comparison: JPEG

For the elevations of the Earth there was no obvious comparison. For photos there is one: **JPEG**, the format of practically every digital camera [6]. JPEG, too, breaks an image down into waves: it divides the image into blocks of 8 × 8 pixels and describes each block as a sum of waves. It stores the long waves accurately, the short ones coarsely or not at all. A **quality** setting from 1 to 100 determines how coarsely.

For each network, the app searches for the highest JPEG quality whose file is not larger than the network. Both are thus compared at the same file size.

## Result

Measured with 1 250 × 1 250 pixels on an Apple M4 (MPS). *Small* with three runs per value (seeds 0 to 2; the mean is given, the runs were at most 0.2 dB apart), *Medium* after 1 000 steps likewise, otherwise one run.

| Preset | File | 1 000 steps | 3 000 steps | 6 000 steps | JPEG of the same size |
|---|---:|---:|---:|---:|---:|
| *Small* | 162 KB | 21.5 dB | 23.0 dB | 24.0 dB | **27.9 dB** (quality 26) |
| *Medium* | 644 KB | 23.7 dB | 26.1 dB | 27.9 dB | **36.3 dB** (quality 91) |
| *Large* | 2.5 MB | 24.6 dB | 27.9 dB | 31.7 dB | **38.6 dB** (quality 95, only 829 KB) |

Computing time: *Small* needs about 45 seconds for 1 000 steps, *Medium* about 3 minutes, *Large* about 7 minutes; for 6 000 steps, correspondingly 6, 15 and 44 minutes. A JPEG is done in milliseconds.

![Chart: PSNR in dB against file size, logarithmic from 50 KB to beyond 4.5 MB. The grey JPEG line rises from 22 dB at 53 KB through 28 dB at 158 KB to 38.6 dB at 829 KB. Below it, three vertical groups of three points each for the networks after 1 000, 3 000 and 6 000 steps: green Small at 162 KB with 21.5, 23.0 and 24.0 dB, blue Medium at 644 KB with 23.7, 26.1 and 27.9 dB, purple Large at 2.5 MB with 24.6, 27.9 and 31.7 dB. All points lie well below the JPEG line. On the far right an orange dashed line at 4.5 MB for the uncompressed image.](../img/en/ortho-messung.svg)

Three things stand out:

- **JPEG wins clearly.** At the same size, JPEG is always ahead: by almost 4 dB for *Small*, by over 8 dB for *Medium*. The 27.9 dB that *Medium* reaches after 6 000 steps, JPEG already manages with 158 KB – a quarter of the size. *Large*, at 2.5 MB, is more than half the size of the uncompressed image; JPEG does not reach that size even at quality 95.
- **Larger networks need more training.** From 1 000 to 6 000 steps, *Small* gains 2.5 dB, *Medium* over 4 dB, *Large* 7 dB. After 1 000 steps the three presets are only 3 dB apart, although *Large* has 16 times as many weights as *Small*. A large network can do more, but takes a long time to show it.
- **The small network soon runs out, the large one is far from done.** From 3 000 to 6 000 steps, *Small* gains only 1 dB, *Large* almost 4 dB. So the large network was not yet finished after 6 000 steps. Even so, at 31.7 dB it stays far behind JPEG: with a third of the size, JPEG already reaches 38.6 dB.

### What it looks like

The difference is visible at first glance. Here is a 150 m × 150 m detail at the Stone Bridge: on the left the original, in the middle *Small* after 1 000 steps, on the right the JPEG of the same size. As in the table, the values refer to the whole image: 21.5 dB for the network, 27.9 dB for the JPEG.

![Three details side by side, each with the Stone Bridge as a vertical light strip on the right, piers in the water, trees and a meadow at the top left. On the left the original, sharp, with individual tree crowns and paths. In the middle the network: softly blurred, the trees as dark green patches, the piers as round light bumps, the colours paler. On the right the JPEG: the trees and paths are easy to make out, but broken into small squares; block edges are visible at the bridge.](../img/en/ortho-bruecke.svg)

The two methods make quite different errors:

- **The network blurs.** It lacks the short waves. Roofs, tree crowns and cars run into each other; the bridge and the large streets remain recognisable. The colours are striking too: the red roofs of the old town become greyish brown. The network first learns a kind of average colour over many pixels and only later where exactly the red areas begin and end.
- **JPEG gets blocky.** Edges stay sharp, but with strong compression the 8 × 8 pixel blocks become visible, and within a block colours become flat.

### Why a photo is so much harder

Chapter 5 showed: a network learns long waves quickly and short ones only slowly. With the elevations of the Earth, this is benign. Most of it lies in long waves – continents, ocean basins, mountain ranges – and peaks and valleys are a small remainder.

An aerial photo of a town is the opposite. Every roof edge jumps from red to grey within one pixel, every tree crown is a dark patch a few metres across, every car a bright dot. Almost everything important is short waves. A wavelength of two pixels is 1.6 m here – that is 625 waves per kilometre, far more than the typical 20 from σ.

JPEG is built for exactly such images, and it works differently: it does not have to learn anything, but writes out the waves of each block directly. The network, by contrast, has to describe the whole image at once with a fixed number of weights – including the many small things that occur nowhere else.

## What is it good for, then?

As pure compression for aerial photos, a network in this simple form is therefore no good. That is an honest result and at the same time a boundary: neural fields are good for smooth fields such as elevations, temperatures or distances (chapters 1 to 13), less so for images full of edges.

Research has not given up on the problem. COIN by Dupont and colleagues also stores photos as a network and beat JPEG at very strong compression [7]. For satellite images with many spectral channels there are dedicated methods such as ImpliSat [8]. They go to more effort than this chapter – ImpliSat, for example, adapts the waves of the input to each channel separately.

And a network has properties that JPEG lacks. It is a function: you can query it at any position (E, N), even between the pixels. And it can be extended – by a third coordinate. Chapter 15 does that and turns buildings into a field f(x, y, z). Chapter 16 shows, with *Gaussian splatting*, a completely different way to store the same aerial photo and the same buildings: without a network, as many small blobs of colour.

## Try it out

The **Orthophoto** tab needs a download and a lot of computing time and therefore only runs in the local version of the app, not online.

### Loading the data

Open the **Orthophoto** tab. The first time, the app asks whether it may download the tile. Click **Download (18 MB)**.

### Training the network

| Setting | Value | Meaning |
|---|---|---|
| *Pixels* | 1250 × 1250 · 0.8 m | resolution to which the tile is scaled down |
| *Network* | Medium | one of the three presets from the table above |
| *Steps* | 1000 | number of training steps |
| *Device* | auto | graphics chip, if available |

Below the settings, the app states the size of the image and of the network.

![The settings in the Orthophoto tab: Pixels 1250 × 1250 · 0.8 m, Network Medium, Steps 1000, Device auto. Below them the line 1,562,500 pixels × 3 colours = 4.5 MB uncompressed · network: 329,475 weights = 643.5 KB in float16, plus the attribution Bayerische Vermessungsverwaltung – www.geodaten.bayern.de (CC BY 4.0). Below that the Train button.](../img/app-ortho-einstellungen.png)

Click **Train**. A bar shows the progress and the PSNR; after about 3 minutes the network is finished. In our run it then reached 23.8 dB. Four numbers then appear at the top: the PSNR of the network, the file size, the ratio and the PSNR of a JPEG of the same size. Below them, the original, the network and the JPEG are shown side by side.

![The Orthophoto tab after training: at the top PSNR of the network 23.8 dB, File size (float16) 643.5 KB, Compression ratio 7 : 1 and JPEG of the same size (quality 91) 36.3 dB. Below, the whole tile three times side by side: Original (DOP40), Network and JPEG, 618.0 KB. At this size all three look almost the same; in the network the trees on the islands and the water of the Danube are a little blotchy.](../img/app-ortho-ergebnis.png)

### Comparing details

Under **Detail** you choose a 100 m × 100 m section: the cathedral, the Stone Bridge or the Old Town Hall. On the far right, a fourth image shows the error of the network: black means exact, bright means far off. The errors sit at edges and details; the network gets large uniform areas right.

![Detail Cathedral: four sections around the cathedral. In the original, roofs, lanes, the cathedral square and the large shadow of the cathedral. In the network only blurred reddish and greenish-grey patches. The JPEG looks almost like the original. In the error image of the network, the edges of the roofs and the many details of the cathedral glow brightly; the large shadow and the area of the cathedral square stay dark.](../img/app-ortho-ausschnitt.png)

## Ideas to explore further

- Train *Small* with 500 × 500 pixels (2 m per pixel). How much does the PSNR improve when the image itself is already coarser? Does that bring the network closer to JPEG?
- Compare cathedral, bridge and town hall in the error image. Where are the largest errors – at edges, in shadows, on trees?
- Train *Medium* once with 1 000 and once with 6 000 steps. Which things only appear in the detail after long training?
- Think about it: an elevation model of the same tile (say roofs and trees as a surface model) – would it be easier or harder for a network than the photo?
- What would it take to store two neighbouring tiles: one network for both or two networks? What happens at the border?

## Further reading

- [1] Wikipedia: [Orthophoto](https://en.wikipedia.org/wiki/Orthophoto) – how an aerial photo becomes an orthophoto true to scale
- [2] Bayerische Vermessungsverwaltung: [OpenData – Digitales Orthophoto 40 cm](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dop40) – download of the tiles (in German)
- [3] Bayerische Vermessungsverwaltung: [Nutzungsbedingungen](https://geodaten.bayern.de/odd/m/3/html/nutzungsbedingungen.html) – CC BY 4.0 licence and attribution (in German)
- [4] Wikipedia: [Universal Transverse Mercator coordinate system](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system) – zones, easting, northing, scale factor
- [5] Bayerische Vermessungsverwaltung: [Abbildungsverzerrungen bei UTM](https://www.ldbv.bayern.de/mam/ldbv/dateien/utm_abbildungsverzerrungen.pdf) – how much UTM distances deviate in Bavaria (PDF, in German)
- [6] Wikipedia: [JPEG](https://en.wikipedia.org/wiki/JPEG) – how JPEG breaks an image into blocks and waves
- [7] Dupont et al.: [COIN: COmpression with Implicit Neural representations](https://arxiv.org/abs/2103.03123), 2021 – photos stored as a neural network
- [8] Cho et al.: [Fourier-Modulated Implicit Neural Representation for Multispectral Satellite Image Compression](https://arxiv.org/abs/2506.01234), IGARSS 2025 – ImpliSat, satellite images as a neural field
- Tancik et al.: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739), NeurIPS 2020 – Fourier features, also for images
- Wikipedia: [Regensburg Cathedral](https://en.wikipedia.org/wiki/Regensburg_Cathedral) – the largest building on the tile

---

[Overview](README.md) · Back: [Chapter 13 · Location encoders as embeddings](13-location-encoder.md) · Next: [Chapter 15 · Buildings as a 3D field](15-gebaeude-als-3d-feld.md)

*This tutorial was created by Gerhard Völkl using Claude Code (Anthropic). It is meant to help you learn more about storing geodata in neural networks. Suggestions for improvement are welcome at gkvoelkl@nelson-games.de.*
