# Storing geodata in neural networks
# Chapter 15 · Buildings as a 3D field

*[Deutsche Fassung](../de/15-gebaeude-als-3d-feld.md) · Translated from the German original.*

![In a dark workshop, a desk lamp lights a wooden board with two models of a Gothic cathedral with two pointed towers. On the left stands a small model made of light stone, next to it Aiko, the glowing neural network. On the right stands a larger model with a house with a red roof in front of it; it rests on glowing, translucent cubes, and its towers shimmer as if made of glass. Chester, wearing a blue cap, holds up a glowing cube with tweezers to place it on the model.](../img/kapitel-15-chester.jpg)

**Contents:**

* Buildings as a function in three dimensions: easting, northing and height as input, distance to the nearest wall as output
* How a 3D city model becomes a voxel grid and a distance field – and buildings again from that
* How accurately a network captures the old town around Regensburg Cathedral, and what it costs compared with the city model


## Starting point

So far every field in this tutorial had two inputs: a place on the Earth or on a map. Buildings need a third, the height. Whether a point lies inside a house depends not only on where it is on the map, but also on how high it is: below the roof you are inside, above it outside.

This chapter uses two ideas shown earlier:

1. **The distance field from chapter 12.** There it was the distance to the coast, with a sign. Here it is the distance to the nearest wall or roof – negative inside, positive outside. Chapter 11 introduced it with *DeepSDF*, a method that was designed for shapes in space from the start [5].
2. **The plane from chapter 14.** For an area of a few hundred metres, the Earth is flat. The coordinates are again easting and northing in metres, plus the height.

So the network is a function f(E, N, H): three numbers in, one distance out. The area is a 300 m × 300 m section around Regensburg Cathedral, in the middle of the orthophoto tile from chapter 14.

## The 3D city model

### LoD2

The Bavarian surveying administration provides a 3D building model for all of Bavaria, free of charge under the CC BY 4.0 licence [1]. It is called **LoD2** (*Level of Detail 2*). The levels mean:

- **LoD1:** Every building is a block with a flat roof.
- **LoD2:** Every building has its roof shape – gable roof, hip roof, shed roof and so on. Windows, balconies and dormers are missing.

![The LoD2 city model around Regensburg Cathedral as a 3D view in the app, at an angle from the south-west. The buildings are coloured brown, darker the higher they are. In the middle the cathedral stands out with its two pointed towers and a slender ridge turret. Around it are blocks of houses with gable and hip roofs, with white lanes and courtyards in between. There are no windows, balconies or other details.](../img/app-gebaeude-lod2.png)

The roof shapes are chosen from a fixed catalogue of standard roofs. Ridges and eaves are usually accurate to 20 to 30 cm; where the roof shape is strongly simplified, in individual cases more than 1 m off [2].

The model is stored in the **CityGML** format [3, 4]. CityGML is a text file in XML format. Each building consists of surfaces: a ground surface, several walls and several roof surfaces. Each surface is a polygon, described by the coordinates of its corners – easting, northing and height in metres.

### The section

One tile of the city model covers 2 km × 2 km and is 113 MB in size. The app takes from it a square of **300 m × 300 m** around the cathedral, namely all buildings that lie wholly or partly inside it:

| Property | Value |
|---|---|
| Area | 300 m × 300 m around Regensburg Cathedral |
| Buildings | 226 |
| Corner points | 36 948 |
| CityGML of these buildings | 7.3 MB |
| Tallest building | the cathedral: 107 m from the ground to the tips of the towers |

The section is densely built up: on a good half of the area there is a building. Most houses are 10 to 20 m high. The cathedral towers far above them.

![On the left the orthophoto from chapter 14, cropped to 300 m × 300 m around the cathedral. On top of it, in orange, the outlines of the buildings from the city model; they fit the roofs in the photo closely. On the right the same area from above, each building in the colour of its height: most houses are blue to turquoise, 10 to 20 m high. In the middle lies the cathedral as a green cross, its two towers at the west end glow yellow. Lanes, squares and courtyards are white.](../img/en/gebaeude-ausschnitt.svg)

*Data: Bayerische Vermessungsverwaltung – www.geodaten.bayern.de, CC BY 4.0*

## From city model to field

### Voxels instead of pixels

A network learns from examples: place in, value out. For the elevations of the Earth a grid supplied these examples, for the orthophoto the pixels. Buildings need a grid in three dimensions. Its cells are small cubes; they are called **voxels** (from *volume* and *pixel*) [6].

The app lays a grid of cubes with an edge length of **1 m** over the section:

- 300 cubes to the east, 300 to the north and 120 upwards, that is up to 120 m above the lowest point.
- That makes **10 800 000 voxels**.

For each voxel, the app decides whether it lies inside a building. To do so, for each column of voxels it finds the building in whose footprint the column stands and computes how high the roof is exactly there. All voxels between the ground of the building and the roof lie **inside**, all others **outside**. The walls are always vertical. That fits LoD2: a city model at this level knows no oriels or canopies.

Of the 10.8 million voxels, **7.6 %** lie inside. The rest is air: above the roofs, in the lanes and courtyards. The terrain itself is not part of the model; a street here is simply air down to the bottom edge of the grid.

### The distance

As in chapter 12, inside and outside become a distance. For each voxel the app asks: **How far is it from here to the nearest voxel on the other side?** In a building, that is the distance to the nearest wall or to the roof; in the air, the distance to the nearest building. Inside it gets a minus, outside a plus. The surface of the buildings is then exactly where the field is 0.

As in chapter 12, the distance is **limited**. The default is ±5 m: a voxel in the middle of a large building gets −5 m, a voxel high above the roofs +5 m. The measurement further below shows why that helps.

![At the top a vertical west–east section through the cathedral and its south tower, as a grid of voxels: red inside the building, white the air. The tower rises as a narrow red spire above 100 m, next to it the lower aisle and a building with a pointed roof; to the left and right smaller houses. At the bottom the same section as a distance field, limited to ±5 m: dark red deep inside, lighter towards the wall, white at walls and roofs, light blue outside and dark blue after 5 m. A black line follows the surface where the distance is 0.](../img/en/gebaeude-voxel.svg)

### The neural network

The neural network has the same shape as in chapter 14, with two changes: it gets three numbers instead of two, and it outputs one number instead of three colours.

Easting, northing and height are treated alike. The app subtracts the corner of the grid and converts all three coordinates with **the same scale**: 300 m correspond to the range −1 to 1. The height thus only goes from −1 to −0.2. If you stretched it separately to −1 to 1, a metre upwards would be two and a half times as long for the network as a metre to the east. The waves of the Fourier features would then be shorter upwards than sideways.

For the Fourier features, **σ = 20** again. A typical wave is therefore about 300 m / 20 = **15 m** long – as long as an old-town house is high.

The app has three presets:

| Preset | Frequencies | Layers × neurons | Weights | File (float16) |
|---|---:|---|---:|---:|
| *Small* | 128 | 4 × 128 | 82 561 | 161 KB |
| *Medium* | 256 | 4 × 256 | 328 961 | 643 KB |
| *Large* | 256 | 5 × 512 | 1 313 793 | 2.5 MB |

*Small* has exactly as many weights as the standard example from chapter 1.

Training works as usual: in each step the app draws 65 536 of the 10.8 million voxels at random and improves the weights so that the distance fits better.

## Measures for shapes

### The correct side is not enough

In chapter 12 the most important measure was the **correct side**: how many cells does the network classify correctly as land or sea? Here this measure is of little use. 92.4 % of all voxels are air. A network that says "outside" everywhere and knows not a single building would already get 92.4 % right.

### IoU

The usual measure for shapes is therefore the **IoU** (*intersection over union*, also called the Jaccard index) [7]. You take two sets of voxels: the voxels that lie inside in the city model, and those that the network sees as inside. Then you divide

- the space that **both** see as inside,
- by the space that **at least one** of the two sees as inside.

An example: a house has 1 000 m³. The network hits 900 m³ of it and also puts 100 m³ next to it, where there is no house. Both see 900 m³ as house, at least one of them 1 100 m³. The IoU is 900 / 1 100 = **0.82**. An IoU of 1 means the two shapes are identical; 0 means they have nothing in common. A network without buildings has an IoU of 0.

### Surface offset

The IoU says how much space is right, but not where the errors lie. For that, as in chapter 12, the app measures the **offset**: it determines the network's surface, that is all places where its distance is 0, and asks for each point of this surface how far it is from the true surface. The median and the value below which 90 % of the points lie are given.

This measure has a limit: the grid has 1 m voxels. Values of about 0.5 m and less – half a voxel – it can hardly tell apart any more.

### A surface again from the field

To see the network's buildings and to measure the offset, you need the surface as a surface in space. The method for this is called **marching cubes** [8]. It goes through the grid cube by cube. Where the field is partly positive and partly negative at the corners of a cube, the surface must pass through this cube. Marching cubes inserts a few triangles there so that they pass exactly through the places where the field becomes 0. Together, the triangles form a closed surface. For the buildings of the section, that is about half a million triangles.

The app uses the *scikit-image* library for this [9] and shows the result as a 3D view with *Plotly*.

## Result

Measured with the 1 m grid on an Apple M4 (MPS), with a limit of ±5 m. Each setting was trained three times (seeds 0 to 2). The mean is given, in brackets the smallest and the largest value of the three runs. *Large* with 6 000 steps was not measured; 3 000 steps alone take half an hour.

**IoU:**

| Preset | File | 1 000 steps | 3 000 steps | 6 000 steps |
|---|---:|---:|---:|---:|
| *Small* | 161 KB | 0.40 (0.31–0.46) | 0.70 (0.61–0.78) | 0.79 (0.73–0.84) |
| *Medium* | 643 KB | 0.89 (0.87–0.93) | 0.95 (0.92–0.97) | 0.97 (0.96–0.98) |
| *Large* | 2.5 MB | 0.93 (0.92–0.95) | 0.98 (0.98–0.98) | – |

**Surface offset, median / 90 %:**

| Preset | 1 000 steps | 3 000 steps | 6 000 steps |
|---|---:|---:|---:|
| *Small* | 1.5 m / 4.1 m | 0.8 m / 2.4 m | 0.6 m / 1.7 m |
| *Medium* | 0.5 m / 0.8 m | 0.4 m / 0.6 m | 0.3 m / 0.5 m |
| *Large* | 0.4 m / 0.6 m | 0.2 m / 0.5 m | – |

Computing time for one run: *Small* needs about 50 seconds for 1 000 steps, *Medium* just under 2 minutes, *Large* 7 to 9 minutes. For 3 000 steps it is about 3 minutes, 7 minutes and 26 to 35 minutes.

![Chart: IoU against the number of training steps, 1 000, 3 000 and 6 000. Three lines with points, each with a vertical bar for the range of the three runs. Green Small rises from 0.40 through 0.70 to 0.79, with long bars. Blue Medium rises from 0.89 through 0.95 to 0.97, with short bars. Purple Large rises from 0.93 to 0.98 and ends at 3 000 steps.](../img/en/gebaeude-messung.svg)

Three things stand out:

- ***Small* is not enough for buildings.** After 1 000 steps, less than half of the space matches. Even after 6 000 steps, every tenth point of the surface is more than 1.7 m off – a lot for houses 10 to 20 m high. The same number of weights stored the elevations of the whole Earth in chapter 1. 226 buildings with their roofs have far more edges relative to their size.
- ***Small* depends strongly on chance.** The three runs are up to 0.17 apart. With the seed, the frequencies of the Fourier features and the initial weights change. A small network cannot learn all buildings, and which ones it learns first depends on this chance. From *Medium* on, the runs are close together.
- ***Medium* captures the buildings well.** After 3 000 steps, half of the surface is less than 0.4 m off, 90 % less than 0.6 m. That is close to the limit the 1 m grid can measure at all. *Large* is only slightly better, but takes four times as long.

### The limit

How strongly should the distance be limited? Measured with *Medium* and 3 000 steps, again three runs each:

| Limit | IoU | Offset, median | Offset, 90 % |
|---|---:|---:|---:|
| ±2 m | **0.98** (0.98–0.99) | 0.2 m | 0.5 m |
| ±5 m | 0.95 (0.92–0.97) | 0.4 m | 0.6 m |
| ±10 m | 0.88 (0.86–0.93) | 0.5 m | 0.9 m |
| ±20 m | 0.81 (0.77–0.85) | 0.5 m | 1.5 m |

**The stronger the limit, the more accurate the buildings** – just as with the coast in chapter 12. With ±2 m, *Medium* is as good as *Large* with ±5 m, at a quarter of the computing time and a quarter of the file size.

The reason is the same as in chapter 12. Without a limit, the network would have to know for every point in the air how far away the nearest house is – above the roofs up to 100 m. These large distances do not matter for the shape of the buildings, but they cost weights. With a limit, almost the whole space is simply +2 m or −2 m. The network can then spend its weights on the narrow strip along the walls and roofs that matters.

The price is the same too: a network with a limit of ±2 m only knows whether a point is more than 2 m from the nearest building, but not how far.

### What it looks like

The app shows the city model and the network side by side as a 3D view. Here both from the same viewpoint close to the cathedral, *Medium* after 1 000 steps with a limit of ±5 m (IoU 0.87):

![Two 3D views of the same houses around the cathedral, same viewpoint. On the left the city model: smooth, vertical walls, straight roof edges, the two cathedral towers with clean tips and a thin ridge turret on the roof. On the right the network: the same buildings and lanes, but the edges are rounded and frayed, the walls slightly wavy, the cathedral towers jagged at the edges, and only a short piece of the ridge turret is left.](../img/app-gebaeude-dom-nah.png)

With *Medium* after 1 000 steps, all larger buildings are there: the cathedral with its two towers, the blocks of houses with their courtyards, the lanes in between. But the shapes are soft: edges are rounded, walls and roofs slightly wavy, as if kneaded from clay. The waves of the Fourier features become visible here. On the cathedral towers, the very tip comes loose as a small piece and floats freely above the tower, because the tower below becomes too thin for the network to capture it fully.

A horizontal slice through the field shows a peculiarity of the network: small rings above low roofs, where the city model has only air. There the distance is just above 0, and the small noise of the network is enough to push it below 0 at individual places. In the 3D view these places appear as small crumbs above the roofs, in the slice as a swarm of small black rings. With more steps or a stronger limit they become rarer.

## What the buildings cost as a network

Finally, again the honest comparison of file sizes:

| Representation | Size |
|---|---:|
| CityGML of the 226 buildings | 7.3 MB |
| CityGML, packed with gzip | 596 KB |
| Voxel grid inside/outside, 1 bit per voxel | 1.3 MB |
| Voxel grid inside/outside, packed with gzip | **38 KB** |
| Distance field, 2 bytes per voxel | 20.6 MB |
| Network *Small* / *Medium* / *Large*, float16 | 161 KB / 643 KB / 2.5 MB |

The *Medium* network is about a twelfth of the size of the CityGML file. That sounds good, but it is not a fair comparison: CityGML is text, and text packs well. With gzip the file is smaller than the network – and more accurate, because it knows every corner to the centimetre.

The voxel grid is even clearer. Packed, it is only 38 KB, smaller than any network. Buildings are easy for gzip: a column of voxels is inside up to the roof and outside above it, and many columns side by side look alike. So, as in chapter 14: **as pure compression, the network does not pay off.**

What the network offers is something else. It is a function that can be queried at any point, even between the voxels, without first unpacking a grid or searching through polygons. A single query answers: does this point lie in a building? How far is it to the nearest wall – up to the limit of ±5 m? Such questions are asked, for example, by programs that plan the flight of a drone through a city. And the surface is smooth instead of stepped like the voxels.

In research this representation is widespread for individual shapes. DeepSDF learns distance fields for furniture, cars and aeroplanes, and usually one network for many shapes at once [5]. *Occupancy networks* learn, instead of the distance, only the question "inside or outside?" [10]. A later chapter takes up this idea: one shared network for many tiles.

## Try it out

The **Buildings 3D** tab needs a large download and a lot of computing time and therefore only runs in the local version of the app, not online.

### Loading the data

Open the **Buildings 3D** tab. The first time, the app asks whether it may download the tile of the city model. Click **Download (108 MB)**. The app then builds the voxel grid and the distance field; the first time this takes about a minute.

### Training the network

| Setting | Value | Meaning |
|---|---|---|
| *Voxel size* | 1 m | edge length of the voxels; 2 m is faster but coarser |
| *Network* | Medium | one of the three presets from the table above |
| *Limit the distance to* | ±5 m | limit of the distance |
| *Steps* | 1000 | number of training steps |
| *Device* | auto | graphics chip, if available |

Below the settings, the app states the number of buildings and corner points, the size of the CityGML file, the size of the grid and the size of the network.

![The settings in the Buildings 3D tab: Voxel size 1 m, Network Medium, Limit the distance to ±5 m, Steps 1000, Device auto. Below them the line 226 buildings · 36,948 corner points · CityGML 7.3 MB · grid 300 × 300 × 120 = 10,800,000 voxels · network: 328,961 weights = 642.5 KB in float16, plus the attribution. Below that the Train button.](../img/app-gebaeude-einstellungen.png)

Click **Train**. A bar shows the progress and the share of voxels on the correct side; after a good minute and a half the network is finished. Four numbers appear at the top: the IoU, the offset as median and for 90 %, and the size of the network compared with the CityGML file. In our run the IoU was 0.87.

Below them are two 3D views side by side: on the left the city model, on the right the network. The colour shows the height above the ground. With the mouse you can rotate and zoom both views.

![The Buildings 3D tab after training: at the top Volume overlap (IoU) 0.87, Surface offset (median) 0.5 m, Surface offset (90 %) 0.9 m and Network vs CityGML 642.5 KB · 12 : 1. Below, two 3D views of the old town from the south-west, on the left LoD2 building models, on the right Network. Both show the same blocks of houses and the cathedral in the middle; in the network the edges are rounded, the walls slightly wavy, and above the cathedral towers the tips float as small detached pieces.](../img/app-gebaeude-3d.png)

### Slice through the field

At the very bottom the app shows a horizontal slice through the distance field, on the left from the city model, on the right from the network. With **Height above the lowest ground** you choose the height. Red means inside, blue outside, white close to a wall; the black line is the 0 line, that is the walls at this height.

![At the top the slider Height above the lowest ground at 10 m. Below it two horizontal slices at this height. On the left LoD2: red blocks of houses on a blue background, light blue areas above roofs just below 10 m, the black line follows the walls exactly. On the right Network: the same blocks, the colours blotchy and the line a little softer. In the light blue areas above low roofs there is a swarm of small black rings.](../img/app-gebaeude-schnitt.png)

## Ideas to explore further

- Set **Limit the distance to** to ±2 m and train *Medium* again. Do the tips of the cathedral towers still float freely? Do the crumbs disappear?
- Train *Small* twice in a row with the same settings. Are the same buildings missing both times?
- Move the slice slowly upwards. At what height are only the cathedral and a few tall houses left? Where is the network least certain there?
- Set *Voxel size* to 2 m. The grid then has only an eighth of the voxels. Does this make the network worse – or only the measurement coarser?
- Think about it: the network knows only buildings. What would you have to do for it to know the terrain as well, for example the slope down to the Danube?

## Further reading

- [1] Bayerische Vermessungsverwaltung: [OpenData – 3D-Gebäudemodelle LoD2](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2) – download of the tiles (in German)
- [2] Bayerische Vermessungsverwaltung: [Hinweise zu den LoD2-Daten](https://geodaten.bayern.de/odd/m/3/pdf/hinweise_daten_lod2_download.pdf) – content, roof shapes and accuracy (PDF, in German)
- [3] Wikipedia: [CityGML](https://en.wikipedia.org/wiki/CityGML) – the format for 3D city models and its levels of detail
- [4] Open Geospatial Consortium: [CityGML](https://www.ogc.org/standards/citygml/) – the standard
- [5] Park et al.: [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103), CVPR 2019 – signed distance functions as a neural field
- [6] Wikipedia: [Voxel](https://en.wikipedia.org/wiki/Voxel) – grid points in three dimensions
- [7] Wikipedia: [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) – intersection over union as a measure of similarity
- [8] Wikipedia: [Marching cubes](https://en.wikipedia.org/wiki/Marching_cubes) – how a field becomes a surface of triangles
- [9] scikit-image: [skimage.measure.marching_cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) – the function the app uses
- [10] Mescheder et al.: [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828), CVPR 2019 – "inside or outside?" as a neural field
- Curless and Levoy: [A Volumetric Method for Building Complex Models from Range Images](https://graphics.stanford.edu/papers/volrange/), SIGGRAPH 1996 – truncated distance fields for 3D scans
- Wikipedia: [Regensburg Cathedral](https://en.wikipedia.org/wiki/Regensburg_Cathedral) – the tallest building in the section

---

[Overview](README.md) · Back: [Chapter 14 · Orthophotos in UTM](14-orthofotos-in-utm.md) · Next: [Chapter 16 · Gaussian splatting](16-gaussian-splatting.md)

*This tutorial was created by Gerhard Völkl using Claude Code (Anthropic). It is meant to help you learn more about storing geodata in neural networks. Suggestions for improvement are welcome at gkvoelkl@nelson-games.de.*
