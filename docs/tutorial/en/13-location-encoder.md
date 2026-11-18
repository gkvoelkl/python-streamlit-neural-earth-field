# Storing geodata in neural networks
# Chapter 13 · Location encoders as embeddings

*[Deutsche Fassung](../de/13-location-encoder.md) · Translated from the German original.*

![In an old study with bookshelves, Chester, wearing a blue cap, crouches on a large wooden globe and holds a string of small, colourfully glowing beads. The string arcs down to Aiko, a glowing neural network, who holds its end against a small thermometer.](../img/kapitel-13-chester.jpg)

**Contents:**

* What a *location encoder* is: a network that translates a place into a list of numbers
* How to test whether these numbers are any good: with a new task and only a few training places


## Starting point

In chapters 1 to 12 a network always stored exactly **one** field: the elevations of the Earth or the distance to the coast. All cells of the grid took part in training the neural network. The finished network can reproduce exactly this one trained quantity. For another quantity, such as temperature, you would have to train a new network – again with all cells.

Chapter 11 described another approach (approach 2): a neural network translates a place into a fixed number of numbers, say 256.

Similar places get similar numbers. These numbers are called an *embedding*, and the neural network that computes them is a *location encoder*.

A location encoder does not learn one particular quantity. It is trained once, and afterwards its numbers are used for many different tasks. For each new task, a second, small network is added. Its input is not longitude and latitude but the 256 numbers of the encoder, and from them it learns, for example, the temperature. The encoder itself stays unchanged.

Because its numbers already reveal a lot about a place – is it in the desert, in the forest, in the mountains? –, the small network needs only a few examples: about 1 000 places instead of all cells. What the encoder has once learned about the Earth is thus **transferred** to a new task. This is called *transfer learning*, and the new task a *downstream task*.

The difference from the approach so far:

**Storing** means a network learns a single field as accurately as possible.

**Transferring** (*transfer learning*) means a network provides a general description of places, with which new tasks can be learned from few examples.

![Two rows of boxes with arrows. At the top, "Storing": lon, lat leads into the elevation network with 82 561 weights, trained on all 2.3 million cells, and from there to the elevation. At the bottom, "Transferring": lon, lat leads into a fixed location encoder trained beforehand, such as SatCLIP, from there to 256 numbers, then into a small network that learns the new task from 1 000 places, and finally to temperature, or precipitation or population.](../img/en/encoder-speichern-uebertragen.svg)

This chapter tries it out. The new task is the **mean annual temperature** on land.

## What a location encoder provides

The Fourier features from chapter 5 turn a place into 256 numbers, the spherical harmonics from chapter 6 into 100 or more, depending on the degree. These numbers divide up the Earth, but carry no additional meaning. Still, they are simple location encoders.

A pretrained encoder, by contrast, has already learned something about the Earth.

The best-known example is **SatCLIP** by Klemmer and colleagues [1]. It was trained with 100 000 satellite images from Sentinel-2, spread roughly evenly over all land areas.

During training, two networks learn at the same time: one turns a satellite image into 256 numbers, the other turns the place where the image was taken into 256 numbers. The goal is that an image and its place give numbers as similar as possible, and different places give numbers as different as possible. After training, the image is no longer needed: the place alone yields numbers that contain what the landscape there looks like.

A trick shows the effect: from the many numbers of each place, take the three directions in which they differ most (the *principal components*) and paint them as red, green and blue.

![Two world maps in bright colours, the seas pale. On the left the spherical harmonics: the colours change only slowly, almost all continents are the same olive brown, only Canada is pink and Siberia light blue to black. On the right SatCLIP: the rainforests of the Amazon and the Congo are almost black, the coniferous forests of Canada and Siberia pink, the Sahara and Arabia light green, Australia light blue, Antarctica green.](../img/en/encoder-embedding-farben.svg)

With SatCLIP the colours form regions that match climate and landscape. The rainforests in South America and Africa have similar colours although an ocean lies between them; so do the coniferous forests in Canada and Siberia.

### SatCLIP from the inside

The location encoder of SatCLIP is made of familiar components. It consists of two parts [2]:

1. **Spherical harmonics** up to degree 9 – as in chapter 6. That makes 100 numbers per place.
2. A small **SIREN** – as in chapter 7 – with two layers of 512 neurons each and an output layer with 256 neurons.

Together that is 445 696 weights. Rußwurm and colleagues showed that exactly this combination of spherical harmonics and SIREN makes a particularly good location encoder [2]. The two deep-dive chapters of this tutorial thus add up to a current method from research.

Microsoft provides the trained weights under the MIT licence [3]. The app uses the smallest version (*ResNet18, L = 10*), and of it only the location encoder; the network for the images is not needed.

## The task: temperature from a few places

The temperatures come from the dataset by Willmott and Matsuura of the University of Delaware (version 5.01) [4, 5]. It is based on measurements at weather stations. Their values were converted to a grid with points 0.5° apart (about 56 km). Between the stations, an elevation model helped: the assumption was that it gets 6 °C colder per kilometre of elevation [5].

![Section through a mountain range with two weather stations and a grid point on the summit. Station A lies in the valley at 400 m and measures 11.0 °C, station B at 1 600 m measures 3.4 °C. Arrows lead from both stations down to sea level: with 6 °C per kilometre, they become 13.4 and 13.0 °C. Two arrows along sea level meet below the grid point at 13.2 °C. An orange arrow leads from there up to the grid point at 2 200 m according to the elevation model: 13.2 minus 2.2 times 6 gives 0.0 °C. On the right, the four steps Measure, Convert to sea level, Spread between the stations and To the elevation of the grid point.](../img/en/encoder-stationen-gitter.svg)

The long-term means from 1981 to 2010 are used, averaged over the twelve months. The dataset covers land only, including Antarctica: **85 794 land cells** with values from −56.1 °C in Antarctica to +35.0 °C.

From these cells, **training places** are drawn at random – 100, 1 000 or 10 000. As in training the elevation network, area counts: a cell in Siberia is smaller than one at the equator and is drawn correspondingly less often (*area-weighted sampling*, chapter 2). Testing is done on **all remaining** cells. The measure is the area-weighted RMSE in °C (chapter 9).

For comparison: always predicting the mean of the training places is off by **18.6 °C** on average. Every input must therefore stay well below that.

### Six inputs, one small network

Each input feeds the same small network: two hidden layers of 64 neurons each, 2 000 training steps with all training places at once. The numbers of each input are first brought to mean 0 and standard deviation 1, so that no input has an advantage because of its scale.

| Input | Numbers per place | Source |
|---|---:|---|
| lon/lat | 2 | longitude and latitude, scaled to −1 to 1 |
| Fourier features | 256 | 128 random stripes on the sphere (chapter 5), σ = 2 |
| Spherical harmonics | 100 | up to degree 9 (chapter 6), as in SatCLIP |
| SatCLIP | 256 | pretrained location encoder [1] |
| Elevation network (hidden layer) | 128 | the last hidden layer of the standard example |
| lon/lat + elevation | 3 | longitude, latitude and the elevation the standard example outputs at this place |

The last two inputs use the elevation network from chapters 1 to 10 (standard example). The hidden layer is an attempt to use the elevation network itself as a location encoder: the 128 numbers before its output layer also describe every place. The input "lon/lat + elevation", by contrast, uses only what the elevation network actually stored – the elevation.

## Result

Measured with three runs per row: each run draws different training places and starts the small network with different weights. The mean is given.

| Input | 100 places | 1 000 places | 10 000 places |
|---|---:|---:|---:|
| lon/lat | 5.1 °C | 3.5 °C | 3.3 °C |
| Fourier features | 12.4 °C | 4.3 °C | 2.1 °C |
| Spherical harmonics | 8.1 °C | 3.6 °C | 2.0 °C |
| SatCLIP | 5.2 °C | 3.6 °C | 2.0 °C |
| Elevation network (hidden layer) | 20.0 °C | 17.5 °C | 10.6 °C |
| lon/lat + elevation | **4.4 °C** | **2.2 °C** | **2.0 °C** |

![Line chart: error in °C against the number of training places 100, 1 000 and 10 000, logarithmic. Five lines fall from left to right. Lowest is lon/lat + elevation with 4.4, 2.2 and 2.0 °C. lon/lat and SatCLIP almost coincide at 100 and 1 000 places; at 10 000 places lon/lat falls behind with 3.3 °C. Fourier features start at 12.4 °C and catch up by 10 000 places. The red line of the elevation network lies above the dashed line for the mean at 18.6 °C at 100 places.](../img/en/encoder-orte-fehler.svg)

Four things stand out:

- **With few places, two numbers are hard to beat.** lon/lat and SatCLIP are level: 5.1 and 5.2 °C at 100 places, 3.5 and 3.6 °C at 1 000. Temperature depends mainly on latitude, and that is already fully contained in lon/lat. Pretraining still helps: compared with the spherical harmonics that SatCLIP itself starts from, the error at 100 places drops from 8.1 to 5.2 °C. The Fourier features are furthest off with 100 places: in their 256 numbers the small network finds patterns that do not exist.
- **With many places, many numbers pay off.** At 10 000 places, Fourier features, spherical harmonics and SatCLIP are at 2.0 to 2.1 °C, while lon/lat stays at 3.3 °C. From two numbers the small network can hardly form fine patterns – the same problem as with the simple network in chapter 2.
- **The hidden layer of the elevation network is no good as an embedding.** With 100 places it is even worse than simply predicting the mean. The 128 numbers are tuned precisely so that the output layer computes the elevation from them – with all its fine details. How warm a place is can hardly be read from them. **Storing is not transferring:** a network that stores a field as accurately as possible does not thereby provide a generally useful description of places.
- **The elevation itself helps a lot.** Longitude, latitude and elevation – three numbers – beat every other input with 100 and 1 000 places. That is domain knowledge: in the air it gets colder with height, by about 6.5 °C per kilometre on average [6]. The small network only has to find this relationship, not guess it from the place. The comparison is not entirely fair, because the temperature data themselves were produced with an elevation model (see above). Along the way, a use of the elevation network shows up: not as an embedding, but as a compact supplier of the elevation for any coordinate.

The maps for 1 000 training places show where the errors lie:

![Three world maps of the error in red and blue, each including Antarctica. At the top lon/lat, error 3.5 °C: large red and blue areas, for example in Greenland and in wide bands across Antarctica, plus strong errors in the Andes, the Rocky Mountains and the Himalayas. In the middle SatCLIP, error 3.7 °C: the errors are distributed differently but are not smaller; they sit in the mountains and at the edges of Greenland and Antarctica. At the bottom lon/lat + elevation, error 2.3 °C: most areas are light; only in Tibet, in Greenland and in one patch of Antarctica do clear errors remain.](../img/en/encoder-fehlerkarten.svg)

For lon/lat and for SatCLIP, the largest errors sit in the mountains. From the satellite images, SatCLIP knows the landscape, but the elevation only roughly. This is also due to the resolution: the finest spherical harmonics up to degree 9 have waves about 4 000 km long. The SIREN behind them does form finer patterns from them, but for a narrow mountain range like the Andes that is not enough. The developers themselves point out that SatCLIP is meant for questions on the whole Earth, not for small areas [3]. For finer tasks there is a version with spherical harmonics up to degree 39 (*L = 40*).

So for temperature SatCLIP brings little, because temperature depends mainly on latitude and elevation. A pretrained encoder helps where the task is related to what it has learned – for SatCLIP, what the landscape looks like in satellite images. Klemmer and colleagues test SatCLIP on nine tasks, including population density and the occurrence of animal species [1].

## What it costs

| | Weights |
|---|---:|
| SatCLIP, location encoder | 445 696 |
| small network on SatCLIP (256 numbers) | 20 673 |
| small network on lon/lat + elevation (3 numbers) | 4 481 |
| elevation network (standard example) | 82 561 |

The encoder is large, but it is trained only once and then used for many tasks. For each new task only the small network is added. So the calculation is different from chapters 1 to 12: there, what counted was how small the file for a single field is. Here, what counts is how few **examples** a new task needs. Measurements from weather stations are expensive; if 1 000 places are enough instead of 10 000, a lot is gained.

## Try it out

The **Location Encoder** tab needs two downloads and therefore only runs in the local version of the app, not online.

### Training the elevation network

The two inputs from the elevation network only exist once one has been trained. Train the standard example: **Dataset** *Elevation (ETOPO 2022)*, **Grid resolution** 1080 × 2160, preset **Fourier MLP (recommended)**, **Steps** 4 000, then **Train**. Without an elevation network, the tab compares the remaining four inputs.

### Loading the data and comparing

Open the **Location Encoder** tab. The first time, the app asks whether it may download the temperatures from the University of Delaware (about 19 MB) and the SatCLIP location encoder (about 57 MB). Click **Download (76 MB)**.

| Setting | Value | Meaning |
|---|---|---|
| *Training places* | 1000 | number of places the small network learns from |
| *Inputs* | all | the inputs being compared |
| *Seed* | 0 | seed for the choice of places and the weights of the small network |

Click **Compare**. After a few seconds, a bar chart shows the error of each input; the dashed line is the error when you always predict the mean.

![The Location Encoder tab: at the top the settings Training places 1000, Inputs with lon/lat, Fourier features, Spherical harmonics, SatCLIP and Elevation network (hidden layer), Seed 0 and the Compare button. Below, horizontal bars: lon/lat + elevation 2.26 °C, lon/lat 3.52 °C, SatCLIP 3.65 °C, Spherical harmonics 3.69 °C, Fourier features 4.53 °C and Elevation network (hidden layer) 17.59 °C, just below the dashed line always the mean at 18.7 °C.](../img/app-encoder-vergleich.png)

The numbers differ slightly from the table above: there the mean of three runs is given, here you see a single one.

### Looking at the maps

Under **Map for** you choose an input. On the left are the measured temperatures with the training places as black dots, on the right the prediction. Below, a third map shows the error.

![Two world maps side by side. On the left the measured mean annual temperature, from dark blue in Antarctica and Greenland to dark red in the Sahara, with many black dots for the 1 000 training places. On the right the prediction from SatCLIP, which looks very similar to the left map on a large scale.](../img/app-encoder-karten.png)

## Ideas to explore further

- Set *Training places* to 100 and change the *Seed* several times. Which input varies most?
- Compare the error maps of SatCLIP and *lon/lat + elevation* in East Africa and the Himalayas. Where does the elevation help, where not?
- Train *Fourier MLP, very small* instead of the standard example. Does the input *lon/lat + elevation* get worse when the elevation network knows the elevation less accurately?
- Train the elevation network for only 500 steps. Is its hidden layer then a better embedding than after 4 000 steps?
- Think about which other task could benefit more from SatCLIP than temperature – for example one in which the landscape matters more than latitude and elevation.

## Further reading

- [1] Klemmer et al.: [SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery](https://arxiv.org/abs/2311.17179), AAAI 2025 – location encoder pretrained with Sentinel-2 images
- [2] Rußwurm et al.: [Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks](https://arxiv.org/abs/2310.06743), ICLR 2024 – spherical harmonics and SIREN as a location encoder
- [3] Microsoft: [SatCLIP on GitHub](https://github.com/microsoft/satclip) and [weights on Hugging Face](https://huggingface.co/microsoft/SatCLIP-ResNet18-L10) – code and trained weights, MIT licence
- [4] NOAA PSL: [University of Delaware Air Temperature & Precipitation](https://psl.noaa.gov/data/gridded/data.UDel_AirT_Precip.html) – monthly means of air temperature on land, 0.5°, version 5.01
- [5] NCAR Climate Data Guide: [Global (land) precipitation and temperature: Willmott & Matsuura, University of Delaware](https://climatedataguide.ucar.edu/climate-data/global-land-precipitation-and-temperature-willmott-matsuura-university-delaware) – how the dataset was made, no usage restrictions
- [6] Wikipedia: [Lapse rate](https://en.wikipedia.org/wiki/Lapse_rate) – how temperature decreases with height
- Mai et al.: [A Review of Location Encoding for GeoAI: Methods and Applications](https://arxiv.org/abs/2111.04006), IJGIS 2022 – overview of location encoders
- Rußwurm: [locationencoder on GitHub](https://github.com/marccoru/locationencoder) – code for [2] with further encoders for comparison

---

[Overview](README.md) · Back: [Chapter 12 · The coast as a distance field](12-kueste-als-abstandsfeld.md) · Next: [Chapter 14 · Orthophotos in UTM](14-orthofotos-in-utm.md)

*This tutorial was created by Gerhard Völkl using Claude Code (Anthropic). It is meant to help you learn more about storing geodata in neural networks. Suggestions for improvement are welcome at gkvoelkl@nelson-games.de.*
