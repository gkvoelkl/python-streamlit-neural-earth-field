# Tutorial: Storing geodata in neural networks

*[Deutsche Fassung](../de/README.md) · The tutorial was written in German. Chapters 1 to 9 are translated; the other chapters follow and link to the German original until then.*

This tutorial shows step by step how to pack the elevation model of the entire Earth into the weights of a small neural network – and why the weights then are the file. Each chapter combines an idea with experiments in the companion app in this repository. The tutorial deliberately shows no program code – if you want to look inside, everything is in the `src/` folder.

**Audience:** people with solid Python skills and some numpy. You need to know neural networks at most from introductions; no experience with geodata is required.

## Getting started

For a first look, the app is also online, without installing anything: <https://python-app-neural-earth-field.streamlit.app>. It runs there on a shared CPU without a graphics unit and therefore works with much less data: coarser grids (default 360 × 720 instead of 1080 × 2160) and lighter training. Its numbers therefore differ from those in the tutorial. For the experiments in the tutorial, your own installation is better:

```bash
git clone https://github.com/gkvoelkl/python-streamlit-neural-earth-field.git
cd python-streamlit-neural-earth-field
uv sync
uv run streamlit run app.py
```

On the first start, the app asks whether it may download the global elevation model ETOPO 2022 (456 MB); it does not run without these data. If you prefer to download the file yourself, the command is in the [repository README](https://github.com/gkvoelkl/python-streamlit-neural-earth-field/blob/main/README_en.md#data).

The app is in English. Terms from the user interface appear in italics, for example *Idea & Budget*.

## Contents

### Part 1 · Basics

1. [Geodata as a function instead of a grid](01-geodaten-als-funktion.md)
2. [Training the neural network](02-training.md)
3. [Architecture of a neural network](03-architektur.md)

### Part 2 · Better networks for geodata

4. [The Earth is a sphere](04-kugelkoordinaten.md)
5. [Elevations as waves](05-fourier-features.md)

Chapter 5 completes the core of the tutorial.

### Extensions and deep dives

The following chapters build on chapters 1 to 5. Read or skip them as you like.

6. [Spherical harmonics](06-kugelflaechenfunktionen.md)
7. [SIREN](07-siren.md)
8. [The context field](08-kontextfeld.md)
9. [Measuring compression ratio and error](09-rate-und-fehler.md)
10. Querying and exporting – [German](../de/10-abfragen-und-exportieren.md)
11. Context and outlook – [German](../de/11-einordnung-und-ausblick.md)

**Appendix**

- Measurements – [German](../de/anhang-messungen.md)

---

*This tutorial was created by Gerhard Völkl using Claude Code (Anthropic). It is meant to help you learn more about storing geodata in neural networks. Suggestions for improvement are welcome at gkvoelkl@nelson-games.de.*
