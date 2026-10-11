# Tutorial: Storing geodata in neural networks

*[Deutsche Fassung](../de/README.md) · The tutorial was written in German. Chapter 1 is translated; the other chapters follow and link to the German original until then.*

This tutorial shows step by step how to pack the elevation model of the entire Earth into the weights of a small neural network – and why the weights then are the file. Each chapter combines an idea with experiments in the companion app in this repository. The tutorial deliberately shows no program code – if you want to look inside, everything is in the `src/` folder.

**Audience:** people with solid Python skills and some numpy. You need to know neural networks at most from introductions; no experience with geodata is required.

## Getting started

```bash
git clone https://github.com/gkvoelkl/python-streamlit-neural-earth-field.git
cd python-streamlit-neural-earth-field
uv sync
uv run streamlit run app.py
```

Optional but recommended: download the global elevation model ETOPO 2022 (456 MB) into `data/`, as described in the [repository README](https://github.com/gkvoelkl/python-streamlit-neural-earth-field/blob/main/README_en.md#data). Without the file, everything runs on a synthetic Earth.

The app is in English. Terms from the user interface appear in italics, for example *Idea & Budget*.

## Contents

### Part 1 · Basics

1. [Geodata as a function instead of a grid](01-geodaten-als-funktion.md)
2. Training the neural network – [German](../de/02-training.md)
3. Architecture of a neural network – [German](../de/03-architektur.md)

### Part 2 · Better networks for geodata

4. The Earth is a sphere – [German](../de/04-kugelkoordinaten.md)
5. Elevations as waves – [German](../de/05-fourier-features.md)

Chapter 5 completes the core of the tutorial.

### Extensions and deep dives

The following chapters build on chapters 1 to 5. Read or skip them as you like.

6. Deep dive: spherical harmonics – [German](../de/06-kugelflaechenfunktionen.md)
7. Deep dive: SIREN – [German](../de/07-siren.md)
8. The context field – [German](../de/08-kontextfeld.md)
9. Measuring rate and error – [German](../de/09-rate-und-fehler.md)
10. Querying and exporting – [German](../de/10-abfragen-und-exportieren.md)
11. Context and outlook – [German](../de/11-einordnung-und-ausblick.md)

**Appendix**

- Measurements – [German](../de/anhang-messungen.md)

---

*This tutorial was created by Gerhard Völkl using Claude Code (Anthropic). It is meant to help you learn more about storing geodata in neural networks. Suggestions for improvement are welcome at gkvoelkl@nelson-games.de.*
