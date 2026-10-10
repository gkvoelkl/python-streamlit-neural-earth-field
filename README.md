<p align="center">
  <img src="./docs/tutorial/img/kapitel-01-chester.jpg" alt="Chester, the squirrel with the blue cap, holds Aiko, a small glowing neural network; a spiral flows into Aiko from a globe made of pixels." width="640">
</p>

<h1 align="center">The Earth as a Neural Field</h1>

<p align="center">
  <b>The elevation model of the whole Earth, stored in the weights of a small neural network.</b><br>
  <b>Das Höhenmodell der ganzen Erde, gespeichert in den Gewichten eines kleinen neuronalen Netzes.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white" alt="Python 3.14">
  <img src="https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.14">
  <img src="https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit 1.64">
  <img src="https://img.shields.io/badge/Data-ETOPO%202022-0B3A66" alt="Data: ETOPO 2022">
  <a href="https://gkvoelkl.github.io/python-streamlit-neural-earth-field/"><img src="https://img.shields.io/badge/Tutorial-Deutsch%20%7C%20English-2A86B0" alt="Tutorial: Deutsch | English"></a>
  <a href="https://github.com/gkvoelkl/python-streamlit-neural-earth-field/actions/workflows/docs.yml"><img src="https://github.com/gkvoelkl/python-streamlit-neural-earth-field/actions/workflows/docs.yml/badge.svg" alt="Website build"></a>
  <a href="https://claude.com/claude-code"><img src="https://img.shields.io/badge/Claude%20Code-D97757?logo=claude&logoColor=white" alt="Claude Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://gkvoelkl.github.io/python-streamlit-neural-earth-field/en/">Tutorial (English)</a> ·
  <a href="https://gkvoelkl.github.io/python-streamlit-neural-earth-field/">Tutorial (Deutsch)</a> ·
  <a href="README_en.md">README (English)</a> ·
  <a href="README_de.md">README (Deutsch)</a>
</p>

---

## English

The elevation model of the entire Earth – 2.3 million values, 4.4 MB – stored in
the weights of a small neural network of 323 KB. Instead of a table of
elevations you keep a function `f(lon, lat) → elevation`, trained deliberately on
exactly this one dataset: **the weights are the file**. A Streamlit app lets you
try it out, and a step-by-step tutorial without program code explains every idea.

```bash
uv sync
uv run streamlit run app.py
```

- **Tutorial:** [website](https://gkvoelkl.github.io/python-streamlit-neural-earth-field/en/) · [Markdown](docs/tutorial/en/README.md) – written in German, being translated into English
- **Details:** data download, the three tricks, results, tests → [README_en.md](README_en.md)

## Deutsch

Das Höhenmodell der ganzen Erde – 2,3 Millionen Werte, 4,4 MB – gespeichert in den
Gewichten eines kleinen neuronalen Netzes von 323 KB. Statt einer Tabelle von
Höhen behält man eine Funktion `f(lon, lat) → Höhe`, absichtlich auf genau diesen
einen Datensatz trainiert: **Die Gewichte sind die Datei.** Eine Streamlit-App
lädt zum Ausprobieren ein, und ein Schritt-für-Schritt-Tutorial ohne Programmcode
erklärt jede Idee.

```bash
uv sync
uv run streamlit run app.py
```

- **Tutorial:** [Website](https://gkvoelkl.github.io/python-streamlit-neural-earth-field/) · [Markdown](docs/tutorial/de/README.md)
- **Details:** Daten herunterladen, die drei Kniffe, Ergebnisse, Tests → [README_de.md](README_de.md)

---

<sub>MIT License · ETOPO 2022 data: NOAA NCEI, DOI <a href="https://doi.org/10.25921/fd45-gt74">10.25921/fd45-gt74</a> · Chester and Aiko images generated with Kling AI</sub>
