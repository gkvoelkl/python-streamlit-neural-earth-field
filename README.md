# The Earth as a Neural Field · Die Erde als neuronales Feld

**English** · [Deutsch](#deutsch)

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

---

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

MIT License · [LICENSE](LICENSE) · ETOPO 2022 data: NOAA NCEI, DOI [10.25921/fd45-gt74](https://doi.org/10.25921/fd45-gt74)
