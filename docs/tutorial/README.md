# Tutorial: Geodaten in neuronalen Netzen speichern

Dieses Tutorial zeigt Schritt für Schritt, wie man das Höhenmodell der ganzen Erde in die Gewichte eines kleinen neuronalen Netzes packt – und warum die Gewichte dann die Datei sind. Jedes Kapitel verbindet eine Idee mit Experimenten in der Begleit-App aus diesem Repo. Programmcode zeigt das Tutorial bewusst nicht – wer hineinschauen will, findet alles im Ordner `src/`.

**Zielgruppe:** Leute mit soliden Python-Kenntnissen und etwas numpy. Neuronale Netze muss man höchstens aus Einführungen kennen, Erfahrung mit Geodaten ist nicht nötig.

## Vorbereitung

```bash
uv sync
uv run streamlit run app.py
```

Optional, aber empfohlen: das globale Höhenmodell ETOPO 2022 (456 MB) nach `data/` laden, wie in der [README des Repos](../../README.md#data) beschrieben. Ohne die Datei läuft alles mit einer synthetischen Erde.

Die App ist auf Englisch. Begriffe aus der Oberfläche stehen im Tutorial im Original und kursiv, etwa *Idea & Budget*.

## Inhalt

### Teil 1 · Grundlagen

1. [Geodaten als Funktion statt als Raster](01-geodaten-als-funktion.md)
2. [Training des neuronalen Netzes](02-training.md)
3. [Architektur eines neuronalen Netzes](03-architektur.md)

### Teil 2 · Bessere Netze für Geodaten *(in Arbeit)*

4. [Die Erde ist eine Kugel](04-kugelkoordinaten.md)
5. [Räumliche Frequenzen und Fourier-Features](05-fourier-features.md)
6. Vertiefung: Kugelflächenfunktionen *(in Arbeit)*
7. Vertiefung: SIREN *(in Arbeit)*
8. Das Kontextfeld *(in Arbeit)*

### Teil 3 · Messen, nutzen, einordnen *(in Arbeit)*

9. Rate und Fehler messen *(in Arbeit)*
10. Abfragen und exportieren *(in Arbeit)*
11. Einordnung und Ausblick *(in Arbeit)*

**Anhang**

- Messungen *(in Arbeit)*
