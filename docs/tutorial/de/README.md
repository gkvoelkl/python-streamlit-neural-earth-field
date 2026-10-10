# Tutorial: Geodaten in neuronalen Netzen speichern

*[English version](../en/README.md) – Kapitel 1 ist übersetzt, die übrigen folgen.*

Dieses Tutorial zeigt Schritt für Schritt, wie man das Höhenmodell der ganzen Erde in die Gewichte eines kleinen neuronalen Netzes packt – und warum die Gewichte dann die Datei sind. Jedes Kapitel verbindet eine Idee mit Experimenten in der Begleit-App aus diesem Repo. Programmcode zeigt das Tutorial bewusst nicht – wer hineinschauen will, findet alles im Ordner `src/`.

**Zielgruppe:** Leute mit soliden Python-Kenntnissen und etwas numpy. Neuronale Netze muss man höchstens aus Einführungen kennen, Erfahrung mit Geodaten ist nicht nötig.

## Vorbereitung

```bash
uv sync
uv run streamlit run app.py
```

Optional, aber empfohlen: das globale Höhenmodell ETOPO 2022 (456 MB) nach `data/` laden, wie in der [README des Repos](https://github.com/gkvoelkl/python-streamlit-neural-earth-field/blob/main/README_de.md#daten) beschrieben. Ohne die Datei läuft alles mit einer synthetischen Erde.

Die App ist auf Englisch. Begriffe aus der Oberfläche stehen im Tutorial im Original und kursiv, etwa *Idea & Budget*.

## Inhalt

### Teil 1 · Grundlagen

1. [Geodaten als Funktion statt als Raster](01-geodaten-als-funktion.md)
2. [Training des neuronalen Netzes](02-training.md)
3. [Architektur eines neuronalen Netzes](03-architektur.md)

### Teil 2 · Bessere Netze für Geodaten

4. [Die Erde ist eine Kugel](04-kugelkoordinaten.md)
5. [Höhen als Wellen](05-fourier-features.md)

Mit Kapitel 5 ist der Kern des Tutorials abgeschlossen.

### Ergänzungen und Vertiefungen

Die folgenden Kapitel bauen auf den Kapiteln 1 bis 5 auf. Sie lassen sich je nach Interesse einzeln lesen oder überspringen.

6. [Vertiefung: Kugelflächenfunktionen](06-kugelflaechenfunktionen.md)
7. [Vertiefung: SIREN](07-siren.md)
8. [Das Kontextfeld](08-kontextfeld.md)
9. [Rate und Fehler messen](09-rate-und-fehler.md)
10. [Abfragen und exportieren](10-abfragen-und-exportieren.md)
11. [Einordnung und Ausblick](11-einordnung-und-ausblick.md)

**Anhang**

- [Messungen](anhang-messungen.md)
