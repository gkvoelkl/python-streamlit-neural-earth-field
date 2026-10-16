# Tutorial: Geodaten in neuronalen Netzen speichern

*[English version](../en/README.md) – Kapitel 1 bis 5 sind übersetzt, die übrigen folgen.*

Dieses Tutorial zeigt Schritt für Schritt, wie man das Höhenmodell der ganzen Erde in die Gewichte eines kleinen neuronalen Netzes packt – und warum die Gewichte dann die Datei sind. Jedes Kapitel verbindet eine Idee mit Experimenten in der Begleit-App aus diesem Repo. Programmcode zeigt das Tutorial bewusst nicht – wer hineinschauen will, findet alles im Ordner `src/`.

**Zielgruppe:** Leute mit soliden Python-Kenntnissen und etwas numpy. Neuronale Netze muss man höchstens aus Einführungen kennen, Erfahrung mit Geodaten ist nicht nötig.

## Vorbereitung

Zum ersten Hineinschauen gibt es die App auch online, ohne Installation: <https://python-app-neural-earth-field.streamlit.app>. Sie läuft dort auf einer geteilten CPU ohne Grafikchip und arbeitet deshalb mit erheblich weniger Daten: gröbere Raster (voreingestellt 360 × 720 statt 1080 × 2160) und ein leichteres Training. Ihre Zahlen weichen darum von denen im Tutorial ab. Für die Experimente des Tutorials ist die eigene Installation besser:

```bash
git clone https://github.com/gkvoelkl/python-streamlit-neural-earth-field.git
cd python-streamlit-neural-earth-field
uv sync
uv run streamlit run app.py
```

Beim ersten Start fragt die App, ob sie das globale Höhenmodell ETOPO 2022 (456 MB) herunterladen darf; ohne diese Daten läuft sie nicht. Wer die Datei lieber selbst lädt, findet den Befehl in der [README des Repos](https://github.com/gkvoelkl/python-streamlit-neural-earth-field/blob/main/README_de.md#daten).

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

6. [Kugelflächenfunktionen](06-kugelflaechenfunktionen.md)
7. [SIREN](07-siren.md)
8. [Das Kontextfeld](08-kontextfeld.md)
9. [Rate und Fehler messen](09-rate-und-fehler.md)
10. [Abfragen und exportieren](10-abfragen-und-exportieren.md)
11. [Einordnung und Ausblick](11-einordnung-und-ausblick.md)

**Anhang**

- [Messungen](anhang-messungen.md)
