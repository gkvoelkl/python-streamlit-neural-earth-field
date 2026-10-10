# Planung: Tutorial „Geodaten in neuronalen Netzen speichern“

Arbeitsdokument für die Autoren, nicht Teil des Tutorials. Stand: 3. Oktober 2026.

## Regeln

- **Format:** Markdown im Repo, deutsche Kapitel unter `docs/tutorial/de/`, eine Datei pro Kapitel (`NN-titel.md`), Bilder gemeinsam in `docs/tutorial/img/` (aus einem Kapitel verlinkt als `../img/…`).
- **Zweisprachig (seit 10. Oktober 2026):** Deutsch ist das Original. Die englische Fassung liegt unter `docs/tutorial/en/` mit denselben Dateinamen und beginnt mit dem Hinweis „Translated from the German original“. Beide Fassungen verlinken sich oben gegenseitig (`../en/…` bzw. `../de/…`); noch nicht übersetzte Kapitel verlinkt die englische Fassung auf das deutsche Original. Englische Diagramme liegen in `img/en/` und entstehen mit `tools/translate_svg.py` aus den deutschen SVGs; die Übersetzungen stehen in `tools/svg-en.json`. Chester-Bilder und App-Screenshots gelten für beide Sprachen.
- **Website:** MkDocs Material mit `mkdocs-static-i18n` (`mkdocs.yml` im Wurzelordner), Deutsch an der Wurzel, Englisch unter `en/`. Neue Kapitel in `nav` und in `nav_translations` eintragen; `mkdocs build --strict` muss ohne Warnung durchlaufen.
- **Vorgehen:** ein Kapitel nach dem anderen. Anmerkungen beim Gegenlesen als `/* … */` direkt in den Text; sie werden eingearbeitet und entfernt.
- **Kein Programmcode** im Tutorial. Erklärt wird mit Text, einfachen Rechenbeispielen, Diagrammen und Screenshots der App.
- **Keine Übungen, keine Lösungen.** Jedes Kapitel endet mit „Ideen zum Weiterprobieren“: offene Vorschläge für die App.
- **Messungen** (Trainingsergebnisse, Fehlerwerte) kommen gesammelt in den Anhang am Schluss, nicht in die Kapitel.
- **Chester:** Jedes Kapitel beginnt mit einem eigenen Bild von Chester, erzeugt mit Kling AI und `img/chester-referenz.png` als Referenz. Prompts stehen in [`img/chester-prompts.md`](img/chester-prompts.md).
- **Zahlen erklären, bevor sie auftauchen.** Bezugsgröße ist das Standardbeispiel: ETOPO 2022 auf 1080 × 2160 gemittelt, Vorlage *Fourier MLP (recommended)*, 82 561 Gewichte.
- **Quellen** im Text als [1], [2] …, am Kapitelende „Weiterführende Links“. Jede Quelle vorher prüfen.
- **Figuren:** Chester (Eichhörnchen mit blauer Kappe) und Aiko, ein leuchtendes neuronales Netz mit Ärmchen und Beinchen. Beide Namen dürfen im Text vorkommen, wo es passt. Das Wort „Eichel“ wird nicht verwendet (Doppeldeutigkeit im Deutschen).
- **Sprache:** Tutorial auf Deutsch, Begriffe aus der englischen App im Original und kursiv.
- **Disclaimer:** Jedes Kapitel endet unter der Navigation mit dem kursiven Satz „Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. …“ (Wortlaut wie in Kapitel 1).

## Zielgruppe

Leute mit soliden Python-Kenntnissen und etwas numpy, die neuronale Netze höchstens aus Einführungen kennen. Erfahrung mit Geodaten ist nicht nötig.

## Kapitel

| Nr. | Teil | Kapitel | Stand |
|---:|---:|---|---|
| 1 | 1 | Geodaten als Funktion statt als Raster | fertig |
| 2 | 1 | Training des neuronalen Netzes | fertig |
| 3 | 1 | Architektur eines neuronalen Netzes | fertig |
| 4 | 2 | Die Erde ist eine Kugel | fertig |
| 5 | 2 | Höhen als Wellen | fertig |
| 6 | E | Vertiefung: Kugelflächenfunktionen | fertig |
| 7 | E | Vertiefung: SIREN | fertig |
| 8 | E | Das Kontextfeld | fertig |
| 9 | E | Rate und Fehler messen | Entwurf |
| 10 | E | Abfragen und exportieren | Entwurf |
| 11 | E | Einordnung und Ausblick | Entwurf |
| Anhang | | Messungen | Entwurf |

**Teil 1 · Grundlagen:** Kapitel 1 bis 3. **Teil 2 · Bessere Netze für Geodaten:** Kapitel 4 und 5. Kapitel 1 bis 5 sind der Kern des Tutorials; Kapitel 5 endet mit „Sie haben es geschafft!“, einem Bild von Chester und Aiko im Ziel (`img/kapitel-05-ziel.jpg`) und einer Liste der Ergänzungen. **E · Ergänzungen und Vertiefungen:** Kapitel 6 bis 11, einzeln lesbar; die Nummern bleiben.

### 1 · Geodaten als Funktion statt als Raster — fertig

- **Lernziel:** Ein Höhenraster als Funktion begreifen, seine Rohgröße ausrechnen und die Kompressionsrate ehrlich bestimmen.
- **Inhalt:** ETOPO 2022 und seine Zeilen und Spalten; vom Raster zur Funktion; was Gewichte sind (Straßenbeispiel); das Standardbeispiel; Lernen andersherum; Rohgröße und Netzgröße; Verkleinern auf die Arbeitsauflösung.
- **In der App:** *Idea & Budget*, Regler *Grid resolution*, Vorlagen vergleichen.
- **Chester:** packt die gepixelte Erde in Aiko, ein kleines leuchtendes neuronales Netz.

### 2 · Training des neuronalen Netzes — fertig

- **Lernziel:** Verstehen, wie das Netz trainiert wird, warum es hier keine Testdaten gibt und was Erfolg stattdessen heißt. Den Begriff Overfitting erst im Kapitel einführen, nicht im Titel.
- **Inhalt:** Training in Schritten: zufällige Gitterzellen ziehen, Fehler messen, Gewichte ein wenig verbessern. Fehlermaße RMSE, MAE und R² in Worten. Warum Zellen am Pol kleiner sind als am Äquator und deshalb mit cos(Breite) gewichtet werden; gewichteter RMSE. Fehlerkurve lesen.
- **In der App:** *Training*, einmal mit und einmal ohne *Area-weighted sampling*.
- **Ideen:** Schritte und Lernrate variieren; beobachten, wann die Fehlerkurve abflacht.
- **Chester:** als Trainer mit Klemmbrett; Aiko (mit Ärmchen und Beinchen) stemmt eine Hantel.

### 3 · Architektur eines neuronalen Netzes — fertig

- **Lernziel:** Sehen, was ein lineares Modell kann und warum ein Netz versteckte Schichten braucht.
- **Inhalt:** 0 versteckte Schichten = gewichtete Summe der Eingaben; ohne Kodierung eine schräge Ebene durch die Erde. Der Knick zwischen den Schichten macht daraus ein MLP. Anknüpfung an das Straßenbeispiel aus Kapitel 1.
- **In der App:** *Hidden layers* auf 0, Kodierung `none`, dann mehr Schichten.
- **Ideen:** Parameterzahl bei 0, 1 und 4 Schichten vergleichen; Ergebnis auf der Karte ansehen.
- **Chester:** als Baumeister in der Werkstatt; links ein einfaches Brett, in der Mitte ein Gebirge, das Aiko aus Leuchtstäben und Knoten wie ein Netz gebaut hat.

### 4 · Die Erde ist eine Kugel — fertig

- **Lernziel:** Die Naht bei ±180° und die Pole als Problem der Eingabe erkennen.
- **Inhalt:** (lon, lat) liegt auf einem Rechteck, die Erde ist eine Kugel. Der Punkt (x, y, z) auf der Einheitskugel hat weder Naht noch entartete Pole.
- **In der App:** Eingabe `lonlat` gegen `xyz`, Fehlerkarte an der Datumsgrenze vergleichen.
- **Ideen:** Gezielt den Pazifik bei 180° auf der Fehlerkarte ansehen.
- **Chester:** als Schneider; er wölbt eine flache Weltkarte über eine Kugel und näht die Kante zusammen, Aiko hält den Faden.

### 5 · Höhen als Wellen — fertig

- **Lernziel:** Räumliche Frequenz und *spectral bias* verstehen, σ bewusst wählen.
- **Inhalt:** Auch Höhen haben Frequenzen: Wellen pro Strecke, von Kontinenten bis zu Graten. Ein einfaches Netz lernt lange Wellen schnell, kurze kaum. Fourier-Features liefern die Wellen fertig; die Zufallsmatrix entsteht aus einem Startwert und wird nicht gespeichert.
- **In der App:** *Plain MLP* gegen *Fourier MLP*; σ auf 1, 6 und 20.
- **Ideen:** Bei welchem σ wird die Fehlerkarte körnig?
- **Chester:** spielt Harfe; die Wellen ziehen über die Erde zu Aiko, die sie auffängt (Chester erzeugt die Wellen, Aiko bekommt sie).

### 6 · Vertiefung: Kugelflächenfunktionen — fertig

- **Lernziel:** Die „echte“ Fourier-Basis der Kugel kennen und ihren Preis abschätzen.
- **Inhalt:** Kugelflächenfunktionen bis Grad L, vollständig und orthogonal, aber (L+1)² Eingaben. Mit 0 Schichten ist das genau die klassische Kugelfunktionsentwicklung der Geodäsie; das Training findet dabei die exakte Lösung.
- **In der App:** *Spherical harmonics (L = 16)* und *Spherical harmonic expansion (L = 40, linear)*.
- **Ideen:** L variieren und auf Parameterzahl und Trainingsdauer achten.
- **Chester:** setzt als Mosaikkünstler eine Erdkugel aus Kacheln zusammen, Aiko steht mit dem Sortierkasten daneben.

### 7 · Vertiefung: SIREN — fertig

- **Lernziel:** Sinus als Aktivierung als Alternative zur Kodierung kennen.
- **Inhalt:** Sinus statt Knick in jeder Schicht, ohne Fourier-Features. Ohne passende Startbelegung lernt das Netz die Trainingspunkte auswendig und rauscht dazwischen; empfindlich beim Speichern in float16.
- **In der App:** *SIREN (sine activation)* gegen *Fourier MLP*.
- **Ideen:** Beide Vorlagen gleich lange trainieren und die Fehlerkarten vergleichen.
- **Chester:** surft auf Sinuswellen.

### 8 · Das Kontextfeld — fertig

- **Lernziel:** Nur den Rest lernen statt alles – und die Bytes dafür ehrlich mitzählen.
- **Inhalt:** Ein grobes 90 × 180-Gitter (31,6 KB) liefert Kontinente und Ozeanbecken; das Netz lernt nur die Abweichung. Kostet Bytes, gewinnt aber bei gleicher Dateigröße bei allen drei Netzgrößen (Messung 4 000 Schritte, je ein Lauf); der Gewinn ist beim großen Netz am größten.
- **In der App:** *Fourier MLP + context field*, Auflösung des Kontextfelds variieren.
- **Ideen:** Ab welcher Netzgröße lohnt sich das Kontextfeld nicht mehr?
- **Chester:** malt nur noch die Details in eine grob vorgezeichnete Karte.

### 9 · Rate und Fehler messen — Entwurf

- **Lernziel:** Eine Rate-Distortion-Kurve erzeugen und lesen.
- **Inhalt:** Mehrere Netzbreiten auf demselben Gitter; Kompressionsrate gegen Fehler. Unten rechts wäre ideal. PSNR in Worten.
- **In der App:** *Rate & Error*, mit und ohne Kontextfeld.
- **Ideen:** Welche Rate erreicht man bei einem bestimmten Fehler?
- **Chester:** wiegt Aiko gegen einen Kartenstapel auf einer Waage (Prompt in `img/chester-prompts.md`).
- **Messung:** neun Netzbreiten 16–256, je 2 000 Schritte, ohne und mit Kontextfeld 90 × 180, je ein Lauf, Gitter 1080 × 2160, Mac mit MPS. Zum Vergleich die App-Voreinstellung mit 600 Schritten.

### 10 · Abfragen und exportieren — Entwurf

- **Lernziel:** Das Feld als kontinuierliche Funktion nutzen und als Datei weitergeben.
- **Inhalt:** Abfrage an beliebigen Koordinaten – ehrlich: keine echte Super-Resolution, nur glatte Interpolation. Export in float32 und float16 und was float16 kostet. Die App zeigt ein Ladebeispiel; das Tutorial verweist darauf, ohne Code abzudrucken.
- **In der App:** *Super-Resolution* und *Export*.
- **Ideen:** Einen kleinen Ausschnitt sehr fein abfragen und mit dem Trainingsgitter vergleichen.
- **Chester:** verschickt Aiko als Päckchen (Prompt in `img/chester-prompts.md`).
- **Messung:** Standardbeispiel, *Fourier MLP + context field* und *SIREN*, je 4 000 Schritte, ein Lauf, Gitter 1080 × 2160, Mac mit MPS. float16-Abweichung auf dem ganzen Gitter; Super-Resolution-Test gegen das Gitter 2160 × 4320.

### 11 · Einordnung und Ausblick — Entwurf

- **Lernziel:** Die Ergebnisse gegen die Literatur einordnen.
- **Inhalt:** Huang und Hoefler erreichen 300- bis über 3000-fache Kompression auf glatten Klimafeldern mit Zeitachse; Topographie ist zerklüftet und hat keine. Weiterführend: Hash-Gitter, Quantisierung schon beim Training, Zeitreihen f(lon, lat, t).
- **Ausblick Vektordaten** (Wunsch vom 10. Oktober 2026): Punkte, Linien, Flächen. Weg 1: als Feld (Dichte, Abstand, vorzeichenbehafteter Abstand; SINR, DeepSDF, Geo2Vec), Küste als 0-m-Linie des Netzes. Weg 2: Encoder für Machine Learning (Space2Vec, SatCLIP, Polygon-Encoder, Poly2Vec).
- **Ideen:** Welches eigene Geodatenset würde sich gut eignen, welches schlecht?
- **Chester:** blickt vom Ast über eine Landschaft mit vielen Datenbäumen; im Tal leuchten Punkte, Wege und eingezäunte Flächen als Bild für Vektordaten (Prompt in `img/chester-prompts.md`).

### Anhang · Messungen — Entwurf

- Alle Trainingsergebnisse an einer Stelle: Varianten, Dateigröße, Kompressionsrate, gewichteter RMSE, PSNR, Trainingsdauer.
- Vorhanden aus der Entwicklung: Vergleich Fourier-MLP, Kugelflächenfunktionen, SIREN und lineare Modelle auf 1080 × 2160 (je ein Lauf) sowie die Tabelle aus der README auf 2160 × 4320.
- Rechner und Einstellungen für jede Messung angeben.
- **Messung (10. Oktober 2026):** alle Vorlagen plus die Netze aus Kapitel 3, 1080 × 2160, 4 000 Schritte, je 3 Läufe (Startwerte 0–2; L = 40 nur 1 Lauf), Auswertung auf dem ganzen Gitter, Apple M4, MPS. Dazu float16 und eine Übersicht der Messungen aus den Kapiteln.

## Offene Punkte

- [ ] Kapitel 2 bis 11 und den Anhang übersetzen, jeweils mit den Diagrammen (Einträge in `tools/svg-en.json`).
- [ ] Begriffsliste Deutsch–Englisch, damit die Übersetzung einheitlich bleibt (z. B. Kontextfeld → *context field*, Startbelegung → *initialisation*, Gewichte → *weights*, Rate → *compression ratio*).
- [ ] E-Mail-Adresse im Disclaimer vor der Veröffentlichung prüfen (öffentlich sichtbar).
- [ ] Nutzungsbedingungen von Kling für die Veröffentlichung der Chester-Bilder prüfen.

- [x] Messungen mit mehreren Startwerten wiederholen: im Anhang 3 Läufe je Variante, Mittelwert und Spanne.
- [ ] Eine kleinere Variante der Experimente für Rechner ohne GPU beschreiben (Gitter 360 × 720, 1000 Schritte)?
- [ ] Kapitel 6: Training mit L = 40 dauert etwa 0,2 s pro Schritt. Für ein Experiment zu langsam?
