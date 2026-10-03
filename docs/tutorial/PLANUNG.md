# Planung: Tutorial „Geodaten in neuronalen Netzen speichern“

Arbeitsdokument für die Autoren, nicht Teil des Tutorials. Stand: 3. Oktober 2026.

## Regeln

- **Format:** Markdown im Repo unter `docs/tutorial/`, eine Datei pro Kapitel (`NN-titel.md`), Bilder in `img/`.
- **Vorgehen:** ein Kapitel nach dem anderen. Anmerkungen beim Gegenlesen als `/* … */` direkt in den Text; sie werden eingearbeitet und entfernt.
- **Kein Programmcode** im Tutorial. Erklärt wird mit Text, einfachen Rechenbeispielen, Diagrammen und Screenshots der App.
- **Keine Übungen, keine Lösungen.** Jedes Kapitel endet mit „Ideen zum Weiterprobieren“: offene Vorschläge für die App.
- **Messungen** (Trainingsergebnisse, Fehlerwerte) kommen gesammelt in den Anhang am Schluss, nicht in die Kapitel.
- **Chester:** Jedes Kapitel beginnt mit einem eigenen Bild von Chester, erzeugt mit Kling AI und `img/chester-referenz.png` als Referenz. Prompts stehen in [`img/chester-prompts.md`](img/chester-prompts.md).
- **Zahlen erklären, bevor sie auftauchen.** Bezugsgröße ist das Standardbeispiel: ETOPO 2022 auf 1080 × 2160 gemittelt, Vorlage *Fourier MLP (recommended)*, 82 561 Gewichte.
- **Quellen** im Text als [1], [2] …, am Kapitelende „Weiterführende Links“. Jede Quelle vorher prüfen.
- **Sprache:** Tutorial auf Deutsch, Begriffe aus der englischen App im Original und kursiv.

## Zielgruppe

Leute mit soliden Python-Kenntnissen und etwas numpy, die neuronale Netze höchstens aus Einführungen kennen. Erfahrung mit Geodaten ist nicht nötig.

## Kapitel

| Nr. | Kapitel | Stand |
|---:|---|---|
| 1 | Geodaten als Funktion statt als Raster | fertig |
| 2 | Training des neuronalen Netzes | Entwurf |
| 3 | Das einfachste Modell | geplant |
| 4 | Koordinaten gehören auf die Kugel | geplant |
| 5 | Räumliche Frequenzen und Fourier-Features | geplant |
| 6 | Vertiefung: Kugelflächenfunktionen | geplant |
| 7 | Vertiefung: SIREN | geplant |
| 8 | Das Kontextfeld | geplant |
| 9 | Rate und Fehler messen | geplant |
| 10 | Abfragen und exportieren | geplant |
| 11 | Einordnung und Ausblick | geplant |
| Anhang | Messungen | geplant |

Kapitel 6 und 7 sind Vertiefungen und lassen sich überspringen.

### 1 · Geodaten als Funktion statt als Raster — fertig

- **Lernziel:** Ein Höhenraster als Funktion begreifen, seine Rohgröße ausrechnen und die Kompressionsrate ehrlich bestimmen.
- **Inhalt:** ETOPO 2022 und seine Zeilen und Spalten; vom Raster zur Funktion; was Gewichte sind (Straßenbeispiel); das Standardbeispiel; Lernen andersherum; Rohgröße und Netzgröße; Verkleinern auf die Arbeitsauflösung.
- **In der App:** *Idea & Budget*, Regler *Grid resolution*, Vorlagen vergleichen.
- **Chester:** packt die gepixelte Erde in eine leuchtende Eichel.

### 2 · Training des neuronalen Netzes — Entwurf

- **Lernziel:** Verstehen, wie das Netz trainiert wird, warum es hier keine Testdaten gibt und was Erfolg stattdessen heißt. Den Begriff Overfitting erst im Kapitel einführen, nicht im Titel.
- **Inhalt:** Training in Schritten: zufällige Gitterzellen ziehen, Fehler messen, Gewichte ein wenig verbessern. Fehlermaße RMSE, MAE und R² in Worten. Warum Zellen am Pol kleiner sind als am Äquator und deshalb mit cos(Breite) gewichtet werden; gewichteter RMSE. Fehlerkurve lesen.
- **In der App:** *Training*, einmal mit und einmal ohne *Area-weighted sampling*.
- **Ideen:** Schritte und Lernrate variieren; beobachten, wann die Fehlerkurve abflacht.
- **Chester:** als Trainer mit Klemmbrett; die leuchtende Netz-Eichel aus Kapitel 1 (mit Ärmchen und Beinchen) stemmt eine Hantel.

### 3 · Das einfachste Modell

- **Lernziel:** Sehen, was ein lineares Modell kann und warum ein Netz versteckte Schichten braucht.
- **Inhalt:** 0 versteckte Schichten = gewichtete Summe der Eingaben; ohne Kodierung eine schräge Ebene durch die Erde. Der Knick zwischen den Schichten macht daraus ein MLP. Anknüpfung an das Straßenbeispiel aus Kapitel 1.
- **In der App:** *Hidden layers* auf 0, Kodierung `none`, dann mehr Schichten.
- **Ideen:** Parameterzahl bei 0, 1 und 4 Schichten vergleichen; Ergebnis auf der Karte ansehen.
- **Chester:** versucht, ein Gebirge mit einem einzigen Lineal nachzubauen.

### 4 · Koordinaten gehören auf die Kugel

- **Lernziel:** Die Naht bei ±180° und die Pole als Problem der Eingabe erkennen.
- **Inhalt:** (lon, lat) liegt auf einem Rechteck, die Erde ist eine Kugel. Der Punkt (x, y, z) auf der Einheitskugel hat weder Naht noch entartete Pole.
- **In der App:** Eingabe `lonlat` gegen `xyz`, Fehlerkarte an der Datumsgrenze vergleichen.
- **Ideen:** Gezielt den Pazifik bei 180° auf der Fehlerkarte ansehen.
- **Chester:** näht eine flache Weltkarte zu einem Ball zusammen.

### 5 · Räumliche Frequenzen und Fourier-Features

- **Lernziel:** Räumliche Frequenz und *spectral bias* verstehen, σ bewusst wählen.
- **Inhalt:** Auch Höhen haben Frequenzen: Wellen pro Strecke, von Kontinenten bis zu Graten. Ein einfaches Netz lernt lange Wellen schnell, kurze kaum. Fourier-Features liefern die Wellen fertig; die Zufallsmatrix entsteht aus einem Startwert und wird nicht gespeichert.
- **In der App:** *Plain MLP* gegen *Fourier MLP*; σ auf 1, 6 und 20.
- **Ideen:** Bei welchem σ wird die Fehlerkarte körnig?
- **Chester:** zupft an Saiten, die über die Erdkugel gespannt sind.

### 6 · Vertiefung: Kugelflächenfunktionen

- **Lernziel:** Die „echte“ Fourier-Basis der Kugel kennen und ihren Preis abschätzen.
- **Inhalt:** Kugelflächenfunktionen bis Grad L, vollständig und orthogonal, aber (L+1)² Eingaben. Mit 0 Schichten ist das genau die klassische Kugelfunktionsentwicklung der Geodäsie; das Training findet dabei die exakte Lösung.
- **In der App:** *Spherical harmonics (L = 16)* und *Spherical harmonic expansion (L = 40, linear)*.
- **Ideen:** L variieren und auf Parameterzahl und Trainingsdauer achten.
- **Chester:** sortiert Muster auf einer Kugel wie Kacheln eines Mosaiks.

### 7 · Vertiefung: SIREN

- **Lernziel:** Sinus als Aktivierung als Alternative zur Kodierung kennen.
- **Inhalt:** Sinus statt Knick in jeder Schicht, ohne Fourier-Features. Funktioniert nur mit passender Startbelegung der Gewichte; empfindlich beim Speichern in float16.
- **In der App:** *SIREN (sine activation)* gegen *Fourier MLP*.
- **Ideen:** Beide Vorlagen gleich lange trainieren und die Fehlerkarten vergleichen.
- **Chester:** surft auf Sinuswellen.

### 8 · Das Kontextfeld

- **Lernziel:** Nur den Rest lernen statt alles – und die Bytes dafür ehrlich mitzählen.
- **Inhalt:** Ein grobes 90 × 180-Gitter (31,6 KB) liefert Kontinente und Ozeanbecken; das Netz lernt nur die Abweichung. Kostet Bytes, bringt bei hoher Kompression am meisten.
- **In der App:** *Fourier MLP + context field*, Auflösung des Kontextfelds variieren.
- **Ideen:** Ab welcher Netzgröße lohnt sich das Kontextfeld nicht mehr?
- **Chester:** malt nur noch die Details in eine grob vorgezeichnete Karte.

### 9 · Rate und Fehler messen

- **Lernziel:** Eine Rate-Distortion-Kurve erzeugen und lesen.
- **Inhalt:** Mehrere Netzbreiten auf demselben Gitter; Kompressionsrate gegen Fehler. Unten rechts wäre ideal. PSNR in Worten.
- **In der App:** *Rate & Error*, mit und ohne Kontextfeld.
- **Ideen:** Welche Rate erreicht man bei einem bestimmten Fehler?
- **Chester:** wiegt Eicheln gegen Kartenstapel auf einer Waage.

### 10 · Abfragen und exportieren

- **Lernziel:** Das Feld als kontinuierliche Funktion nutzen und als Datei weitergeben.
- **Inhalt:** Abfrage an beliebigen Koordinaten – ehrlich: keine echte Super-Resolution, nur glatte Interpolation. Export in float32 und float16 und was float16 kostet. Die App zeigt ein Ladebeispiel; das Tutorial verweist darauf, ohne Code abzudrucken.
- **In der App:** *Super-Resolution* und *Export*.
- **Ideen:** Einen kleinen Ausschnitt sehr fein abfragen und mit dem Trainingsgitter vergleichen.
- **Chester:** verschickt seine Eichel als Päckchen.

### 11 · Einordnung und Ausblick

- **Lernziel:** Die Ergebnisse gegen die Literatur einordnen.
- **Inhalt:** Huang und Hoefler erreichen 300- bis über 3000-fache Kompression auf glatten Klimafeldern mit Zeitachse; Topographie ist zerklüftet und hat keine. Weiterführend: Hash-Gitter, Quantisierung schon beim Training, Zeitreihen f(lon, lat, t).
- **Ideen:** Welches eigene Geodatenset würde sich gut eignen, welches schlecht?
- **Chester:** blickt vom Ast über eine Landschaft mit vielen Datenbäumen.

### Anhang · Messungen

- Alle Trainingsergebnisse an einer Stelle: Varianten, Dateigröße, Kompressionsrate, gewichteter RMSE, PSNR, Trainingsdauer.
- Vorhanden aus der Entwicklung: Vergleich Fourier-MLP, Kugelflächenfunktionen, SIREN und lineare Modelle auf 1080 × 2160 (je ein Lauf) sowie die Tabelle aus der README auf 2160 × 4320.
- Rechner und Einstellungen für jede Messung angeben.

## Offene Punkte

- [ ] Messungen mit mehreren Startwerten wiederholen (z. B. 3 Läufe je Variante) und Mittelwerte angeben?
- [ ] Eine kleinere Variante der Experimente für Rechner ohne GPU beschreiben (Gitter 360 × 720, 1000 Schritte)?
- [ ] Kapitel 6: Training mit L = 40 dauert etwa 0,2 s pro Schritt. Für ein Experiment zu langsam?
