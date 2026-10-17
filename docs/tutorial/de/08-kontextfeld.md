# Geodaten in neuronalen Netzen speichern
# Kapitel 8 · Das Kontextfeld

*[English version](../en/08-kontextfeld.md)*

![Chester mit blauer Kappe malt mit einem feinen Pinsel Einzelheiten auf eine große Erdkugel in einem hellen Atelier; daneben hält Aiko, ein leuchtendes neuronales Netz, die Farbpalette.](../img/kapitel-08-chester.jpg)

**Inhalt:**

* Wie eine sehr grobe Höhenkarte dem Netz die Kontinente und Ozeanbecken abnimmt
* Warum diese Karte in die Dateigröße gehört
* Ob sich der zusätzliche Speicher lohnt

Statt die ganze Erde aus dem Nichts zu lernen, bekommt das neuronale Netz eine sehr grobe Höhenkarte mit und lernt dazu nur, was noch fehlt. Diese grobe Karte kostet zusätzlichen Speicher. Dieses Kapitel ergänzt Kapitel 5. Sie können es auch überspringen.

## Ausgangspunkt

Bisher muss das Netz alle Höhen der Erde komplett lernen: die Kontinente und Ozeanbecken genauso wie die Gebirge und Täler. Die Höhen der Erde schwanken dabei gewaltig: Zwischen dem tiefsten Punkt des Marianengrabens und dem Gipfel des Mount Everest liegen fast 20 km.

Ein großer Teil dieser Schwankung ist aber leicht zu beschreiben: Hier liegt ein Kontinent, dort ein tiefes Becken im Meer. Dafür braucht man keine feine Auflösung.

Die Idee dieses Kapitels: Diese groben Formen liefert man dem neuronalen Netz fertig mit, und dieses kümmert sich nur noch um den Rest.

## Das Kontextfeld

Das **Kontextfeld** ist eine sehr grobe Höhenkarte der Erde. In der Vorlage der App hat sie 90 × 180 Gitterpunkte, also alle 2° einen Wert – am Äquator etwa alle 220 km. Jeder Wert ist der Mittelwert der Höhen in seiner Zelle. Zwischen den Gitterpunkten werden die Werte geradlinig verbunden.

![Drei Karten des Alpenraums von 2° bis 18° Ost und 40° bis 52° Nord, jeweils mit weißen Gitterlinien alle 2° und einem Punkt in jeder Zellmitte. Links das Original mit Alpenbogen, Po-Ebene und Mittelmeer. In der Mitte das Kontextfeld: jede Zelle einfarbig mit ihrer gespeicherten Höhe, zum Beispiel 1 520 m in der Zelle mit dem Alpenhauptkamm und −2 650 m im Mittelmeer. Rechts dieselben Werte, zwischen den Gitterpunkten geradlinig verbunden: eine weiche Karte, auf der die Alpen nur als heller Rücken zu sehen sind.](../img/kontextfeld-gitter.png)

Die Abbildung zeigt das am Beispiel der Alpen:

1. **Original:** die Höhen des Standardgitters. Die weißen Gitterlinien teilen die Karte in Zellen von 2° × 2°.
2. **Kontextfeld:** Für jede Zelle wird nur eine Zahl gespeichert, der Mittelwert aller Höhen in der Zelle. Er gehört zum Gitterpunkt in der Zellmitte. Die Zelle mit dem Alpenhauptkamm bekommt so 1 520 m, obwohl ihr höchster Gipfel, der Ortler, 3 905 m hoch ist – Täler und Gipfel gleichen sich im Mittel aus.
3. **Dazwischen verbunden:** Für einen Ort zwischen den Gitterpunkten werden die Werte der vier nächsten Gitterpunkte gemischt, je näher, desto stärker. So entstehen keine Stufen an den Zellgrenzen, sondern eine weiche Fläche.

Damit lässt sich jede Höhe in zwei Teile zerlegen:

```math
\text{Höhe} = \text{Kontextfeld} + \text{Rest}
```

![Drei Weltkarten nebeneinander: Das Kontextfeld 90 × 180 zeigt Kontinente und Ozeanbecken weich gezeichnet. Plus der Rest, den das Netz lernt: fast überall weiß, nur an Küsten, Gebirgen, Inselbögen und Tiefseegräben rote und blaue Linien. Gleich die Höhe im Original.](../img/kontextfeld-welt.png)

Das Kontextfeld enthält schon die Kontinente und Ozeanbecken. Im Rest bleibt fast nur übrig, was auf kleinem Raum passiert: Küsten, Gebirge, Inselbögen und Tiefseegräben. Im Mittel weicht der Rest nur um rund 450 m von null ab – statt 2 500 m.

Am Alpenprofil aus Kapitel 3 sieht das so aus:

![Oben das Alpenprofil bei 46,4° Nord in Grau, darüber das Kontextfeld als orange Linie mit Stützstellen alle 2°: eine flache Kurve, die über den Alpen nur auf etwa 1 000 m ansteigt. Unten der Rest: über den Alpen bis zu 1 600 m positiv, in den Tälern und im Flachland leicht negativ.](../img/kontextfeld-profil.png)

Das Kontextfeld ist viel zu grob für die Gipfel und Täler – aber es weiß, dass hier ein Gebirge liegt. Die Einzelheiten muss das Netz weiterhin selbst lernen.

## Wie das Netz das Kontextfeld nutzt

Für jeden Ort schaut die App zuerst im Kontextfeld nach. Dieser Wert wird auf zwei Arten verwendet:

1. **Als zusätzliche Eingabe.** Neben den 256 Zahlen der Fourier-Features aus Kapitel 5 bekommt das Netz die Höhe aus dem Kontextfeld als 257. Zahl. So weiß es, ob es gerade über einem Gebirge oder einem Ozeanbecken rechnet.
2. **Als Grundlage für das Ergebnis.** Das Netz gibt nicht mehr die Höhe aus, sondern nur den Rest. Die App zählt beides zusammen:

```math
\text{Höhe} = \text{Kontextfeld} + \text{Ausgabe des Netzes}
```

Für das Training ändert sich nichts: Das Netz wird weiterhin mit den echten Höhen verglichen, und die Gewichte werden so nachjustiert, dass die Summe stimmt.

## Ehrlich zählen: Das Kontextfeld gehört zur Datei

Das Kontextfeld ist ein Teil der gespeicherten Daten. Ohne es kann man aus den Gewichten keine einzige Höhe mehr berechnen. Es muss deshalb in die Dateigröße und in die Kompressionsrate eingehen.

Die App speichert das Kontextfeld wie das Raster aus Kapitel 1, mit 2 Byte pro Wert: 90 × 180 = 16 200 Werte × 2 Byte = 32 400 Byte, also 31,6 KB.

Dazu kommt das Netz. Die zusätzliche Eingabe kostet in der ersten Schicht ein Gewicht pro Neuron, bei 128 Neuronen also 128 Gewichte mehr: 82 561 + 128 = 82 689 Gewichte. Zusammen:

| Teil | Größe |
|---|---:|
| 82 689 Gewichte × 4 Byte | 330 756 Byte |
| Kontextfeld 90 × 180 × 2 Byte | 32 400 Byte |
| Mittelwert, Standardabweichung und Startwert der Fourier-Features | 12 Byte |
| **Datei** | **363 168 Byte = 354,7 KB** |

Die Kompressionsrate sinkt dadurch von 14 auf 13. Genau diese Zahlen zeigt die App im Reiter *Idea & Budget*:

![Die vier Kennzahlen im Reiter Idea & Budget für die Vorlage Fourier MLP + context field: 2 332 800 Gitterpunkte, 82 689 Parameter, float32 plus 31,7 KB Zusatz ergibt 354,7 KB, Kompressionsrate 13.](../img/app-kontextfeld-budget.png)

Die App weist die 31,7 KB als *overhead* aus: das Kontextfeld plus die 12 Byte für Mittelwert, Standardabweichung und Startwert.

Nicht jede Veröffentlichung zählt so. Mostajeran und Kollegen geben ihrem Netz ebenfalls Zusatzdaten als Kontext mit, rechnen deren Speicher aber nicht in die Kompressionsrate ein [1]. Für einen fairen Vergleich sollte man deshalb immer nachsehen, was in einer Kompressionsrate mitgezählt ist.

## Lohnt sich das?

Das Kontextfeld kostet 31,6 KB. Dieselben Bytes könnte man auch in ein etwas breiteres Netz stecken. Ein fairer Vergleich stellt deshalb Varianten mit **gleicher Dateigröße** nebeneinander: Netze mit Kontextfeld und Netze ohne Kontextfeld, die dafür ein paar Neuronen mehr pro Schicht haben.

Zuerst das Standardbeispiel auf der Karte:

![Drei Karten von Europa: das Original, das Netz ohne Kontextfeld und das Netz mit Kontextfeld 90 × 180. Ohne Kontextfeld sind Küsten und Gebirge stark verwaschen, die Alpen sind ein weicher Fleck. Mit Kontextfeld sind die Umrisse von Italien, Griechenland und der Iberischen Halbinsel deutlich schärfer, die Alpen und Pyrenäen klar erkennbar.](../img/europa-kontextfeld.png)

Mit Kontextfeld werden Küsten und Gebirge deutlich schärfer, obwohl die Datei nur um 10 % wächst. Das Netz muss seine Gewichte nicht mehr für die Lage der Kontinente aufwenden und hat mehr für die Einzelheiten übrig.

Für drei Netzgrößen, jeweils mit gleicher Dateigröße:

| Vorlage | Datei | Kompressionsrate | Fehler ohne Kontextfeld | Fehler mit Kontextfeld 90 × 180 |
|---|---:|---:|---:|---:|
| *Fourier MLP, very small* | 75 KB | 61 | 339 m | 302 m |
| *Fourier MLP (recommended)* | 355 KB | 13 | 241 m | 185 m |
| *Fourier MLP, large* | 1 575 KB | 2,9 | 181 m | 121 m |

Als Fehler steht hier der flächengewichtete RMSE aus Kapitel 2. Bei den Netzen ohne Kontextfeld wurden die Schichten etwas breiter gemacht (70 statt 48, 136 statt 128 und 260 statt 256 Neuronen), damit die Datei genauso groß ist. Jede Zeile ist ein einzelner Trainingslauf mit 4 000 Schritten; die Werte schwanken von Lauf zu Lauf etwas.

Bei allen drei Größen gewinnt das Kontextfeld, und zwar deutlich: Das Standardnetz mit Kontextfeld (355 KB) ist genauso genau wie das große Netz *Fourier MLP, large* ohne Kontextfeld (1 542 KB, laut Anhang 185 m im Mittel von drei Läufen) – bei weniger als einem Viertel der Dateigröße.

### Wie fein sollte das Kontextfeld sein?

Die App bietet drei Auflösungen. Je feiner das Kontextfeld, desto mehr Einzelheiten liefert es – und desto mehr kostet es:

| Kontextfeld | Abstand der Gitterpunkte | Größe | Datei mit Standardnetz | Kompressionsrate | Fehler |
|---|---:|---:|---:|---:|---:|
| ohne | – | – | 323 KB | 14,1 | 245 m |
| 45 × 90 | 4° | 7,9 KB | 331 KB | 13,8 | 204 m |
| 90 × 180 | 2° | 31,6 KB | 355 KB | 12,8 | 185 m |
| 180 × 360 | 1° | 126,6 KB | 450 KB | 10,1 | 167 m |

Schon das sehr grobe Kontextfeld 45 × 90 mit nur 7,9 KB senkt den Fehler um ein Sechstel. Jede Verdopplung der Auflösung vervierfacht die Größe des Kontextfelds, bringt aber immer weniger. Wie man solche Abwägungen zwischen Dateigröße und Fehler systematisch darstellt, zeigt [Kapitel 9](09-rate-und-fehler.md).

## Ausprobieren

Wählen Sie in der Seitenleiste unter **2 · Network** die Vorlage **Fourier MLP + context field** und klappen Sie **Architecture details** auf.

Ganz unten ist **Provide coarse context field** eingeschaltet. Darunter legt **Context field resolution** fest, wie viele Gitterpunkte das Kontextfeld von Nord nach Süd hat: 45, 90 oder 180.

   <img src="../img/app-kontextfeld-einstellungen.png" alt="Ausschnitt der Seitenleiste: Neurons per layer 128, Provide coarse context field eingeschaltet, Context field resolution 90 auf einer Skala von 45 bis 180." width="300">

Vergleichen Sie im Reiter **Idea & Budget** die Dateigröße mit der Vorlage **Fourier MLP (recommended)**.

Trainieren Sie dann und sehen Sie sich im Reiter **Maps & Error** die Fehlerkarte an. Wo liegen die größten Fehler jetzt?

## Ideen zum Weiterprobieren

- Die Auflösung des Kontextfelds auf 45 und auf 180 stellen. Wie viel größer wird die Datei, und wie viel genauer die Karte?
- Bei der Vorlage *Fourier MLP, very small* das Kontextfeld 180 × 360 einschalten. Das Kontextfeld ist dann fast dreimal so groß wie das Netz. Wer liefert jetzt die meisten Einzelheiten?
- Im Reiter *Rate & Error* den Vergleich *Also with context field (90 × 180)* einschalten – mehr dazu in [Kapitel 9](09-rate-und-fehler.md).

## Weiterführende Links

- [1] Mostajeran et al.: [Context-aware implicit neural representations to compress Earth systems model data](https://pmc.ncbi.nlm.nih.gov/articles/PMC12271555/), Scientific Reports 15, 25932 (2025) – Netz mit Topographie und mittlerer Temperatur als Kontext für Klimadaten; der Kontext wird nicht in die Kompressionsrate eingerechnet (Englisch)
- Wikipedia: [Bilineare Interpolation](https://de.wikipedia.org/wiki/Bilineare_Interpolation) – wie die Werte zwischen den Gitterpunkten des Kontextfelds entstehen

---

[Übersicht](README.md) · Zurück: [Kapitel 7 · SIREN](07-siren.md) · Weiter: [Kapitel 9 · Rate und Fehler messen](09-rate-und-fehler.md)

*Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. Es soll helfen, mehr über das Thema „Geodaten in neuronalen Netzen speichern“ zu lernen. Verbesserungsvorschläge gerne an gkvoelkl@nelson-games.de.*
