# Kapitel 2 · Training des neuronalen Netzes

![Chester mit blauer Kappe und Klemmbrett als Trainer. Vor ihm stemmt eine leuchtende Eichel mit dünnen Ärmchen, Beinchen und Turnschuhen eine Langhantel über den Kopf; im Körper der Eichel leuchten feine Netzlinien. Im Hintergrund schwebt eine Erdkugel aus Pixeln.](img/kapitel-02-chester.jpg)

**Inhalt:** Wie das Netz seine Gewichte lernt, warum es hier alles auswendig lernen soll, warum die Pole einen Ausgleich brauchen und woran man sieht, wie gut das Training läuft.

## Ausgangspunkt

Aus Kapitel 1 haben wir zwei Dinge: ein Höhenraster mit 2 332 800 Werten und ein Netz mit 82 561 Gewichten. Am Anfang sind diese Gewichte zufällige Zahlen, und das Netz liefert für jede Koordinate Unsinn.

**Training** heißt: Gewichte finden, mit denen das Netz die Höhen des Rasters möglichst genau wiedergibt.

## Lernen heißt: Gewichte nachstellen

Zurück zur ansteigenden Straße aus Kapitel 1. Angenommen, wir kennen die beiden Gewichte 300 und 0,05 noch nicht und raten zunächst 0 und 0. Dann sagt die Formel für jede Stelle der Straße 0 m voraus – weit weg von den gemessenen Höhen.

Jetzt fragen wir für jedes Gewicht: Wird der Fehler kleiner oder größer, wenn ich es ein wenig erhöhe? Und dann stellen wir es ein kleines Stück in die günstige Richtung. Das wiederholen wir, bis die Formel zu den Messwerten passt. Schritt für Schritt wandern die beiden Gewichte so auf 300 und 0,05 zu.

Dafür muss man nicht ausprobieren. Die Mathematik liefert für alle Gewichte gleichzeitig, in welche Richtung und wie stark der Fehler auf eine Änderung reagiert (die Ableitung). Das Verfahren heißt **Gradientenabstieg**.

Ein Bild aus dem Gelände hilft: Man steht im Nebel an einem Hang und will ins Tal. Man sieht das Tal nicht, spürt aber unter den Füßen, wo es bergab geht – und macht einen kleinen Schritt in diese Richtung. Das Tal ist der kleinste Fehler, die Position sind die Gewichte. Beim Netz hat diese Landschaft nicht zwei, sondern 82 561 Richtungen.

**Die Lernrate** bestimmt, wie groß jeder Schritt ist. Zu große Schritte springen über das Tal hinweg, zu kleine kommen kaum voran. Die App geht dabei in drei Phasen vor:

1. In den ersten 100 Schritten wächst die Schrittweite langsam an, damit die zufälligen Startgewichte nicht gleich weit danebenspringen.
2. Dann arbeitet sie mit der eingestellten Lernrate.
3. Bis zum Ende schrumpft die Schrittweite allmählich auf 2 % davon, damit sich die Gewichte zum Schluss fein einstellen können.

Genauer gesagt verwendet die App eine Weiterentwicklung des Gradientenabstiegs namens **Adam** [1], die für jedes Gewicht eine eigene, passende Schrittweite findet.

## Ein Trainingsschritt

![Ein Trainingsschritt in vier Teilen: Stichprobe von 32 768 Zellen ziehen, das Netz rechnet für jede Zelle eine Höhe, der Fehler zur echten Höhe wird gemessen, alle Gewichte werden ein Stück nachgestellt. Danach beginnt der nächste Schritt mit neuen Zellen, insgesamt 4 000-mal.](img/training-schleife.svg)

Jeder Schritt arbeitet nicht mit allen 2,3 Millionen Rasterzellen, sondern mit einer **Stichprobe** von 32 768 zufällig gezogenen. Das ist viel schneller, und für die Richtung, in die die Gewichte wandern sollen, genügt eine Stichprobe dieser Größe.

Weil jeder Schritt neue Zellen zieht, kommen über das ganze Training trotzdem alle an die Reihe: 4 000 Schritte × 32 768 Zellen ergeben gut 131 Millionen Ziehungen, also im Mittel rund 56 pro Zelle.

## Auswendiglernen ist hier erwünscht

Normalerweise soll ein neuronales Netz aus Beispielen etwas Allgemeines lernen und es auf neue Fälle anwenden. Deshalb hält man einen Teil der Daten zurück und prüft damit, ob das Netz auch unbekannte Fälle richtig einschätzt. Passt sich ein Netz zu genau an seine Trainingsdaten an und versagt dann bei neuen, heißt das **Overfitting** – eigentlich der klassische Fehler beim maschinellen Lernen.

Hier ist es genau umgekehrt. Es gibt keine neuen Fälle: Das Raster *ist* der vollständige Datensatz. Das Netz soll ihn so genau wie möglich auswendig lernen, und deshalb wird nichts zurückgehalten. Gemessen wird, wie gut das Netz das Raster wiedergibt.

Ganz gelingt das nie. Im Standardbeispiel muss jedes Gewicht im Mittel 28 Rasterzellen „erklären“. Für jede Einzelheit reicht das nicht – das Netz glättet. Wie stark, entscheidet vor allem die Größe des Netzes, also die Kompressionsrate aus Kapitel 1.

## Die Pole brauchen einen Ausgleich

![Gleiche Gradzahl, verschiedene Fläche: Eine Rasterzelle von 1° × 1° ist am Äquator 111 km breit, bei 45° Breite 79 km und bei 70° nur 38 km. Jenseits von 60° Breite liegt ein Drittel der Rasterzeilen, aber nur 13 % der Erdoberfläche.](img/zellflaeche.svg)

Jede Zeile des Rasters hat gleich viele Zellen, egal ob sie am Äquator oder nahe am Pol liegt. Die Zellen sind aber verschieden groß: Zu den Polen hin rücken die Längengrade zusammen, die Zellen werden schmaler. Ihre Breite schrumpft mit dem Kosinus der geografischen Breite.

Zieht man die Zellen einfach gleichmäßig, bekommen die Polargebiete im Training ein Gewicht, das ihrer Fläche nicht entspricht – jenseits von 60° Breite rund zweieinhalbmal zu viel. Das Netz würde viel Mühe auf Arktis und Antarktis verwenden und dafür bei den Kontinenten sparen.

Die App gleicht das aus: Sie zieht die Zeilen mit einer Wahrscheinlichkeit, die zur Fläche ihrer Zellen passt. Zeilen am Äquator kommen häufiger dran, Zeilen am Pol seltener. In der Seitenleiste heißt das *Area-weighted sampling*.

## Den Fehler messen

Für jede Rasterzelle gibt es einen Fehler: die Höhe, die das Netz ausrechnet, minus die Höhe im Raster. Um aus Millionen solcher Einzelfehler eine Zahl zu machen, zeigt die App mehrere Maße:

- **RMSE** (*root mean square error*): jeden Fehler quadrieren, den Mittelwert bilden, daraus die Wurzel ziehen. Das Ergebnis ist wieder in Metern. Durch das Quadrieren zählen große Fehler stärker als kleine.

  ```math
  \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\text{Netz}_i - \text{Raster}_i)^2}
  ```

- **MAE** (*mean absolute error*): der mittlere Fehler ohne Vorzeichen, ebenfalls in Metern. Er ist meist kleiner als der RMSE, weil Ausreißer weniger ins Gewicht fallen.
- **Gewichteter RMSE** (*weighted RMSE*): wie der RMSE, aber jede Zelle zählt nach ihrer Fläche. So rechnet auch die Wetterforschung, etwa im Vergleichsdatensatz WeatherBench [2]. Für die ganze Erde ist das das fairste Maß.
- **R²**: der Anteil der Höhenunterschiede, den das Netz erklärt. 1 heißt perfekt; 0 hieße, das Netz wäre nicht besser als überall dieselbe mittlere Höhe.

Während des Trainings misst die App diese Werte regelmäßig an einer festen Stichprobe von 200 000 Zellen – immer denselben, damit die Werte vergleichbar sind. Das ganze Raster wird erst ganz am Ende durchgerechnet; sonst würde das Messen länger dauern als das Lernen.

Die **Fehlerkurve** zeigt den RMSE über die Trainingsschritte. Ihre Höhenachse ist logarithmisch: Jeder gleich große Abstand nach unten bedeutet, dass der Fehler um denselben Faktor kleiner geworden ist. Typisch ist ein steiler Abfall in den ersten paar hundert Schritten und danach eine lange, flache Verbesserung.

## Ausprobieren

1. In der Seitenleiste unter *3 · Training* stehen die Einstellungen für das Training. Für das Standardbeispiel bleiben sie, wie sie sind.

   <img src="img/app-training-einstellungen.png" alt="Seitenleiste, Abschnitt 3 · Training: Steps 4000, Points per step 32768, Learning rate 0.003, Häkchen bei Area-weighted sampling, Device auto, darunter der Knopf Train." width="300">

   | Einstellung | Bedeutung |
   |---|---|
   | *Steps* | Anzahl der Trainingsschritte |
   | *Points per step* | Größe der Stichprobe je Schritt |
   | *Learning rate* | Schrittweite beim Nachstellen der Gewichte |
   | *Area-weighted sampling* | Ausgleich für die Pole |
   | *Device* | Rechenwerk: *auto* nimmt die Grafikeinheit, wenn es eine gibt (*mps* auf Apple-Rechnern, *cuda* bei NVIDIA), sonst den Prozessor (*cpu*) |

2. Auf *🚀 Train* klicken und den Reiter *Training* öffnen. Oben läuft ein Fortschrittsbalken, darunter stehen die aktuellen Fehlermaße und die Fehlerkurve, die mit jedem Messpunkt wächst.

   ![Der Reiter Training während des Trainings, nach 1101 von 4000 Schritten: Fortschrittsbalken, die Kennzahlen RMSE, weighted RMSE, MAE und R² sowie die Fehlerkurve, die nach einem steilen Abfall flach weiterfällt.](img/app-training-laeuft.png)

3. Warten. Auf einem Apple-M-Rechner dauert das Standardbeispiel drei bis fünf Minuten, auf einem Rechner ohne Grafikeinheit deutlich länger. Dann stellt man in der Seitenleiste besser ein kleineres Raster ein.

4. Am Ende rechnet die App das ganze Raster einmal durch und meldet *Done*. Das trainierte Netz bleibt erhalten, bis man neu trainiert; die Reiter *Globe 3D*, *Maps & Error* und *Super-Resolution* zeigen jetzt das Ergebnis.

   ![Der Reiter Training nach Abschluss: alle 4000 Schritte durchlaufen, die Fehlerkurve über das ganze Training und die grüne Meldung, dass die Reiter Globe, Maps und Super-Resolution das Ergebnis zeigen.](img/app-training-fertig.png)

Ändert man nach dem Training die Gitterauflösung, passt das Netz nicht mehr zum Raster. Die App weist dann darauf hin, dass neu trainiert werden muss.

## Ideen zum Weiterprobieren

- Die *Learning rate* auf 0.01 und auf 0.0001 stellen und die Fehlerkurven vergleichen. Wann geht es schneller, wann wird es unruhig?
- Mit wenigen *Points per step* (4096) und mit sehr vielen (131072) trainieren. Was ändert sich an der Dauer, was an der Kurve?
- 1 000 statt 4 000 Schritte – oder 10 000. Lohnt sich das längere Training?
- Das Häkchen bei *Area-weighted sampling* entfernen und neu trainieren. Wie verändern sich RMSE und gewichteter RMSE im Vergleich?
- Nach dem Training einen Blick in den Reiter *Maps & Error* werfen: Wo auf der Erde liegen die größten Fehler?
- Unter *Device* einmal *cpu* statt *auto* wählen und die Trainingsdauer vergleichen.

## Weiterführende Links

- [1] Kingma und Ba: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014
- [2] Rasp et al.: [WeatherBench: A benchmark dataset for data-driven weather forecasting](https://arxiv.org/abs/2002.00469), 2020 – definiert den flächengewichteten RMSE
- Google Machine Learning Crash Course: [Gradientenabstieg](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent?hl=de) – Schritt-für-Schritt-Erklärung an einem Beispiel
- 3Blue1Brown: [Gradient descent, how neural networks learn](https://www.3blue1brown.com/lessons/gradient-descent) – anschauliches Video (Englisch)

---

[Übersicht](README.md) · Zurück: [Kapitel 1 · Geodaten als Funktion statt als Raster](01-geodaten-als-funktion.md) · Weiter: Kapitel 3 · Das einfachste Modell *(in Arbeit)*
