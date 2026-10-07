# Kapitel 6 · Vertiefung: Kugelflächenfunktionen

![Chester mit blauer Kappe setzt mit einer Pinzette Mosaiksteine in eine Erdkugel aus vielen blauen, grünen und ockerfarbenen Kacheln. Links steht Aiko, ein leuchtendes neuronales Netz, vor einem Sortierkasten; auf dem Tisch liegen weitere Mosaiksteine.](img/kapitel-06-chester.jpg)

**Inhalt:** Die Streifenmuster aus Kapitel 5 sind zufällig gewählt. Es gibt aber auch einen festen, geordneten Satz von Mustern, der genau zur Kugel passt: die sogenannte Kugelflächenfunktionen. Dieses Kapitel ist eine Vertiefung. Sie können es gerne überspringen.

## Ausgangspunkt

In Kapitel 5 wurden 128 Streifenmuster mit zufälliger Richtung und zufälligem Abstand über die Erde gelegt. Das funktioniert gut, ist aber ein Notbehelf: Gerade Streifen passen nicht genau auf eine Kugel.

Mathematiker kennen seit langem einen geordneten Satz von solchen Mustern, der genau für die Kugel gemacht ist. Diese heißen **Kugelflächenfunktionen** (englisch *spherical harmonics*). Sie sind das Gegenstück zur Fourier-Analyse aus Kapitel 5.

## Kugelflächenfunktionen

Die Kugelflächenfunktionen sind nach ihrem **Grad** geordnet. Jeder Grad bringt andere Muster, jedes feiner als die davor:

![Kugelflächenfunktionen der Grade 0 bis 3 auf Erdkugeln, als Pyramide angeordnet. Grad 0: ein einziges Muster, die ganze Kugel gleich. Grad 1: drei Muster, jeweils eine Hälfte positiv, eine negativ. Grad 2: fünf Muster mit vier Feldern oder Bändern. Grad 3: sieben noch feinere Muster. Bis Grad 3 sind es zusammen 16 Muster.](img/kugelflaechen-grad0-3.png)

Wie bei den Streifenmustern in Kapitel 5 ergibt sich auch hier eine Welle, wenn man quer über die Kugel geht und die Werte aufträgt:

![Zwei Kugelflächenfunktionen vom Grad 3 auf Erdkugeln. Links ein Muster aus Bändern mit einer Linie von Süd nach Nord; darunter der Verlauf entlang der Linie: eine Welle, die zum Nordpol hin auf den Höchstwert steigt. Rechts ein Muster aus Sektoren mit einer Linie entlang des Äquators; darunter eine gleichmäßige Welle mit einem Gipfel bei 0° und Tälern bei 60° West und 60° Ost – rund um die Erde genau drei Wellen.](img/kugelflaeche-grad3-schnitt.png)

Anders als die geraden Streifen aus Kapitel 5 passen diese Wellen genau auf die Kugel: Entlang des Äquators geht beim Sektoren-Muster eine ganze Zahl von Wellen auf – beim Grad 3 genau drei.

- **Grad 0:** ein einziges Muster, überall gleich. Es steht für die mittlere Höhe.
- **Grad 1:** drei Muster. Jedes teilt die Kugel in eine positive und eine negative Hälfte – eines nach Nord und Süd, zwei nach Ost und West.
- **Grad 2:** fünf Muster, die Kugel ist in vier Felder oder in Bänder geteilt.
- **Jeder weitere Grad** bringt **2 · Grad + 1** neue Muster, und die Felder werden kleiner.

Bis zu einem **Grad *L*** gibt es also **(*L* + 1)²** Muster. Die feinste Welle hat ungefähr eine Länge von 40 000 km geteilt durch *L* – dem Erdumfang geteilt durch den Grad:

| höchster Grad *L* | Muster insgesamt | kürzeste Welle |
|---:|---:|---:|
| 2 | 9 | etwa 20 000 km |
| 8 | 81 | etwa 5 000 km |
| 16 | 289 | etwa 2 500 km |
| 40 | 1 681 | etwa 1 000 km |
| 100 | 10 201 | etwa 400 km |

Anders als die zufälligen Streifen aus Kapitel 5 sind diese Muster **vollständig**: Bis zu ihrem Grad fehlt keine Welle, und keine kommt doppelt vor. Und sie sind **orthogonal** – jedes Muster beschreibt etwas, das kein anderes beschreibt. Deshalb nutzt die Geodäsie sie seit Jahrzehnten, um die Form und das Schwerefeld der Erde zu beschreiben. Das Erdschweremodell EGM2008 etwa besteht aus den Gewichten von Kugelflächenfunktionen bis zum Grad 2190 [1].

## Die Erde aus Kugelflächenfunktionen

Mit diesen Mustern kann man die Erde zusammensetzen – genau wie das Alpenprofil in Kapitel 5 aus Wellen. Jedes Muster bekommt ein Gewicht, alle werden zusammengezählt. Je höher der Grad, desto mehr Muster stehen zur Verfügung und desto schärfer wird die Erde:

![Fünf Weltkarten: das Original und die Erde, zusammengesetzt aus Kugelflächenfunktionen bis Grad 2, 8, 20 und 40. Bis Grad 2 ist nur ein grober Unterschied zwischen Nordhalbkugel und Pazifik zu sehen, bis Grad 8 erscheinen Kontinente als Flecken, bis Grad 20 sind ihre Umrisse erkennbar, bis Grad 40 auch Halbinseln, Gebirge und die Antarktis.](img/erde-aus-kugelflaechen.png)

Das ist das einfachste Modell aus Kapitel 3, nur mit anderen Eingaben: **eine gewichtete Summe**, ohne versteckte Schichten und ohne Knicke. Jedes Muster ist eine Eingabe, jedes Gewicht gehört zu einem Muster.

Bei Kugelflächenfunktionen lassen sich die passenden Gewichte sogar direkt ausrechnen, ohne Training – so ist die Abbildung oben entstanden. Trainiert man stattdessen ein Netz ohne versteckte Schichten mit diesen Eingaben, findet das Training dieselben Gewichte. Ein solches Netz ist also genau die klassische Darstellung der Geodäsie.

## Kugelflächenfunktionen als Eingabe für ein Netz

Statt der zufälligen Streifen aus Kapitel 5 kann man die Kugelflächenfunktionen auch als Eingabe für ein Netz mit versteckten Schichten verwenden. Für einen Ort schaut man – wie in Kapitel 5 – in jedem Muster nach, welchen Wert es an dieser Stelle hat. Bis Grad 16 sind das 289 Zahlen statt 256.

Das Netz kann die Muster dann mit seinen Knicken weiter kombinieren und so auch Formen bilden, die feiner sind als die kürzeste Welle der Eingabe.

Die App bietet dafür zwei Vorlagen:

| Vorlage | Grad *L* | Eingaben | versteckte Schichten | Gewichte | Größe | Kompressionsrate |
|---|---:|---:|---:|---:|---:|---:|
| *Spherical harmonic expansion (L = 40, linear)* | 40 | 1 681 | 0 | 1 682 | 6,6 KB | 693 |
| *Spherical harmonics (L = 16)* | 16 | 289 | 4 | 86 785 | 339 KB | 13 |
| zum Vergleich: *Fourier MLP (recommended)*, Kapitel 5 | – | 256 | 4 | 82 561 | 323 KB | 14 |

Die lineare Variante ist winzig: Sie braucht nur ein Gewicht pro Muster und ist rund 700-mal kleiner als das Raster. Dafür zeigt sie keine Einzelheiten unter etwa 1 000 km.

## Der Preis der Vollständigkeit

So elegant die Kugelflächenfunktionen sind – für feine Einzelheiten werden sie teuer. Weil sie vollständig sind, braucht jede feinere Welle *alle* Richtungen auf einmal: Für Wellen von 400 km Länge braucht man über 10 000 Muster, für 100 km bereits über 160 000. Jedes Muster ist eine Eingabe, und jede Eingabe kostet in der ersten Schicht eines Netzes ein Gewicht pro Neuron.

Die zufälligen Streifen aus Kapitel 5 sind da sparsamer. Sie decken nicht jede Richtung systematisch ab, erreichen aber schon mit 128 Mustern auch kurze Wellen. Das Netz kann dann aus vielen Richtungen die passenden auswählen.

| | zufällige Streifen (Kapitel 5) | Kugelflächenfunktionen |
|---|---|---|
| Auswahl der Muster | zufällig, aus einem Startwert | fest und geordnet |
| passen genau auf die Kugel | annähernd | ja |
| vollständig bis zu einer Wellenlänge | nein | ja |
| Zahl der Eingaben für kurze Wellen | gering (z. B. 256) | wächst sehr schnell |
| Rechenaufwand | gering | hoch, vor allem bei großem *L* |

Welche Variante die Höhen besser trifft, hängt von der Netzgröße und vom Grad ab. Die Ergebnisse stehen im Anhang mit den Messungen.

## Ausprobieren

1. In der Seitenleiste unter *2 · Network* die Vorlage *Spherical harmonic expansion (L = 40, linear)* wählen und *Architecture details* aufklappen.
2. Unter *Positional encoding* steht jetzt `spherical`. Der Regler *Maximum degree L* legt den höchsten Grad fest. Darunter zeigt die App die Zahl der Muster und die kürzeste Welle. *Hidden layers* steht auf 0: ein lineares Modell.

   <img src="img/app-kugelflaechen-einstellungen.png" alt="Seitenleiste mit aufgeklappten Architecture details: Vorlage Spherical harmonic expansion, Positional encoding spherical, Maximum degree L 40, darunter 1681 features und finest wavelength ungefähr 1,000 km, Hidden layers 0 und der Hinweis Linear model." width="300">

3. Trainieren und im Reiter *Maps & Error* das Ergebnis ansehen. Es sollte ähnlich aussehen wie die Karte „bis Grad 40“ oben.
4. Dann die Vorlage *Spherical harmonics (L = 16)* wählen, trainieren und mit *Fourier MLP (recommended)* aus Kapitel 5 vergleichen.

**Hinweis zur Rechenzeit:** Bei großem Grad wird das Training langsam, weil für jeden Ort sehr viele Muster berechnet werden müssen. Mit *L* = 40 dauert ein Trainingsschritt ein Vielfaches eines Schritts mit Fourier-Features. Für erste Versuche lohnt sich ein kleineres *L* oder eine kleinere *Grid resolution*.

## Ideen zum Weiterprobieren

- Beim linearen Modell *Maximum degree L* auf 8, 20 und 40 stellen und die Karten mit der Abbildung oben vergleichen.
- Beim linearen Modell *Hidden layers* von 0 auf 2 erhöhen. Was passiert mit der Zahl der Gewichte – und mit den Einzelheiten?
- Bei *Spherical harmonics (L = 16)* den Grad auf 8 senken. Wie viel schlechter wird das Ergebnis, und wie viel kleiner die Datei?

## Weiterführende Links

- [1] Wikipedia: [Earth Gravitational Model](https://en.wikipedia.org/wiki/Earth_Gravitational_Model) – das Schweremodell EGM2008 der US-Behörde NGA, Kugelflächenfunktionen bis Grad 2190 (Englisch)
- Wikipedia: [Kugelflächenfunktionen](https://de.wikipedia.org/wiki/Kugelfl%C3%A4chenfunktionen) – mathematische Grundlagen, mit Abbildungen der ersten Grade
- Rußwurm et al.: [Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks](https://arxiv.org/abs/2310.06743), ICLR 2024 – Kugelflächenfunktionen als Eingabe für neuronale Netze (Englisch)

---

[Übersicht](README.md) · Zurück: [Kapitel 5 · Höhen als Wellen](05-fourier-features.md) · Weiter: Kapitel 7 · Vertiefung: SIREN *(in Arbeit)*
