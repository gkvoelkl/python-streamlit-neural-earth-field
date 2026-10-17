# Geodaten in neuronalen Netzen speichern
# Kapitel 10 · Abfragen und exportieren

*[English version](../en/10-abfragen-und-exportieren.md)*

![Auf einem Holztisch steht ein offener Pappkarton, aus dem Aiko, ein leuchtendes neuronales Netz, fröhlich mit beiden Armen winkt. Daneben drückt Chester mit blauer Kappe ein kleines Versandetikett an den Karton. Rechts steht eine Erdkugel aus Mosaiksteinen, im Hintergrund ein Regal mit weiteren Paketen.](../img/kapitel-10-chester.jpg)

**Inhalt:**

* Das Netz an beliebigen Koordinaten nach der Höhe fragen
* Warum feiner abfragen keine zusätzlichen Einzelheiten liefert
* Das Netz als Datei speichern – in float32 oder float16 – und was float16 kostet

Dieses Kapitel ergänzt die Kapitel 1 bis 5. Sie können es auch überspringen.

## Ausgangspunkt

Nach dem Training steckt die Erde in **82 561** Gewichten. Bisher hat die App mit diesen Gewichten nur das Trainingsgitter nachgerechnet, um Karten und Fehler zu zeigen.

Zwei Dinge fehlen noch, damit das Netz wirklich die Datei ersetzt: Man muss es nach jeder beliebigen Stelle fragen können, und man muss es speichern und weitergeben können.

## Das Netz an einer beliebigen Stelle nach der Höhe fragen

Das neuronale Netz ist eine Funktion (Kapitel 1). Es bekommt eine Länge und eine Breite als Eingabe und liefert daraus eine Höhe als Ergebnis. Dazwischen passiert, was die Kapitel 3 bis 5 beschrieben haben: Die Koordinate wird zum Punkt (x, y, z) auf der Kugel, daraus werden 256 Wellen der Fourier-Features, und die laufen durch die vier Schichten des Netzes.

Nirgends in dieser Rechnung kommt das Gitter vor. Das Netz kennt keine Zeilen und Spalten. Es rechnet für 47,421° Nord genauso wie für 47,4° oder 47,42137°.

Ein paar Abfragen beim Standardbeispiel, nach 4 000 Trainingsschritten:

| Ort | Koordinate | echte Höhe | Zelle im Gitter 2160 × 4320 | Zelle im Trainingsgitter 1080 × 2160 | Netz |
|---|---|---:|---:|---:|---:|
| Mount Everest | 86,925° O · 27,988° N | 8 849 m | 6 345 m | 5 776 m | 4 673 m |
| Challengertief im Marianengraben | 142,2° O · 11,35° N | −10 935 m | −9 816 m | −9 369 m | −9 268 m |
| Zugspitze | 10,985° O · 47,421° N | 2 962 m | 1 250 m | 1 475 m | 1 323 m |

Das sieht zunächst erschreckend aus: Der Mount Everest ist über 4 000 m zu niedrig. Das liegt aber nicht allein am neuronalen Netz: Den größten Teil davon, gut 3 000 m, verliert der Gipfel schon im Trainingsgitter.

Eine Zelle im Trainingsgitter ist etwa 19 km breit, und ihr Wert ist der Mittelwert aller Höhen in dieser Fläche (Kapitel 1). Darin liegen der Gipfel, die Flanken und die Täler darum herum – im Mittel 5 776 m. Das Netz hat diesen Mittelwert gelernt, nicht den Gipfel, und rundet ihn zusätzlich noch etwas ab.

Bei der Zugspitze sieht man das noch deutlicher: Selbst im feineren Gitter mit 9 km breiten Zellen bleibt von 2 962 m nicht einmal die Hälfte übrig. Einzelne Gipfel sind für ein Gitter dieser Auflösung zu schmal.

## Feiner abfragen als trainiert

Weil das Netz eine Funktion ist, kann man es auch auf einem Gitter abfragen, das viel feiner ist als das Trainingsgitter – zum Beispiel alle 1/60°, wie im Original von ETOPO. Bekommt man dann wieder die feinen Einzelheiten zurück?

Nein. Das Profil durch die Alpen zeigt es:

![Höhenprofil bei 46,4° Nord von 2° bis 18° Ost. Hellbraun das feinere Gitter 2160 × 4320 mit vielen spitzen Gipfeln und Tälern bis knapp 3 000 m. Blau das Trainingsgitter 1080 × 2160 als Treppe, etwas weniger zackig. Orange das Netz, alle 1/60° abgefragt: eine glatte, weiche Kurve, die über den Alpen nur bis etwa 2 000 m reicht und keinen der einzelnen Gipfel zeigt.](../img/abfrage-profil.svg)

Die orange Kurve ist sehr fein abgefragt, 960 Punkte auf 16 Grad. Trotzdem ist sie glatt. Das Netz liefert zwischen den Punkten, die es im Training gesehen hat, weiche Übergänge, aber keine neuen Gipfel und Täler. Woher sollte es sie auch kennen? Im Training kamen sie nie vor.

Nachmessen lässt sich das mit dem Gitter 2160 × 4320, das die App ebenfalls anbietet. Es ist doppelt so fein wie das Trainingsgitter und hat Werte, die das Netz nie gesehen hat:

| Vergleich mit dem feineren Gitter 2160 × 4320 | gewichteter RMSE |
|---|---:|
| Netz, auf dem feinen Gitter abgefragt | 272 m |
| Netz, nur auf dem Trainingsgitter abgefragt, jeder Wert auf 2 × 2 feine Zellen verteilt | 276 m |
| Trainingsgitter selbst, jeder Wert auf 2 × 2 feine Zellen verteilt | 128 m |

Das feine Abfragen bringt gerade einmal 4 m. Das grobe Trainingsgitter liegt mit 128 m sogar viel näher am feinen Gitter als das Netz. Was das Netz beim Lernen nicht getroffen hat, kann es beim Abfragen nicht nachliefern.

Die App nennt den Reiter, in dem man das ausprobiert, *Super-Resolution*. Gemeint ist damit üblicherweise ein Verfahren, das echte zusätzliche Einzelheiten erzeugt. Das leistet das Netz nicht, und die App sagt das auch selbst in einem Hinweis unter den Karten.

Nützlich ist das feine Abfragen trotzdem. Man kann Höhen genau an den Stellen ausrechnen, die man braucht – entlang einer Wanderroute, an den Messpunkten einer Wetterstation oder auf einem Gitter in jeder beliebigen Auflösung –, ohne vorher ein Raster umrechnen zu müssen.

## Das neuronale Netz als Datei

Damit jemand anderes das Netz nutzen kann, muss es in eine Datei. In dieser Datei steht alles, was man zum Nachrechnen der Höhen braucht:

- **der Bauplan:** Vorlage, Zahl der Schichten und Neuronen, Eingabeformat, Kodierung und σ – also die Einstellungen aus der Seitenleiste,
- **die Gewichte:** die 82 561 Zahlen, um die es in diesem Tutorial geht,
- **Mittelwert und Standardabweichung der Höhen**, mit denen das Netz normiert wurde (Kapitel 1),
- **das Kontextfeld**, falls das Netz eines hat (Kapitel 8), als ganze Meter mit 2 Byte pro Wert,
- **die Größe des Trainingsgitters**, damit man weiß, worauf das Netz trainiert wurde.

Was *nicht* in der Datei steht: die Zufallsmatrix der Fourier-Features. Sie entsteht beim Laden neu aus dem Startwert, der im Bauplan steht (Kapitel 5).

Die App speichert das im Format von **PyTorch**, der Bibliothek, mit der das Netz rechnet. Die Datei hat die Endung `.pt`. Für das Standardbeispiel ist sie 334 527 Byte groß, also 326,7 KB. Das sind gut 4 KB mehr als die 330 256 Byte, mit denen Kapitel 1 gerechnet hat: Die Datei enthält zusätzlich Namen für jede Schicht und etwas Verwaltung. Die Kompressionsrate zählt diese Verwaltung nicht mit, denn sie hängt nur vom Dateiformat ab und nicht vom Verfahren.

## float32 oder float16

Bisher belegte jedes Gewicht 4 Byte (float32, Kapitel 1). Es gibt auch Zahlen mit 2 Byte, **float16**. Damit wird die Datei halb so groß. Der Preis: float16 speichert weniger Stellen.

| | float32 | float16 |
|---|---:|---:|
| Speicher pro Zahl | 4 Byte | 2 Byte |
| genaue Dezimalstellen | etwa 7 | etwa 3 bis 4 |
| größte Zahl | etwa 3,4 · 10³⁸ | 65 504 |
| Beispiel 0,123456789 | 0,12345679 | 0,12347 |

Für die Gewichte reicht float16 meist aus. Sie sind kleine Zahlen – beim Standardbeispiel liegen praktisch alle zwischen −1 und 1 –, und kein einzelnes Gewicht entscheidet über eine Höhe. Jede Höhe entsteht aus Tausenden von Gewichten, und deren Rundungsfehler gleichen sich zum großen Teil aus.

Wie viel float16 tatsächlich kostet, zeigt der Vergleich auf dem ganzen Trainingsgitter, jeweils nach 4 000 Schritten:

| Vorlage | Datei float32 | Datei float16 | Rate float32 → float16 | Abweichung float16 von float32: im Mittel | bei 99 % der Zellen höchstens | größte | gewichteter RMSE float32 → float16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| *Fourier MLP (recommended)* | 323 KB | 161 KB | 14 → 28 | 0,5 m | 2,6 m | 8 m | 245 m → 245 m |
| *Fourier MLP + context field* | 355 KB | 193 KB | 13 → 24 | 0,2 m | 1,6 m | 6 m | 191 m → 191 m |
| *SIREN (sine activation)* | 305 KB | 153 KB | 15 → 30 | 13 m | 63 m | 239 m | 278 m → 279 m |

Die Dateigrößen sind hier wie in Kapitel 1 ohne die Verwaltung des Dateiformats gerechnet. Jede Zeile ist ein einzelner Trainingslauf.

**Beim Fourier-MLP ist float16 fast umsonst.** Die Rate verdoppelt sich von 14 auf 28, und der Fehler ändert sich nicht einmal auf den Meter. Die Abweichung von der float32-Fassung ist im Mittel ein halber Meter – verglichen mit 245 m Fehler gegenüber dem Original verschwindend klein.

**Beim Kontextfeld wächst die Rate weniger.** Das Kontextfeld ist schon in ganzen Metern mit 2 Byte gespeichert und wird durch float16 nicht kleiner. Nur das Netz schrumpft, die Rate steigt deshalb nur von 13 auf 24.

**Beim SIREN-Netz ist float16 deutlich teurer.** Die Abweichungen sind etwa 25-mal so groß wie beim Fourier-MLP, im schlimmsten Fall 239 m. Der Grund steht in Kapitel 7: Das SIREN-Netz multipliziert in jeder Schicht die Summe mit ω₀ = 30, bevor der Sinus kommt. Ein kleiner Rundungsfehler in einem Gewicht wird dadurch 30-mal größer und verschiebt die Welle spürbar – und das von Schicht zu Schicht. Insgesamt steigt der Fehler zwar nur um einen Meter, aber an einzelnen Stellen springt die Höhe um mehr als 200 m.

Noch kleiner würde die Datei mit 8 Bit oder weniger pro Gewicht. Dann reicht einfaches Runden nach dem Training aber nicht mehr; das Netz muss schon beim Training lernen, mit den gröberen Zahlen zurechtzukommen. [Kapitel 11](11-einordnung-und-ausblick.md) kommt darauf zurück.

## Die Datei weitergeben

Wer die Datei bekommt, braucht noch den Programmcode des Projekts. Die Datei enthält nur Zahlen und den Bauplan; wie aus dem Bauplan wieder ein Netz wird, steht im Ordner `src/` des Repositorys. Der Reiter *Export* zeigt unter den beiden Download-Knöpfen ein kurzes Beispiel in Python, das eine Datei lädt und die Höhe am Mount Everest abfragt.

Eine Vorsicht gehört dazu: Das Dateiformat `.pt` kann beim Laden nicht nur Zahlen, sondern auch Programmcode ausführen. Die Dokumentation von PyTorch rät deshalb ausdrücklich, keine Dateien aus nicht vertrauenswürdigen Quellen zu laden [1]. Eigene Dateien und solche von Leuten, denen man vertraut, sind kein Problem.

## Ausprobieren

Trainieren Sie das Standardbeispiel: Vorlage **Fourier MLP (recommended)**, **Steps** 4 000, dann **Train**. Die beiden Reiter dieses Kapitels zeigen erst nach einem Training etwas.

Öffnen Sie den Reiter **Super-Resolution**. Dort lassen sich vier Dinge einstellen:

- **Longitude (centre)** und **Latitude (centre)** – die Mitte des Ausschnitts, voreingestellt 10° Ost und 46° Nord, also die Alpen,
- **Window [°]** – wie viele Grad der Ausschnitt breit und hoch ist, 2 bis 90,
- **Points per axis** – wie viele Punkte das Netz pro Richtung abfragt, 128 bis 512.

![Die Einstellungen im Reiter Super-Resolution: Longitude (centre) 10,00, Latitude (centre) 46,00, Window 20 Grad, Points per axis 256.](../img/app-abfrage-einstellungen.png)

Vergleichen Sie die beiden Karten. Links das Trainingsgitter in diesem Ausschnitt, bei 20° sind das 120 × 120 Zellen. Rechts das Netz, auf 256 × 256 Punkten abgefragt. Die rechte Karte ist feiner gerastert, aber weicher – Küsten und Gebirge sind verwaschen.

![Zwei Karten von Mitteleuropa und dem Mittelmeer, jeweils 0° bis 20° Ost und 36° bis 56° Nord. Links das Trainingsgitter mit 120 × 120 Zellen: scharfe Küsten, der Alpenbogen mit weißen Gipfeln, Sardinien, Korsika und Italien klar erkennbar. Rechts das Netz auf 256 × 256 Punkten: glatte Farbflächen, die Alpen als weicher brauner Rücken, Küsten und Inseln verschwommen.](../img/app-abfrage-karten.png)

Öffnen Sie dann den Reiter **Export**. Oben stehen drei Zahlen: die Größe des Rasters, die Datei in float32 und die Datei in float16, jeweils mit der Kompressionsrate darunter. Mit den beiden Knöpfen darunter lassen sich die Dateien herunterladen.

![Der Reiter Export für das Standardbeispiel: Raw grid (int16) 4.4 MB, Model float32 326.7 KB mit Rate 14, Model float16 165.4 KB mit Rate 28, darunter die Knöpfe Weights float32 (.pt) und Weights float16 (.pt).](../img/app-export.png)

## Ideen zum Weiterprobieren

- Im Reiter *Super-Resolution* das Fenster auf 2° stellen. Wie viele Zellen des Trainingsgitters bleiben links übrig, und wie sieht die rechte Karte mit 512 Punkten pro Achse aus?
- Den Ausschnitt auf den Himalaya legen (etwa 87° Ost, 28° Nord) und die höchste Farbe in beiden Karten vergleichen.
- Die Vorlage *SIREN (sine activation)* trainieren und im Reiter *Export* die beiden Dateigrößen ansehen. Warum ist der Unterschied zum Fourier-MLP bei der Dateigröße klein, bei der Genauigkeit in float16 aber groß?
- Mit *Fourier MLP + context field* trainieren und die Rate in float16 mit der des Standardbeispiels vergleichen. Welcher Teil der Datei schrumpft nicht mit?
- Wer Python kennt: das Ladebeispiel aus dem Reiter *Export* ausführen und die Höhe am eigenen Wohnort abfragen.

## Weiterführende Links

- [1] PyTorch-Dokumentation: [torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html) – Laden gespeicherter Netze, mit dem Hinweis, keine Dateien aus nicht vertrauenswürdigen Quellen zu laden (Englisch)
- Wikipedia: [Half-precision floating-point format](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) – Aufbau von float16 mit 16 Bit, größte Zahl 65 504, gut drei Dezimalstellen (Englisch)
- NOAA NCEI: [ETOPO Global Relief Model](https://www.ncei.noaa.gov/products/etopo-global-relief-model) – das Original mit Zellen bis hinunter zu 15 Bogensekunden, falls man echte Einzelheiten braucht

---

[Übersicht](README.md) · Zurück: [Kapitel 9 · Kompressionsrate und Fehler messen](09-rate-und-fehler.md) · Weiter: [Kapitel 11 · Einordnung und Ausblick](11-einordnung-und-ausblick.md)

*Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. Es soll helfen, mehr über das Thema „Geodaten in neuronalen Netzen speichern“ zu lernen. Verbesserungsvorschläge gerne an gkvoelkl@nelson-games.de.*
