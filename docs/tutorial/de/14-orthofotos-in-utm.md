# Geodaten in neuronalen Netzen speichern
# Kapitel 14 · Orthofotos in UTM

*[English version](../en/14-orthofotos-in-utm.md)*

![Chester mit blauer Kappe und Aiko, das leuchtende neuronale Netz, schweben im Korb eines bunt gestreiften Heißluftballons über einer Altstadt mit roten Dächern. Chester hält eine große Kamera, Aiko ein quadratisches Foto von Fluss und Häusern. Im Hintergrund ein gotischer Dom mit zwei Türmen und eine alte Steinbrücke über einen breiten Fluss; über der Landschaft liegt ein zartes, leuchtendes Quadratgitter.](../img/kapitel-14-chester.jpg)

**Inhalt:**

* Ein Luftbild als Funktion: X- und Y-Wert als Eingabe und Rot, Grün und Blau als Ergebnis
* Warum für einen Kilometer die Erde eben ist und was UTM-Koordinaten sind
* Wie gut ein neuronales Netz ein Foto speichert? 


## Ausgangspunkt

In den Kapiteln 1 bis 13 ging es immer um die ganze Erde: ein Gitter aus Längen- und Breitengraden, eine Kugel, Zellen von einigen Kilometern. Viele Geodaten sehen ganz anders aus. Sie decken nur ein kleines Gebiet ab, dafür aber sehr fein. Das bekannteste Beispiel sind **Luftbilder**.

Dieses Kapitel speichert ein Luftbild von einem Quadratkilometer in einem neuronalen Netz. Drei Dinge ändern sich dabei gegenüber den bisherigen Kapiteln:

1. **Ebene statt Kugel.** Das Netz bekommt Koordinaten in Metern auf einer Ebene, keinen Punkt auf einer Kugel.
2. **Drei Ausgaben statt einer.** Statt einer Höhe kommen drei Farben heraus: Rot, Grün und Blau.
3. **Ein ganz anderes Bild.** Ein Foto einer Stadt ist viel detailreicher als ein Höhenmodell der Erde.

## Das Orthofoto

Wenn ein Luftbild aufgenommen wird, ist es – genauso wie jedes andere Foto – verzerrt. Am Rand sieht man Häuser etwas von der Seite, und ein Hang erscheint je nach Blickwinkel gestaucht oder gedehnt. Ein **Orthofoto** ist ein Luftbild, das mit einem Geländemodell so entzerrt wurde, dass es maßstabsgetreu auf eine Karte passt [1]. Jedes Pixel hat dann eine feste Lage in Metern. Man kann damit die Entfernung zwischen zwei Punkten messen.

![Zwei Schnitte durch dasselbe Gelände mit einem roten Haus links und einem Hügel rechts; auf dem Gelände liegen blaue Punkte in gleichen Abständen, darunter ein Streifen „Bild auf der Karte“. Links das Luftbild: Alle Strahlen laufen von den Punkten zur Kamera oben in der Mitte. Verlängert bis zur Ebene, rücken hohe Punkte im Bild nach außen – die Punkte auf dem Hügel liegen im Streifen weiter auseinander, die Dachecken des Hauses rutschen vom Haus weg. Rechts das Orthofoto: Jeder Punkt geht senkrecht nach unten, die Punkte im Streifen liegen in gleichen Abständen, das Dach genau über dem Haus.](../img/ortho-entzerrung.svg)

Die Bayerische Vermessungsverwaltung stellt ihre Orthofotos seit 2023 kostenlos bereit, unter der Lizenz CC BY 4.0 [2, 3]. Dieses Kapitel verwendet **DOP40** (Digitales Orthophoto, ein Bildpunkt entspricht 40 x 40 cm in der Realität):

| Eigenschaft | Wert |
|---|---|
| Kachel | 1 km × 1 km, Altstadt von Regensburg |
| Pixel | 2 500 × 2 500, je 40 cm × 40 cm |
| Farben | Rot, Grün, Blau, je 1 Byte (0 bis 255) |
| Aufnahme | Bildflug am 11. Mai 2025 |
| Datei | GeoTIFF, etwa 18 MB |

Auf der Kachel liegen der Dom, die Steinerne Brücke, die Donau mit zwei Inseln und Hunderte von Dächern.

![Luftbild der Kachel: oben die Häuser von Stadtamhof, in der Mitte die Donau mit zwei Inseln, über die von oben nach unten die lange Steinerne Brücke führt, rechts eine zweite, schmale Brücke. Unten die dicht bebaute Altstadt mit roten Dächern, engen Gassen und dem Dom. Rechts am Südufer ein großes weißes Gebäude.](../img/ortho-kachel.jpg)

*Daten: Bayerische Vermessungsverwaltung – www.geodaten.bayern.de, CC BY 4.0*

Für das Training rechnet die App die Kachel auf 1 250 × 1 250 Pixel herunter; je 2 × 2 Pixel werden gemittelt. Ein Pixel entspricht dann 80 x 80 cm. 

Das Bild ist damit kleiner und das Training schneller, aber Dächer, Bäume und Autos sind immer noch gut zu erkennen. Unkomprimiert sind das 1 250 · 1 250 · 3 = 4 687 500 Byte, also **4,5 MB**.

## Ebene statt Kugel

### Warum die Kugel hier keine Rolle spielt

Kapitel 4 hat erklärt, warum die Erde für ein Netz eine Kugel sein muss: Längen- und Breitengrade sind am Pol verzerrt, und am Datumsrand springen die Längen von +180° auf −180°. Für einen Kilometer ist das alles bedeutungslos. Auf 500 m Abstand weicht die Kugel nur etwa **2 cm** von einer Ebene ab – weniger als ein Zehntel Pixel. Für ein so kleines Gebiet ist die Erde praktisch eben.

### UTM: Koordinaten in Metern

Für Karten in Metern gibt es Kartenprojektionen. In Deutschland ist das amtliche System **ETRS89 / UTM** [4]. UTM teilt die Erde in Streifen von 6° Breite, die *Zonen*. Jede Zone hat einen eigenen Mittelmeridian, eine Nord-Süd-Linie in der Mitte. Ein Ort wird mit zwei Zahlen in Metern beschrieben:

- Der **Ostwert** (*Easting*, E) ist der Abstand vom Mittelmeridian. Damit er nie negativ wird, zählt der Mittelmeridian als 500 000 m.
- Der **Nordwert** (*Northing*, N) ist der Abstand vom Äquator.

Bayern liegt größtenteils in Zone 32 mit dem Mittelmeridian 9° Ost. Regensburg liegt eigentlich schon in Zone 33, die Vermessungsverwaltung rechnet aber ganz Bayern in Zone 32 [5]. Die Kachel beginnt links unten bei **E = 726 000 m** und **N = 5 434 000 m**: 226 km östlich des Mittelmeridians und 5 434 km nördlich des Äquators. Daher kommt auch der Dateiname der Kachel, `32726_5434`: Zone 32, Ostwert 726 km, Nordwert 5 434 km.

![Links eine Karte von Mitteleuropa mit Nord- und Ostsee oben und der Adria unten. Zwei senkrechte Streifen sind gefärbt: Zone 32 von 6° bis 12° Ost in Blau mit dem gestrichelten Mittelmeridian 9°, Zone 33 von 12° bis 18° Ost in Orange mit dem Mittelmeridian 15°. Regensburg liegt als roter Punkt knapp östlich von 12°, also schon im orangen Streifen. Rechts die Kachel als Luftbild, mit der linken unteren Ecke E 726 000 m, N 5 434 000 m und der rechten oberen Ecke E 727 000 m, N 5 435 000 m; darunter Kachel 32726_5434: 1 000 m × 1 000 m, Zone 32, Ostwert 726 km, Nordwert 5 434 km.](../img/ortho-utm.svg)

Ganz ohne Verzerrung geht es auch hier nicht: Eine Ebene kann die Kugel nie genau abbilden. In Bayern sind Strecken auf der UTM-Karte um bis zu 12 cm pro 100 m länger oder kürzer als in der Natur [5]. Innerhalb einer Kachel von einem Kilometer ist dieser Faktor aber praktisch gleich. Das Bild ist nur ein klein wenig gleichmäßig gestreckt, und das spielt für das Netz keine Rolle.

### Was das neuronale Netz bekommt

Das Netz bekommt den Ostwert und den Nordwert, beide auf −1 bis 1 umgerechnet: −1 am linken bzw. unteren Rand der Kachel, +1 am rechten bzw. oberen. Dafür zieht die App zuerst die linke untere Ecke ab.

Dieser Schritt ist wichtiger, als er aussieht. Ein Netz rechnet in float32, und float32 speichert nur etwa 7 Stellen. Ein Nordwert wie 5 434 000,4 m hat aber 8. In float32 lassen sich Nordwerte in dieser Größe nur in Schritten von **50 cm** darstellen – gröber als ein Pixel. Erst wenn die Ecke abgezogen ist, bleiben kleine Zahlen übrig, die float32 genau genug speichert. Die Ecke selbst gehört zur Datei des Netzes: zwei Zahlen mit je 8 Byte.

Kugelkoordinaten (Kapitel 4) und Kugelflächenfunktionen (Kapitel 6) fallen damit weg. Die Fourier-Features aus Kapitel 5 bleiben: Sie funktionieren in der Ebene genauso, sogar einfacher.

## Drei Farben statt einer Höhe

Das Netz hat dieselbe Form wie das Fourier-MLP der früheren Kapitel. Nur die letzte Schicht hat drei Neuronen statt einem: für Rot, Grün und Blau. Eine Funktion namens *Sigmoid* drückt jede Ausgabe in den Bereich 0 bis 1; das ergibt mal 255 die Farbwerte. So kann das Netz nie eine Farbe unter 0 oder über 255 ausgeben.

Die App hat drei Vorlagen:

| Vorlage | Frequenzen | Schichten × Neuronen | Gewichte | Datei (float16) | Rate |
|---|---:|---|---:|---:|---:|
| *Small* | 128 | 4 × 128 | 82 819 | 162 KB | 28 |
| *Medium* | 256 | 4 × 256 | 329 475 | 644 KB | 7,1 |
| *Large* | 256 | 5 × 512 | 1 314 819 | 2,5 MB | 1,8 |

Die Datei rechnet mit float16, also 2 Byte pro Gewicht (Kapitel 10), dazu die Ecke der Kachel und der Startwert der Fourier-Features. *Small* ist fast genau so groß wie das Standardbeispiel aus Kapitel 1 mit 82 561 Gewichten und hat in float16 dieselbe Rate: 28.

Für die Fourier-Features gilt **σ = 20**. In der Ebene bedeutet das: Eine typische Welle passt etwa 20-mal auf die Kantenlänge der Kachel, ist also etwa 50 m lang. Weil die Frequenzen zufällig gezogen werden, sind auch deutlich kürzere dabei.

Trainiert wird wie in Kapitel 2: In jedem Schritt zieht die App zufällig 65 536 Pixel und verbessert die Gewichte so, dass die Farben besser passen. Alle Pixel zählen gleich viel – auf einer ebenen Kachel sind sie ja alle gleich groß.

## Maße für Bilder

### PSNR

Für Bilder ist **PSNR** das übliche Maß (Kapitel 9). Die Spannweite ist hier 255, der Bereich eines Farbwerts. Gemessen wird über alle Pixel und alle drei Farben. Zur Orientierung:

| PSNR | mittlerer Fehler | Eindruck |
|---:|---:|---|
| 20 dB | etwa 26 Stufen | deutlich verschwommen, Farben verfälscht |
| 30 dB | etwa 8 Stufen | auf den ersten Blick wie das Original |
| 40 dB | etwa 3 Stufen | kaum vom Original zu unterscheiden |

Wie in Kapitel 9: +6 dB heißt, der Fehler ist halb so groß.

### Der Vergleich: JPEG

Für die Höhen der Erde gab es keinen naheliegenden Vergleich. Für Fotos gibt es einen: **JPEG**, das Format praktisch jeder Digitalkamera [6]. Auch JPEG zerlegt ein Bild in Wellen: Es teilt das Bild in Blöcke von 8 × 8 Pixeln und beschreibt jeden Block als Summe von Wellen. Die langen Wellen speichert es genau, die kurzen grob oder gar nicht. Eine Einstellung **Qualität** von 1 bis 100 legt fest, wie grob.

Die App sucht zu jedem Netz die höchste JPEG-Qualität, deren Datei nicht größer ist als das Netz. So werden beide bei gleicher Dateigröße verglichen.

## Ergebnis

Gemessen mit 1 250 × 1 250 Pixeln auf einem Apple M4 (MPS). *Small* mit drei Läufen je Wert (Startwerte 0 bis 2; angegeben ist der Mittelwert, die Läufe lagen höchstens 0,2 dB auseinander), *Medium* nach 1 000 Schritten ebenfalls, sonst ein Lauf.

| Vorlage | Datei | 1 000 Schritte | 3 000 Schritte | 6 000 Schritte | JPEG gleicher Größe |
|---|---:|---:|---:|---:|---:|
| *Small* | 162 KB | 21,5 dB | 23,0 dB | 24,0 dB | **27,9 dB** (Qualität 26) |
| *Medium* | 644 KB | 23,7 dB | 26,1 dB | 27,9 dB | **36,3 dB** (Qualität 91) |
| *Large* | 2,5 MB | 24,6 dB | 27,9 dB | 31,7 dB | **38,6 dB** (Qualität 95, nur 829 KB) |

Rechenzeit: *Small* braucht für 1 000 Schritte etwa 45 Sekunden, *Medium* etwa 3 Minuten, *Large* etwa 7 Minuten; für 6 000 Schritte entsprechend 6, 15 und 44 Minuten. Ein JPEG ist in Millisekunden fertig.

![Diagramm: PSNR in dB über der Dateigröße, logarithmisch von 50 KB bis über 4,5 MB. Die graue JPEG-Linie steigt von 22 dB bei 53 KB über 28 dB bei 158 KB bis 38,6 dB bei 829 KB. Darunter drei senkrechte Gruppen von je drei Punkten für die Netze nach 1 000, 3 000 und 6 000 Schritten: grün Small bei 162 KB mit 21,5, 23,0 und 24,0 dB, blau Medium bei 644 KB mit 23,7, 26,1 und 27,9 dB, violett Large bei 2,5 MB mit 24,6, 27,9 und 31,7 dB. Alle Punkte liegen deutlich unter der JPEG-Linie. Ganz rechts eine orange gestrichelte Linie bei 4,5 MB für das unkomprimierte Bild.](../img/ortho-messung.svg)

Drei Dinge fallen auf:

- **JPEG gewinnt klar.** Bei gleicher Größe liegt JPEG immer vorn: knapp 4 dB bei *Small*, über 8 dB bei *Medium*. Die 27,9 dB, die *Medium* nach 6 000 Schritten erreicht, schafft JPEG schon mit 158 KB – einem Viertel der Größe. *Large* ist mit 2,5 MB über halb so groß wie das unkomprimierte Bild; JPEG kommt selbst bei Qualität 95 nicht auf diese Größe.
- **Größere Netze brauchen mehr Training.** Von 1 000 auf 6 000 Schritte gewinnt *Small* 2,5 dB, *Medium* über 4 dB, *Large* 7 dB. Nach 1 000 Schritten liegen die drei Vorlagen nur 3 dB auseinander, obwohl *Large* 16-mal so viele Gewichte hat wie *Small*. Ein großes Netz kann mehr, braucht aber lange, bis es das zeigt.
- **Beim kleinen Netz ist bald Schluss, beim großen noch lange nicht.** Von 3 000 auf 6 000 Schritte gewinnt *Small* nur 1 dB, *Large* dagegen fast 4 dB. Das große Netz war nach 6 000 Schritten also noch nicht fertig. Trotzdem bleibt es mit 31,7 dB weit hinter JPEG: Schon mit einem Drittel der Größe erreicht JPEG 38,6 dB.

### Wie das aussieht

Der Unterschied ist auf den ersten Blick zu sehen. Hier ein Ausschnitt von 150 m × 150 m an der Steinernen Brücke, links das Original, in der Mitte *Small* nach 1 000 Schritten, rechts das JPEG gleicher Größe. Die Werte in Klammern gelten wie in der Tabelle für das ganze Bild: 21,5 dB für das Netz, 27,9 dB für das JPEG.

![Drei Ausschnitte nebeneinander, jeweils mit der Steinernen Brücke als senkrechtem hellem Streifen rechts, Pfeilern im Wasser, Bäumen und einer Wiese links oben. Links das Original, scharf, mit einzelnen Baumkronen und Wegen. In der Mitte das Netz: weich verschwommen, die Bäume als dunkelgrüne Flecken, die Pfeiler als runde helle Beulen, die Farben blasser. Rechts das JPEG: die Bäume und Wege gut erkennbar, aber in kleine Quadrate zerlegt; an der Brücke sind Blockkanten zu sehen.](../img/ortho-bruecke.svg)

Die beiden Verfahren machen ganz verschiedene Fehler:

- **Das Netz verschwimmt.** Ihm fehlen die kurzen Wellen. Dächer, Baumkronen und Autos verlaufen ineinander; die Brücke und die großen Straßen bleiben erkennbar. Auffällig sind auch die Farben: Die roten Dächer der Altstadt werden graubraun. Das Netz lernt zuerst eine Art Mittelfarbe über viele Pixel und erst später, wo genau die roten Flächen anfangen und aufhören.
- **JPEG wird blockig.** Kanten bleiben scharf, aber bei starker Kompression sind die Blöcke von 8 × 8 Pixeln zu sehen, und innerhalb eines Blocks werden Farben flächig.

### Warum ein Foto so viel schwerer ist

Kapitel 5 hat gezeigt: Ein Netz lernt lange Wellen schnell und kurze nur langsam. Bei den Höhen der Erde ist das gutmütig. Der größte Teil steckt in langen Wellen – Kontinente, Ozeanbecken, Gebirge –, und Gipfel und Täler sind ein kleiner Rest.

Ein Luftbild einer Stadt ist das Gegenteil. Jede Dachkante springt innerhalb eines Pixels von Rot auf Grau, jede Baumkrone ist ein dunkler Fleck von wenigen Metern, jedes Auto ein heller Punkt. Fast alles Wichtige sind kurze Wellen. Eine Wellenlänge von zwei Pixeln sind hier 1,6 m – das sind 625 Wellen auf einen Kilometer, weit mehr als die typischen 20 aus σ.

JPEG ist für genau solche Bilder gebaut, und es arbeitet anders: Es muss nichts lernen, sondern schreibt die Wellen jedes Blocks direkt aus. Das Netz dagegen muss mit einer festen Zahl von Gewichten das ganze Bild auf einmal beschreiben – auch die vielen kleinen Dinge, die nirgends sonst wieder vorkommen.

## Wozu dann?

Als reine Kompression für Luftbilder taugt ein Netz in dieser einfachen Form also nicht. Das ist ein ehrliches Ergebnis und zugleich eine Abgrenzung: Neuronale Felder sind gut für glatte Felder wie Höhen, Temperaturen oder Abstände (Kapitel 1 bis 13), weniger für Bilder voller Kanten.

Die Forschung hat das Problem nicht aufgegeben. COIN von Dupont und Kollegen speichert Fotos ebenfalls als Netz und war damit bei sehr starker Kompression besser als JPEG [7]. Für Satellitenbilder mit vielen Spektralkanälen gibt es eigene Verfahren wie ImpliSat [8]. Sie treiben mehr Aufwand als dieses Kapitel – ImpliSat etwa passt die Wellen der Eingabe jedem Kanal einzeln an.

Und ein Netz hat Eigenschaften, die JPEG fehlen. Es ist eine Funktion: Man kann es an jeder Stelle (E, N) abfragen, auch zwischen den Pixeln. Und es lässt sich erweitern – um eine dritte Koordinate. Kapitel 15 tut das und macht aus Gebäuden ein Feld f(x, y, z). Kapitel 16 zeigt mit *Gaussian Splatting* eine ganz andere Art, dasselbe Luftbild und dieselben Gebäude zu speichern: ohne Netz, als viele kleine Farbflecken.

## Ausprobieren

Der Reiter **Orthophoto** braucht einen Download und viel Rechenzeit und läuft deshalb nur in der lokalen Fassung der App, nicht online.

### Daten laden

Öffnen Sie den Reiter **Orthophoto**. Beim ersten Mal fragt die App, ob sie die Kachel herunterladen darf. Klicken Sie auf **Download (18 MB)**.

### Netz trainieren

| Einstellung | Wert | Bedeutung |
|---|---|---|
| *Pixels* | 1250 × 1250 · 0.8 m | Auflösung, auf die die Kachel heruntergerechnet wird |
| *Network* | Medium | eine der drei Vorlagen aus der Tabelle oben |
| *Steps* | 1000 | Zahl der Trainingsschritte |
| *Device* | auto | Grafikchip, wenn vorhanden |

Unter den Einstellungen nennt die App die Größe des Bildes und des Netzes.

![Die Einstellungen im Reiter Orthophoto: Pixels 1250 × 1250 · 0.8 m, Network Medium, Steps 1000, Device auto. Darunter die Zeile 1,562,500 pixels × 3 colours = 4.5 MB uncompressed · network: 329,475 weights = 643.5 KB in float16, dazu die Quellenangabe Bayerische Vermessungsverwaltung – www.geodaten.bayern.de (CC BY 4.0). Darunter der Knopf Train.](../img/app-ortho-einstellungen.png)

Klicken Sie auf **Train**. Ein Balken zeigt den Fortschritt und den PSNR; nach etwa 3 Minuten ist das Netz fertig. Bei uns lag es danach bei 23,8 dB. Dann erscheinen oben vier Zahlen: der PSNR des Netzes, die Dateigröße, die Rate und der PSNR eines JPEG gleicher Größe. Darunter stehen das Original, das Netz und das JPEG nebeneinander.

![Der Reiter Orthophoto nach dem Training: oben PSNR of the network 23.8 dB, File size (float16) 643.5 KB, Compression ratio 7 : 1 und JPEG of the same size (quality 91) 36.3 dB. Darunter die ganze Kachel dreimal nebeneinander: Original (DOP40), Network und JPEG, 618.0 KB. In dieser Größe sehen alle drei fast gleich aus; beim Netz sind die Bäume auf den Inseln und das Wasser der Donau etwas fleckig.](../img/app-ortho-ergebnis.png)

### Ausschnitte vergleichen

Unter **Detail** wählen Sie einen Ausschnitt von 100 m × 100 m: den Dom, die Steinerne Brücke oder das Alte Rathaus. Ganz rechts zeigt ein viertes Bild den Fehler des Netzes: Schwarz heißt genau, hell heißt weit daneben. Die Fehler sitzen an Kanten und Einzelheiten; große gleichmäßige Flächen trifft das Netz gut.

![Detail Cathedral: vier Ausschnitte um den Dom. Im Original Dächer, Gassen, der Domplatz und der große Schatten des Doms. Im Netz nur verschwommene rötliche und grüngraue Flecken. Das JPEG sieht fast aus wie das Original. Im Fehlerbild des Netzes leuchten die Kanten der Dächer und die vielen Einzelheiten am Dom hell; der große Schatten und die Fläche des Domplatzes bleiben dunkel.](../img/app-ortho-ausschnitt.png)

## Ideen zum Weiterprobieren

- Trainieren Sie *Small* mit 500 × 500 Pixeln (2 m pro Pixel). Wie viel besser wird der PSNR, wenn das Bild selbst schon gröber ist? Ist das Netz damit näher an JPEG?
- Vergleichen Sie im Fehlerbild Dom, Brücke und Rathaus. Wo liegen die größten Fehler – an Kanten, in Schatten, auf Bäumen?
- Trainieren Sie *Medium* einmal mit 1 000 und einmal mit 6 000 Schritten. Welche Dinge tauchen im Ausschnitt erst nach langem Training auf?
- Überlegen Sie: Ein Höhenmodell derselben Kachel (etwa Dächer und Bäume als Oberflächenmodell) – wäre es für ein Netz leichter oder schwerer als das Foto?
- Was wäre zu tun, um zwei benachbarte Kacheln zu speichern: ein Netz für beide oder zwei Netze? Was passiert an der Grenze?

## Weiterführende Links

- [1] Wikipedia: [Orthofoto](https://de.wikipedia.org/wiki/Orthofoto) – wie aus einem Luftbild ein maßstabsgetreues Orthofoto wird
- [2] Bayerische Vermessungsverwaltung: [OpenData – Digitales Orthophoto 40 cm](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dop40) – Download der Kacheln
- [3] Bayerische Vermessungsverwaltung: [Nutzungsbedingungen](https://geodaten.bayern.de/odd/m/3/html/nutzungsbedingungen.html) – Lizenz CC BY 4.0 und Quellenangabe
- [4] Wikipedia: [UTM-Koordinatensystem](https://de.wikipedia.org/wiki/UTM-Koordinatensystem) – Zonen, Ostwert, Nordwert, Maßstabsfaktor
- [5] Bayerische Vermessungsverwaltung: [Abbildungsverzerrungen bei UTM](https://www.ldbv.bayern.de/mam/ldbv/dateien/utm_abbildungsverzerrungen.pdf) – wie stark UTM-Strecken in Bayern abweichen (PDF)
- [6] Wikipedia: [JPEG](https://de.wikipedia.org/wiki/JPEG) – wie JPEG ein Bild in Blöcke und Wellen zerlegt
- [7] Dupont et al.: [COIN: COmpression with Implicit Neural representations](https://arxiv.org/abs/2103.03123), 2021 – Fotos als neuronales Netz gespeichert (Englisch)
- [8] Cho et al.: [Fourier-Modulated Implicit Neural Representation for Multispectral Satellite Image Compression](https://arxiv.org/abs/2506.01234), IGARSS 2025 – ImpliSat, Satellitenbilder als neuronales Feld (Englisch)
- Tancik et al.: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739), NeurIPS 2020 – Fourier-Features, auch für Bilder (Englisch)
- Wikipedia: [Regensburger Dom](https://de.wikipedia.org/wiki/Regensburger_Dom) – das größte Bauwerk auf der Kachel

---

[Übersicht](README.md) · Zurück: [Kapitel 13 · Location Encoder als Embedding](13-location-encoder.md) · Weiter: [Kapitel 15 · Gebäude als 3D-Feld](15-gebaeude-als-3d-feld.md)

*Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. Es soll helfen, mehr über das Thema „Geodaten in neuronalen Netzen speichern“ zu lernen. Verbesserungsvorschläge gerne an gkvoelkl@nelson-games.de.*
