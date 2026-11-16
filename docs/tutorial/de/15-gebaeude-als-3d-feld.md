# Geodaten in neuronalen Netzen speichern
# Kapitel 15 · Gebäude als 3D-Feld

*[English version](../en/15-gebaeude-als-3d-feld.md)*

![In einer dunklen Werkstatt beleuchtet eine Schreibtischlampe eine Holzplatte mit zwei Modellen eines gotischen Doms mit zwei spitzen Türmen. Links steht ein kleines Modell aus hellem Stein, daneben Aiko, das leuchtende neuronale Netz. Rechts steht ein größeres Modell mit einem Haus mit rotem Dach davor; es ruht auf leuchtenden, durchscheinenden Würfeln, und auch seine Türme schimmern wie aus Glas. Chester mit blauer Kappe hält mit einer Pinzette einen leuchtenden Würfel in die Höhe, um ihn auf das Modell zu setzen.](../img/kapitel-15-chester.jpg)

**Inhalt:**

* Gebäude als Funktion in drei Dimensionen: Ostwert, Nordwert und Höhe als Eingabe, Abstand zur nächsten Wand als Ausgabe
* Wie aus einem 3D-Stadtmodell ein Würfelgitter und ein Abstandsfeld wird – und daraus wieder Gebäude
* Wie genau ein Netz die Altstadt um den Regensburger Dom trifft und was es im Vergleich zum Stadtmodell kostet


## Ausgangspunkt

Bisher hatte jedes Feld in diesem Tutorial zwei Eingaben: einen Ort auf der Erde oder auf einer Karte. Gebäude brauchen eine dritte, die Höhe. Ob ein Punkt in einem Haus liegt, hängt nicht nur davon ab, wo er auf der Karte liegt, sondern auch, wie hoch: Unter dem Dach ist man drinnen, darüber draußen.

Dieses Kapitel verwendet zwei bisher gezeigte Ideen:

1. **Das Abstandsfeld aus Kapitel 12.** Dort war es der Abstand zur Küste, mit Vorzeichen. Hier ist es der Abstand zur nächsten Wand oder zum nächsten Dach – innen negativ, außen positiv. Kapitel 11 hat sie mit *DeepSDF* vorgestellt, einem Verfahren, das von Anfang an für Formen im Raum gedacht war [5].
2. **Die Ebene aus Kapitel 14.** Für ein Gebiet von wenigen hundert Metern ist die Erde eben. Die Koordinaten sind wieder Ostwert und Nordwert in Metern, dazu kommt die Höhe.

Das Netz ist also eine Funktion f(E, N, H): drei Zahlen hinein, ein Abstand heraus. Das Gebiet ist ein Ausschnitt von 300 m × 300 m um den Regensburger Dom, mitten in der Orthofoto-Kachel aus Kapitel 14.

## Das 3D-Stadtmodell

### LoD2

Die Bayerische Vermessungsverwaltung stellt für ganz Bayern ein 3D-Gebäudemodell bereit, kostenlos unter der Lizenz CC BY 4.0 [1]. Es heißt **LoD2** (*Level of Detail 2*). Die Stufen bedeuten:

- **LoD1:** Jedes Gebäude ist ein Klotz mit flachem Dach.
- **LoD2:** Jedes Gebäude hat seine Dachform – Satteldach, Walmdach, Pultdach und so weiter. Fenster, Balkone und Gauben fehlen.

![Das LoD2-Stadtmodell um den Regensburger Dom als 3D-Ansicht in der App, schräg von Südwesten. Die Gebäude sind braun eingefärbt, je höher, desto dunkler. In der Mitte ragt der Dom mit seinen zwei spitzen Türmen und einem schmalen Dachreiter heraus. Um ihn herum Häuserblöcke mit Sattel- und Walmdächern, dazwischen weiße Gassen und Höfe. Fenster, Balkone und andere Einzelheiten gibt es nicht.](../img/app-gebaeude-lod2.png)

Die Dachformen sind dabei aus einem festen Katalog von Standarddächern gewählt. Firste und Traufen liegen meist auf 20 bis 30 cm genau, wo die Dachform stark vereinfacht ist, in Einzelfällen über 1 m daneben [2].

Gespeichert ist das Modell im Format **CityGML** [3, 4]. CityGML ist eine Textdatei im XML-Format. Jedes Gebäude besteht aus Flächen: einer Grundfläche, mehreren Wänden und mehreren Dachflächen. Jede Fläche ist ein Vieleck, beschrieben durch die Koordinaten seiner Ecken – Ostwert, Nordwert und Höhe in Metern.

### Der Ausschnitt

Eine Kachel des Stadtmodells deckt 2 km × 2 km ab und ist 113 MB groß. Die App nimmt daraus ein Quadrat von **300 m × 300 m** um den Dom, und zwar alle Gebäude, die ganz oder teilweise darin liegen:

| Eigenschaft | Wert |
|---|---|
| Gebiet | 300 m × 300 m um den Regensburger Dom |
| Gebäude | 226 |
| Eckpunkte | 36 948 |
| CityGML dieser Gebäude | 7,3 MB |
| höchstes Gebäude | der Dom: 107 m vom Boden bis zur Spitze der Türme |

Der Ausschnitt ist dicht bebaut: Auf gut der Hälfte der Fläche steht ein Gebäude. Die meisten Häuser sind 10 bis 20 m hoch. Der Dom ragt weit heraus.

![Links das Orthofoto aus Kapitel 14, beschnitten auf 300 m × 300 m um den Dom. Darüber in Orange die Umrisse der Gebäude aus dem Stadtmodell; sie passen eng auf die Dächer im Foto. Rechts dasselbe Gebiet von oben, jedes Gebäude in der Farbe seiner Höhe: Die meisten Häuser sind blau bis türkis, 10 bis 20 m hoch. In der Mitte liegt der Dom als grünes Kreuz, seine beiden Türme am Westende leuchten gelb. Gassen, Plätze und Höfe sind weiß.](../img/gebaeude-ausschnitt.svg)

*Daten: Bayerische Vermessungsverwaltung – www.geodaten.bayern.de, CC BY 4.0*

## Vom Stadtmodell zum Feld

### Würfel statt Pixel

Ein Netz lernt aus Beispielen: Ort hinein, Wert heraus. Für die Höhen der Erde lieferte ein Raster diese Beispiele, für das Orthofoto die Pixel. Für Gebäude braucht es ein Raster in drei Dimensionen. Seine Zellen sind kleine Würfel; sie heißen **Voxel** (von *volume* und *pixel*) [6].

Die App legt über den Ausschnitt ein Gitter aus Würfeln von **1 m** Kantenlänge:

- 300 Würfel nach Osten, 300 nach Norden und 120 nach oben, also bis 120 m über dem tiefsten Punkt.
- Das sind **10 800 000 Würfel**.

Für jeden Würfel entscheidet die App, ob er in einem Gebäude liegt. Dazu sucht sie für jede Säule von Würfeln das Gebäude, in dessen Grundriss die Säule steht, und berechnet, wie hoch das Dach genau dort ist. Alle Würfel zwischen dem Boden des Gebäudes und dem Dach liegen **innen**, alle anderen **außen**. Die Wände sind dabei immer senkrecht. Das passt zu LoD2: Ein Stadtmodell dieser Stufe kennt keine Erker oder Vordächer.

Von den 10,8 Millionen Würfeln liegen **7,6 %** innen. Der Rest ist Luft: über den Dächern, in den Gassen und Höfen. Das Gelände selbst gehört nicht zum Modell; eine Straße ist hier einfach Luft bis zum unteren Rand des Gitters.

### Der Abstand

Aus innen und außen wird wie in Kapitel 12 ein Abstand. Für jeden Würfel fragt die App: **Wie weit ist es von hier bis zum nächsten Würfel auf der anderen Seite?** In einem Gebäude ist das der Abstand zur nächsten Wand oder zum Dach, in der Luft der Abstand zum nächsten Gebäude. Innen bekommt er ein Minus, außen ein Plus. Die Oberfläche der Gebäude ist dann genau dort, wo das Feld 0 ist.

Wie in Kapitel 12 wird der Abstand **begrenzt**. Voreingestellt sind ±5 m: Ein Würfel mitten in einem großen Gebäude bekommt −5 m, ein Würfel hoch über den Dächern +5 m. Warum das hilft, zeigt die Messung weiter unten.

![Oben ein senkrechter Schnitt in West-Ost-Richtung durch den Dom und seinen Südturm, als Gitter aus Würfeln: Rot im Gebäude, Weiß die Luft. Der Turm ragt als schmale rote Spitze über 100 m auf, daneben das niedrigere Seitenschiff und ein Bau mit spitzem Dach; links und rechts kleinere Häuser. Unten derselbe Schnitt als Abstandsfeld, begrenzt auf ±5 m: dunkelrot tief im Inneren, zur Wand hin heller, an Wänden und Dächern weiß, außen hellblau und nach 5 m dunkelblau. Eine schwarze Linie folgt der Oberfläche, wo der Abstand 0 ist.](../img/gebaeude-voxel.svg)

### Das neuronale Netz

Das neuronale Netz hat dieselbe Form wie in Kapitel 14, mit zwei Änderungen: Es bekommt drei Zahlen statt zwei, und es gibt eine Zahl aus statt drei Farben.

Ostwert, Nordwert und Höhe werden gleich behandelt. Die App zieht die Ecke des Gitters ab und rechnet alle drei Koordinaten mit **demselben Maßstab** um: 300 m entsprechen dem Bereich −1 bis 1. Die Höhe reicht damit nur von −1 bis −0,2. Würde man sie eigens auf −1 bis 1 strecken, wäre ein Meter nach oben für das Netz zweieinhalbmal so lang wie ein Meter nach Osten. Die Wellen der Fourier-Features wären dann nach oben kürzer als zur Seite.

Für die Fourier-Features gilt wieder **σ = 20**. Eine typische Welle ist damit etwa 300 m / 20 = **15 m** lang – so lang wie ein Altstadthaus hoch ist.

Die App hat drei Vorlagen:

| Vorlage | Frequenzen | Schichten × Neuronen | Gewichte | Datei (float16) |
|---|---:|---|---:|---:|
| *Small* | 128 | 4 × 128 | 82 561 | 161 KB |
| *Medium* | 256 | 4 × 256 | 328 961 | 643 KB |
| *Large* | 256 | 5 × 512 | 1 313 793 | 2,5 MB |

*Small* hat genau so viele Gewichte wie das Standardbeispiel aus Kapitel 1.

Trainiert wird wie gewohnt: In jedem Schritt zieht die App zufällig 65 536 der 10,8 Millionen Würfel und verbessert die Gewichte so, dass der Abstand besser passt.

## Maße für Formen

### Die richtige Seite reicht nicht

In Kapitel 12 war das wichtigste Maß die **richtige Seite**: Wie viele Zellen ordnet das Netz richtig als Land oder Meer ein? Hier taugt dieses Maß wenig. 92,4 % aller Würfel sind Luft. Ein Netz, das überall „außen“ sagt und kein einziges Gebäude kennt, hätte schon 92,4 % richtig.

### IoU

Das übliche Maß für Formen ist deshalb die **IoU** (*Intersection over Union*, auch Jaccard-Koeffizient) [7]. Man nimmt zwei Mengen von Würfeln: die Würfel, die im Stadtmodell innen liegen, und die, die das Netz innen sieht. Dann teilt man

- den Raum, den **beide** innen sehen,
- durch den Raum, den **mindestens einer** von beiden innen sieht.

Ein Beispiel: Ein Haus hat 1 000 m³. Das Netz trifft davon 900 m³ und legt außerdem 100 m³ daneben, wo kein Haus steht. Beide sehen 900 m³ als Haus, mindestens einer 1 100 m³. Die IoU ist 900 / 1 100 = **0,82**. Eine IoU von 1 heißt, die beiden Formen sind gleich; 0 heißt, sie haben nichts gemeinsam. Ein Netz ohne Gebäude hat die IoU 0.

### Versatz der Oberfläche

Die IoU sagt, wie viel Raum stimmt, aber nicht, wo die Fehler liegen. Dafür misst die App wie in Kapitel 12 den **Versatz**: Sie bestimmt die Oberfläche des Netzes, also alle Stellen, an denen sein Abstand 0 ist, und fragt für jeden Punkt dieser Oberfläche, wie weit er von der echten Oberfläche entfernt ist. Angegeben werden der Median und der Wert, unter dem 90 % der Punkte liegen.

Eine Grenze hat dieses Maß: Das Gitter hat Würfel von 1 m. Werte von etwa 0,5 m und weniger – ein halber Würfel – kann es kaum noch auseinanderhalten.

### Aus dem Feld wird wieder eine Oberfläche

Um die Gebäude des Netzes zu sehen und den Versatz zu messen, braucht man die Oberfläche als Fläche im Raum. Das Verfahren dafür heißt **Marching Cubes** [8]. Es geht Würfel für Würfel durch das Gitter. Wo das Feld an den Ecken eines Würfels teils positiv und teils negativ ist, muss die Oberfläche durch diesen Würfel gehen. Marching Cubes setzt dort ein paar Dreiecke ein, so dass sie genau durch die Stellen gehen, an denen das Feld 0 wird. Zusammen ergeben die Dreiecke eine geschlossene Oberfläche. Für die Gebäude des Ausschnitts sind das etwa eine halbe Million Dreiecke.

Die App verwendet dafür die Bibliothek *scikit-image* [9] und zeigt das Ergebnis mit *Plotly* als 3D-Ansicht.

## Ergebnis

Gemessen mit dem 1-m-Gitter auf einem Apple M4 (MPS), mit Begrenzung auf ±5 m. Jede Einstellung wurde dreimal trainiert (Startwerte 0 bis 2). Angegeben ist der Mittelwert, in Klammern der kleinste und der größte Wert der drei Läufe. *Large* mit 6 000 Schritten wurde nicht gemessen; schon 3 000 Schritte dauern eine halbe Stunde.

**IoU:**

| Vorlage | Datei | 1 000 Schritte | 3 000 Schritte | 6 000 Schritte |
|---|---:|---:|---:|---:|
| *Small* | 161 KB | 0,40 (0,31–0,46) | 0,70 (0,61–0,78) | 0,79 (0,73–0,84) |
| *Medium* | 643 KB | 0,89 (0,87–0,93) | 0,95 (0,92–0,97) | 0,97 (0,96–0,98) |
| *Large* | 2,5 MB | 0,93 (0,92–0,95) | 0,98 (0,98–0,98) | – |

**Versatz der Oberfläche, Median / 90 %:**

| Vorlage | 1 000 Schritte | 3 000 Schritte | 6 000 Schritte |
|---|---:|---:|---:|
| *Small* | 1,5 m / 4,1 m | 0,8 m / 2,4 m | 0,6 m / 1,7 m |
| *Medium* | 0,5 m / 0,8 m | 0,4 m / 0,6 m | 0,3 m / 0,5 m |
| *Large* | 0,4 m / 0,6 m | 0,2 m / 0,5 m | – |

Rechenzeit für einen Lauf: *Small* braucht für 1 000 Schritte etwa 50 Sekunden, *Medium* knapp 2 Minuten, *Large* 7 bis 9 Minuten. Für 3 000 Schritte sind es etwa 3 Minuten, 7 Minuten und 26 bis 35 Minuten.

![Diagramm: IoU über der Zahl der Trainingsschritte, 1 000, 3 000 und 6 000. Drei Linien mit Punkten, jeweils mit einem senkrechten Balken für die Spanne der drei Läufe. Grün Small steigt von 0,40 über 0,70 auf 0,79, mit langen Balken. Blau Medium steigt von 0,89 über 0,95 auf 0,97, mit kurzen Balken. Violett Large steigt von 0,93 auf 0,98 und endet bei 3 000 Schritten.](../img/gebaeude-messung.svg)

Drei Dinge fallen auf:

- ***Small* reicht für Gebäude nicht.** Nach 1 000 Schritten stimmt weniger als die Hälfte des Raums überein. Auch nach 6 000 Schritten liegt jeder zehnte Punkt der Oberfläche mehr als 1,7 m daneben – bei Häusern von 10 bis 20 m Höhe ist das viel. Dieselbe Zahl von Gewichten hat in Kapitel 1 die Höhen der ganzen Erde gespeichert. 226 Gebäude mit ihren Dächern haben im Verhältnis zu ihrer Größe weit mehr Kanten.
- ***Small* hängt stark vom Zufall ab.** Die drei Läufe liegen bis zu 0,17 auseinander. Mit dem Startwert ändern sich die Frequenzen der Fourier-Features und die ersten Gewichte. Ein kleines Netz kann nicht alle Gebäude lernen, und welche es zuerst lernt, hängt von diesem Zufall ab. Ab *Medium* liegen die Läufe eng beieinander.
- ***Medium* trifft die Gebäude gut.** Nach 3 000 Schritten liegt die Hälfte der Oberfläche weniger als 0,4 m daneben, 90 % weniger als 0,6 m. Das ist nahe an der Grenze, die das 1-m-Gitter überhaupt messen kann. *Large* ist nur wenig besser, braucht aber viermal so lange.

### Die Begrenzung

Wie stark soll man den Abstand begrenzen? Gemessen mit *Medium* und 3 000 Schritten, ebenfalls je drei Läufe:

| Begrenzung | IoU | Versatz, Median | Versatz, 90 % |
|---|---:|---:|---:|
| ±2 m | **0,98** (0,98–0,99) | 0,2 m | 0,5 m |
| ±5 m | 0,95 (0,92–0,97) | 0,4 m | 0,6 m |
| ±10 m | 0,88 (0,86–0,93) | 0,5 m | 0,9 m |
| ±20 m | 0,81 (0,77–0,85) | 0,5 m | 1,5 m |

**Je stärker die Begrenzung, desto genauer die Gebäude** – genau wie bei der Küste in Kapitel 12. Mit ±2 m ist *Medium* so gut wie *Large* mit ±5 m, bei einem Viertel der Rechenzeit und einem Viertel der Dateigröße.

Der Grund ist derselbe wie in Kapitel 12. Ohne Begrenzung müsste das Netz für jeden Punkt in der Luft wissen, wie weit das nächste Haus entfernt ist – über den Dächern bis zu 100 m. Diese großen Abstände sind für die Form der Gebäude unwichtig, kosten aber Gewichte. Mit Begrenzung ist fast der ganze Raum einfach +2 m oder −2 m. Das Netz kann seine Gewichte dann für den schmalen Streifen an den Wänden und Dächern ausgeben, auf den es ankommt.

Der Preis ist auch derselbe: Ein Netz mit Begrenzung auf ±2 m weiß nur noch, ob ein Punkt weiter als 2 m vom nächsten Gebäude entfernt ist, aber nicht, wie weit.

### Wie das aussieht

Die App zeigt das Stadtmodell und das Netz nebeneinander als 3D-Ansicht. Hier beide aus demselben Blickwinkel nah am Dom, *Medium* nach 1 000 Schritten mit Begrenzung auf ±5 m (IoU 0,87):

![Zwei 3D-Ansichten derselben Häuser um den Dom, gleicher Blickwinkel. Links das Stadtmodell: glatte, senkrechte Wände, gerade Dachkanten, die zwei Domtürme mit sauberen Spitzen und ein dünner Dachreiter auf dem Dach. Rechts das Netz: dieselben Gebäude und Gassen, aber die Kanten sind gerundet und ausgefranst, die Wände leicht wellig, die Domtürme an den Rändern zackig, und vom Dachreiter ist nur ein kurzes Stück übrig.](../img/app-gebaeude-dom-nah.png)

Bei *Medium* nach 1 000 Schritten sind alle größeren Gebäude da: der Dom mit seinen beiden Türmen, die Häuserblöcke mit ihren Höfen, die Gassen dazwischen. Die Formen sind aber weich: Kanten sind gerundet, Wände und Dächer leicht wellig, wie aus Ton geknetet. Die Wellen der Fourier-Features sind hier sichtbar. An den Domtürmen löst sich die oberste Spitze als kleines Stück ab und schwebt frei über dem Turm, weil der Turm darunter zu dünn wird, als dass das Netz ihn ganz trifft.

Ein waagrechter Schnitt durch das Feld zeigt eine Eigenart des Netzes: kleine Ringe über niedrigen Dächern, wo im Stadtmodell nur Luft ist. Dort liegt der Abstand knapp über 0, und das kleine Rauschen des Netzes reicht, um ihn an einzelnen Stellen unter 0 zu drücken. Im 3D-Bild erscheinen diese Stellen als kleine Krümel über den Dächern, im Schnitt als Gewimmel kleiner schwarzer Ringe. Mit mehr Schritten oder einer stärkeren Begrenzung werden sie seltener.

## Was die Gebäude als Netz kosten

Zum Schluss wieder der ehrliche Vergleich der Dateigrößen:

| Darstellung | Größe |
|---|---:|
| CityGML der 226 Gebäude | 7,3 MB |
| CityGML, mit gzip gepackt | 596 KB |
| Würfelgitter innen/außen, 1 Bit je Würfel | 1,3 MB |
| Würfelgitter innen/außen, mit gzip gepackt | **38 KB** |
| Abstandsfeld, 2 Byte je Würfel | 20,6 MB |
| Netz *Small* / *Medium* / *Large*, float16 | 161 KB / 643 KB / 2,5 MB |

Das Netz *Medium* ist etwa ein Zwölftel so groß wie die CityGML-Datei. Das klingt gut, ist aber kein fairer Vergleich: CityGML ist Text, und Text lässt sich gut packen. Mit gzip ist die Datei kleiner als das Netz – und dabei genauer, denn sie kennt jede Ecke auf Zentimeter.

Noch deutlicher ist das Würfelgitter. Gepackt ist es nur 38 KB groß, kleiner als jedes Netz. Gebäude sind für gzip leicht: Eine Säule von Würfeln ist bis zum Dach innen und darüber außen, und viele Säulen nebeneinander sehen gleich aus. Wie in Kapitel 14 gilt also: **Als reine Kompression lohnt sich das Netz nicht.**

Was das Netz bietet, ist etwas anderes. Es ist eine Funktion, die man an jedem Punkt abfragen kann, auch zwischen den Würfeln, ohne vorher ein Gitter zu entpacken oder Vielecke zu durchsuchen. Eine einzige Abfrage beantwortet: Liegt dieser Punkt in einem Gebäude? Wie weit ist es bis zur nächsten Wand – bis zur Grenze von ±5 m? Solche Fragen stellen etwa Programme, die den Flug einer Drohne durch eine Stadt planen. Und die Oberfläche ist glatt statt treppenförmig wie die Würfel.

In der Forschung ist diese Darstellung für einzelne Formen weit verbreitet. DeepSDF lernt Abstandsfelder für Möbel, Autos und Flugzeuge, und meist ein Netz für viele Formen zugleich [5]. *Occupancy Networks* lernen statt des Abstands nur die Frage „innen oder außen?“ [10]. Kapitel 21 greift diese Idee auf: ein gemeinsames Netz für viele Kacheln.

## Ausprobieren

Der Reiter **Buildings 3D** braucht einen großen Download und viel Rechenzeit und läuft deshalb nur in der lokalen Fassung der App, nicht online.

### Daten laden

Öffnen Sie den Reiter **Buildings 3D**. Beim ersten Mal fragt die App, ob sie die Kachel des Stadtmodells herunterladen darf. Klicken Sie auf **Download (108 MB)**. Danach baut die App das Würfelgitter und das Abstandsfeld; das dauert beim ersten Mal etwa eine Minute.

### Netz trainieren

| Einstellung | Wert | Bedeutung |
|---|---|---|
| *Voxel size* | 1 m | Kantenlänge der Würfel; 2 m geht schneller, ist aber gröber |
| *Network* | Medium | eine der drei Vorlagen aus der Tabelle oben |
| *Limit the distance to* | ±5 m | Begrenzung des Abstands |
| *Steps* | 1000 | Zahl der Trainingsschritte |
| *Device* | auto | Grafikchip, wenn vorhanden |

Unter den Einstellungen nennt die App die Zahl der Gebäude und Eckpunkte, die Größe der CityGML-Datei, die Größe des Gitters und die Größe des Netzes.

![Die Einstellungen im Reiter Buildings 3D: Voxel size 1 m, Network Medium, Limit the distance to ±5 m, Steps 1000, Device auto. Darunter die Zeile 226 buildings · 36,948 corner points · CityGML 7.3 MB · grid 300 × 300 × 120 = 10,800,000 voxels · network: 328,961 weights = 642.5 KB in float16, dazu die Quellenangabe. Darunter der Knopf Train.](../img/app-gebaeude-einstellungen.png)

Klicken Sie auf **Train**. Ein Balken zeigt den Fortschritt und den Anteil der Würfel auf der richtigen Seite; nach gut eineinhalb Minuten ist das Netz fertig. Oben erscheinen vier Zahlen: die IoU, der Versatz im Median und für 90 % und die Größe des Netzes im Vergleich zur CityGML-Datei. Bei uns lag die IoU bei 0,87.

Darunter stehen zwei 3D-Ansichten nebeneinander: links das Stadtmodell, rechts das Netz. Die Farbe zeigt die Höhe über dem Boden. Mit der Maus können Sie beide Ansichten drehen und vergrößern.

![Der Reiter Buildings 3D nach dem Training: oben Volume overlap (IoU) 0.87, Surface offset (median) 0.5 m, Surface offset (90 %) 0.9 m und Network vs CityGML 642.5 KB · 12 : 1. Darunter zwei 3D-Ansichten der Altstadt von Südwesten, links LoD2 building models, rechts Network. Beide zeigen dieselben Häuserblöcke und in der Mitte den Dom; beim Netz sind die Kanten gerundet, die Wände leicht wellig, und über den Domtürmen schweben die Spitzen als kleine abgelöste Stücke.](../img/app-gebaeude-3d.png)

### Schnitt durch das Feld

Ganz unten zeigt die App einen waagrechten Schnitt durch das Abstandsfeld, links aus dem Stadtmodell, rechts aus dem Netz. Mit **Height above the lowest ground** wählen Sie die Höhe. Rot heißt innen, Blau außen, Weiß nahe an einer Wand; die schwarze Linie ist die 0-Linie, also die Wände in dieser Höhe.

![Oben der Regler Height above the lowest ground auf 10 m. Darunter zwei waagrechte Schnitte in dieser Höhe. Links LoD2: rote Häuserblöcke auf blauem Grund, hellblaue Flächen über Dächern knapp unter 10 m, die schwarze Linie folgt genau den Wänden. Rechts Network: dieselben Blöcke, die Farben fleckig und die Linie etwas weicher. In den hellblauen Flächen über niedrigen Dächern wimmelt es von kleinen schwarzen Ringen.](../img/app-gebaeude-schnitt.png)

## Ideen zum Weiterprobieren

- Stellen Sie **Limit the distance to** auf ±2 m und trainieren Sie *Medium* erneut. Schweben die Spitzen der Domtürme noch frei? Verschwinden die Krümel?
- Trainieren Sie *Small* zweimal hintereinander mit denselben Einstellungen. Fehlen beide Male dieselben Gebäude?
- Schieben Sie den Schnitt langsam nach oben. In welcher Höhe sind nur noch der Dom und wenige hohe Häuser übrig? Wo ist das Netz dort am unsichersten?
- Stellen Sie *Voxel size* auf 2 m. Das Gitter hat dann nur ein Achtel der Würfel. Wird das Netz dadurch schlechter – oder nur die Messung gröber?
- Überlegen Sie: Das Netz kennt nur Gebäude. Was müsste man tun, damit es auch das Gelände kennt, etwa den Abfall zur Donau?

## Weiterführende Links

- [1] Bayerische Vermessungsverwaltung: [OpenData – 3D-Gebäudemodelle LoD2](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2) – Download der Kacheln
- [2] Bayerische Vermessungsverwaltung: [Hinweise zu den LoD2-Daten](https://geodaten.bayern.de/odd/m/3/pdf/hinweise_daten_lod2_download.pdf) – Inhalt, Dachformen und Genauigkeit (PDF)
- [3] Wikipedia: [CityGML](https://de.wikipedia.org/wiki/CityGML) – das Format für 3D-Stadtmodelle und seine Detailstufen
- [4] Open Geospatial Consortium: [CityGML](https://www.ogc.org/standards/citygml/) – der Standard (Englisch)
- [5] Park et al.: [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103), CVPR 2019 – vorzeichenbehaftete Abstandsfunktionen als neuronales Feld (Englisch)
- [6] Wikipedia: [Voxel](https://de.wikipedia.org/wiki/Voxel) – Rasterpunkte in drei Dimensionen
- [7] Wikipedia: [Jaccard-Koeffizient](https://de.wikipedia.org/wiki/Jaccard-Koeffizient) – Schnitt durch Vereinigung als Maß für Ähnlichkeit
- [8] Wikipedia: [Marching Cubes](https://de.wikipedia.org/wiki/Marching_Cubes) – wie aus einem Feld eine Oberfläche aus Dreiecken wird
- [9] scikit-image: [skimage.measure.marching_cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) – die Funktion, die die App verwendet (Englisch)
- [10] Mescheder et al.: [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828), CVPR 2019 – „innen oder außen?“ als neuronales Feld (Englisch)
- Curless und Levoy: [A Volumetric Method for Building Complex Models from Range Images](https://graphics.stanford.edu/papers/volrange/), SIGGRAPH 1996 – begrenzte Abstandsfelder für 3D-Scans (Englisch)
- Wikipedia: [Regensburger Dom](https://de.wikipedia.org/wiki/Regensburger_Dom) – das höchste Gebäude im Ausschnitt

---

[Übersicht](README.md) · Zurück: [Kapitel 14 · Orthofotos in UTM](14-orthofotos-in-utm.md) · Weiter: [Kapitel 16 · Gaussian Splatting](16-gaussian-splatting.md)

*Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. Es soll helfen, mehr über das Thema „Geodaten in neuronalen Netzen speichern“ zu lernen. Verbesserungsvorschläge gerne an gkvoelkl@nelson-games.de.*
