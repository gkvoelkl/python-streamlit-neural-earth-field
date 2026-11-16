# Geodaten in neuronalen Netzen speichern
# Kapitel 16 · Gaussian Splatting

*[English version](../en/16-gaussian-splatting.md)*

![In einer abendlichen Werkstatt sprüht Chester mit blauer Kappe mit einer Airbrush-Pistole einen Strahl aus gelben, ovalen Farbflecken auf eine Leinwand. Auf der Leinwand ist eine Altstadt von oben gemalt: ein breiter blauer Fluss, eine steinerne Bogenbrücke, rote Dächer und zwei Kirchtürme. Neben Chester steht Aiko, das leuchtende neuronale Netz, und breitet die Arme aus. Rechts auf einem Tisch leuchtet ein durchscheinendes Modell eines gotischen Doms mit zwei Türmen.](../img/kapitel-16-chester.jpg)

**Inhalt:**

* Eine andere Art, Geodaten zu speichern: kein Netz, sondern viele kleine Gaußglocken
* Das Luftbild aus Kapitel 14 als Summe von Gaußglocken – im Vergleich mit dem Netz und mit JPEG
* Die Gebäude aus Kapitel 15 als Glocken im Raum
* Was das berühmte 3D Gaussian Splatting zusätzlich macht


## Ausgangspunkt

Seit Kapitel 1 steckt alles in einem neuronalen Netz. Jeder Ort, jedes Pixel, jeder Würfel läuft durch dieselben Gewichte. Ändert das Training ein Gewicht, ändert sich das Ergebnis überall ein wenig.

Das hat in Kapitel 14 eine Grenze gezeigt: Das Netz zeichnet das Luftbild weich. Viele kleine Dinge, die nur an einer Stelle vorkommen – ein Auto, eine Dachkante, ein Baum –, muss das Netz mit Gewichten beschreiben, die gleichzeitig für alle anderen Pixel gebraucht werden.

Dieses Kapitel probiert das Gegenteil aus: Man speichert das Bild als viele kleine Bausteine, und jeder Baustein ist nur für seine Umgebung zuständig. Die Bausteine sind **Gaußglocken**. Das Verfahren heißt **Gaussian Splatting**. Bekannt wurde es 2023 durch Kerbl und Kollegen, die damit ganze Szenen aus Fotos in Echtzeit darstellen [1]. Hier geht es zuerst um die einfachste Form: ein Bild in der Ebene [2].

## Was eine Gaußglocke ist

Eine Gaußglocke ist ein weicher Fleck: in der Mitte am stärksten, nach außen gleichmäßig schwächer, ohne harten Rand [3]. In der Ebene hat sie die Form einer Ellipse. Beschrieben wird sie durch **acht Zahlen**:

| Zahlen | Bedeutung |
|---:|---|
| 2 | die Mitte: Ostwert und Nordwert |
| 2 | zwei Breiten: wie weit der Fleck in die eine und in die andere Richtung reicht |
| 1 | eine Drehung: in welche Richtung die Ellipse zeigt |
| 3 | eine Farbe: Rot, Grün und Blau |

Das Bild ist die **Summe** aller Glocken. Jede Glocke gibt ihre Farbe an die Pixel in ihrer Nähe ab, in der Mitte voll, weiter außen immer weniger. Wo sich Glocken überlappen, addieren sich die Farben. Dieses Auftragen von Flecken heißt im Englischen *splatting* – „Kleckse auftragen“.

![Drei Felder mit je einer leuchtenden Glocke auf schwarzem Grund, die Mitte als gelber Punkt und eine Breite als gestrichelte Ellipse: eine runde, große grüne Glocke, eine lange, schmale rote Glocke und eine lange blaue Glocke, um 40° gedreht. Daneben ihre Summe: Wo sich die Glocken überlappen, mischen sich die Farben zu Gelb und Türkis. Darunter ein Kasten: Eine Glocke = 8 Zahlen, Mitte (2), zwei Breiten (2), Drehung (1), Farbe Rot, Grün, Blau (3).](../img/splat-glocken.svg)

Der Unterschied zum neuronalen Netz liegt darin, wer wofür zuständig ist. Eine Glocke wirkt nur auf ein kleines Fenster von Pixeln um ihre Mitte. Ändert das Training eine Glocke, ändert sich nur dort etwas. Und eine Glocke kann sich ihre Form selbst suchen: lang und schmal an einer Kante, rund und groß auf einer gleichmäßigen Fläche.

## Wie die Glocken lernen

Trainiert wird wie in Kapitel 2, nur sind die Gewichte jetzt die Zahlen der Glocken:

1. **Start:** Die Glocken liegen auf einem regelmäßigen Gitter über der Kachel, jede mit der Farbe des Pixels unter ihrer Mitte.
2. **Ein Schritt:** Alle Glocken werden aufs Bild gemalt, das Ergebnis wird mit dem Luftbild verglichen. Dann werden alle Zahlen ein wenig verbessert: Mitten wandern, Breiten wachsen oder schrumpfen, Ellipsen drehen sich, Farben ändern sich.
3. **Wiederholen**, bis das Bild passt.

![Drei Ausschnitte von 100 m × 100 m an der Steinernen Brücke nebeneinander. Links der Start: Die Glocken liegen als gleich große gelbe Kreise auf einem regelmäßigen Gitter, das Bild darunter ist verschwommen und kariert. In der Mitte nach 1 000 Schritten: Die Kreise sind verschoben, viele zu Ellipsen gestreckt und gedreht; entlang der hellen Brücke rechts und entlang des Uferwegs unten liegen sie dicht und schmal in Richtung der Kante, das Bild darunter zeigt Wiese, Bäume, Wasser und Brücke. Rechts das Original als Ziel des Trainings.](../img/splat-training.svg)

Anders als bei den Netzen vergleicht jeder Schritt das **ganze** Bild, nicht nur eine Stichprobe von Pixeln. Das geht, weil jede Glocke nur ihr kleines Fenster ausrechnen muss – bis zur dreifachen Breite. Damit das Fenster klein bleibt, darf eine Glocke höchstens gut halb so breit werden (das 0,6-Fache) wie der Abstand der Glocken am Start.

Damit der Vergleich mit Kapitel 14 fair ist, gibt es drei Vorlagen mit derselben Dateigröße wie die drei Netze. Jede Zahl wird wie dort in float16 gespeichert, also mit 2 Byte:

| Vorlage | Glocken | Datei | Netz aus Kapitel 14 |
|---|---:|---:|---:|
| *Small* | 10 201 | 159 KB | 162 KB |
| *Medium* | 41 209 | 644 KB | 644 KB |
| *Large* | 164 025 | 2,5 MB | 2,5 MB |

## Ergebnis: das Luftbild

Gemessen mit derselben Kachel wie in Kapitel 14, 1 250 × 1 250 Pixel, auf einem Apple M4 (MPS), je ein Lauf. Angegeben ist der PSNR des ganzen Bildes; die Werte für das Netz stammen aus Kapitel 14.

| Vorlage | Glocken, 1 000 Schritte | Glocken, 3 000 Schritte | Netz, 1 000 Schritte | Netz, 3 000 Schritte | JPEG gleicher Größe |
|---|---:|---:|---:|---:|---:|
| *Small* | **22,6 dB** | **23,4 dB** | 21,5 dB | 23,0 dB | 27,9 dB |
| *Medium* | **25,9 dB** | **27,4 dB** | 23,7 dB | 26,1 dB | 36,3 dB |
| *Large* | **30,9 dB** | **33,3 dB** | 24,6 dB | 27,9 dB | 38,6 dB (nur 829 KB) |

![Diagramm: PSNR in dB über der Dateigröße, logarithmisch von 50 KB bis über 4,5 MB. Die graue JPEG-Linie steigt von 22 dB bei 53 KB bis 38,6 dB bei 829 KB. Bei 160 KB, 644 KB und 2,5 MB stehen je zwei Paare von Zeichen für 1 000 und 3 000 Schritte: leere Kreise für das Netz, gefüllte Rauten für die Glocken. Die Rauten liegen jeweils über den Kreisen, am deutlichsten bei 2,5 MB mit 30,9 und 33,3 dB gegen 24,6 und 27,9 dB. Alle Zeichen liegen unter der JPEG-Linie.](../img/splat-messung.svg)

Drei Dinge fallen auf:

- **Die Glocken schlagen das Netz bei jeder Größe.** Bei *Small* ist der Vorsprung klein, gut 1 dB. Je mehr Glocken, desto größer wird er: Bei *Large* sind es nach 1 000 Schritten 6 dB – der Fehler ist nur noch halb so groß. Viele kleine Bausteine können viele kleine Dinge an ihrem Ort beschreiben; ein Netz muss dafür alle Gewichte zugleich abstimmen.
- **JPEG bleibt vorn.** Auch *Large* liegt noch fast 8 dB hinter einem JPEG, das ein Drittel so groß ist. JPEG ist seit über 30 Jahren auf Fotos abgestimmt und packt seine Zahlen zusätzlich verlustfrei zusammen. Die Glocken dagegen speichern jede Zahl einfach mit 2 Byte. Zhang und Kollegen kommen mit genau solchen Glocken in die Nähe der Netz-Verfahren aus der Forschung, wenn sie die Zahlen geschickt runden und packen [2].
- **Training kostet mehr Zeit.** 1 000 Schritte dauern bei allen drei Vorlagen 6 bis 8 Minuten, gleich lang, weil jeder Schritt das ganze Bild malt. Das Netz *Small* braucht dafür 45 Sekunden. Erst bei *Large* sind beide ungefähr gleich schnell.

### Wie das aussieht

Auch hier zeigt der Ausschnitt an der Steinernen Brücke den Unterschied:

![Vier Ausschnitte nebeneinander, jeweils mit der Steinernen Brücke als senkrechtem hellem Streifen rechts, Pfeilern im Wasser, Bäumen und einer Wiese links oben. Links das Original, scharf. Daneben das Netz Small mit 21,5 dB: weich verschwommen, die Pfeiler als runde Beulen. Dann die Glocken Small mit 22,6 dB: ebenfalls weich, aber die Brücke, der Uferweg und die Wege auf der Wiese sind schärfer, die Bäume bestehen aus Flecken. Rechts das JPEG mit 27,9 dB: scharf, aber in kleine Blöcke zerlegt.](../img/splat-bruecke.svg)

Die Glocken machen eine dritte Art von Fehler, neben der Unschärfe des Netzes und den Blöcken von JPEG: Das Bild wirkt **fleckig**. Gleichmäßige Flächen wie das Wasser der Donau sind gut getroffen. Kanten dagegen bestehen aus vielen ovalen Flecken nebeneinander. Die Brücke und die Uferwege sind schärfer als beim Netz, weil sich dort die Glocken dichter legen und in Richtung der Kante strecken.

Das zeigt die App direkt: Sie zeichnet für einen Ausschnitt jede Glocke als Ellipse ein. Ganz lang und schmal werden die Glocken dabei nicht – ihre Breite ist begrenzt, damit das Fenster klein bleibt.

![Der Ausschnitt an der Steinernen Brücke, stark vergrößert und weich verschwommen, darüber jede Glocke als dünne gelbe Ellipse. Auf dem Wasser und der Wiese liegen fast runde Ellipsen gleichmäßig nebeneinander. Rechts entlang der hellen Brücke und unten entlang des Uferwegs sind die Ellipsen leicht in Richtung der Kante gestreckt und liegen dichter.](../img/app-splat-ellipsen.png)

## Gebäude als Glocken im Raum

Dieselbe Idee funktioniert in drei Dimensionen. Statt Farben addieren die Glocken jetzt, **wie viel Gebäude** an einer Stelle ist. Wo die Summe größer als ½ ist, zählt ein Würfel als innen, sonst als außen. Die Oberfläche der Gebäude ist dort, wo die Summe genau ½ ist – wie die 0-Linie beim Abstand in Kapitel 15.

Eine Glocke im Raum hat wieder **acht Zahlen**: die Mitte (drei Zahlen), drei Breiten, eine Drehung um die senkrechte Achse und eine Stärke. Eine Drehung genügt, weil die Wände im Stadtmodell LoD2 immer senkrecht stehen. Die drei Vorlagen haben wieder genau die Dateigröße der Netze aus Kapitel 15: 10 320, 41 120 und 164 224 Glocken.

Einen Unterschied zum Netz muss man offen nennen: Die Glocken **starten innerhalb der Gebäude**, an zufällig gezogenen Würfeln. Das Netz in Kapitel 15 begann dagegen ohne jedes Wissen über die Gebäude. Dieser Start ist beim Gaussian Splatting üblich – das echte 3D-Verfahren beginnt mit einer Punktwolke, die aus den Fotos berechnet wurde [1]. Er erklärt aber einen Teil des Vorsprungs.

Gemessen mit dem 1-m-Gitter auf einem Apple M4 (MPS), je ein Lauf (Startwert 0). Die Werte für das Netz stammen aus Kapitel 15 (Begrenzung ±5 m, Mittel aus drei Läufen). „Start“ ist die IoU der Glocken vor dem ersten Trainingsschritt.

| Vorlage | Datei | Glocken: Start | Glocken: 300 Schritte | Versatz Median / 90 % | Netz: 1 000 Schritte | Netz: 3 000 Schritte |
|---|---:|---:|---:|---:|---:|---:|
| *Small* | 161 KB | 0,71 | **0,93** | 0,33 m / 0,53 m | 0,40 | 0,70 |
| *Medium* | 643 KB | 0,80 | **0,97** | 0,15 m / 0,50 m | 0,89 | 0,95 |
| *Large* | 2,5 MB | 0,87 | **1,00** | 0,09 m / 0,28 m | 0,93 | 0,98 |

*Medium* mit 1 000 statt 300 Schritten erreicht eine IoU von 0,99 und einen Versatz von 0,11 m / 0,42 m. Rechenzeit für 300 Schritte: *Small* gut 7 Minuten, *Medium* knapp 10, *Large* 13 Minuten; *Medium* mit 1 000 Schritten 34 Minuten.

Drei Dinge fallen auf:

- **Die Glocken treffen die Gebäude viel besser als das Netz.** Schon *Small* liegt nach 300 Schritten mit 0,93 über dem viermal so großen Netz *Medium* nach 1 000 Schritten (0,89). *Large* trifft praktisch jeden Würfel (IoU 0,997). Der Versatz liegt bei *Medium* und *Large* unter einem halben Meter und damit an der Grenze dessen, was das 1-m-Gitter überhaupt messen kann (Kapitel 15).
- **Ein guter Teil davon ist der Start.** Schon vor dem ersten Schritt erreichen die Glocken eine IoU von 0,71 bis 0,87 – mehr als das Netz *Small* nach 6 000 Schritten (0,79). Die Glocken bekommen beim Start zu wissen, wo Gebäude sind; das Netz muss es selbst herausfinden. Das Training bringt danach trotzdem noch viel: Bei *Medium* steigt die IoU von 0,80 auf 0,97.
- **Gebäude liegen den Glocken.** Häuser sind Blöcke mit senkrechten Wänden. Eine gestreckte Glocke, die sich um die Hochachse dreht, passt gut in eine Hauswand; viele davon füllen einen Block. Ein Netz muss dieselben Kanten aus Wellen zusammensetzen, die über den ganzen Ausschnitt reichen.

Zum Vergleich mit Kapitel 15 derselbe Blickwinkel nah am Dom, *Medium* mit 300 Schritten (IoU 0,97):

![Oben die Zahlen Volume overlap (IoU) 0.97, Surface offset (median) 0.2 m, Surface offset (90 %) 0.5 m und Gaussians vs CityGML 642.5 KB · 12 : 1. Darunter zwei 3D-Ansichten derselben Häuser um den Dom aus demselben Blickwinkel. Links das Stadtmodell mit glatten Wänden, den zwei Domtürmen und einem dünnen Dachreiter. Rechts die Glocken: kaum vom Stadtmodell zu unterscheiden, die Kanten nur wenig weicher; vom Dachreiter ist ein kurzes Stück übrig, über einer Turmspitze schwebt ein winziger Punkt.](../img/app-splat-gebaeude.png)

Im Vergleich mit dem Netz in Kapitel 15 sind die Wände gerade, die Kanten kaum gerundet, und die Domtürme behalten ihre Form.

## Was echtes 3D Gaussian Splatting zusätzlich macht

Das Verfahren von Kerbl und Kollegen geht von Fotos aus, nicht von einem fertigen Raster [1]. Daraus ergeben sich drei Unterschiede zu diesem Kapitel:

- **Die Glocken werden von der Seite gesehen.** Für jedes Foto wird berechnet, wie die Glocken von der Kamera aus aussehen. Dafür werden sie nach ihrer Entfernung sortiert, und nähere verdecken fernere teilweise. In diesem Kapitel genügt eine einfache Summe, weil das Luftbild von oben kommt und die Gebäude als Würfelgitter vorliegen.
- **Die Farbe hängt von der Blickrichtung ab.** Ein Dach glänzt aus einer Richtung, aus einer anderen nicht. Dafür speichert jede Glocke ihre Farbe als Summe von Kugelflächenfunktionen – denselben Wellen auf der Kugel wie in Kapitel 6.
- **Es sind Millionen von Glocken.** Kerbl und Kollegen arbeiten mit 1 bis 5 Millionen Glocken je Szene. Während des Trainings werden Glocken geteilt, wo Einzelheiten fehlen, und entfernt, wo sie überflüssig sind.

Für ganze Städte aus Luftbildern ist das aktive Forschung. Die Schwierigkeit ist vor allem der Speicher: Eine Stadt braucht so viele Glocken, dass man sie in Blöcke aufteilen muss [4].

## Was es kostet

| | Größe |
|---|---:|
| Luftbild, unkomprimiert (1 250 × 1 250 Pixel) | 4,5 MB |
| Glocken *Small* / *Medium* / *Large* | 159 KB / 644 KB / 2,5 MB |
| JPEG mit dem PSNR der Glocken *Medium* (27,4 dB) | unter 160 KB |
| CityGML der 226 Gebäude (Kapitel 15) | 7,3 MB |
| Glocken für die Gebäude *Small* / *Medium* / *Large* | 161 KB / 643 KB / 2,5 MB |

Als Kompression für Luftbilder lohnen sich die Glocken in dieser einfachen Form so wenig wie das Netz: JPEG erreicht dieselbe Qualität mit einem Viertel der Größe. Was die Glocken bieten, ist etwas anderes. Sie sind schnell zu zeichnen, weil jede nur ihr kleines Fenster braucht – Zhang und Kollegen zeichnen ein Bild über tausendmal pro Sekunde [2]. Und man kann jede Glocke einzeln ansehen, verschieben oder löschen. In einem Netz gibt es kein Gewicht, das „die Brücke“ ist.

## Ausprobieren

Der Reiter **Gaussian Splatting** braucht die Daten aus den Reitern *Orthophoto* und *Buildings 3D* und viel Rechenzeit. Er läuft deshalb nur in der lokalen Fassung der App, nicht online.

### Luftbild

Wählen Sie oben **Aerial photo (2D)**.

| Einstellung | Wert | Bedeutung |
|---|---|---|
| *Pixels* | 1250 × 1250 · 0.8 m | Auflösung, wie in Kapitel 14 |
| *Gaussians* | Small · 10,201 | eine der drei Vorlagen aus der Tabelle oben |
| *Steps* | 1000 | Zahl der Trainingsschritte |
| *Device* | auto | Grafikchip, wenn vorhanden |

![Der Reiter Gaussian Splatting: oben die Auswahl Store mit Aerial photo (2D) und Buildings (3D), darunter ein kurzer Text und die Einstellungen Pixels 1250 × 1250 · 0.8 m, Gaussians Small · 10,201, Steps 1000 und Device auto. Darunter die Zeile 1,562,500 pixels × 3 colours = 4.5 MB uncompressed · 10,201 Gaussians × 8 numbers = 159.4 KB in float16 (about the size of the Small network in the Orthophoto tab) und die Quellenangabe, darunter der Knopf Train.](../img/app-splat-einstellungen.png)

Klicken Sie auf **Train**. Nach etwa 6 Minuten erscheinen der PSNR der Glocken, die Dateigröße und der PSNR eines JPEG gleicher Größe. Haben Sie im Reiter *Orthophoto* das gleich große Netz trainiert, steht auch dessen PSNR daneben. Darunter folgen Original, Glocken und JPEG nebeneinander, dann ein Ausschnitt mit dem Fehlerbild und ganz unten die Glocken als Ellipsen.

![Der Reiter nach dem Training: oben PSNR of the Gaussians 22.6 dB, File size (float16) 159.4 KB, Compression ratio 29 : 1 und JPEG of the same size (quality 26) 27.9 dB. Darunter die ganze Kachel dreimal nebeneinander: Original (DOP40), Gaussians und JPEG, 158.1 KB. Bei den Glocken sind die Dächer der Altstadt weich und fleckig, Fluss, Brücken und Inseln gut zu erkennen.](../img/app-splat-ergebnis.png)

### Gebäude

Wählen Sie oben **Buildings (3D)**, dann *Voxel size* 1 m, *Gaussians* Medium und *Steps* 300, und klicken Sie auf **Train**. Nach knapp 10 Minuten ist das Training fertig. Die App zeigt danach IoU und Versatz der Oberfläche und die Gebäude als 3D-Ansicht neben dem Stadtmodell.

## Ideen zum Weiterprobieren

- Vergleichen Sie im Ausschnitt mit den Ellipsen den Dom, die Brücke und das Rathaus. Wo werden die Glocken lang und schmal, wo rund?
- Trainieren Sie *Small* mit 500 × 500 Pixeln. Die Glocken sind dann im Verhältnis zum Bild genauso groß, aber es gibt weniger Einzelheiten. Wie nahe kommen sie an JPEG heran?
- Trainieren Sie im Reiter *Orthophoto* das Netz *Large* und im Reiter *Gaussian Splatting* die Glocken *Large*. Welche Fehler machen beide an derselben Stelle?
- Schauen Sie im Ausschnitt mit den Ellipsen, wie weit die Glocken beim Training von ihrem Startgitter weggewandert sind. Sammeln sie sich dort, wo es viele Einzelheiten gibt?
- Wie würde man ein Höhenmodell der ganzen Erde mit Glocken speichern? Was passiert an der Naht bei ±180° (Kapitel 4)?

## Weiterführende Links

- [1] Kerbl et al.: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079), ACM Transactions on Graphics 2023 – Szenen aus Fotos als Millionen von 3D-Glocken, Darstellung in Echtzeit ([Projektseite](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)) (Englisch)
- [2] Zhang et al.: [GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](https://arxiv.org/abs/2403.08551), ECCV 2024 – Bilder als Summe von 2D-Glocken mit 8 Zahlen je Glocke (Englisch)
- [3] Wikipedia: [Gauß-Funktion](https://de.wikipedia.org/wiki/Gau%C3%9F-Funktion) – die Glockenkurve und ihre Breite
- [4] HUG: [Hierarchical Urban Gaussian Splatting with Block-Based Reconstruction for Large-Scale Aerial Scenes](https://arxiv.org/abs/2504.16606), 2025 – ganze Stadtgebiete aus Luftbildern (Englisch)
- Dupont et al.: [COIN: COmpression with Implicit Neural representations](https://arxiv.org/abs/2103.03123), 2021 – das Gegenstück mit einem Netz, wie in Kapitel 14 (Englisch)

---

[Übersicht](README.md) · Zurück: [Kapitel 15 · Gebäude als 3D-Feld](15-gebaeude-als-3d-feld.md) · Weiter: [Anhang · Messungen](anhang-messungen.md)

*Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. Es soll helfen, mehr über das Thema „Geodaten in neuronalen Netzen speichern“ zu lernen. Verbesserungsvorschläge gerne an gkvoelkl@nelson-games.de.*
