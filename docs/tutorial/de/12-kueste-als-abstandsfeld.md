# Geodaten in neuronalen Netzen speichern
# Kapitel 12 · Die Küste als Abstandsfeld

*[English version](../en/12-kueste-als-abstandsfeld.md)*

![An einem sonnigen Strand steht Aiko, ein leuchtendes neuronales Netz, im flachen türkisen Wasser und hält das Ende eines gelben Maßbands. Chester mit blauer Kappe sitzt oben auf einer grünen Düne und spannt das Band mit beiden Pfoten. Im Wasser ziehen sich leuchtende Linien in Wellen vom Ufer weg.](../img/kapitel-12-chester.jpg)

**Inhalt:**

* Aus einer Küstenlinie – also Vektordaten – ein Feld machen: den Abstand zur Küste mit Vorzeichen
* Warum ein neuronales Netz die Küstenlinie genauer wiedergibt als die Höhen
* Was es bringt, den Abstand zu begrenzen – und was die Küste als neuronales Netz im Vergleich zu den Vektordaten an Speicher kostet


## Ausgangspunkt

In Kapitel 11 war die Küste die Linie, an der das Höhennetz genau 0 m ausgibt. Das Standardbeispiel ordnete damit 97 % der Erdoberfläche richtig als Land oder Meer ein, in Küstennähe (Höhen zwischen −200 m und +200 m) aber nur 80 %.

Das ist kein Zufall. Das Höhennetz wurde nie auf die Küstenlinie trainiert, sondern generell auf alle Höhen. Das Training des neuronalen Netzes bewertet einen Fehler von 100 m in den Alpen genauso wie einen Fehler von 100 m in den Niederlanden.

In den Alpen machen 100 m nicht viel aus. Im flachen Land an der Nordsee verschiebt derselbe Fehler die 0-m-Linie um viele Kilometer.

Kapitel 11 hat einen Ausweg beschrieben: Vektordaten in ein Feld verwandeln, das genau die gewünschte Linie enthält, und dieses Feld lernen (Weg 1). Dieses Kapitel probiert das an der Küste aus.

## Die Küste als Vektordaten

Die Küstenlinie in diesem Kapitel stammt aus **Natural Earth**, einem freien Kartendatensatz (*public domain*) [1]. Das Beispiel verwendet den Maßstab 1:50 Millionen; er ist für die Gitter der App fein genug.

In diesem Maßstab besteht die Küstenlinie aus gut 1 400 Linienzügen mit zusammen **60 416 Punkten**. Jeder Punkt ist ein Paar aus Länge und Breite. Mit 4 Byte pro Zahl (float32) sind das 483 328 Byte, also **472 KB**.

Dazu kommen die Landflächen von Natural Earth. Sie werden nur gebraucht, um zu entscheiden, auf welcher Seite der Küstenlinie ein Ort liegt.

![Links Korsika und Sardinien als beige Landflächen auf hellblauem Meer. Ihre Küsten sind schwarze Linien mit vielen kleinen orangen Punkten. Ein blauer Rahmen markiert den Nordosten Sardiniens. Rechts ist dieser Ausschnitt vergrößert: Die Küste besteht aus orangen Punkten, die durch gerade schwarze Strecken verbunden sind. Drei aufeinanderfolgende Punkte sind schwarz umrandet, darunter stehen ihre Koordinaten, etwa 9,54° Ost und 41,05° Nord.](../img/kueste-vektordaten.svg)

## Von der Linie zum Feld

Aus der Linie wird ein Feld, indem man für jede Zelle des Gitters fragt: **Wie weit ist es von hier bis zur nächsten Küste?** Die Messung entlang der Erdoberfläche erfolgt in Kilometern.

Damit man Land und Meer unterscheiden kann, bekommt der Abstand ein Vorzeichen: an Land positiv, im Meer negativ – genauso wie bei der Höhe. Die Küste ist dann wie in Kapitel 11 die Linie, an der das Feld genau 0 ist. Kapitel 11 hat diese Darstellung als vorzeichenbehaftete Abstandsfunktion (*signed distance function*, SDF) vorgestellt [2]; dort war das Innere einer Fläche negativ. Welches Vorzeichen man wählt, ist für das Netz gleichgültig.

![Links Korsika und Sardinien auf einem groben Gitter. Die Zellen an Land sind beige bis hellbraun, die Zellen im Meer hellblau bis dunkelblau, je nach Abstand zur Küste. Vier Zellen sind schwarz umrandet. Von jeder führt eine orange Strecke zum nächsten Punkt der Küste, darüber steht der Abstand: +46 km in der Mitte Sardiniens, +12 km auf Korsika, −69 km westlich und −90 km südöstlich von Sardinien. Rechts die vier Schritte, mit denen der Wert einer Zelle entsteht, und eine Farbskala von −120 km (Meer) bis +120 km (Land).](../img/kueste-linie-zu-feld.svg)

Die App berechnet das Feld beim ersten Aufruf selbst. Sie legt dazu entlang der Küstenlinie Punkte im Abstand von höchstens 2 km, sucht für jede der 2 332 800 Zellen den nächsten davon und prüft mit den Landflächen, ob die Zelle an Land liegt. Das dauert einige Sekunden; danach liegt das Ergebnis im Zwischenspeicher.

![Weltkarte des Abstands zur Küste. Die Kontinente sind beige an der Küste und werden zum Inneren hin immer dunkler braun, am dunkelsten in Zentralasien, in der Sahara und in der Mitte Nordamerikas. Die Meere sind an der Küste fast weiß und werden nach außen immer dunkler blau. Um kleine Inseln mitten im Ozean liegen helle Kreise, die wie Blasen im dunklen Blau aussehen.](../img/kueste-abstand-welt.svg)

Die Werte reichen von **−2 830 km** im Südpazifik bis **+2 473 km** in Zentralasien. Der Punkt im Südpazifik ist im Maßstab 1:50 Millionen noch weiter vom Land entfernt als der bekannte Point Nemo, der 2 688 km von der nächsten Küste liegt [3]. Der Grund: Kleine Inseln wie die Pitcairninseln fehlen in diesem Maßstab.

## Was dieses Feld besonders macht

Ein Schnitt entlang 40° Nord zeigt, wie der Abstand verläuft:

![Profil des Abstands zur Küste entlang 40° Nord von 20° West bis 30° Ost. Über dem Atlantik sinkt die blaue Kurve bis auf −697 km, steigt dann gleichmäßig an, kreuzt an der Küste Portugals die 0-Linie und erreicht in Spanien 354 km. Danach fällt sie gerade bis zur Mittelmeerküste ab. Über dem Mittelmeer schwankt sie zwischen etwa −180 km und kleinen positiven Spitzen bei den Inseln. Gestrichelte orange Linien markieren ±200 km.](../img/kueste-abstand-profil.svg)

Drei Dinge fallen auf:

- **Die Kurve steigt und fällt überall ähnlich steil.** Geht man 1 km direkt von der Küste weg, wächst der Abstand um genau 1 km; steiler wird es nie. Das Feld hat deshalb keine feinen Einzelheiten wie ein Gebirge. Es besteht vor allem aus langen Wellen (Kapitel 5), und lange Wellen lernt ein neuronales Netz leicht.
- **An der Küste geht die Kurve gerade durch die 0.** Dort gibt es keinen Sprung und keinen Knick. Das Netz kann die Nullstelle deshalb genau setzen, ohne dort besondere Mühe zu haben.
- **Knicke liegen woanders:** genau in der Mitte zwischen zwei Küsten, zum Beispiel zwischen den Balearen und Sardinien, und auf dem höchsten Punkt im Landesinneren. Ein Knick ist ein Stück kurze Welle, und kurze Wellen kosten Gewichte. In der Weltkarte sieht man außerdem, dass schon eine kleine Insel mitten im Ozean einen Kreis von vielen hundert Kilometern prägt – überall dort ist sie die nächste Küste.

![Profil des Abstands zur Küste entlang 40,5° Nord von Spanien bis Italien, mit gleichem Maßstab auf beiden Achsen. Die blaue Kurve kreuzt an vier Küsten die 0-Linie, markiert mit weißen Kreisen; dort liegt sie eng an gestrichelten Linien mit Steigung 1. Drei rote Kreise markieren spitze Knicke: unten bei −171 km in der Mitte zwischen den Balearen und Sardinien, oben bei +54 km auf dem höchsten Punkt Sardiniens und unten bei −143 km in der Mitte zwischen Sardinien und Italien.](../img/kueste-knicke.svg)

## Höhennetz gegen Abstandsnetz

Der Vergleich verwendet zweimal dasselbe Netz mit demselben Training: das Standardbeispiel (*Fourier MLP (recommended)*, 82 561 Gewichte, Gitter 1080 × 2160, 4 000 Schritte, je ein Lauf). Das eine Netz lernt die Höhen aus ETOPO, das andere den Abstand zur Küste. Bei beiden wird die 0-Linie mit der Küstenlinie von Natural Earth verglichen.

Gemessen werden drei Dinge:

- **Richtige Seite:** Bei welchem Anteil der Erdoberfläche sagt das Netz Land, wo Land ist, und Meer, wo Meer ist?
- **In Küstenzellen:** derselbe Anteil, aber nur für die Zellen, durch die die wahre Küste verläuft – also die Zellen, die einen Nachbarn auf der anderen Seite der Küste haben.
- **Küstenversatz:** Wie weit liegen die Zellen der zurückgewonnenen Küste von der wahren Küstenlinie entfernt? Angegeben sind der Median und der Wert, unter dem 90 % der Küste liegen.

| | Höhenraster ETOPO selbst | Höhennetz | Abstandsnetz |
|---|---:|---:|---:|
| Richtige Seite | 99,3 % | 97,1 % | 98,8 % |
| In Küstenzellen | 84,8 % | 58,7 % | 64,6 % |
| Küstenversatz, Median | 7,1 km | 28,3 km | 9,9 km |
| Küstenversatz, 90 % | 17 km | 244 km | 25 km |

Die erste Spalte ist kein Netz, sondern das Höhenraster selbst. Schon seine 0-m-Linie weicht von Natural Earth ab: Die beiden Datensätze haben verschiedene Quellen, und eine Zelle von 19 km ist ein Mittelwert über Land und Meer. Selbst ein perfektes Netz könnte deshalb nicht genauer werden als das Raster, aus dem es lernt. Das Abstandsraster stimmt mit Natural Earth per Konstruktion überein; seine Küstenzellen liegen im Median 6,4 km, zu 90 % höchstens 14 km von der Linie entfernt – das ist die Grenze, die ein Gitter mit 19 km breiten Zellen setzt.

Die Zahlen hier sind strenger als in Kapitel 11. Dort zählten als Küste alle Zellen mit Höhen zwischen −200 m und +200 m, hier nur die Zellen direkt an der Küstenlinie.

**Das Abstandsnetz trifft die Küste deutlich besser.** Der Median des Versatzes sinkt von 28 auf 10 km. Noch größer ist der Unterschied bei den 90 %: von 244 km auf 25 km. Das sieht man auch auf der Karte:

![Zwei Karten von Mitteleuropa und dem Mittelmeer mit der wahren Küstenlinie in Schwarz und der Küste aus dem Netz in Orange. Links das Höhennetz: Die orange Linie umrandet große Phantomseen in den Niederlanden und in Norddeutschland, kleinere in der Po-Ebene und in Ungarn, Mallorca fehlt, Sizilien ist zu klein. Rechts das Abstandsnetz: Die orange Linie folgt der schwarzen Küste eng, Phantomseen gibt es nicht, Korsika und Sardinien sind erkennbar, Mallorca ist nur ein kleiner Fleck.](../img/kueste-hoehe-gegen-abstand.svg)

Das Höhennetz erfindet im Landesinneren **Phantomseen**: Wo das Land flach und niedrig ist, wie in den Niederlanden, reicht sein Fehler, um die Höhe unter 0 m zu drücken. In einem Ausschnitt von 4° bis 8° Ost und 51° bis 53,5° Nord hält das Höhennetz 151 von 311 Landzellen für Meer, das Abstandsnetz nur 5. Solche Fehler liegen weit weg von jeder Küste – daher die 244 km.

Kleine Inseln haben es in beiden Netzen schwer. Mallorca belegt im Gitter 15 Zellen. Das Höhennetz findet davon keine, das Abstandsnetz 6. Eine kleine Insel ist im Abstandsfeld nur ein flacher Hügel von wenigen Kilometern, umgeben von tiefen Meereswerten, und wird beim Lernen leicht geglättet.

## Den Abstand begrenzen

Für die Küstenlinie ist es gleichgültig, ob ein Punkt im Pazifik 2 000 oder 2 800 km vom Land entfernt ist. Trotzdem verwendet das Netz Gewichte dafür, diese Werte zu lernen.

Die Abhilfe ist einfach: Man **begrenzt** den Abstand, etwa auf ±200 km. Alles, was weiter von der Küste entfernt ist, bekommt einfach den Wert +200 km oder −200 km. Im Profil oben sind das die beiden gestrichelten Linien: Die Kurve wird dort abgeschnitten. Diese Idee heißt *truncated signed distance function* und ist weit verbreitet, etwa bei Curless und Levoy, die 1996 auf diese Weise 3D-Scans zusammengesetzt haben [4].

Die App bietet dafür den Regler **Limit the distance to**. Gemessen mit dem Standardbeispiel wie oben:

| Begrenzung | Richtige Seite | In Küstenzellen | Küstenversatz, Median | Küstenversatz, 90 % |
|---|---:|---:|---:|---:|
| keine | 98,8 % | 64,6 % | 9,9 km | 25 km |
| ±500 km | 99,0 % | 68,7 % | 8,3 km | 20 km |
| ±200 km | 99,2 % | 71,5 % | 7,6 km | 18 km |
| ±100 km | 99,3 % | 74,0 % | 7,3 km | 17 km |
| ±50 km | 99,3 % | 76,0 % | 7,0 km | 16 km |
| ±25 km | 99,4 % | 78,9 % | 6,7 km | 15 km |
| ±10 km | 99,4 % | 79,0 % | 6,8 km | 15 km |
| ±1 km | 99,5 % | 82,6 % | 6,8 km | 15 km |

**Je stärker die Begrenzung, desto genauer die Küste.** Ab etwa ±25 km ist der Versatz fast so klein wie beim Abstandsraster selbst (6,4 km und 14 km). Mehr kann ein Gitter mit 19 km breiten Zellen nicht hergeben. Nur der Anteil in den Küstenzellen steigt noch weiter.

![Zwei Karten desselben Ausschnitts, beide aus dem Abstandsnetz. Links ohne Begrenzung, rechts mit Begrenzung auf ±25 km. Rechts folgt die orange Linie der schwarzen Küste noch enger, auch an der Adria und um Sizilien, und Mallorca ist als kleine Insel erkennbar.](../img/kueste-begrenzung.svg)

Bei ±1 km ist vom Abstand fast nichts mehr übrig: Jede Zelle ist entweder +1 oder −1, außer in einem schmalen Streifen an der Küste. Das Netz lernt dann eine reine **Land-Meer-Maske**. Für die Küstenlinie ist das sogar am besten.

Was man dabei verliert, ist der Abstand selbst. Ein neuronales Netz ohne Begrenzung beantwortet für jeden Ort der Erde die Frage „Wie weit ist es von hier bis zur Küste?“ – im Mittel auf rund 10 km genau (gewichteter RMSE 10,3 km). Ein Netz mit Begrenzung auf ±25 km weiß nur noch, ob ein Ort weiter als 25 km von der Küste entfernt ist. Welche Variante richtig ist, hängt also von der Frage ab, mit welcher Genauigkeit das neuronale Netz beantworten soll.

## Verschiedene neuronale Netze

Wie in den früheren Kapiteln lassen sich auch die verschiedenen Architekturen ausprobieren:

| Netz | Begrenzung | Datei | Richtige Seite | In Küstenzellen | Küstenversatz, Median | Küstenversatz, 90 % |
|---|---|---:|---:|---:|---:|---:|
| *Fourier MLP (recommended)* | keine | 323 KB | 98,8 % | 64,6 % | 9,9 km | 25 km |
| *SIREN (sine activation)* | keine | 305 KB | 98,8 % | 64,1 % | 10,5 km | 27 km |
| *Fourier MLP + context field* | keine | 355 KB | 99,3 % | 73,5 % | 7,3 km | 17 km |
| *Fourier MLP (recommended)* | ±100 km | 323 KB | 99,3 % | 74,0 % | 7,3 km | 17 km |
| *SIREN (sine activation)* | ±100 km | 305 KB | 98,7 % | 64,6 % | 10,5 km | 27 km |
| *Fourier MLP + context field* | ±100 km | 355 KB | 99,5 % | 79,5 % | 6,7 km | 15 km |

**Das Kontextfeld hilft wie die Begrenzung.** Es liefert dem Netz die großen Abstände über den Ozeanen fast umsonst (Kapitel 8). Das Netz kann seine Gewichte deshalb für die Küste ausgeben. Mit Begrenzung zusammen ist es die beste Variante der Tabelle.

**Das SIREN-Netz profitiert nicht.** Es liegt ohne Begrenzung etwa beim Fourier-MLP und wird durch die Begrenzung nicht besser. Woran das liegt, wurde hier nicht weiter untersucht.

## Was die Küste als neuronales Netz kostet

Zum Schluss der ehrliche Vergleich der Dateigrößen:

| | Größe |
|---|---:|
| Küstenlinie als Vektordaten (60 416 Punkte × 2 Zahlen × 4 Byte) | 472 KB |
| Abstandsraster 1080 × 2160 (2 Byte pro Zelle) | 4,4 MB |
| Netz, float32 | 323 KB |
| Netz, float16 | 161 KB |

Das Netz ist kleiner als die Vektordaten. Trotzdem ist es keine gute Kompression der Küstenlinie:

- **Es ist ungenauer.** Die Vektordaten kennen jeden Punkt der Küste genau. Das Netz setzt die Küste im besten Fall auf etwa 7 km genau, kleine Inseln fehlen.
- **Die Vektordaten ließen sich ebenfalls verkleinern.** Speichert man statt jedes Punktes nur den kleinen Schritt zum vorherigen Punkt, braucht jede Zahl weniger Byte.
- **Es kennt keine Ordnung.** Die Vektordaten sagen, welche Punkte in welcher Reihenfolge eine Insel bilden. Im Feld gibt es nur eine Nulllinie; wer wieder Linienzüge braucht, muss sie erst aus dem Gitter herauslesen.

Nützlich ist das Abstandsnetz dort, wo man das Feld ohnehin braucht: Wie weit ist dieser Ort von der Küste entfernt? Liegt er an Land oder im Meer? Beides beantwortet das Netz für jede beliebige Koordinate mit einer einzigen Abfrage, ohne Vektordaten und ohne Raster.

## Ausprobieren

Wählen Sie in der Seitenleiste unter **1 · Data** bei **Dataset** die Einstellung **Distance to the coast (Natural Earth)**. Beim ersten Mal fragt die App, ob sie die Küstendaten von Natural Earth herunterladen darf (etwa 3 MB). Klicken Sie auf **Download Natural Earth (3 MB)**. Danach berechnet die App das Abstandsfeld; das dauert einige Sekunden.

Unter **Grid resolution** steht jetzt zusätzlich der Regler **Limit the distance to**, voreingestellt auf *no limit*.

![Die Seitenleiste mit dem Abschnitt 1 · Data: Dataset mit den Einstellungen Elevation (ETOPO 2022) und Distance to the coast (Natural Earth), darunter die Quelle, Grid resolution 1080 × 2160 und Limit the distance to mit no limit.](../img/app-kueste-seitenleiste.png)

Trainieren Sie das Standardbeispiel: Vorlage **Fourier MLP (recommended)**, **Steps** 4 000, dann **Train**. Die Reiter *Training*, *Maps & Error* und *Super-Resolution* zeigen jetzt Kilometer statt Meter.

Öffnen Sie dann den Reiter **Coastline**. Oben stehen die drei Maße aus diesem Kapitel, darunter die drei Dateigrößen:

![Der Reiter Coastline: Correct side, … in coastal cells, Coast offset (median) und Coast offset (90 %), darunter Coastline as vector data 472.0 KB, Network 322.5 KB und Grid of the field (int16) 4.4 MB.](../img/app-kueste-masse.png)

Darunter zeigt eine Karte den gewählten Ausschnitt: im Hintergrund das Trainingsgitter, in Schwarz die Küstenlinie von Natural Earth, in Orange die 0-Linie des Netzes. Mit **Longitude (centre)** und **Latitude (centre)** wählen Sie die Mitte des Ausschnitts, mit **Window [°]** seine Höhe in Grad; breit ist er doppelt so viel.

![Die Karte im Reiter Coastline für den voreingestellten Ausschnitt von 10° West bis 30° Ost und 36° bis 56° Nord: das Abstandsfeld in Blau und Braun, am dunkelsten in Osteuropa, darüber die schwarze Küstenlinie und die orange Linie des Netzes, die meist eng an der schwarzen liegt. Kleine Inseln wie die Balearen und die Inseln der Ägäis haben keine orange Linie.](../img/app-kueste-karte.png)

Stellen Sie dann **Limit the distance to** auf ±25 km und trainieren Sie erneut. Vergleichen Sie die Zahlen im Reiter *Coastline*.

Zum Vergleich mit dem Höhennetz wählen Sie bei **Dataset** wieder **Elevation (ETOPO 2022)** und trainieren erneut. Der Reiter *Coastline* vergleicht dann die 0-m-Linie der Höhe mit der Küstenlinie von Natural Earth.

## Ideen zum Weiterprobieren

- Schieben Sie den Ausschnitt im Reiter *Coastline* nach Norwegen (etwa 10° Ost, 64° Nord) oder Indonesien (etwa 115° Ost, 0°). Wo verliert das Netz die Küste zuerst?
- Vergleichen Sie *Fourier MLP, very small* und *Fourier MLP, large* beim Küstenversatz. Lohnt sich das größere Netz für die Küste mehr als für die Höhen?
- Trainieren Sie auf dem Gitter 2160 × 4320. Sinkt der Küstenversatz um die Hälfte, wie die halbe Zellbreite vermuten lässt?
- Suchen Sie im Reiter *Maps & Error* die Knicke in der Mitte zwischen zwei Küsten. Liegt dort auch der größte Fehler?
- Überlegen Sie, wie man Flüsse als Feld darstellen könnte. Ein Fluss hat kein Innen und kein Außen – welches Vorzeichen bekäme der Abstand dann?

## Weiterführende Links

- [1] Natural Earth: [Coastline 1:50m](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-coastline/) – Küstenlinie und Landflächen, gemeinfrei ([Nutzungsbedingungen](https://www.naturalearthdata.com/about/terms-of-use/)) (Englisch)
- [2] Park et al.: [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103), CVPR 2019 – vorzeichenbehaftete Abstandsfunktionen als neuronales Feld (Englisch)
- [3] Wikipedia: [Pol der Unzugänglichkeit](https://de.wikipedia.org/wiki/Pol_der_Unzug%C3%A4nglichkeit) – Point Nemo und die anderen Orte, die am weitesten von der Küste entfernt sind
- [4] Curless und Levoy: [A Volumetric Method for Building Complex Models from Range Images](https://graphics.stanford.edu/papers/volrange/), SIGGRAPH 1996 – abgeschnittene vorzeichenbehaftete Abstände für 3D-Scans (Englisch)
- Chu und Shahabi: [Geo2Vec: Shape- and Distance-Aware Neural Representation of Geospatial Entities](https://arxiv.org/abs/2508.19305), 2025 – Abstandsfelder für Punkte, Linien und Flächen in Geodaten (Englisch)
- Wikipedia: [Signed distance function](https://en.wikipedia.org/wiki/Signed_distance_function) – Definition und Eigenschaften (Englisch)

---

[Übersicht](README.md) · Zurück: [Kapitel 11 · Einordnung und Ausblick](11-einordnung-und-ausblick.md) · Weiter: [Anhang · Messungen](anhang-messungen.md)

*Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. Es soll helfen, mehr über das Thema „Geodaten in neuronalen Netzen speichern“ zu lernen. Verbesserungsvorschläge gerne an gkvoelkl@nelson-games.de.*
