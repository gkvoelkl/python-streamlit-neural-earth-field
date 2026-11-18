# Geodaten in neuronalen Netzen speichern
# Kapitel 13 · Location Encoder als Embedding

*[English version](../en/13-location-encoder.md)*

![In einem alten Arbeitszimmer mit Bücherregalen hockt Chester mit blauer Kappe auf einem großen Holzglobus und hält eine Kette aus kleinen, bunt leuchtenden Perlen. Die Kette spannt sich in einem Bogen hinunter zu Aiko, einem leuchtenden neuronalen Netz, das ihr Ende an einem kleinen Thermometer in der Hand hält.](../img/kapitel-13-chester.jpg)

**Inhalt:**

* Was ein *Location Encoder* ist: ein Netz, das einen Ort in eine Liste von Zahlen übersetzt
* Wie man prüft, ob diese Zahlen etwas taugen: mit einer neuen Aufgabe und wenigen Trainingsorten


## Ausgangspunkt

In den Kapiteln 1 bis 12 hat ein Netz immer genau **ein** Feld gespeichert: die Höhen der Erde oder den Abstand zur Küste. Im Training des neuronalen Netzes waren alle Zellen des Gitters mit dabei. Das fertige Netz kann genau diese eine trainierte Größe wiedergeben. Für eine andere Größe, etwa die Temperatur, müsste man ein neues Netz trainieren – wieder mit allen Zellen.

Kapitel 11 hat einen anderen Weg beschrieben (Weg 2): Ein neuronales Netz übersetzt einen Ort in eine feste Zahl von Zahlen, etwa 256.

Ähnliche Orte bekommen dabei ähnliche Zahlen. Diese Zahlen heißen *Embedding*, das neuronale Netz, das sie berechnet, *Location Encoder*.

Ein Location Encoder lernt keine bestimmte Größe. Er wird einmal trainiert, und danach verwendet man seine Zahlen für viele verschiedene Aufgaben. Für jede neue Aufgabe kommt ein zweites, kleines Netz dazu. Es bekommt nicht Länge und Breite als Eingabe, sondern die 256 Zahlen des Encoders, und lernt daraus zum Beispiel die Temperatur. Der Encoder selbst bleibt dabei unverändert.

Weil seine Zahlen schon viel über einen Ort verraten – liegt er in der Wüste, im Wald, im Gebirge? –, braucht das kleine Netz nur wenige Beispiele: etwa 1 000 Orte statt aller Zellen. Was der Encoder einmal über die Erde gelernt hat, wird so auf eine neue Aufgabe **übertragen**. Im Englischen heißt das *transfer learning*, die neue Aufgabe *downstream task*.

Der Unterschied zum bisherigen Vorgehen: 

**Speichern** heißt, ein Netz lernt ein einziges Feld so genau wie möglich. 

**Übertragen** (*transfer learning*) heißt, ein Netz liefert eine allgemeine Beschreibung der Orte, mit der sich neue Aufgaben aus wenigen Beispielen lernen lassen.

![Zwei Reihen von Kästen mit Pfeilen. Oben „Speichern“: lon, lat führt in das Höhennetz mit 82 561 Gewichten, trainiert auf allen 2,3 Millionen Zellen, und von dort zur Höhe. Unten „Übertragen“: lon, lat führt in einen festen, vorher trainierten Location Encoder wie SatCLIP, von dort zu 256 Zahlen, dann in ein kleines Netz, das die neue Aufgabe aus 1 000 Orten lernt, und schließlich zur Temperatur oder zu Niederschlag oder Bevölkerung.](../img/encoder-speichern-uebertragen.svg)

Dieses Kapitel probiert das aus. Die neue Aufgabe ist die **Jahresmitteltemperatur** an Land.

## Was ein Location Encoder liefert

Die Fourier-Features aus Kapitel 5 machen aus einem Ort 256 Zahlen, die Kugelflächenfunktionen aus Kapitel 6 je nach Grad 100 oder mehr. Diese Zahlen unterteilen die Erde, haben aber keine zusätzliche Bedeutung. Trotzdem sind dies einfache Location Encoder.

Ein vortrainierter Encoder dagegen hat bereits etwas über die Erde gelernt. 

Das bekannteste Beispiel ist **SatCLIP** von Klemmer und Kollegen [1]. Es wurde mit 100 000 Satellitenbildern von Sentinel-2 trainiert, die etwa gleichmäßig über alle Landflächen verteilt sind. 

Beim Training lernen zwei Netze gleichzeitig: Das eine macht aus einem Satellitenbild 256 Zahlen, das andere aus dem Ort, an dem das Bild aufgenommen wurde. Ziel ist, dass zu einem Bild und seinem Ort möglichst ähnliche Zahlen herauskommen und zu verschiedenen Orten möglichst verschiedene. Nach dem Training braucht man das Bild nicht mehr: Der Ort allein liefert Zahlen, in denen steckt, wie die Landschaft dort aussieht.

Wie sich das auswirkt, zeigt ein Trick: Von den vielen Zahlen jedes Ortes nimmt man die drei Richtungen, in denen sie sich am stärksten unterscheiden (die *Hauptkomponenten*), und malt sie als Rot, Grün und Blau.

![Zwei Weltkarten in bunten Farben, die Meere blass. Links die Kugelflächenfunktionen: Die Farben ändern sich nur langsam, fast alle Kontinente sind gleich olivbraun, nur Kanada ist rosa und Sibirien hellblau bis schwarz. Rechts SatCLIP: Die Regenwälder am Amazonas und am Kongo sind fast schwarz, die Nadelwälder Kanadas und Sibiriens rosa, die Sahara und Arabien hellgrün, Australien hellblau, die Antarktis grün.](../img/encoder-embedding-farben.svg)

Bei SatCLIP bilden die Farben Gebiete, die zu Klima und Landschaft passen. Die Regenwälder in Südamerika und Afrika haben ähnliche Farben, obwohl ein Ozean dazwischenliegt; ebenso die Nadelwälder in Kanada und Sibirien.

### SatCLIP von innen

Der Location Encoder von SatCLIP hat bereits bekannte Komponenten. Er besteht aus zwei Teilen [2]:

1. **Kugelflächenfunktionen** bis zum Grad 9 – wie in Kapitel 6. Das sind 100 Zahlen pro Ort.
2. Ein kleines **SIREN** – wie in Kapitel 7 – mit zwei Schichten zu je 512 Neuronen und einer Ausgabeschicht mit 256 Neuronen.

Zusammen sind das 445 696 Gewichte. Rußwurm und Kollegen haben gezeigt, dass genau diese Kombination aus Kugelflächenfunktionen und SIREN ein besonders guter Location Encoder ist [2]. Die beiden Vertiefungskapitel dieses Tutorials ergeben zusammen also ein aktuelles Verfahren aus der Forschung.

Microsoft stellt die trainierten Gewichte unter der MIT-Lizenz bereit [3]. Die App verwendet die kleinste Fassung (*ResNet18, L = 10*) und davon nur den Location Encoder; das Netz für die Bilder wird nicht gebraucht.

## Die Aufgabe: Temperatur aus wenigen Orten

Die Temperaturen stammen aus dem Datensatz von Willmott und Matsuura von der University of Delaware (Version 5.01) [4, 5]. Er beruht auf Messungen an Wetterstationen. Ihre Werte wurden auf ein Gitter mit Punkten im Abstand von 0,5° (etwa 56 km) umgerechnet. Zwischen den Stationen half dabei ein Höhenmodell: Angenommen wurde, dass es pro Kilometer Höhe 6 °C kälter wird [5].

![Schnitt durch ein Gebirge mit zwei Wetterstationen und einem Gitterpunkt auf dem Gipfel. Station A liegt im Tal auf 400 m und misst 11,0 °C, Station B auf 1 600 m misst 3,4 °C. Pfeile führen von beiden Stationen hinunter auf Meereshöhe: Mit 6 °C pro Kilometer werden daraus 13,4 und 13,0 °C. Zwei Pfeile entlang der Meereshöhe treffen sich unter dem Gitterpunkt bei 13,2 °C. Ein oranger Pfeil führt von dort hinauf zum Gitterpunkt auf 2 200 m laut Höhenmodell: 13,2 minus 2,2 mal 6 ergibt 0,0 °C. Rechts die vier Schritte Messen, Auf Meereshöhe umrechnen, Zwischen den Stationen verteilen und Auf die Höhe des Gitterpunkts.](../img/encoder-stationen-gitter.svg)

Verwendet werden die langjährigen Mittelwerte von 1981 bis 2010, gemittelt über die zwölf Monate. Der Datensatz deckt nur Land ab, einschließlich der Antarktis: **85 794 Landzellen** mit Werten von −56,1 °C in der Antarktis bis +35,0 °C.

Aus diesen Zellen werden zufällig **Trainingsorte** gezogen – 100, 1 000 oder 10 000. Wie beim Training des Höhennetzes zählt dabei die Fläche: Eine Zelle in Sibirien ist kleiner als eine am Äquator und wird entsprechend seltener gezogen (*Area-weighted sampling*, Kapitel 2). Geprüft wird an **allen übrigen** Zellen. Das Maß ist der flächengewichtete RMSE in °C (Kapitel 9).

Zum Vergleich: Wer immer nur den Mittelwert der Trainingsorte vorhersagt, liegt im Mittel **18,6 °C** daneben. Jede Eingabe muss also deutlich darunter bleiben.

### Sechs Eingaben, ein kleines Netz

Jede Eingabe füttert dasselbe kleine Netz: zwei verborgene Schichten mit je 64 Neuronen, 2 000 Trainingsschritte mit allen Trainingsorten auf einmal. Die Zahlen jeder Eingabe werden vorher auf Mittelwert 0 und Streuung 1 gebracht, damit keine Eingabe wegen ihres Maßstabs im Vorteil ist.

| Eingabe | Zahlen pro Ort | Woher |
|---|---:|---|
| lon/lat | 2 | Länge und Breite, auf −1 bis 1 gebracht |
| Fourier-Features | 256 | 128 zufällige Streifen auf der Kugel (Kapitel 5), σ = 2 |
| Kugelflächenfunktionen | 100 | bis Grad 9 (Kapitel 6), wie in SatCLIP |
| SatCLIP | 256 | vortrainierter Location Encoder [1] |
| Höhennetz (verborgene Schicht) | 128 | die letzte verborgene Schicht des Standardbeispiels |
| lon/lat + Höhe | 3 | Länge, Breite und die Höhe, die das Standardbeispiel an diesem Ort ausgibt |

Die letzten beiden Eingaben verwenden das Höhennetz aus Kapitel 1 bis 10 (Standardbeispiel). Die verborgene Schicht ist der Versuch, das Höhennetz selbst als Location Encoder zu nutzen: Die 128 Zahlen vor seiner Ausgabeschicht beschreiben ja auch jeden Ort. Die Eingabe „lon/lat + Höhe“ nutzt dagegen nur, was das Höhennetz eigentlich gespeichert hat – die Höhe.

## Ergebnis

Gemessen mit drei Läufen je Zeile: Jeder Lauf zieht andere Trainingsorte und startet das kleine Netz mit anderen Gewichten. Angegeben ist der Mittelwert.

| Eingabe | 100 Orte | 1 000 Orte | 10 000 Orte |
|---|---:|---:|---:|
| lon/lat | 5,1 °C | 3,5 °C | 3,3 °C |
| Fourier-Features | 12,4 °C | 4,3 °C | 2,1 °C |
| Kugelflächenfunktionen | 8,1 °C | 3,6 °C | 2,0 °C |
| SatCLIP | 5,2 °C | 3,6 °C | 2,0 °C |
| Höhennetz (verborgene Schicht) | 20,0 °C | 17,5 °C | 10,6 °C |
| lon/lat + Höhe | **4,4 °C** | **2,2 °C** | **2,0 °C** |

![Liniendiagramm: Fehler in °C über der Zahl der Trainingsorte 100, 1 000 und 10 000, logarithmisch. Fünf Linien fallen von links nach rechts. Am tiefsten liegt lon/lat + Höhe mit 4,4, 2,2 und 2,0 °C. lon/lat und SatCLIP liegen bei 100 und 1 000 Orten fast aufeinander, bei 10 000 Orten bleibt lon/lat mit 3,3 °C zurück. Fourier-Features beginnen bei 12,4 °C und holen bis 10 000 Orte auf. Die rote Linie des Höhennetzes liegt bei 100 Orten über der gestrichelten Linie für den Mittelwert bei 18,6 °C.](../img/encoder-orte-fehler.svg)

Vier Dinge fallen auf:

- **Mit wenigen Orten sind zwei Zahlen schwer zu schlagen.** lon/lat und SatCLIP liegen gleichauf: 5,1 und 5,2 °C bei 100 Orten, 3,5 und 3,6 °C bei 1 000. Die Temperatur hängt vor allem von der Breite ab, und die steckt in lon/lat schon vollständig drin. Das Vortraining hilft trotzdem: Gegenüber den Kugelflächenfunktionen, mit denen SatCLIP selbst beginnt, sinkt der Fehler bei 100 Orten von 8,1 auf 5,2 °C. Die Fourier-Features liegen mit 100 Orten am weitesten daneben: In ihren 256 Zahlen findet das kleine Netz Muster, die es gar nicht gibt.
- **Mit vielen Orten zahlen sich viele Zahlen aus.** Bei 10 000 Orten liegen Fourier-Features, Kugelflächenfunktionen und SatCLIP bei 2,0 bis 2,1 °C, lon/lat bleibt bei 3,3 °C. Aus zwei Zahlen kann das kleine Netz feine Muster nur schwer formen – dasselbe Problem wie beim einfachen Netz in Kapitel 2.
- **Die verborgene Schicht des Höhennetzes taugt nicht als Embedding.** Mit 100 Orten ist sie sogar schlechter, als einfach den Mittelwert vorherzusagen. Die 128 Zahlen sind genau darauf abgestimmt, dass die Ausgabeschicht aus ihnen die Höhe berechnet – mit all ihren feinen Einzelheiten. Wie warm es an einem Ort ist, lässt sich aus ihnen kaum herauslesen. **Speichern ist nicht übertragen:** Ein Netz, das ein Feld möglichst genau speichert, liefert dabei keine allgemein brauchbare Beschreibung der Orte.
- **Die Höhe selbst hilft sehr.** Länge, Breite und Höhe – drei Zahlen – schlagen mit 100 und 1 000 Orten jede andere Eingabe. Das ist Fachwissen: In der Luft wird es nach oben hin im Mittel etwa 6,5 °C pro Kilometer kälter [6]. Das kleine Netz muss diesen Zusammenhang nur noch finden, nicht aus dem Ort erraten. Ganz fair ist der Vergleich nicht, denn auch die Temperaturdaten selbst wurden mit einem Höhenmodell erzeugt (siehe oben). Nebenbei zeigt sich ein Nutzen des Höhennetzes: nicht als Embedding, sondern als kompakter Lieferant der Höhe für jede Koordinate.

Wo die Fehler liegen, zeigen die Karten für 1 000 Trainingsorte:

![Drei Weltkarten des Fehlers in Rot und Blau, jeweils mit der Antarktis. Oben lon/lat, Fehler 3,5 °C: große rote und blaue Flächen, etwa in Grönland und in breiten Bändern über der Antarktis, dazu starke Fehler in Anden, Rocky Mountains und Himalaya. In der Mitte SatCLIP, Fehler 3,7 °C: die Fehler sind anders verteilt, aber nicht kleiner; sie sitzen in den Gebirgen und an den Rändern Grönlands und der Antarktis. Unten lon/lat + Höhe, Fehler 2,3 °C: die meisten Flächen sind hell, nur in Tibet, in Grönland und in einem Fleck der Antarktis bleiben deutliche Fehler.](../img/encoder-fehlerkarten.svg)

Bei lon/lat und bei SatCLIP sitzen die größten Fehler in den Gebirgen. SatCLIP kennt aus den Satellitenbildern zwar die Landschaft, aber die Höhe nur ungefähr. Das liegt auch an der Auflösung: Die feinsten Kugelflächenfunktionen bis Grad 9 haben Wellen von etwa 4 000 km Länge. Das SIREN dahinter formt daraus zwar feinere Muster, doch für ein schmales Gebirge wie die Anden reicht das nicht. Die Entwickler weisen selbst darauf hin, dass SatCLIP für Fragen auf der ganzen Erde gedacht ist, nicht für kleine Gebiete [3]. Für feinere Aufgaben gibt es eine Fassung mit Kugelflächenfunktionen bis Grad 39 (*L = 40*).

Für die Temperatur bringt SatCLIP also wenig, weil sie vor allem von Breite und Höhe abhängt. Ein vortrainierter Encoder hilft dort, wo die Aufgabe mit dem zusammenhängt, was er gelernt hat – bei SatCLIP mit dem, wie die Landschaft auf Satellitenbildern aussieht. Klemmer und Kollegen prüfen SatCLIP an neun Aufgaben, darunter auch Bevölkerungsdichte und das Vorkommen von Tierarten [1].

## Was das kostet

| | Gewichte |
|---|---:|
| SatCLIP, Location Encoder | 445 696 |
| kleines Netz auf SatCLIP (256 Zahlen) | 20 673 |
| kleines Netz auf lon/lat + Höhe (3 Zahlen) | 4 481 |
| Höhennetz (Standardbeispiel) | 82 561 |

Der Encoder ist groß, wird aber nur einmal trainiert und dann für viele Aufgaben verwendet. Für jede neue Aufgabe kommt nur das kleine Netz dazu. Die Rechnung ist also eine andere als in den Kapiteln 1 bis 12: Dort zählte, wie klein die Datei für ein einziges Feld ist. Hier zählt, wie wenige **Beispiele** eine neue Aufgabe braucht. Messdaten von Wetterstationen sind teuer; wenn 1 000 Orte reichen statt 10 000, ist viel gewonnen.

## Ausprobieren

Der Reiter **Location Encoder** braucht zwei Downloads und läuft deshalb nur in der lokalen Fassung der App, nicht online.

### Höhennetz trainieren

Die beiden Eingaben aus dem Höhennetz gibt es erst, wenn eines trainiert ist. Trainieren Sie das Standardbeispiel: **Dataset** *Elevation (ETOPO 2022)*, **Grid resolution** 1080 × 2160, Vorlage **Fourier MLP (recommended)**, **Steps** 4 000, dann **Train**. Ohne Höhennetz vergleicht der Reiter die übrigen vier Eingaben.

### Daten laden und vergleichen

Öffnen Sie den Reiter **Location Encoder**. Beim ersten Mal fragt die App, ob sie die Temperaturen der University of Delaware (etwa 19 MB) und den Location Encoder von SatCLIP (etwa 57 MB) herunterladen darf. Klicken Sie auf **Download (76 MB)**.

| Einstellung | Wert | Bedeutung |
|---|---|---|
| *Training places* | 1000 | Zahl der Orte, aus denen das kleine Netz lernt |
| *Inputs* | alle | die Eingaben, die verglichen werden |
| *Seed* | 0 | Startwert für die Auswahl der Orte und die Gewichte des kleinen Netzes |

Klicken Sie auf **Compare**. Nach einigen Sekunden zeigt ein Balkendiagramm den Fehler jeder Eingabe; die gestrichelte Linie ist der Fehler, wenn man immer den Mittelwert vorhersagt.

![Der Reiter Location Encoder: oben die Einstellungen Training places 1000, Inputs mit lon/lat, Fourier features, Spherical harmonics, SatCLIP und Elevation network (hidden layer), Seed 0 und der Knopf Compare. Darunter waagrechte Balken: lon/lat + elevation 2,26 °C, lon/lat 3,52 °C, SatCLIP 3,65 °C, Spherical harmonics 3,69 °C, Fourier features 4,53 °C und Elevation network (hidden layer) 17,59 °C, knapp unter der gestrichelten Linie always the mean bei 18,7 °C.](../img/app-encoder-vergleich.png)

Die Zahlen weichen etwas von der Tabelle oben ab: Dort ist der Mittelwert aus drei Läufen angegeben, hier sehen Sie einen einzigen.

### Karten ansehen

Unter **Map for** wählen Sie eine Eingabe. Links stehen die gemessenen Temperaturen mit den Trainingsorten als schwarze Punkte, rechts die Vorhersage. Darunter zeigt eine dritte Karte den Fehler.

![Zwei Weltkarten nebeneinander. Links die gemessene Jahresmitteltemperatur, von Dunkelblau in der Antarktis und in Grönland bis Dunkelrot in der Sahara, mit vielen schwarzen Punkten für die 1 000 Trainingsorte. Rechts die Vorhersage aus SatCLIP, die der linken Karte im Großen sehr ähnlich sieht.](../img/app-encoder-karten.png)

## Ideen zum Weiterprobieren

- Stellen Sie *Training places* auf 100 und wechseln Sie mehrmals den *Seed*. Welche Eingabe schwankt am stärksten?
- Vergleichen Sie die Fehlerkarten von SatCLIP und *lon/lat + elevation* in Ostafrika und im Himalaya. Wo hilft die Höhe, wo nicht?
- Trainieren Sie statt des Standardbeispiels *Fourier MLP, very small*. Wird die Eingabe *lon/lat + elevation* schlechter, wenn das Höhennetz die Höhe ungenauer kennt?
- Trainieren Sie das Höhennetz nur 500 Schritte. Taugt seine verborgene Schicht dann besser als Embedding als nach 4 000 Schritten?
- Überlegen Sie, welche andere Aufgabe von SatCLIP mehr profitieren könnte als die Temperatur – etwa eine, bei der die Landschaft wichtiger ist als Breite und Höhe.

## Weiterführende Links

- [1] Klemmer et al.: [SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery](https://arxiv.org/abs/2311.17179), AAAI 2025 – Location Encoder, vortrainiert mit Sentinel-2-Bildern (Englisch)
- [2] Rußwurm et al.: [Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks](https://arxiv.org/abs/2310.06743), ICLR 2024 – Kugelflächenfunktionen und SIREN als Location Encoder (Englisch)
- [3] Microsoft: [SatCLIP auf GitHub](https://github.com/microsoft/satclip) und [Gewichte auf Hugging Face](https://huggingface.co/microsoft/SatCLIP-ResNet18-L10) – Programmcode und trainierte Gewichte, MIT-Lizenz (Englisch)
- [4] NOAA PSL: [University of Delaware Air Temperature & Precipitation](https://psl.noaa.gov/data/gridded/data.UDel_AirT_Precip.html) – Monatsmittel der Lufttemperatur an Land, 0,5°, Version 5.01 (Englisch)
- [5] NCAR Climate Data Guide: [Global (land) precipitation and temperature: Willmott & Matsuura, University of Delaware](https://climatedataguide.ucar.edu/climate-data/global-land-precipitation-and-temperature-willmott-matsuura-university-delaware) – wie der Datensatz entstanden ist, ohne Nutzungsbeschränkungen (Englisch)
- [6] Wikipedia: [Temperaturgradient](https://de.wikipedia.org/wiki/Temperaturgradient) – wie die Temperatur mit der Höhe abnimmt
- Mai et al.: [A Review of Location Encoding for GeoAI: Methods and Applications](https://arxiv.org/abs/2111.04006), IJGIS 2022 – Überblick über Location Encoder (Englisch)
- Rußwurm: [locationencoder auf GitHub](https://github.com/marccoru/locationencoder) – Programmcode zu [2] mit weiteren Encodern zum Vergleich (Englisch)

---

[Übersicht](README.md) · Zurück: [Kapitel 12 · Die Küste als Abstandsfeld](12-kueste-als-abstandsfeld.md) · Weiter: [Kapitel 14 · Orthofotos in UTM](14-orthofotos-in-utm.md)

*Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. Es soll helfen, mehr über das Thema „Geodaten in neuronalen Netzen speichern“ zu lernen. Verbesserungsvorschläge gerne an gkvoelkl@nelson-games.de.*
