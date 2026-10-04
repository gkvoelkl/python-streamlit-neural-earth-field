# Kapitel 1 · Geodaten als Funktion statt als Raster

![Chester, das Eichhörnchen mit der blauen Kappe, hält Aiko in den Pfoten, ein kleines leuchtendes neuronales Netz. Aus einer Erdkugel, die aus kleinen Pixel-Quadraten besteht, fließt eine Spirale in Aiko hinein.](img/kapitel-01-chester.jpg)

![Das Netz ersetzt das Raster und ist 14-mal kleiner: Höhenraster mit 2 332 800 Werten wird trainiert zu einem neuronalen Feld mit 82 561 Gewichten, das an jeder Koordinate abgefragt werden kann. Gespeichert werden 323 KB statt 4,4 MB.](img/geodaten-zu-netz.svg)

**Inhalt:** Ein Höhenraster als Funktion `h(lon, lat)` begreifen, seine Rohgröße ausrechnen und die Kompressionsrate eines Netzes ehrlich bestimmen.

## Die Idee

Ein digitales Höhenmodell ist eine Tabelle: Für jede Zelle eines regelmäßigen Gitters aus Längen- und Breitengraden steht dort eine Höhe in Metern.

![Ein Höhenraster ist eine Tabelle: Links ein Ausschnitt mit Alpen und Ligurischem Meer als Zahlen, sechs Zeilen für die Breitengrade 48,5° bis 43,5° Nord und acht Spalten für die Längengrade 5,5° bis 12,5° Ost. Rechts dieselben Werte als Farben, von Blau für das Meer über Grün bis Braun für die Alpen.](img/raster-als-tabelle.svg)

Das globale Modell ETOPO 2022 hat in der Fassung, die wir verwenden, 10 800 × 21 600 Zellen, also rund **233 Millionen Werte**.

### ETOPO 2022

ETOPO 2022 ist das globale Reliefmodell der US-Behörde NOAA (National Centers for Environmental Information), veröffentlicht 2022. Es setzt Höhen an Land, Wassertiefen am Meeresboden und Küstenlinien aus vielen regionalen und weltweiten Quellen zu einem einzigen, lückenlosen Raster zusammen – von der Tiefseerinne bis zum Gipfel.

- **Auflösungen:** NOAA bietet das Modell mit Zellen von 15, 30 und 60 Bogensekunden an. Wir nutzen die 60-Bogensekunden-Fassung: eine Zelle ist 1/60 Grad groß, am Äquator etwa 1,85 km. Die Datei ist 456 MB groß.
- **Zwei Varianten:** *Ice Surface* zeigt Grönland und die Antarktis mit der Oberfläche ihrer Eisschilde, *Bedrock* den Fels darunter. Wir verwenden die Eisoberfläche.
- **Bezug:** Die Höhen sind Meter über dem Geoid EGM2008, also grob über dem mittleren Meeresspiegel; negative Werte liegen darunter. Die Lage bezieht sich auf WGS 84, das System, das auch GPS verwendet.

**Was die Zeilen und Spalten bedeuten:**

- **Zeilen** sind Breitenstreifen. 180 Breitengrade von Süd- bis Nordpol × 60 Zellen je Grad ergeben 10 800 Zeilen. In der Datei liegt der südlichste Streifen (−89,99°) in der ersten Zeile, der nördlichste in der letzten.
- **Spalten** sind Längenstreifen. 360 Längengrade × 60 ergeben 21 600 Spalten, beginnend bei −180° an der Datumsgrenze und nach Osten fortlaufend.
- **Eine Zelle** in Zeile *i* und Spalte *j* enthält die Höhe des Feldes, das zwischen diesen Breiten- und Längengraden liegt.

Wer eine Höhe nachschlagen will, rechnet also die Koordinate in eine Zeilen- und eine Spaltennummer um und liest den Wert ab.

### Vom Raster zur Funktion

Diese Tabelle ist aber nur eine Stichprobe von etwas Kontinuierlichem: der Höhe als Funktion des Ortes.

```math
h : (\text{lon}, \text{lat}) \mapsto \text{Höhe in Metern}
```

Wenn wir eine Funktion `f` finden, die an allen Gitterpunkten ungefähr die richtige Höhe liefert, brauchen wir die Tabelle nicht mehr – wir speichern stattdessen die Funktion. Ein neuronales Netz ist genau so eine Funktion, festgelegt durch seine Gewichte θ:

```math
f_\theta(\text{lon}, \text{lat}) \approx h(\text{lon}, \text{lat})
```

### Was sind Gewichte?

Gewichte sind die Zahlen, die eine Rechenvorschrift festlegen – die Faktoren in einer Funktion.

**Ein einfaches Beispiel:** Eine Straße steigt auf 5 km Länge gleichmäßig an. Alle 10 m wurde die Höhe gemessen, das sind rund 500 Messwerte. Beginnt die Straße auf 300 m und steigt sie je Meter um 5 cm, lassen sich alle 500 Werte durch eine einzige Formel ersetzen:

```math
\text{Höhe}(x) = 300 + 0{,}05 \cdot x \qquad (x = \text{Strecke in Metern})
```

Statt 500 Zahlen speichern wir nur zwei: **300** und **0,05**. Das sind die Gewichte dieser Funktion. Wer die Höhe nach 2,5 km wissen will, rechnet sie aus: 300 + 0,05 · 2 500 = 425 m. Die Tabelle braucht man nicht mehr.

Echtes Gelände steigt nicht gleichmäßig. Es hat Täler, Grate und Küsten, und dafür reicht eine Gerade nicht. Ein neuronales Netz baut deshalb aus vielen solchen einfachen Stücken eine biegsame Funktion zusammen:

- Ein **Neuron** nimmt seine Eingaben, multipliziert jede mit einem eigenen Gewicht, addiert alles und einen Grundwert und knickt das Ergebnis dann ab (zum Beispiel: negative Werte werden zu 0). Ohne diesen Knick blieben auch viele Neuronen zusammen nur eine Gerade.
- Mehrere Neuronen nebeneinander bilden eine **Schicht**, mehrere Schichten hintereinander das **Netz**. Jede Schicht rechnet mit den Ergebnissen der vorherigen weiter.
- Schon ein winziges Netz mit 2 Eingaben (Länge, Breite), 3 Neuronen und 1 Ausgabe (Höhe) hat 13 Gewichte: 2 · 3 + 3 für die Neuronen und 3 · 1 + 1 für die Ausgabe.

Gespeichert wird nur θ, die Gesamtheit aller Gewichte. **Die Gewichte sind die Daten.** Wer die Höhe an einer Stelle wissen will, setzt die Koordinate ins Netz ein und rechnet sie aus – genau wie bei der Straße, nur mit mehr Zahlen.

### Unser Standardbeispiel

Damit das Speicher spart, müssen die Gewichte θ deutlich weniger Bytes belegen als die ursprüngliche Tabelle.

Für die Rechnungen in diesem Tutorial verwenden wir die Voreinstellung der Begleit-App als **Standardbeispiel**:

- **Daten:** ETOPO 2022, gemittelt auf 1080 × 2160 Zellen (je 1/6 Grad, etwa 19 km). Das sind 2 332 800 Höhenwerte.
- **Netz:** die Vorlage *Fourier MLP (recommended)* mit 82 561 Gewichten. Ihr Aufbau ist Thema späterer Kapitel.

Gespeichert sind das 323 KB statt 4,4 MB; wie sich diese Größen berechnen, steht im nächsten Abschnitt. Der Preis: Das Netz trifft die Höhen nicht exakt. Wie groß der Fehler ist und wovon er abhängt, ist das Thema der folgenden Kapitel.

### Lernen andersherum

Dieses Vorgehen stellt die Herangehensweise des maschinellen Lernens mit neuronalen Netzen auf den Kopf.

#### Standard-Training bei neuronalen Netzen

Die Input-Daten sind Beispiele, aus denen das neuronale Netz lernt und von diesen wenigen Daten auf alle anderen schließt.

#### Training zum Speichern von Geo-Informationen in neuronalen Netzen

Die Input-Daten enthalten alle bekannten Werte. Das neuronale Netz soll nach dem Training diese möglichst richtig wiedergeben.

| gewöhnliches Training | Training in diesem Fall |
|---|---|
| auf neue Daten verallgemeinern | genau **einen** Datensatz auswendig lernen |
| kleines Modell, viele Daten | das Modell **sind** die Daten |
| Erfolg = Geringer Fehler bei Testdaten | Erfolg = Geringer Rekonstruktionsfehler |

In der Literatur heißt so ein Netz *Implicit Neural Representation* (INR) oder *neural field*. Xie et al. beschreiben neuronale Felder in ihrem Überblicksartikel als Netze, die eine Größe in Abhängigkeit von Raum- und Zeitkoordinaten darstellen [1]; bekannt gemacht haben den Begriff INR unter anderem Sitzmann et al. [2]. Dass man auf diese Weise Geodaten komprimieren kann, zeigen Huang und Hoefler für Wetter- und Klimadaten [3] – die direkte Vorlage für dieses Tutorial.

## Kompressionsrate ermitteln

Eine Kompressionsrate ist nur so gut wie ihre Buchhaltung. Wir vergleichen zwei Größen.

**Rohgröße des Rasters** = Anzahl Gitterpunkte × Bytes pro Wert.

Die ETOPO-Datei selbst speichert jede Höhe als float32 (4 Byte). Für Höhen in ganzen Metern – von −10 935 m im Marianengraben bis 8 849 m am Mount Everest – genügt aber int16 (2 Byte, Bereich −32 768 bis 32 767).

Wir rechnen deshalb mit **2 Byte pro Wert**. Das ist die vorsichtige Wahl: Gegen float32 gerechnet sähe jede Kompressionsrate doppelt so gut aus.

**Größe des Netzes** = alles, was man zum Zurückrechnen braucht:

- die Gewichte, je 4 Byte (float32),
- Mittelwert und Standardabweichung der Höhen, mit denen das Netz normiert wurde (2 × 4 Byte) – ohne sie liefert das Netz keine Meter,
- weitere Zutaten, die in späteren Kapiteln dazukommen, etwa ein Startwert für Zufallszahlen (4 Byte) oder ein grobes Kontextgitter.

Daraus folgen zwei Kennzahlen:

```math
\text{Kompressionsrate} = \frac{\text{Rohgröße}}{\text{Netzgröße}}
\qquad
\text{Bits pro Gitterpunkt} = \frac{8 \cdot \text{Netzgröße}}{\text{Anzahl Gitterpunkte}}
```

Für das Standardbeispiel: 2 332 800 Punkte × 2 Byte = 4 665 600 Byte Rohgröße. Das Netz belegt 82 561 × 4 Byte für die Gewichte plus 12 Byte für Normierung und Startwert, zusammen 330 256 Byte. Das ergibt eine Rate von **14,1** und 1,13 Bits pro Gitterpunkt. Eine Rate unter 1 heißt: Das Netz ist größer als die Daten, es wird nichts gespart.

## Vereinfachen: Das Raster verkleinern

233 Millionen Punkte aus dem ursprünglichen ETOPO-Raster sind für die Verarbeitung auf dem Laptop zu viel.

![Die Begleit-App nach dem Start: links die Seitenleiste mit Datenquelle, Gitterauflösung, Netz-Vorlage und Trainingseinstellungen, rechts der Reiter Idea & Budget.](img/app-uebersicht.jpg)

Die Begleit-App reduziert das ETOPO-Raster deshalb auf eine wählbare Arbeitsauflösung zwischen 180 × 360 (1°, etwa 111 km pro Zelle) und 2160 × 4320 (1/12°, etwa 9 km).

Dafür fasst die App Blöcke von Zellen zusammen und nimmt deren Mittelwert; das Ergebnis landet in `data/cache/`, damit es beim nächsten Start wieder da ist.

Mit diesen Daten arbeitet die App weiter. Das Netz soll sie reproduzieren.

## Ausprobieren

1. App starten: `uv run streamlit run app.py`. Fehlt die ETOPO-Datei, arbeitet die App mit einer synthetischen Erde – für dieses Kapitel genügt das.

2. Reiter *Idea & Budget* öffnen. Unter *Your budget – before training* stehen vier Kennzahlen: *Grid points*, *Parameters*, *Compression ratio* und *Bits per grid point*.

   ![Die vier Kennzahlen im Reiter Idea & Budget für das Standardbeispiel: 2 332 800 Gitterpunkte, 82 561 Parameter, Kompressionsrate 14, 1,133 Bits pro Gitterpunkt.](img/app-budget.png)

3. In der Seitenleiste unter *1 · Data* den Regler *Grid resolution* verschieben. Beobachte: Die Zahl der Parameter bleibt gleich, Rohgröße und Kompressionsrate ändern sich.

   <img src="img/app-seitenleiste.png" alt="Seitenleiste der App: unter 1 · Data die Quelle ETOPO 2022 und der Regler Grid resolution auf 1080 × 2160, unter 2 · Network die Vorlage Fourier MLP (recommended)." width="300">

4. Den Regler ganz nach links schieben (180 × 360). Was meldet die App?

5. Unter *2 · Network* die Netzwerk-Architektur des neuronalen Netzes wählen, etwa *Fourier MLP, very small* oder *Fourier MLP, large*, und die Rate vergleichen.

Trainiert wird in diesem Kapitel noch nicht.

Bereits die Wahl der Netzwerk-Architektur legt die Kompressionsrate fest. Das Training entscheidet nur, wie gut die Rekonstruktion wird.

## Ideen zum Weiterprobieren

- Den Regler *Grid resolution* Stufe für Stufe nach links schieben: Ab welcher Auflösung ist das Netz größer als die Daten?
- Die feinste Auflösung 2160 × 4320 wählen und auf *Bits per grid point* achten. Was bedeutet es, wenn das Netz für einen Gitterpunkt weniger als ein Bit braucht?
- Alle Vorlagen unter *2 · Network* durchgehen und die Parameterzahl vergleichen. Welche spart am meisten, welche gar nicht?
- Unter *Architecture details* die *Neurons per layer* verdoppeln. Wächst die Parameterzahl auf das Doppelte – oder stärker?
- Zwischen den Quellen *ETOPO 2022* und *Synthetic* wechseln. Warum ändert sich am Budget nichts?

## Weiterführende Links

- NOAA NCEI: [ETOPO Global Relief Model](https://www.ncei.noaa.gov/products/etopo-global-relief-model) – Beschreibung, Download und Nutzerhandbuch, Zitat: DOI [10.25921/fd45-gt74](https://doi.org/10.25921/fd45-gt74)
- [1] Xie et al.: [Neural Fields in Visual Computing and Beyond](https://arxiv.org/abs/2111.11426), 2022 – Überblick über neuronale Felder
- [2] Sitzmann et al.: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661), 2020
- [3] Huang und Hoefler: [Compressing multidimensional weather and climate data into neural networks](https://arxiv.org/abs/2210.12538), 2023
- Tancik et al.: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739), 2020 – Grundlage für ein späteres Kapitel

---

[Übersicht](README.md) · Weiter: [Kapitel 2 · Training des neuronalen Netzes](02-training.md)
