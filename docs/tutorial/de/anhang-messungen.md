# Geodaten in neuronalen Netzen speichern
# Anhang · Messungen

**Inhalt:**

* Alle Vorlagen der App und die Netze aus den Kapiteln 3 bis 5 im Vergleich, gemessen unter gleichen Bedingungen
* Was float16 bei jedem Netz kostet
* Die Messungen aus den einzelnen Kapiteln an einer Stelle, mit ihren Einstellungen

## So wurde gemessen

Damit die Zahlen vergleichbar sind, liefen alle Netze der Haupttabelle unter denselben Bedingungen:

| | |
|---|---|
| **Rechner** | Apple M4 mit 32 GB Arbeitsspeicher, macOS 27.0.1; gerechnet auf dem Grafikchip (*Device* `auto` = MPS) |
| **Software** | Python 3.14.6, PyTorch 2.14.0 |
| **Daten** | ETOPO 2022 (60 Bogensekunden, Eisoberfläche), gemittelt auf das Standardgitter 1080 × 2160 = 2 332 800 Höhenwerte, Rohgröße 4,4 MB (2 Byte pro Wert) |
| **Training** | Voreinstellungen der Seitenleiste: 4 000 Schritte, 32 768 Zellen pro Schritt, Lernrate 0,003 (die ersten 100 Schritte langsam ansteigend, danach allmählich auf 2 % davon fallend), *Area-weighted sampling* eingeschaltet |
| **Läufe** | je Netz drei Läufe mit den Startwerten 0, 1 und 2 für die Startgewichte und die Auswahl der Zellen; nur das lineare Modell mit *L* = 40 einmal, weil ein Lauf 5 Minuten dauert |
| **Auswertung** | nach dem Training auf dem **ganzen** Gitter, nicht nur auf einer Stichprobe |
| **Dateigröße** | wie in Kapitel 1: Gewichte, Normierung, gegebenenfalls Startwert und Kontextfeld, ohne die Verwaltung des Dateiformats `.pt` (rund 4 KB, Kapitel 10) |

Die Netze aus Kapitel 3 haben als Eingabe Länge und Breite und keine Kodierung; das Netz aus Kapitel 4 ist die Vorlage *Plain MLP (fails on purpose)* mit (x, y, z). Alle anderen Zeilen sind die Vorlagen der App, unverändert.

## Haupttabelle

Beim gewichteten RMSE steht der Mittelwert der drei Läufe, in Klammern der kleinste und der größte Wert. Alle übrigen Fehlermaße und die Dauer sind Mittelwerte. Die Fehlermaße sind in Kapitel 2 erklärt, PSNR in Kapitel 9.

| Netz | Kapitel | Gewichte | Datei | Rate | gew. RMSE | RMSE | MAE | größter Fehler | PSNR | R² | Dauer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 Schichten (Länge, Breite) | 3 | 3 | 20 Byte | 233 280 | 2 386 m (2 386–2 386) | 2 771 m | 2 254 m | 8 593 m | 15,4 dB | −0,092 | 5 s |
| 1 Schicht (Länge, Breite) | 3 | 513 | 2,0 KB | 2 265 | 1 425 m (1 390–1 457) | 1 491 m | 1 159 m | 8 307 m | 20,8 dB | 0,684 | 15 s |
| 4 Schichten (Länge, Breite) | 2, 3 | 50 049 | 196 KB | 23 | 706 m (685–731) | 648 m | 426 m | 7 977 m | 28,0 dB | 0,940 | 55 s |
| *Plain MLP (fails on purpose)*, (x, y, z) | 4 | 50 177 | 196 KB | 23 | 593 m (578–604) | 568 m | 373 m | 7 885 m | 29,2 dB | 0,954 | 58 s |
| *Fourier MLP, very small* | 5 | 10 945 | 43 KB | 107 | 372 m (368–378) | 358 m | 228 m | 5 070 m | 33,2 dB | 0,982 | 29 s |
| *Fourier MLP (recommended)* | 5 | 82 561 | 323 KB | 14 | 245 m (243–248) | 232 m | 141 m | 4 387 m | 37,0 dB | 0,992 | 74 s |
| *Fourier MLP, large* | 5 | 394 753 | 1 542 KB | 3,0 | 185 m (183–187) | 174 m | 102 m | 3 829 m | 39,4 dB | 0,996 | 275 s |
| *Spherical harmonic expansion (L = 40, linear)* | 6 | 1 682 | 6,6 KB | 693 | 631 m | 603 m | 405 m | 7 344 m | 28,7 dB | 0,948 | 305 s |
| *Spherical harmonics (L = 16)* | 6 | 86 785 | 339 KB | 13 | 241 m (240–242) | 227 m | 131 m | 4 332 m | 37,1 dB | 0,993 | 116 s |
| *SIREN (sine activation)* | 7 | 78 081 | 305 KB | 15 | 233 m (215–246) | 223 m | 148 m | 4 110 m | 37,3 dB | 0,993 | 112 s |
| *Fourier MLP + context field* | 8 | 82 689 | 355 KB | 13 | 186 m (185–187) | 179 m | 111 m | 3 701 m | 39,2 dB | 0,995 | 85 s |

## Was die Zahlen zeigen

**Der Weg durch die Kapitel lohnt sich.** Von Kapitel 3 bis Kapitel 5 sinkt der gewichtete RMSE von 2 386 m über 706 m und 593 m auf 245 m. Die Kugelkoordinaten aus Kapitel 4 bringen bei gleicher Netzgröße gut 100 m, die Fourier-Features aus Kapitel 5 noch einmal fast 350 m – bei einer etwas größeren Datei.

**Das einfachste Modell ist schlechter als nichts.** Sein R² ist kleiner als null. Das heißt: Eine Karte, die überall nur die mittlere Höhe zeigt, läge näher am Original als die schräge Ebene. Der Grund ist vermutlich die Gewichtung: Das Netz lernt mit flächengewichteter Auswahl (Kapitel 2), R² zählt aber jede Zelle gleich – auch die vielen kleinen Zellen an den Polen, wo die Ebene besonders weit danebenliegt.

**Das Kontextfeld ist der größte einzelne Gewinn.** Mit 355 KB ist *Fourier MLP + context field* so genau wie *Fourier MLP, large* mit 1 542 KB (186 m gegen 185 m) – bei einem knappen Viertel der Dateigröße. Das bestätigt den Vergleich aus Kapitel 8.

**Ähnlich große Netze liegen nah beieinander.** *Fourier MLP (recommended)*, *Spherical harmonics (L = 16)* und *SIREN* sind zwischen 305 KB und 339 KB groß und liegen zwischen 233 m und 245 m. Der SIREN ist im Mittel am genauesten, schwankt aber am stärksten: Seine drei Läufe liegen 31 m auseinander, die der anderen beiden nur 2 bis 5 m.

**Das lineare Modell mit L = 40 ist winzig.** Mit 6,6 KB erreicht es eine Rate von 693 und ist dabei genauer als das Netz mit 4 Schichten aus Kapitel 3, das 30-mal so groß ist. Die Kugelflächenfunktionen bis Grad 40 enthalten aber keine Welle kürzer als etwa 1 000 km (Kapitel 6); feiner kann es nicht werden.

**Die Läufe sind gut wiederholbar.** Bei den meisten Netzen liegen die drei Läufe nur wenige Meter auseinander. Unterschiede von unter 10 m zwischen zwei Netzen sollte man deshalb nicht überbewerten, wenn sie aus einem einzigen Lauf stammen.

## float16

Dieselben Netze, nach dem Training auf float16 gerundet (Kapitel 10). Mittelwerte der drei Läufe:

| Netz | Datei float16 | Rate float16 | gew. RMSE float32 | gew. RMSE float16 |
|---|---:|---:|---:|---:|
| 0 Schichten (Länge, Breite) | 14 Byte | 333 257 | 2 386,4 m | 2 386,4 m |
| 1 Schicht (Länge, Breite) | 1,0 KB | 4 512 | 1 425,1 m | 1 425,1 m |
| 4 Schichten (Länge, Breite) | 98 KB | 47 | 706,0 m | 706,1 m |
| *Plain MLP (fails on purpose)* | 98 KB | 47 | 593,1 m | 593,1 m |
| *Fourier MLP, very small* | 21 KB | 213 | 371,9 m | 371,9 m |
| *Fourier MLP (recommended)* | 161 KB | 28 | 244,9 m | 244,9 m |
| *Fourier MLP, large* | 771 KB | 5,9 | 184,7 m | 184,7 m |
| *Spherical harmonic expansion (L = 40, linear)* | 3,3 KB | 1 384 | 630,9 m | 630,9 m |
| *Spherical harmonics (L = 16)* | 170 KB | 27 | 240,7 m | 240,7 m |
| *SIREN (sine activation)* | 153 KB | 30 | 233,3 m | 233,7 m |
| *Fourier MLP + context field* | 193 KB | 24 | 186,2 m | 186,2 m |

Bei keinem Netz ändert sich der gewichtete RMSE um mehr als einen halben Meter. Nur beim SIREN ist überhaupt ein Unterschied zu sehen, wie in Kapitel 10 beschrieben. Für die Dateigröße sind Mittelwert, Standardabweichung und Startwert weiter mit je 4 Byte gezählt; beim Runden für die Fehlermessung wurden auch sie auf float16 gerundet.

## Messungen aus den Kapiteln

Einige Kapitel enthalten eigene Messungen für eine bestimmte Frage. Sie wurden mit anderen Einstellungen gerechnet als die Haupttabelle, meist in einem einzigen Lauf:

| Kapitel | Frage | Einstellungen |
|---|---|---|
| [5](05-fourier-features.md) | *spectral bias* am Alpenprofil: einfaches Netz gegen Fourier-Features nach 50, 500 und 3 000 Schritten | nur das Profil bei 46,4° Nord, nicht die ganze Erde |
| [7](07-siren.md) | SIREN mit und ohne Startbelegung nach Sitzmann | nur jeder zweite Punkt des Alpenprofils trainiert, die anderen zur Kontrolle |
| [8](08-kontextfeld.md) | Kontextfeld bei gleicher Dateigröße; Auflösung des Kontextfelds | 1080 × 2160, 4 000 Schritte; Netze ohne Kontextfeld etwas breiter für gleiche Dateigröße |
| [9](09-rate-und-fehler.md) | Rate-Distortion-Kurve | 1080 × 2160, neun Netzbreiten von 16 bis 256, 4 Schichten, σ = 8, 2 000 und 600 Schritte; Fehler auf 200 000 Zellen |
| [10](10-abfragen-und-exportieren.md) | float16 auf dem ganzen Gitter; Abfragen an einzelnen Orten; feiner abfragen gegen das Gitter 2160 × 4320 | 1080 × 2160, 4 000 Schritte |
| [11](11-einordnung-und-ausblick.md) | Land oder Meer: Anteil der Erdoberfläche auf der richtigen Seite der 0-m-Linie | Netz aus Kapitel 10 |

Die Zahlen in den Kapiteln weichen deshalb um einige Meter von der Haupttabelle ab. Zum Beispiel hat das Standardbeispiel in Kapitel 10 einen gewichteten RMSE von 245 m, *Fourier MLP + context field* 191 m statt 186 m; in Kapitel 8 stehen 245 m und 185 m. Solche Unterschiede liegen im Bereich der Schwankung von Lauf zu Lauf.

### Feineres Gitter 2160 × 4320

Aus der Entwicklung stammt eine Messung auf dem feinsten Gitter der App, 2160 × 4320 (17,8 MB Rohgröße). Gerechnet wurde nur 1 500 Schritte lang, ebenfalls auf einem Apple-Chip, einmal pro Netz. Die Netze sind dabei noch nicht fertig trainiert:

| Netz | Datei | Rate | gew. RMSE | PSNR |
|---|---:|---:|---:|---:|
| Fourier MLP, σ = 8 | 322 KB | 57 | 311 m | 35,3 dB |
| + Kontextfeld 90 × 180 | 355 KB | 51 | 257 m | 36,9 dB |
| + Kontextfeld 180 × 360 | 450 KB | 41 | 233 m | 37,8 dB |
| SIREN | 305 KB | 60 | 258 m | 36,8 dB |

Die Raten sind viermal so hoch wie auf dem Standardgitter, weil das Raster viermal so groß ist, das Netz aber gleich bleibt. Der Fehler ist größer, denn das feinere Gitter enthält mehr Einzelheiten. In dieser kurzen Messung liegt der SIREN etwa gleichauf mit dem Kontextfeld 90 × 180; auf dem Standardgitter mit 4 000 Schritten ist das Kontextfeld deutlich besser (186 m gegen 233 m).

## Selbst nachmessen

Alle Netze der Haupttabelle lassen sich in der App nachrechnen: Vorlage wählen, *Steps* auf 4 000 lassen, *Train*. Die Fehlermaße stehen danach im Reiter *Training*, Dateigröße und Rate im Reiter *Idea & Budget*. Auf einem Rechner ohne Grafikchip dauert ein Lauf deutlich länger; für einen ersten Eindruck reicht das Gitter 360 × 720 mit 1 000 Schritten.

Ein einzelner Lauf in der App wird nicht genau dieselben Zahlen liefern wie hier, weil die Startgewichte vom Zufall abhängen. Er sollte aber in der Nähe der angegebenen Spanne liegen.

---

[Übersicht](README.md) · Zurück: [Kapitel 11 · Einordnung und Ausblick](11-einordnung-und-ausblick.md)

*Dieses Tutorial wurde von Gerhard Völkl unter Verwendung von Claude Code (Anthropic) erstellt. Es soll helfen, mehr über das Thema „Geodaten in neuronalen Netzen speichern“ zu lernen. Verbesserungsvorschläge gerne an gkvoelkl@nelson-games.de.*
