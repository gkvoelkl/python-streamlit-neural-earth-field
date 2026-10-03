# Anhang · Lösungen

## Kapitel 1

**1.1** Die Rohgröße ist ny × 2ny Punkte × 2 Byte = 4 · ny² Byte. Das Netz ist kleiner, sobald 4 · ny² > 330 256, also ny > 287.

| Gitter | Rohgröße | Kompressionsrate |
|---|---:|---:|
| 180 × 360 | 129 600 Byte (126,6 KB) | 0,39 |
| 360 × 720 | 518 400 Byte (506 KB) | 1,57 |
| 540 × 1080 | 1 166 400 Byte (1,1 MB) | 3,53 |
| 720 × 1440 | 2 073 600 Byte (2,0 MB) | 6,28 |
| 1080 × 2160 | 4 665 600 Byte (4,4 MB) | 14,13 |
| 2160 × 4320 | 18 662 400 Byte (17,8 MB) | 56,51 |

Ab **360 × 720** ist das Netz kleiner als die Daten. Bei 180 × 360 ist es 2,5-mal so groß, und die App zeigt im Reiter *Idea & Budget* eine rote Warnung.

**1.2** 2160 × 4320 = 9 331 200 Punkte. 8 · 330 256 / 9 331 200 ≈ **0,28 Bits pro Gitterpunkt**. Ein Wert unter 1 heißt: Das Netz verwendet für jeden Punkt weniger als ein einziges Bit – weniger, als man bräuchte, um nur „Land oder Meer“ zu speichern. Das geht nur, weil benachbarte Punkte stark zusammenhängen und das Netz diese Struktur statt der einzelnen Werte speichert.

**1.3** Mit 4 Byte pro Wert wäre die Rohgröße 9 331 200 Byte und die Rate 9 331 200 / 330 256 ≈ **28,3** statt 14,1. Weniger ehrlich ist das, weil niemand Höhen in ganzen Metern als float32 weitergeben müsste: Ein einfaches Umspeichern auf int16 halbiert die Datei ohne jeden Verlust. Eine Kompressionsrate sollte sich mit der besten naheliegenden einfachen Lösung messen, nicht mit einer unnötig großen.
