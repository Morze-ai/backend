# Raw data

Is raw data changed from excel into csv and it's corresponding metadata.json files.

## Formats

### Pomiary codzienne

Pole      Zawartość pola

PSKDSZS - Kod stacji
PSNZWP  - Nazwa stacji
KDNRZK  - Nazwa rzeki/jeziora
COROKH  - Rok hydrologiczny
COMSCH  - Wskaźnik miesiąca w roku hydrologicznym
CODZIEN - Dzień
COSTAN  - Stan wody [cm]
COPRZP  - Przepływ [m^3/s]
COPTMP  - Temperatura wody [st. C]
COMSCK  - Miesiąc kalendarzowy

Stan wody 9999 albo NULL oznacza brak danych w bazie.
Przepływ 99999.999 albo NULL oznacza, że przepływ w tym dniu nie był opracowywany.
Temperatura 99.9 albo NULL oznacza brak danych w bazie, która może wynikać np. z braku pomiarów temperatury na stacji.

Od roku 2024 braki w danych są oznaczane zawsze jako NULL.

### Pomiary zjawisk (zjawiska lodowe, pokrycie roślinne)

PSKDSZS - Kod stacji
PSNZWP  - Nazwa stacji
KDNRZK  - Nazwa rzeki/jeziora
ZJROKH  - Rok hydrologiczny
ZJMSCH  - Wskaźnik miesiąca w roku hydrologicznym
ZJDZIEN - Dzień
ZJGRLD  - Grubość lodu [cm]
ZJKODZJ - Kod zjawiska lodowego (słownik poniżej)
ZJKDPRC - Procent udziału zjawiska lodowego [mnożnik *10; np. 3 oznacza 30% udziału zjawisk lodowych]
ZJZRST  - Kod zarastania (szczegółowe informacje poniżej)

Grubość lodu
0   oznacza brak pomiaru grubości lodu ze względu na brak zjawisk lodowych
999 albo NULL oznacza brak pomiaru grubości lodu przy występowaniu zjawisk lodowych lub (w miesiącach letnich) występowanie zarastania przy braku zjawisk lodowych (tzn. jeśli kod pole zjawiska lodowego jest puste)
Od roku 2024 braki w danych są oznaczane zawsze jako NULL.

Rekordy generowane są tylko dla dni, w których zanotowano zjawiska lodowe lub zarastanie.

Kody zjawisk lodowych - słownik
01 - śryż
02 - kra
03 - lód brzegowy
04 - pokrywa lodowa
05 - zator lodowy
06 - lód brzegowy i śryż
07 - lód brzegowy i kra
08 - śryż i kra
09 - zator śryżowy
32 - lód zatokowy
41 - woda na lodzie
42 - lód pływający (wolny od brzegów)
43 - lód zmurszały (dziurawy)

Kody zarastania
Kod zarastania zapisany jest w układzie dpw

Litery oznaczają rodzaj roślinności
d - roślinność denna zanurzona całkowicie
p - roślinność pływająca
w - roślinność wystająca ponad zwierciadło wody

Przykładowe kody oznaczają:
020: p2
112: d1 p1 w2
213: d2 p1 w3

Cyfry oznaczają stopień zarośnięcia koryta rzeki:
0 – brak roślinności
1 – zarastanie w 1/3 części przekroju
2 – zarastanie w 2/3 części przekroju
3 – zarastanie niemal całkowite lub całkowite

UWAGA!
Suma pokrycia roślinnością różnego typu może być większa niż 1 (czyli 3/3), ponieważ różne rodzaje roślinności mogą się przenikać.
