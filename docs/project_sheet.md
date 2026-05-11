# *Karta projektu studenckiego*

*Wykrywanie warunków sprzyjających wysokim poziomom wody (wezbraniom) na podstawie danych meteorologicznych.*

| Cel | Opracowanie algorytmu, który wskaże czy i w jaki sposób analizowane parametry meteorologiczne mogą sprzyjać wzrostom poziomu wody w portach i zwiększonemu ryzyku wezbrania. |
| --- | --- |
| **Warianty** | **A: analiza zależności meteo- poziom wody (dane: szeregi czasowe)+ prototyp algorytmu B: wykorzystanie uczenia maszynowego do stworzenia algorytmu wykrywania krytycznych epizodów.** |
| **Efekt końcowy** | **Raport (word, pdf) opisujący zależności meteo–poziom wody, tabela (csv) wartości progowych i czynników/sezon, Notebook/skrypt (Jupyter + README): pipeline przygotowania danych + wykrywanie epizodów + generowanie komunikatów „czynnik/ryzyko”.** |

## 1. Wstęp

Mimo tego, że Morze Bałtyckie jest morzem bezpływowym, obserwowane są przypadki drastycznej zmiany poziomów morza. Zmiany te mogą być wynikiem działania jednego czynnika, na przykład wiatru, lub kumulatywnym efektem nakładania się szeregu czynników, których siła i intensywność jest zmienna w czasie. Lokalnie, znaczący wpływ na poziom morza ma batymetria oraz kształt linii brzegowej i obecności infrastruktury. Konstrukcje morskie (np. porty) mogą wspołgenerować zjawiska rezonansowe, dodatkowo zmieniając lokalnie poziom morza. Celem projektu jest zbadanie, w jaki sposób warunki meteorologiczne wpływają na zmiany poziomu wody w kanałach portowych oraz opracowanie algorytmu, który będzie wskazywać, czy obserwowane (lub prognozowane) parametry meteorologiczne mogą sprzyjać wzrostom poziomu wody i zwiększonemu ryzyku wezbrania. Analiza będzie obejmować zmienne meteorologiczne takie jak suma i intensywność opadu, temperatura powietrza, ciśnienie atmosferyczne, a w razie dostępności również pokrywa śnieżna. Uwzględniona zostanie sezonowość oraz różnice mechanizmów w poszczególnych porach roku (np. wezbrania opadowe latem vs roztopowe zimą/wiosną), a zależności będą analizowane także z opóźnieniem czasowym, ponieważ reakcja poziomu wody na warunki meteo może następować po kilku godzinach lub dniach.

## 2. Słowniczek pojęć

**Poziom wody / stan wody** – wysokość zwierciadła wody mierzona na stacji (np. cm), zwykle jako
szereg czasowy.

**Wezbranie / epizod wysokiej wody** – okres, w którym poziom wody przekracza ustalony próg (np.
stan ostrzegawczy/alarmowy lub percentyl 90/95 z danych).

**Opad skumulowany** – suma opadów z ostatnich X godzin/dni (np. 24h, 72h), często lepiej wyjaśnia
reakcję zlewni.

**Opóźnienie (lag)** – przesunięcie w czasie między meteo a reakcją poziomu wody (np. maksimum
poziomu wody 6–48h po opadzie).

**Roztopy** – wzrost dopływu wody z topniejącego śniegu; w danych często widoczny przy dodatnich
temperaturach i sezonie zimowo-wiosennym.

## 3. Przykładowy sposób identyfikacji operacji

**Cel identyfikacji:** wskazać, czy warunki meteo sprzyjają wzrostom poziomu wody oraz który czynnik
jest najbardziej „podejrzany”.

Przykładowe operacje identyfikacyjne (regułowo-statystyczne, interpretowalne):

1. **O1: Epizod opadowy (krótki i intensywny)**
  a. warunek: intensywność opadu > próg (np. percentyl 90) i/lub opad 24h > próg
sezonowy
  b. komunikat: „Wysoka intensywność opadu sprzyja gwałtownemu wzrostowi poziomu
wody.”

2. **O2: Długotrwałe opady + nasycenie zlewni**
  a. warunek: opad skumulowany 72h / 7 dni > próg + opad bieżący > próg
  b. komunikat: „Zlewnia jest nasycona – nawet umiarkowany opad może podnieść poziom
wody.”

3. **O3: Roztopy (mechanizm sezonowy)**
  a. warunek: utrzymujące się temperatury dodatnie (lub szybki wzrost T) w sezonie
zimowo-wiosennym + brak/mały opad (lub opad deszczu na śnieg, jeśli da się uchwycić)
  b. komunikat: „Warunki roztopowe mogą powodować wzrost poziomu wody.”

4. **O4: Zależności sezonowe (różne mechanizmy w różnych porach roku)**
  a. warunek: inne relacje meteo→poziom wody w różnych sezonach (np. latem dominują
wezbrania opadowe, a wiosną roztopowe)
  b. komunikat: „W tym sezonie dominują inne czynniki podnoszące poziom wody.”

## 4. Zadanie projektowe

### 1. Kontrola jakości danych

- braki, anomalia, wartości niefizyczne (poziom wody skoki, ujemne opady, itp.)
- decyzja: usunięcie / interpolacja (ostrożnie) / oznaczenie jako brak

### 2. Ujednolicenie kroku czasowego

- wybór kroku: godzinowy lub dobowy
- agregacja: poziom wody (np. średnia/dobowe maksimum), opad (suma), temperatura
(średnia/min/max)

### 3. Inżynieria cech

- opóźnienia (lag) meteo i opadu (np. 1–48h / 1–7 dni)
- opad skumulowany (24h, 72h, 7 dni) jako proxy nasycenia zlewni
- cechy sezonowe: miesiąc/pora roku (+ ewentualnie sin/cos dnia roku)
- wiatr: sektory lub sin/cos (jeśli ma wpływ lokalny)

### 4. Analiza eksploracyjna

- sezonowość poziomu wody (miesiące/pory roku, dobowo)
- zależności poziom wody–meteo ogólnie i osobno dla sezonów
- analiza opóźnień (kiedy po opadzie rośnie poziom)

### 5. Analiza statystyczna zależności

- korelacje z opóźnieniem (lagged correlations)
- porównania rozkładów (poziom wody przy dużym vs małym opadzie), testy istotności
- proste modele interpretowalne (regresja, GAM) – jeśli zasadne

### 6. Projekt systemu „wskazywania czynników”

- progi sezonowe (percentyle/odchylenia) dla opadu/temperatury/nasycenia
- komunikat: czynnik + kierunek wpływu + „pewność” (np. częstość współwystępowania w
danych historycznych)

### 7. Walidacja i benchmark

- ocena metryk i stabilności wyników w czasie (rok-po-roku)

### 8. Outcome: raport + prototyp

- raport z wynikami i wnioskami
- notebook/skrypt generujący diagnozę dla nowej obserwacji (warunki sprzyjające/nie
sprzyjające + główny czynnik)

## 5. Zakres danych i przygotowanie

W projekcie wykorzystane zostaną szeregi czasowe poziomu wody oraz danych meteorologicznych z
tego samego obszaru i spójnego kroku czasowego (godzinowy lub dobowy). Przygotowanie obejmie:

- synchronizację czasową,
- kontrolę jakości (braki i odstające),
- ujednolicenie jednostek,
- agregacje zgodne z fizyką procesu (opad jako suma, poziom wody np. maksimum dobowe),
- konstrukcję cech pochodnych (opady skumulowane, opóźnienia, cechy sezonowe).

## 6. Format Danych Gotowego Raportu

### 1. Raport / rozdział wyników

- sezonowość poziomu wody
- relacje poziom wody–meteo ogólnie i sezonowo
- interpretacja: jakie warunki sprzyjają wezbraniom

### 2. Tabela „czynników sprzyjających” per sezon

- zima/wiosna/lato/jesień:
  - progi opadu (intensywność/suma), progi opadu skumulowanego, warunki temperatury
(roztopy)
  - typowe opóźnienie reakcji (np. 6–24h po opadzie)

### 3. Prototyp systemu wskazań (notebook/skrypt)

- wejście: meteo (bieżące/prognozowane) + sezon
- wyjście:
  - „ryzyko wysokiej wody: niskie/średnie/wysokie”
  - „główny czynnik: opad intensywny / nasycenie zlewni / roztopy / (wiatr/ciśnienie jeśli
dotyczy)”
  - krótki komentarz: „warunki podobne do historycznych epizodów w tym sezonie”

### Wariant rozszerzony (ML)

Możliwe jest wykorzystanie uczenia maszynowego, o ile ma to statystycznie uzasadniony sens: gdy
dane wskazują na nieliniowe zależności lub interakcje między meteo a poziomem wody, a model
oceniany w walidacji czasowej daje stabilną poprawę względem prostych metod odniesienia.
Warunkiem koniecznym jest interpretowalność (wskazanie czynników) oraz kontrola sezonowości.

## 7. Benchmark i metryki (jak sprawdzamy, czy działa)

**Błąd czasu początku/końca epizodu (onset/offset error):** o ile godzin/dni system myli start i koniec
wezbrania.

**Pokrycie zdarzeń (event coverage / recall):** jaki % rzeczywistych epizodów wysokiej wody został
wykryty.

**Fałszywe alarmy (false alarm rate / precision):** ile wykrytych epizodów nie potwierdziło się w danych
(lub jaki % alarmów jest trafny).

**Błąd piku (peak timing/peak error):** błąd czasu wystąpienia maksimum oraz błąd wysokości
maksimum podczas epizodu.

**Zgodność wartości (dla wersji regresyjnej):** MAE/RMSE dla poziomu wody w epizodach oraz poza
epizodami (osobno).

**Stabilność sezonowa:** metryki liczone osobno dla sezonów i rok-po-roku.

## 8. Wyniki i format oddania

- **Raport** (PDF/DOCX): wykresy sezonowości, zależności meteo–poziom wody, opis epizodów i
    wnioski.
- **Tabela progów i czynników per sezon** (CSV/XLSX).
- **Notebook/skrypt** (Jupyter + README): pipeline przygotowania danych + wykrywanie
epizodów + generowanie komunikatów „czynnik/ryzyko”.
