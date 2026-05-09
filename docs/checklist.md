# Project checklist

## Goal

Zbudować pipeline, który:

wykrywa epizody wysokiej wody
wskazuje główny czynnik (opad / nasycenie / roztopy / inne)
generuje komunikat ryzyka

## Dane które będą potrzebne

Input datasets (IMGW-PIB, Copernicus Climate Data Store - ERA5, port + martwa Wisła):
poziom wody (port + Martwa Wisła, opcjonalnie IMGW-PIB)
nasycenie gleby (IMGW-PIB)
dane meteorologiczne:

- opad
- temperatura
- ciśnienie
- wiatr (opcjonalnie)

## Etapy projektu

### 1. **Zbieranie danych**

Pobranie danych z IMGW-PIB, Copernicus Climate Data Store - ERA5 oraz portu + Martwa Wisła.

### 2. **Przygotowanie danych**

Oczyszczenie, uzupełnienie brakujących wartości i standaryzacja danych.

### 3. **Synchronizacja danych**

Dopasowanie danych do wspólnego formatu i osi czasu.

### 4. **Resampling danych**

Przekształcenie danych do odpowiedniej częstotliwości (godzinowej i dziennej).

### 5. **Agregacja danych**

Obliczenie średnich, sum i innych statystyk dla różnych okresów czasu.

- poziom wody - średnia, max, min
- opad - suma, średnia
- temperatura - średnia, max, min
- ciśnienie - średnia, max, min
- wiatr - średnia, max, min

### 6. **Feature engineering**

Stworzenie nowych cech na podstawie istniejących danych, takich jak wskaźniki nasycenia gleby, wskaźniki opadów itp.

- opad - rain_1h_sum, rain_3h_sum, rain_6h_sum, rain_12h_sum, rain_24h_sum
- temperatura - temp_mean, temp_delta_24h, thaw_flag (czy temperatura przekroczyła 0°C w ciągu ostatnich 24h)
- wiatr - speed, direction

### 7. **Lagi**

Dodanie opóźnień do cech, aby uwzględnić wpływ wcześniejszych warunków na poziom wody.

- rain_lag_1h, rain_lag_3h, rain_lag_6h, rain_lag_12h, rain_lag_24h, rain_lag_48h, rain_lag_72h
- temp_lag
- pressure_lag

### 8. **Sezonowość**

Dodanie cech sezonowych, takich jak miesiąc, dzień tygodnia, pora roku itp.

### 9. **Exploracyjna analiza danych**

Analiza korelacji między cechami a poziomem wody, identyfikacja wzorców i trendów.

- sezonowość poziomu wody
- zależność meteo vs poziom wody
- zależność nasycenia gleby vs poziom wody
- analiza opóźnień (lagów)

> [!NOTE]
> Kluczowe pytania:
> Jakie cechy są najbardziej istotne dla przewidywania poziomu wody?
> Po ilu godzinach od opadów poziom wody zaczyna rosnąć?
> Jakie są typowe wzorce sezonowe dla poziomu wody?

### 10. **Analiza statyczna**

Przeprowadzenie testów statystycznych, takich jak testy korelacji, testy t-Studenta itp., aby ocenić istotność cech.

- Korelacje z lagami
- Testy t-Studenta dla różnych grup (np. wysokie vs niskie poziomy wody)
- Porównanie (np: high rain vs low rain)

### 11. **Definicja epizodów wysokiej wody**

Określenie progu, powyżej którego poziom wody jest uważany za wysoki.

```python
threshold = water_level.quantile(0.95)
```

### 12. **System czujników**

Wskazać czy warunki sprzyjają wzrostowi wody oraz określenie głównego czynnika (opad, nasycenie, roztopy itp.) na podstawie analizy cech.

Przykładowe reguły:

**O1: Epizod opadowy (krótki i intensywny)**
a. warunek: intensywność opadu > próg (np. percentyl 90) i/lub opad 24h > próg
sezonowy
b. komunikat: "Wysoka intensywność opadu sprzyja gwałtownemu wzrostowi poziomu
wody."

**O2: Długotrwałe opady + nasycenie zlewni**
a. warunek: opad skumulowany 72h / 7 dni > próg + opad bieżący > próg
b. komunikat: "Zlewnia jest nasycona – nawet umiarkowany opad może podnieść poziom
wody."

**O3: Roztopy (mechanizm sezonowy)**
a. warunek: utrzymujące się temperatury dodatnie (lub szybki wzrost T) w sezonie
zimowo-wiosennym + brak/mały opad (lub opad deszczu na śnieg, jeśli da się uchwycić)
b. komunikat: "Warunki roztopowe mogą powodować wzrost poziomu wody."

**O4: Zależności sezonowe (różne mechanizmy w różnych porach roku)**
a. warunek: inne relacje meteo→poziom wody w różnych sezonach (np. latem dominują
wezbrania opadowe, a wiosną roztopowe)
b. komunikat: "W tym sezonie dominują inne czynniki podnoszące poziom wody"

### 13. **Wyliczenie confidence**

Na podstawie przypadków historycznych, które spełniały warunki O1-O4, można oszacować prawdopodobieństwo wystąpienia epizodu wysokiej wody i oszacować % pewności.

## Machine learning

### Modele

- Logistic Regression
- Random Forest / XGBoost

### Wymogi modelu

- interpretowalność (ważne cechy, wpływ na decyzję)
- możliwość oszacowania prawdopodobieństwa (confidence)
- SHAP values do analizy wpływu cech

### Walidacja modelu

Podział na zbiór treningowy i testowy:

- Zbiór treningowy: 2021-2023
- Zbiór testowy: 2024-2025

### Metryki oceny

- Recall (czułość) - ważne, aby model wykrywał jak najwięcej epizodów wysokiej wody
- Precision (precyzja) - ważne, aby unikać fałszywych alarmów
- Onset error - czas między przewidywanym a rzeczywistym początkiem epizodu wysokiej wody
- MAE / RMSE dla regresji poziomu wody (opcjonalnie)

### Interpretacja stabilności modelu

- Porównania rok do roku, sezon do sezonu
- Analiza per sezon

## Outputy

### Raport PDF

- Sezonowość
- Zależność Meteo -> poziom wody
- Interpretacja wyników modelu (ważne cechy, wpływ na decyzję)

### CSV

- Timestamp / Sezon
- Czynnik
- Threshold
- Lag

### Notebook

Pipeline:

1. Zbieranie danych
2. Przygotowanie danych
3. Feature engineering
4. Wykrywanie eventów
5. Analiza czynników
6. Generowanie outputu

## Checklista

- [ ] Zbieranie danych
- [x] Przygotowanie danych
- [x] Synchronizacja danych
- [x] Resampling danych
- [x] Agregacja danych (tylko dla poziomu wody. Potrzebuje reszty jak zbierzemy inne datasety)
- [ ] Feature engineering
- [ ] Dodanie lagów
- [ ] Dodanie cech sezonowych
- [ ] Exploracyjna analiza danych
- [ ] Analiza statyczna
- [ ] Definicja epizodów wysokiej wody
- [ ] System czujników (reguły O1-O4)
- [ ] Wyliczenie confidence
- [ ] Budowa modeli machine learning
- [ ] Walidacja modeli
- [ ] Interpretacja wyników
- [ ] Generowanie raportu PDF
- [ ] Generowanie CSV
- [ ] Przygotowanie notebooka z pipeline'em
