PairsTradingSimulation
Paweł Richert

OPIS:
Projekt służący do symulacji, analizy i wizualizacji strategii inwestycyjnej 
Pairs Trading. Aplikacja umożliwia testowanie par aktywów na danych 
historycznych, zarządzanie bazą danych oraz podgląd wyników za pomocą 
interaktywnego dashboardu.

STRUKTURA PLIKÓW I KATALOGÓW

1. Dashboard.py
   Główny punkt wejścia dla interfejsu użytkownika. Służy do wizualizacji 
   wyników strategii i sterowania parametrami symulacji.

2. Simulation.py
   Główny moduł logiczny. Zawiera algorytmy obliczające spread, sygnały 
   wejścia/wyjścia oraz wyniki finansowe (P&L) dla strategii.

3. MultiSimulationOneProcess.py
   Skrypt umożliwiający seryjne uruchamianie wielu symulacji w ramach jednego 
   procesu (np. optymalizacja parametrów dla wielu par).

4. LoadData.py
   Moduł odpowiedzialny za pobieranie danych rynkowych (np. z API finansowych) 
   i ich wstępne czyszczenie.

5. DatabaseManager.py
   Warstwa dostępu do danych. Zarządza zapisem i odczytem danych historycznych 
   oraz wyników symulacji z bazy danych.

6. Config.py
   Plik konfiguracyjny. Tutaj definiowane są parametry strategii, ścieżki do 
   plików oraz ustawienia połączeń.

7. Charts.py
   Moduł generujący wykresy (ceny, spread, z-score, krzywa kapitału) na 
   potrzeby analizy i dashboardu.

8. Logger.py
   Obsługa logowania zdarzeń systemowych i błędów.

INSTRUKCJA INSTALACJI I URUCHOMIENIA

WYMAGANIA:
- Python 3.x
- Biblioteki (przykładowe): pandas, numpy, matplotlib, plotly, streamlit/dash, 
  yfinance, sqlite3.

INSTALACJA:
1. Sklonuj repozytorium:
   git clone https://github.com/Pawelrichert11/PairsTradingSimulation.git

2. Wejdź do katalogu projektu:
   cd PairsTradingSimulation

3. (Opcjonalnie) Utwórz i aktywuj środowisko wirtualne:
   python -m venv venv
   source venv/bin/activate  (Linux/Mac)
   venv\Scripts\activate     (Windows)

4. Zainstaluj zależności:
   pip install -r requirements.txt

URUCHOMIENIE:
- Aby uruchomić Dashboard (wizualizacja):
  streamlit run Dashboard.py
  (lub: python Dashboard.py)

- Aby uruchomić samą symulację w konsoli:
  python Simulation.py