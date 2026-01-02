import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

# Importujemy logikę symulacji oraz zarządcę bazy danych
from Simulation import run_pairs_trading_sim_fast
from DatabaseManager import DatabaseManager

# --- KONFIGURACJA ŚCIEŻEK ---
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_files"
PARQUET_FILE = PROCESSED_DIR / "processed_market_data.parquet"

# --- PARAMETRY STRATEGII ---
LIMIT_PAIRS = 10000    # Ograniczenie liczby par (dla testów ustaw np. 5000, dla pełnej analizy usuń limit)
WINDOW = 20
ENTRY_Z = 2.0
EXIT_Z = 0.5
COMMISSION = 0.001

def generate_diagonal_pairs(tickers, limit):
    """
    Generuje pary metodą 'diagonalną' (najbliżsi sąsiedzi na liście),
    co pozwala szybciej znaleźć skorelowane pary w posortowanych danych.
    """
    n = len(tickers)
    pairs = []
    # k to dystans w liście (1 = sąsiad, 2 = co drugi, itd.)
    for k in range(1, n):
        for i in range(n - k):
            pairs.append((tickers[i], tickers[i + k]))
            if limit and len(pairs) >= limit:
                return pairs
    return pairs

def run_hybrid_simulation():
    """
    Główna pętla symulacji w modelu hybrydowym:
    1. CZYTA z Parquet (Szybkość)
    2. OBLICZA w RAM (Numba)
    3. ZAPISUJE do SQL (Integracja/Wymogi)
    """
    
    # 1. Sprawdzenie dostępności danych (Parquet)
    if not PARQUET_FILE.exists():
        print(f"❌ BŁĄD: Nie znaleziono pliku danych: {PARQUET_FILE}")
        print("   Uruchom najpierw 'LoadData.py', aby przetworzyć dane rynkowe.")
        return

    print(f"📥 Ładowanie cen z pliku Parquet (High Performance I/O)...")
    try:
        prices_df = pd.read_parquet(PARQUET_FILE)
    except Exception as e:
        print(f"❌ Błąd odczytu pliku Parquet: {e}")
        return

    all_tickers = prices_df.columns.tolist()
    
    # 2. Optymalizacja danych pod NumPy/Numba (Słownik tablic)
    # Wyciągamy wartości .values raz, aby nie odwoływać się do DataFrame w pętli
    print("🧠 Konwersja danych do tablic NumPy...")
    data_dict = {t: prices_df[t].values for t in all_tickers}

    # 3. Generowanie par do sprawdzenia
    print(f"🔄 Generowanie listy par (Limit: {LIMIT_PAIRS})...")
    pairs_to_test = generate_diagonal_pairs(all_tickers, LIMIT_PAIRS)
    
    if not pairs_to_test:
        print("⚠️ Brak par do sprawdzenia. Sprawdź czy masz wystarczającą liczbę tickerów.")
        return

    # 4. Inicjalizacja bazy danych (do zapisu wyników)
    db = DatabaseManager()
    final_results = []

    print(f"\n" + "="*50)
    print(f"🚀 START SYMULACJI HYBRYDOWEJ")
    print(f"Liczba par: {len(pairs_to_test)}")
    print(f"Źródło cen: Parquet | Cel wyników: SQL")
    print("="*50 + "\n")

    # 5. Główna pętla obliczeniowa
    pbar = tqdm(pairs_to_test, desc="Symulacja", unit="para", dynamic_ncols=True)
    
    for t1, t2 in pbar:
        # Pobieramy tablice numpy dla danej pary
        price_a = data_dict[t1]
        price_b = data_dict[t2]
        
        # Uruchamiamy szybką symulację (Numba)
        res = run_pairs_trading_sim_fast(
            price_a, price_b, 
            WINDOW, ENTRY_Z, EXIT_Z, COMMISSION
        )
        
        if res:
            ret, trades = res
            
            # Dodatkowo obliczamy korelację (wymóg analizy statystycznej w SQL)
            # Używamy np.corrcoef dla szybkości zamiast pandas
            # np.corrcoef zwraca macierz 2x2, interesuje nas [0,1]
            try:
                # Uwaga: trzeba usunąć NaN, jeśli występują, dla poprawnej korelacji
                valid_mask = ~np.isnan(price_a) & ~np.isnan(price_b)
                if np.sum(valid_mask) > WINDOW:
                    corr = np.corrcoef(price_a[valid_mask], price_b[valid_mask])[0, 1]
                else:
                    corr = 0.0
            except:
                corr = 0.0
            
            final_results.append({
                'ticker_a': t1,
                'ticker_b': t2,
                'wynik_netto': float(ret),          # Konwersja na typ Python float (dla SQL)
                'liczba_transakcji': int(trades),   # Konwersja na typ Python int
                'korelacja': float(corr)
            })

    # 6. Zapis wyników do SQL
    if final_results:
        print(f"\n💾 Zapisywanie {len(final_results)} wyników do bazy SQL...")
        df_results = pd.DataFrame(final_results)
        
        # Sortowanie dla lepszej czytelności przy debugowaniu
        df_results.sort_values(by='wynik_netto', ascending=False, inplace=True)
        
        # Wywołanie metody z DatabaseManager
        db.save_simulation_results(df_results)
        
        print("\n🏆 TOP 10 WYNIKÓW (Zapisano w 'simulation_results'):")
        print("-" * 60)
        pd.options.display.float_format = '{:,.4f}'.format
        print(df_results.head(10).to_string(index=False))
        print("-" * 60)
        print("✅ Proces zakończony sukcesem.")
    else:
        print("\n⚠️ Symulacja zakończona, ale nie znaleziono żadnych zyskownych par.")

if __name__ == "__main__":
    run_hybrid_simulation()