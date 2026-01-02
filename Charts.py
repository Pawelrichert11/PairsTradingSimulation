import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from Simulation import run_full_simulation

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_files"
DATA_SOURCE = PROCESSED_DIR / "processed_market_data.parquet"

TICKER_A = 'APA.US'
TICKER_B = 'APC.US'
COMMISSION = 0.001
WINDOW = 20
ENTRY_Z = 2.0
EXIT_Z = 0.5

def export_trade_history(results, ticker_a, ticker_b, filename='trade_history.csv'):
    """Generuje i zapisuje historię transakcji do folderu processed_files."""
    trades = results[results['Trades_Made'] > 0].copy()
    
    if trades.empty:
        print("Brak transakcji do zapisania.")
        return
    
    # Mapowanie operacji
    def identify_operation(pos):
        if pos == 1: return f"LONG {ticker_a} / SHORT {ticker_b}"
        if pos == -1: return f"SHORT {ticker_a} / LONG {ticker_b}"
        return "EXIT POSITION"

    history = pd.DataFrame(index=trades.index)
    history['Nr_Transakcji'] = range(1, len(trades) + 1)
    history['Cena_' + ticker_a] = trades[ticker_a].round(4)
    history['Cena_' + ticker_b] = trades[ticker_b].round(4)
    history['Operacja'] = trades['Position'].apply(identify_operation)
    history['Saldo_Kumulatywne'] = trades['Cumulative_Return_Net'].round(4)
    history['Zmiana_Saldo_%'] = trades['Cumulative_Return_Net'].pct_change().round(4) * 100

    # Pełna ścieżka do pliku
    output_path = PROCESSED_DIR / filename
    history.to_csv(output_path)
    print(f"✅ Zapisano historię transakcji do: {output_path}")

def get_performance_metrics(results, ticker_a, ticker_b):
    rets_a = results[ticker_a].pct_change()
    rets_b = results[ticker_b].pct_change()
    market_returns = (rets_a + rets_b) / 2
    
    cum_ret_net = results['Cumulative_Return_Net']
    running_max = cum_ret_net.cummax()
    drawdown = (cum_ret_net - running_max) / running_max
    market_cum = (1 + market_returns).fillna(0).cumprod()

    metrics = {
        "Total Return NET (%)": (cum_ret_net.iloc[-1] - 1) * 100,
        "Total Return Market (%)": (market_cum.iloc[-1] - 1) * 100,
        "Max Drawdown NET (%)": drawdown.min() * 100,
        "Total Transactions": results['Trades_Made'].sum(),
        "Total Costs Paid (%)": (results['Transaction_Costs_Value'].sum()) * 100
    }
    return metrics, drawdown, market_cum

if __name__ == "__main__":
    if os.path.exists(DATA_SOURCE):
        all_data = pd.read_parquet(DATA_SOURCE)
        
        # Walidacja tickerów
        missing_tickers = [t for t in [TICKER_A, TICKER_B] if t not in all_data.columns]
        if missing_tickers:
            print(f"❌ BŁĄD: Brakuje tickerów: {missing_tickers}")
            sys.exit()

        print(f"🚀 Uruchamiam symulację dla {TICKER_A} i {TICKER_B}...")
        results = run_full_simulation(all_data, TICKER_A, TICKER_B, WINDOW, ENTRY_Z, EXIT_Z, COMMISSION)
        
        if results is not None:
            export_trade_history(results, TICKER_A, TICKER_B)
            metrics, drawdown_series, market_cum = get_performance_metrics(results, TICKER_A, TICKER_B)

            print(f"\n--- STATYSTYKI PARY ({TICKER_A}/{TICKER_B}) ---")
            for key, value in metrics.items():
                print(f"{key:<25}: {value:.2f}")

            # --- GENEROWANIE WYKRESU ---
            plt.figure(figsize=(14, 12))
            
            # Subplot 1: Wynik Strategii (Equity Curve)
            plt.subplot(2, 1, 1)
            plt.plot(results.index, results['Cumulative_Return_Net'], label='STRATEGIA (NETTO)', color='blue', lw=3)
            plt.plot(results.index, market_cum, label='Benchmark Rynek 50/50', color='black', ls='--')
            plt.title(f'Wynik inwestycji: {TICKER_A} vs {TICKER_B}', fontsize=14)
            plt.ylabel('Skumulowany Zwrot (1.0 = Start)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Subplot 2: Znormalizowane wykresy cenowe spółek (Nałożone na siebie)
            plt.subplot(2, 1, 2)
            # Normalizujemy ceny (dzielimy przez pierwszą wartość), aby oba zaczynały od 1.0
            norm_a = results[TICKER_A] / results[TICKER_A].iloc[0]
            norm_b = results[TICKER_B] / results[TICKER_B].iloc[0]
            
            plt.plot(results.index, norm_a, label=f'Cena (norm): {TICKER_A}', color='green', lw=1.5, alpha=0.8)
            plt.plot(results.index, norm_b, label=f'Cena (norm): {TICKER_B}', color='red', lw=1.5, alpha=0.8)
            
            plt.title(f'Porównanie zachowania cen (znormalizowane)', fontsize=12)
            plt.ylabel('Względna zmiana ceny')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Symulacja nie zwróciła danych.")
    else:
        print(f"❌ Brak pliku danych: {DATA_SOURCE}")