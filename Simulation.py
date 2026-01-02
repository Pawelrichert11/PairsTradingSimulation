import numpy as np
from numba import jit

@jit(nopython=True)
def fast_sim_logic(spread, window, entry_z, exit_z):
    """Błyskawiczna pętla w Numbie - zero overheadu Pythona"""
    n = len(spread)
    positions = np.zeros(n)
    current_pos = 0
    
    # Pre-kalkulacja z-score w pętli
    for i in range(window, n):
        sub = spread[i-window:i]
        mu = np.mean(sub)
        std = np.std(sub)
        
        if std > 0:
            z = (spread[i] - mu) / std
            if current_pos == 0:
                if z < -entry_z: current_pos = 1
                elif z > entry_z: current_pos = -1
            elif current_pos == 1:
                if z > -exit_z: current_pos = 0
            elif current_pos == -1:
                if z < exit_z: current_pos = 0
        positions[i] = current_pos
    return positions

def run_pairs_trading_sim_fast(prices_a, prices_b, window, entry_z, exit_z, commission):
    # 1. FILTROWANIE NaN: Znajdujemy wiersze, gdzie obie spółki mają dane
    # mask to tablica True/False - True tylko tam, gdzie obie ceny są liczbami
    mask = ~np.isnan(prices_a) & ~np.isnan(prices_b)
    
    clean_a = prices_a[mask]
    clean_b = prices_b[mask]

    # 2. SPRAWDZENIE DŁUGOŚCI: Jeśli po usunięciu NaN zostało za mało danych, pomijamy parę
    if len(clean_a) < window + 10:
        return None

    # 3. OBLICZENIA na "czystych" danych
    spread = np.log(clean_a) - np.log(clean_b)
    
    # Obliczamy pozycje
    pos_raw = fast_sim_logic(spread, window, entry_z, exit_z)
    
    # Shift o 1 (wejście na następnej sesji po sygnale)
    pos = np.zeros_like(pos_raw)
    pos[1:] = pos_raw[:-1]
    
    # Koszty i zwroty
    # diff_pos wyłapuje każdą zmianę pozycji (prowizja przy wejściu i wyjściu)
    diff_pos = np.abs(np.diff(pos, prepend=0))
    
    # Log-return: pozycja * zmiana spreadu
    spread_diff = np.diff(spread, prepend=spread[0])
    returns = pos * spread_diff
    
    # Prowizja logarytmiczna: ln(1 - commission)
    # Dodajemy, bo logarytm z liczby < 1 jest ujemny
    log_comm = np.log(1 - commission)
    net_returns = returns + (diff_pos * log_comm)
    
    total_trades = np.sum(diff_pos)
    
    # Jeśli system ani razu nie wszedł w pozycję, zwracamy None
    if total_trades == 0:
        return None
        
    # Sumujemy logarytmiczne stopy zwrotu i zamieniamy na wynik procentowy
    final_return = np.exp(np.sum(net_returns)) - 1
    
    return final_return, int(total_trades)

def run_full_simulation(df, ticker_a, ticker_b, window, entry_z, exit_z, commission):
    # Pobranie czystych danych dla pary
    if ticker_a not in df.columns or ticker_b not in df.columns:
        return None
        
    pair_data = df[[ticker_a, ticker_b]].dropna().copy()
    if len(pair_data) < window + 10:
        return None

    prices_a = pair_data[ticker_a].values
    prices_b = pair_data[ticker_b].values
    
    # Obliczenia spreadu (logarytmiczne)
    spread = np.log(prices_a) - np.log(prices_b)
    
    # Obliczamy pozycje za pomocą Numby
    pos_raw = fast_sim_logic(spread, window, entry_z, exit_z)
    
    # Shift o 1 (wejście na następnej sesji po sygnale)
    pos = np.zeros_like(pos_raw)
    pos[1:] = pos_raw[:-1]
    
    # Koszty i zwroty
    spread_diff = np.diff(spread, prepend=spread[0])
    diff_pos = np.abs(np.diff(pos, prepend=0))
    
    log_returns = pos * spread_diff
    log_costs = diff_pos * np.log(1 - commission)
    
    # Zapisanie wyników do DataFrame (potrzebne do wykresów)
    pair_data['Strategy_Log_Ret'] = log_returns + log_costs
    pair_data['Cumulative_Return_Net'] = np.exp(pair_data['Strategy_Log_Ret'].cumsum())
    pair_data['Trades_Made'] = diff_pos
    pair_data['Position'] = pos  # <--- DODAJ TĘ LINIĘ
    pair_data['Transaction_Costs_Value'] = diff_pos * commission
    
    return pair_data