import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "trading_metadata.db" # Nowa nazwa bazy, lekka

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._prepare_db()

    def _prepare_db(self):
        # Tabela 1: Metadane o spółkach (Sektor, Ryzyko itp.)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tickers_info (
                ticker TEXT PRIMARY KEY,
                volatility_class TEXT,
                avg_volume REAL
            )
        """)
        
        # Tabela 2: Wyniki symulacji
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS simulation_results (
                ticker_a TEXT,
                ticker_b TEXT,
                wynik_netto REAL,
                liczba_transakcji INTEGER,
                korelacja REAL,
                FOREIGN KEY(ticker_a) REFERENCES tickers_info(ticker)
            )
        """)
        self.conn.commit()

    # --- 1. STORE DATA (Zapisywanie metadanych) ---
    def save_tickers_metadata(self, df_metadata):
        """Zapisuje informacje o spółkach (nie ceny!)."""
        df_metadata.to_sql('tickers_info', self.conn, if_exists='replace', index=False)
        print("✅ Zapisano metadane spółek do SQL.")

    def get_all_simulation_results_sorted(self):
            """Pobiera wszystkie wyniki symulacji posortowane malejąco po wyniku netto."""
            query = """
                SELECT ticker_a, ticker_b, wynik_netto, liczba_transakcji, korelacja 
                FROM simulation_results 
                ORDER BY wynik_netto DESC
            """
            return pd.read_sql(query, self.conn)
    
    def save_simulation_results(self, df_results):
        df_results.to_sql('simulation_results', self.conn, if_exists='replace', index=False)
        print("✅ Zapisano wyniki symulacji do SQL.")

    # --- 2. JOIN & FILTER (Demonstracja wymogów) ---
    def get_joined_results(self, min_trades=5):
        """
        DEMONSTRACJA JOIN: Łączy wyniki symulacji z informacją o zmienności spółki A.
        """
        query = f"""
            SELECT 
                r.ticker_a, 
                r.ticker_b, 
                r.wynik_netto, 
                r.liczba_transakcji,
                t.volatility_class -- Kolumna z drugiej tabeli
            FROM simulation_results r
            JOIN tickers_info t ON r.ticker_a = t.ticker
            WHERE r.liczba_transakcji >= {min_trades}
            ORDER BY r.wynik_netto DESC
        """
        return pd.read_sql(query, self.conn)

    # --- 3. AGGREGATION (Demonstracja Group By) ---
    def get_stats_by_volatility(self):
        """
        DEMONSTRACJA AGGREGATION: Średni wynik strategii w zależności od klasy zmienności.
        """
        query = """
            SELECT 
                t.volatility_class, 
                COUNT(*) as count_pairs,
                AVG(r.wynik_netto) as avg_return,
                MAX(r.wynik_netto) as max_return
            FROM simulation_results r
            JOIN tickers_info t ON r.ticker_a = t.ticker
            GROUP BY t.volatility_class
        """
        return pd.read_sql(query, self.conn)