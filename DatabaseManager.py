import pandas as pd
from sqlalchemy import create_engine, text
import Config

class DatabaseManager:
    def __init__(self):
        try:
            self.engine = create_engine(Config.DB_CONNECTION_STRING)
            self._create_tables()
            print(f"Database connected: {Config.DB_PATH}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.engine = None

    def _create_tables(self):
        if self.engine is None:
            return
        queries = [
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                window_size INTEGER,
                std_entry FLOAT,
                total_pairs_tested INTEGER,
                avg_annual_return FLOAT,
                max_annual_return FLOAT,
                avg_sharpe FLOAT,
                avg_num_trades FLOAT,
                profitable_pairs_pct FLOAT
            )
            """
        ]
        
        try:
            with self.engine.connect() as conn:
                for query in queries:
                    conn.execute(text(query))
                conn.commit()
        except Exception as e:
            print(f"Error creating tables: {e}")

    def save_simulation_session(self, metrics):
        if self.engine is None:
            return None

        try:
            df_session = pd.DataFrame([metrics])
            df_session.to_sql('sessions', self.engine, if_exists='append', index=False)
            
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT last_insert_rowid()"))
                session_id = result.scalar()
                return session_id
        except Exception as e:
            print(f"Error saving simulation session: {e}")
            return None