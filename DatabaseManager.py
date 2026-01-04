import pandas as pd
from sqlalchemy import create_engine, text
import Config

class DatabaseManager:
    """
    Manages SQLite database interactions using SQLAlchemy.
    """
    def __init__(self):
        """
        Initializes the database engine using the connection string from Config.
        """
        try:
            self.engine = create_engine(Config.DB_CONNECTION_STRING)
            print(f"Database connected: {Config.DB_PATH}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.engine = None

    def save_tickers_metadata(self, df):
        """
        Saves ticker metadata (volatility class, average volume) to the database.
        Replaces the table if it exists.
        """
        if self.engine is None or df.empty:
            return

        try:
            table_name = 'tickers_metadata'
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            print(f"Saved {len(df)} rows to table '{table_name}'.")
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def get_tickers_metadata(self):
        """
        Retrieves ticker metadata from the database.
        """
        if self.engine is None:
            return pd.DataFrame()

        try:
            query = "SELECT * FROM tickers_metadata"
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return pd.DataFrame()

    def execute_query(self, query):
        """
        Executes a raw SQL query (for debugging or advanced usage).
        """
        if self.engine is None:
            return

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                return result
        except Exception as e:
            print(f"Error executing query: {e}")