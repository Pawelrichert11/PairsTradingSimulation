import pandas as pd
import argparse
import os
import time

def stream_data(file_path, date_range="all", selected_columns=None, delay=0, max_rows=None):
    if not os.path.exists(file_path):
        print(f"❌ Plik nie istnieje: {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # 1. Odczyt pliku
        if ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif ext == '.csv':
            df = pd.read_csv(file_path, parse_dates=['Date'])
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
        else:
            return

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df.sort_index(inplace=True)

        # 2. Filtrowanie po dacie (teraz opcjonalne)
        if date_range and date_range.lower() != 'all':
            try:
                if '-' in date_range:
                    start_year, end_year = date_range.split('-')
                    df = df.loc[start_year:end_year]
                else:
                    df = df.loc[date_range]
            except Exception as e:
                print(f"⚠️  Nieprawidłowy zakres dat: {date_range}. Pokazuję całość.")

        # 3. Wybór kolumn i usuwanie NaN (tylko w nich)
        show_index = False
        if selected_columns:
            if any(c.lower() == 'date' for c in selected_columns):
                show_index = True
                selected_columns = [c for c in selected_columns if c.lower() != 'date']

            existing_cols = [c for c in selected_columns if c in df.columns]
            
            if existing_cols:
                df = df[existing_cols]
                df = df.dropna()
            elif not show_index:
                print("❌ Nie znaleziono wybranych kolumn.")
                return

        # 4. Limit wierszy
        if max_rows:
            df = df.head(max_rows)

        print(f"🚀 Plik: {os.path.basename(file_path)} | Zakres: {date_range} | Wierszy: {len(df)}")
        print("-" * 50)

        # 5. Szerokość kolumn
        col_widths = {}
        index_name = str(df.index.name) if df.index.name else "DATE"
        sample_index_strings = df.index[:10].astype(str)
        sample_index_width = sample_index_strings.map(len).max() if not sample_index_strings.empty else 10
        col_widths['__index__'] = max(len(index_name), sample_index_width) + 2

        sample_cols = df.head(10).astype(str)
        for col in df.columns:
            col_widths[col] = max(len(str(col)), sample_cols[col].map(len).max()) + 2

        # 6. Nagłówek
        header = ""
        if show_index or True: # Zawsze rezerwujemy miejsce na datę dla porządku
            idx_name = index_name.upper()
            header += f"{idx_name:<{col_widths['__index__']}}"
        header += "".join([f"{str(col).upper():<{col_widths[col]}}" for col in df.columns])
        
        print(header)
        print("-" * len(header))

        # 7. Wypisywanie
        for index, row in df.iterrows():
            idx_val = str(index.date()) if hasattr(index, 'date') else str(index)
            line = f"{idx_val:<{col_widths['__index__']}}"
            line += "".join([f"{str(val):<{col_widths[col]}}" for col, val in row.items()])
            print(line)
            
            if delay > 0:
                time.sleep(delay)
                
        print("-" * len(header))
        print("✅ Koniec danych.")

    except Exception as e:
        print(f"❌ Błąd: {e}")

def main():
    parser = argparse.ArgumentParser(description="Logger giełdowy.")
    parser.add_argument("path", type=str, help="Ścieżka do pliku")
    parser.add_argument("columns", nargs="*", help="Nazwy kolumn")
    
    # Przenosimy date_range do flagi --range, domyślnie ustawionej na 'all'
    parser.add_argument("--range", type=str, default="all", help="Zakres dat, np. 2006-2009")
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--rows", type=int, default=None)
    
    args = parser.parse_args()
    stream_data(args.path, args.range, args.columns, args.delay, args.rows)

if __name__ == "__main__":
    main()