import argparse
import pandas as pd
import time
import sys
from datetime import datetime

#File used for development purposes
def stream_data(file_path, date_range, columns, delay, max_rows):
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    if columns:
        existing_cols = [c for c in columns if c in df.columns]
        if not existing_cols:
            print(f"Error: None of the specified columns {columns} exist in the file.")
            print(f"Available columns: {df.columns.tolist()}")
            return
        df = df[existing_cols]

    if date_range != "all":
        try:
            start_year, end_year = map(int, date_range.split('-'))
            
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
            else:
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    mask = (df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= end_year)
                    df = df[mask]
                else:
                    print("Warning: Cannot filter by date (no DateTimeIndex or Date column).")
        except ValueError:
            print("Date format error. Use format YYYY-YYYY, e.g., 2006-2009")

    if max_rows is not None:
        df = df.head(max_rows)
    for index, row in df.iterrows():
        row_str = " | ".join([f"{str(val):<10}" for val in row.values])
        print(f"{index}: {row_str}")
        
        if delay > 0:
            time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description="Parquet logger")
    
    parser.add_argument("path", type=str, help="File path")
    
    parser.add_argument("--cols", action="store_true", help="Display column names only and exit")
    
    parser.add_argument("columns", nargs="*", help="Specific columns to display")
    parser.add_argument("--range", type=str, default="all", help="Data range, ex. 2006-2009")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds between rows")
    parser.add_argument("--rows", type=int, default=None, help="Limit number of rows")
    
    args = parser.parse_args()

    if args.cols:
        try:
            df = pd.read_parquet(args.path)
            print(f"\nColumns in file {args.path}:")
            print("=" * 30)
            for i, col in enumerate(df.columns):
                print(f"{i+1}. {col}")
            print("=" * 30)
            return # Exit, do not start streaming
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    stream_data(args.path, args.range, args.columns, args.delay, args.rows)

if __name__ == "__main__":
    main()