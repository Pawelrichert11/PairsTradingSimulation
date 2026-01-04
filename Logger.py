import argparse
import pandas as pd
import time
import sys
from datetime import datetime

def stream_data(file_path, date_range, columns, delay, max_rows):
    """
    Reads a parquet file and streams it row by row with an optional delay.
    """
    try:
        # Load data
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # 1. Filter columns
    if columns:
        # Check if specified columns exist in the file
        existing_cols = [c for c in columns if c in df.columns]
        if not existing_cols:
            print(f"Error: None of the specified columns {columns} exist in the file.")
            print(f"Available columns: {df.columns.tolist()}")
            return
        df = df[existing_cols]

    # 2. Filter date range (simple handling of "YYYY-YYYY" strings)
    if date_range != "all":
        try:
            start_year, end_year = map(int, date_range.split('-'))
            
            # Check if index is a datetime index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
            else:
                # If index is not date, try parsing if 'Date' column exists
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    mask = (df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= end_year)
                    df = df[mask]
                else:
                    print("Warning: Cannot filter by date (no DateTimeIndex or Date column).")
        except ValueError:
            print("Date format error. Use format YYYY-YYYY, e.g., 2006-2009")

    # 3. Limit number of rows
    if max_rows is not None:
        df = df.head(max_rows)

    # 4. Stream display
    print(f"\nStarting data streaming from: {file_path}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 50)

    for index, row in df.iterrows():
        # Format row values for better readability
        row_str = " | ".join([f"{str(val):<10}" for val in row.values])
        print(f"{index}: {row_str}")
        
        if delay > 0:
            time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description="Parquet logger")
    
    parser.add_argument("path", type=str, help="File path")
    
    # Flag to display column names only
    parser.add_argument("--cols", action="store_true", help="Display column names only and exit")
    
    parser.add_argument("columns", nargs="*", help="Specific columns to display")
    parser.add_argument("--range", type=str, default="all", help="Data range, ex. 2006-2009")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds between rows")
    parser.add_argument("--rows", type=int, default=None, help="Limit number of rows")
    
    args = parser.parse_args()

    # Logic for --cols flag
    if args.cols:
        try:
            # Read only headers for performance
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

    # Normal execution
    stream_data(args.path, args.range, args.columns, args.delay, args.rows)

if __name__ == "__main__":
    main()