#!/usr/bin/env python3
"""
Stock data downloader script using yfinance.
Downloads the last 5 years of daily data for a given symbol.
"""

import yfinance as yf
import pandas as pd
import argparse
from datetime import datetime, timedelta
import os
import glob
import shutil


def archive_existing_files(symbol):
    """
    Move any existing CSV files for the given symbol to an archive folder.
    
    Args:
        symbol (str): Stock symbol to archive files for
    """
    # Create archive directory if it doesn't exist
    archive_dir = os.path.join('data', 'archive')
    os.makedirs(archive_dir, exist_ok=True)
    
    # Find all existing CSV files for this symbol
    pattern = os.path.join('data', f'{symbol}-*.csv')
    existing_files = glob.glob(pattern)
    
    if existing_files:
        print(f"Found {len(existing_files)} existing file(s) for {symbol}:")
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            archive_path = os.path.join(archive_dir, filename)
            
            # If archive file already exists, add timestamp to make it unique
            if os.path.exists(archive_path):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                name, ext = os.path.splitext(filename)
                archive_path = os.path.join(archive_dir, f'{name}_archived_{timestamp}{ext}')
            
            # Move the file to archive
            shutil.move(file_path, archive_path)
            print(f"  Archived: {filename} -> archive/{os.path.basename(archive_path)}")
    else:
        print(f"No existing files found for {symbol}")


def download_stock_data(symbol):
    """
    Download 5 years of daily stock data for the given symbol.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
        str: Path to the saved CSV file
    """
    # Calculate date range (last 5 years)
    end_date = datetime.now() + timedelta(days=1)  # Add 1 day to include today
    start_date = end_date - timedelta(days=5*365)
    
    # Format dates for filename
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Archive any existing files for this symbol first
    archive_existing_files(symbol)
    
    print(f"Downloading {symbol} data from {start_str} to {end_str}...")
    
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        
        if data.empty:
            print(f"No data found for symbol: {symbol}")
            return None
        
        # Transform data to match required columns
        # yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        transformed_data = pd.DataFrame({
            'Symbol': symbol,
            'Time': data.index.strftime('%Y-%m-%d'),
            'Open': data['Open'].values,
            'High': data['High'].values,
            'Low': data['Low'].values,
            'Close': data['Close'].values,  # Official closing price
            'Change': data['Close'].diff().values,
            '%Change': (data['Close'].pct_change() * 100).round(2).values,
            'Volume': data['Volume'].values
        })
        
        # Fill NaN values for first row (no previous day to compare)
        transformed_data = transformed_data.fillna({'Change': 0, '%Change': 0})
        
        # Create filename
        filename = f"{symbol}-{start_str}-{end_str}.csv"
        filepath = os.path.join('data', filename)
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV with exact column order
        column_order = ['Symbol', 'Time', 'Open', 'High', 'Low', 'Close', 'Change', '%Change', 'Volume']
        transformed_data[column_order].to_csv(filepath, index=False)
        
        print(f"Data saved to: {filepath}")
        print(f"Records: {len(data)} days")
        print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        return filepath
        
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download stock data using yfinance')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, MSFT)')
    
    args = parser.parse_args()
    
    # Convert symbol to uppercase
    symbol = args.symbol.upper()
    
    # Download data
    filepath = download_stock_data(symbol)
    
    if filepath:
        print(f"\nSuccess! Data downloaded to: {filepath}")
    else:
        print(f"\nFailed to download data for {symbol}")


if __name__ == "__main__":
    main()