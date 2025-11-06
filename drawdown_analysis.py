import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse
import glob


# --- 1. Helper Functions ---

def find_latest_csv_for_symbol(symbol):
    """
    Find the latest CSV file for a given symbol in the data folder.
    
    Args:
        symbol (str): Stock symbol to search for
        
    Returns:
        str: Path to the latest CSV file, or None if not found
    """
    # Convert symbol to uppercase for consistency
    symbol = symbol.upper()
    
    # Search for CSV files matching the symbol pattern
    pattern = os.path.join('data', f'{symbol}-*.csv')
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        return None
    
    # Sort by modification time (most recent first)
    matching_files.sort(key=os.path.getmtime, reverse=True)
    
    return matching_files[0]


def resolve_input_file(input_arg):
    """
    Resolve the input argument to a CSV file path.
    Can handle either a symbol or a full file path.
    
    Args:
        input_arg (str): Either a stock symbol or a file path
        
    Returns:
        str: Resolved file path
        
    Raises:
        FileNotFoundError: If no suitable file is found
    """
    # Check if it's already a valid file path
    if os.path.isfile(input_arg):
        return input_arg
    
    # Check if it's a file path without extension
    if input_arg.endswith('.csv') and os.path.isfile(input_arg):
        return input_arg
    
    # Treat as symbol and search for latest CSV
    latest_file = find_latest_csv_for_symbol(input_arg)
    
    if latest_file:
        print(f"Found latest file for {input_arg.upper()}: {latest_file}")
        return latest_file
    else:
        # Provide helpful error message
        available_symbols = []
        for csv_file in glob.glob('data/*.csv'):
            filename = os.path.basename(csv_file)
            if '-' in filename:
                symbol = filename.split('-')[0]
                if symbol not in available_symbols:
                    available_symbols.append(symbol)
        
        error_msg = f"No CSV file found for symbol '{input_arg.upper()}'"
        if available_symbols:
            error_msg += f"\nAvailable symbols in data folder: {', '.join(sorted(available_symbols))}"
        else:
            error_msg += "\nNo CSV files found in data folder. Use download_stock_data.py to download data first."
        
        raise FileNotFoundError(error_msg)


# --- 2. Analysis Functions ---

def analyze_symbol(symbol, symbol_df, output_dir):
    """
    Performs all financial analyses for a single symbol.
    Generates a comprehensive HTML report and saves graph files to the output_dir.
    """
    print(f"\n{'=' * 60}")
    print(f"Analyzing Symbol: {symbol}")
    print(f"{'=' * 60}")
    
    # Initialize HTML report content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drawdown Analysis Report - {symbol}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; border-bottom: 2px solid #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .chart {{ text-align: center; margin: 20px 0; }}
            .chart img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Drawdown Analysis Report for {symbol}</h1>
        <div class="summary">
            <p><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%m/%d/%Y')}</p>
            <p><strong>Data Period:</strong> {symbol_df['Date'].min().strftime('%m/%d/%Y')} to {symbol_df['Date'].max().strftime('%m/%d/%Y')}</p>
            <p><strong>Total Records:</strong> {len(symbol_df)}</p>
        </div>
    """

    # --- Data Preparation for this Symbol ---

    # Ensure data is sorted by date, which is crucial for all calculations
    symbol_df = symbol_df.sort_values(by='Date')

    # Get the 'Close' price column as a Series, with 'Date' as the index
    prices = symbol_df.set_index('Date')['Close']

    # Data Quality Check: Ensure we have valid price data to analyze
    if prices.empty or prices.isna().all():
        print(f"No valid price data for {symbol}. Skipping analysis.")
        return

    # --- Clean filename ---
    # Create a "safe" version of the symbol for filenames
    # Replaces special characters like '/' or ':' with '_'
    safe_symbol = re.sub(r'[^a-zA-Z0-9_.-]', '_', symbol)

    # --- Analysis 1: Cumulative Return ---
    print("\n--- 1. Cumulative Return ---")

    # Calculate daily returns, ignoring the first NaN
    # fill_method=None is specified to silence a FutureWarning
    daily_returns = prices.pct_change(fill_method=None)

    # The first daily_return is always NaN, fill it with 0 for cumulative product
    daily_returns.iloc[0] = 0

    # Calculate cumulative product of (1 + daily returns)
    cumulative_return = (1 + daily_returns).cumprod()
    # Name the Series for a clean markdown header
    cumulative_return.name = "CumulativeReturn"
    cumulative_return.index.name = "Date"

    # Store cumulative return data for later use in HTML report
    cumulative_return_data = cumulative_return

    # Graph: Plot Cumulative Return
    plt.figure(figsize=(10, 6))
    cumulative_return.plot(title=f'Cumulative Return for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (1.0 = Initial Investment)')
    plt.grid(True)
    plt.tight_layout()
    graph_path = os.path.join(output_dir, f'{safe_symbol}_cumulative_return.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Graph saved to: {graph_path}")
    
    # Add charts section to HTML
    html_content += f"""
        <h2>Charts</h2>
        <div class="chart">
            <h3>Cumulative Return</h3>
            <img src="{safe_symbol}_cumulative_return.png" alt="Cumulative Return Chart">
        </div>
    """

    # --- Analysis 2: Max Drawdown ---
    print("\n--- 2. Max Drawdown ---")

    # Calculate the running maximum (peak) of the price series
    running_max = prices.cummax()

    # This is the percentage drop from the last peak
    drawdown_series = (prices - running_max) / running_max

    # --- Get current drawdown ---
    current_drawdown = drawdown_series.iloc[-1]

    # Find the highest price (peak) in the entire series
    peak_date = prices.idxmax()
    peak_value = prices.max()
    
    # Find the lowest price (trough) in the entire series
    trough_date = prices.idxmin()
    trough_value = prices.min()
    
    # Calculate the max drawdown as the worst drawdown in the series
    max_drawdown = drawdown_series.min()

    # Add Max Drawdown section to HTML report
    print("Adding Max Drawdown to report...")

    # Create a DataFrame for the HTML table
    max_dd_data = {
        "Metric": ["Max Drawdown", "Peak Date", "Peak Price", "Trough Date", "Trough Price", "Current DD"],
        "Value": [
            f"{max_drawdown:.4%}",
            peak_date.strftime('%m/%d/%Y'),
            f"{peak_value:.2f}",
            trough_date.strftime('%m/%d/%Y'),
            f"{trough_value:.2f}",
            f"{current_drawdown:.4%}"
        ]
    }
    max_dd_df = pd.DataFrame(max_dd_data).set_index("Metric")
    
    html_content += f"""
        <h2>Max Drawdown Analysis</h2>
        {max_dd_df.to_html(table_id='max_drawdown', escape=False)}
    """
    
    # Store max drawdown info for later use in local drawdowns section
    max_dd_info = {
        'max_drawdown': max_drawdown,
        'current_drawdown': current_drawdown
    }

    # Graph: Plot Drawdown Series
    plt.figure(figsize=(10, 6))
    drawdown_series.plot(kind='area', color='red', alpha=0.3, title=f'Drawdown Series for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.tight_layout()
    graph_path = os.path.join(output_dir, f'{safe_symbol}_drawdown_series.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Graph saved to: {graph_path}")
    
    html_content += f"""
        <div class="chart">
            <h3>Drawdown Series</h3>
            <img src="{safe_symbol}_drawdown_series.png" alt="Drawdown Series Chart">
        </div>
    """

    # --- Analysis 3: Local Drawdowns Histogram (Corrected Logic) ---
    print("\n--- 3. Local Drawdowns Histogram ---")

    # A drawdown "event" is a continuous period of being below a peak.
    # We can find these by looking at the drawdown_series.

    # 1. Identify periods *not* in a drawdown (i.e., at a new peak, where drawdown is 0)
    not_in_drawdown = (drawdown_series == 0)

    # 2. Find the *start* of each drawdown event. This is when the series
    #    goes from 0 (at peak) to non-zero (below peak).
    #    We shift `not_in_drawdown` to compare t-1 (yesterday) with t (today).
    #    A start occurs when: yesterday was 0 AND today is not 0.
    drawdown_starts = (not_in_drawdown.shift(1, fill_value=True) & ~not_in_drawdown)

    # 3. Get the dates (index) where these starts occur
    drawdown_start_dates = drawdown_series[drawdown_starts].index

    if drawdown_start_dates.empty:
        print("No drawdown events found (e.g., price always went up).")
        return

    # 4. Now find the end of the previous drawdown event.
    #    This is the day *before* the start of the *next* drawdown event.
    #    We use the date right before the *next* start date as the end_date.

    # Get all dates in the series
    all_dates = prices.index

    local_drawdowns_list = []
    drawdown_durations_list = []

    for i, start_date in enumerate(drawdown_start_dates):
        # Determine the end date for this drawdown period
        if i < len(drawdown_start_dates) - 1:
            # End date is the day *before* the next drawdown starts
            next_start_date = drawdown_start_dates[i + 1]
            end_date_index = all_dates.get_loc(next_start_date) - 1
            end_date = all_dates[end_date_index]
        else:
            # For the last drawdown, it runs to the end of the series
            end_date = all_dates[-1]

        # Get the drawdown series for just this period
        event_period = drawdown_series.loc[start_date:end_date]

        if not event_period.empty:
            # The "max local drawdown" is the minimum value in this period
            local_drawdowns_list.append(event_period.min())
            
            # Calculate duration in days for this drawdown period
            duration_days = len(event_period)
            drawdown_durations_list.append(duration_days)

    if not local_drawdowns_list:
        print("Not enough local drawdowns to calculate histogram.")
        return

    # Convert to pandas Series for easier analysis
    drawdown_table = pd.Series(local_drawdowns_list)
    duration_table = pd.Series(drawdown_durations_list)

    # --- Calculate median, max, and count of local drawdowns ---
    median_local_drawdown = drawdown_table.median()
    max_local_drawdown = drawdown_table.min()  # Max drawdown is the smallest (most negative) number
    num_local_drawdowns = drawdown_table.count()
    
    # --- Calculate duration statistics ---
    avg_duration_days = duration_table.mean()
    median_duration_days = duration_table.median()
    
    # --- Calculate drawdown percentage statistics ---
    avg_drawdown_pct = drawdown_table.mean()

    # Add Local Drawdowns section to HTML report (before charts)
    print("Adding Local Drawdowns summary to report...")

    # Create comprehensive summary table with all requested metrics
    summary_data = {
        "Metric": [
            "Number of Drawdowns", 
            "Average Drawdown %", 
            "Median Drawdown %", 
            "Max Drawdown %",
            "Average Days in Drawdown", 
            "Median Days in Drawdown"
        ],
        "Value": [
            f"{num_local_drawdowns}",
            f"{avg_drawdown_pct:.4%}",
            f"{median_local_drawdown:.4%}",
            f"{max_local_drawdown:.4%}",
            f"{avg_duration_days:.1f}",
            f"{median_duration_days:.1f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data).set_index("Metric")
    
    # Create enhanced statistics table with additional percentiles
    # Convert to positive percentages for better interpretation
    drawdown_table_positive = drawdown_table.abs() * 100  # Convert to positive percentages
    
    # Calculate standard describe() statistics
    describe_stats = drawdown_table_positive.describe()
    
    # Calculate additional percentiles
    p10 = drawdown_table_positive.quantile(0.10)
    p90 = drawdown_table_positive.quantile(0.90)
    
    # Create enhanced statistics dictionary
    enhanced_stats = {
        'count': describe_stats['count'],
        'mean': describe_stats['mean'],
        'std': describe_stats['std'],
        'min': describe_stats['min'],
        '10%': p10,
        '25%': describe_stats['25%'],
        '50%': describe_stats['50%'],
        '75%': describe_stats['75%'],
        '90%': p90,
        'max': describe_stats['max']
    }
    
    # Create DataFrame with enhanced statistics
    describe_df = pd.DataFrame(list(enhanced_stats.items()), columns=['Statistic', 'Value (%)'])
    describe_df = describe_df.set_index('Statistic')
    
    # Format the values as percentages (except count)
    describe_df['Value (%)'] = describe_df['Value (%)'].apply(
        lambda x: f"{int(x)}" if describe_df.index[describe_df['Value (%)'] == x].tolist()[0] == 'count' 
        else f"{x:.2f}%"
    )
    
    html_content += f"""
        <h2>Local Drawdowns Summary</h2>
        {summary_df.to_html(table_id='local_drawdowns_summary', escape=False)}
        
        <h2>Local Drawdowns Statistics</h2>
        {describe_df.to_html(table_id='local_drawdowns_stats', escape=False)}
    """

    # Graph: Plot Histogram of Local Drawdowns
    plt.figure(figsize=(10, 6))
    drawdown_table.plot(kind='hist', bins=20, edgecolor='black', alpha=0.7)

    # Add Red line for current drawdown
    plt.axvline(current_drawdown, color='red', linestyle='--', linewidth=2,
                label=f'Current DD ({current_drawdown:.4%})')
    plt.legend()

    plt.title(f'Histogram of Local Drawdowns for {symbol}')
    plt.xlabel('Drawdown (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    graph_path = os.path.join(output_dir, f'{safe_symbol}_local_drawdowns_hist.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Graph saved to: {graph_path}")
    
    html_content += f"""
        <div class="chart">
            <h3>Local Drawdowns Histogram</h3>
            <img src="{safe_symbol}_local_drawdowns_hist.png" alt="Local Drawdowns Histogram">
        </div>
    """
    

    
    # Graph: Histogram showing average duration by drawdown percentage bins
    plt.figure(figsize=(12, 8))
    
    # Convert to positive percentages for binning
    drawdown_pct_positive = [-x * 100 for x in local_drawdowns_list]
    
    # Create bins for drawdown percentages
    n_bins = min(10, len(set(drawdown_pct_positive)))  # Adaptive binning
    
    # Create a DataFrame for easier manipulation
    dd_data = pd.DataFrame({
        'drawdown_pct': drawdown_pct_positive,
        'duration': drawdown_durations_list
    })
    
    # Create bins and calculate average duration for each bin
    dd_data['dd_bin'] = pd.cut(dd_data['drawdown_pct'], bins=n_bins)
    bin_stats = dd_data.groupby('dd_bin')['duration'].agg(['mean', 'count', 'std']).reset_index()
    
    # Get bin centers for plotting
    bin_centers = [interval.mid for interval in bin_stats['dd_bin']]
    
    # Create bar chart showing average duration by drawdown percentage range
    bars = plt.bar(bin_centers, bin_stats['mean'], 
                   width=[interval.length * 0.8 for interval in bin_stats['dd_bin']], 
                   alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add error bars showing standard deviation
    plt.errorbar(bin_centers, bin_stats['mean'], yerr=bin_stats['std'], 
                fmt='none', color='black', capsize=5, alpha=0.7)
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, bin_stats['count'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bin_stats['std'].iloc[i] + 1, 
                f'n={int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Drawdown Magnitude (%) - Bin Centers')
    plt.ylabel('Average Duration (Days)')
    plt.title(f'Average Drawdown Duration by Magnitude Range for {symbol}')
    plt.grid(True, alpha=0.3)
    
    # Add current drawdown indicator if applicable
    if current_drawdown < 0:
        current_dd_pct = -current_drawdown * 100
        plt.axvline(current_dd_pct, color='red', linestyle='--', linewidth=2, 
                   label=f'Current DD ({current_drawdown:.2%})')
        plt.legend()
    
    plt.tight_layout()
    duration_hist_path = os.path.join(output_dir, f'{safe_symbol}_duration_by_magnitude.png')
    plt.savefig(duration_hist_path)
    plt.close()
    print(f"Duration histogram saved to: {duration_hist_path}")
    
    html_content += f"""
        <div class="chart">
            <h3>Average Duration by Drawdown Magnitude</h3>
            <img src="{safe_symbol}_duration_by_magnitude.png" alt="Average Duration by Drawdown Magnitude">
        </div>
        
        <h2>Detailed Data Series</h2>
        <h3>Cumulative Return Series</h3>
    """
    
    # Create enhanced cumulative return table with drawdown periods
    cumulative_return_df = cumulative_return_data.to_frame()
    
    # Calculate drawdown periods for each day
    drawdown_periods = pd.Series(index=cumulative_return_df.index, dtype='object')
    
    # Use the same logic as local drawdowns to identify periods
    not_in_drawdown = (drawdown_series == 0)
    drawdown_starts = (not_in_drawdown.shift(1, fill_value=True) & ~not_in_drawdown)
    drawdown_start_dates = drawdown_series[drawdown_starts].index
    
    if not drawdown_start_dates.empty:
        period_number = 1
        
        for i, start_date in enumerate(drawdown_start_dates):
            # Determine the end date for this drawdown period
            if i < len(drawdown_start_dates) - 1:
                # End date is the day *before* the next drawdown starts
                next_start_date = drawdown_start_dates[i + 1]
                end_date_index = prices.index.get_loc(next_start_date) - 1
                end_date = prices.index[end_date_index]
            else:
                # For the last drawdown, it runs to the end of the series
                end_date = prices.index[-1]
            
            # Mark all days in this drawdown period
            period_mask = (cumulative_return_df.index >= start_date) & (cumulative_return_df.index <= end_date)
            drawdown_periods.loc[period_mask] = period_number
            period_number += 1
    
    # Add the drawdown period column, replacing NaN with empty strings
    cumulative_return_df['Draw Period #'] = drawdown_periods.fillna('')
    
    html_content += f"""
        {cumulative_return_df.to_html(table_id='cumulative_return', escape=False)}
        
        </body>
        </html>
    """
    
    # Save the complete HTML report
    print("Saving complete HTML report...")
    report_path = os.path.join(output_dir, f'{safe_symbol}_drawdown_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Complete HTML report saved to: {report_path}")


# --- 2. Main Execution ---

def main():
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(
        description='Perform drawdown analysis on stock data',
        epilog='Examples:\n'
               '  python drawdown_analysis.py AAPL\n'
               '  python drawdown_analysis.py data/AAPL-2020-11-08-2025-11-07.csv\n'
               '  python drawdown_analysis.py MSFT --output-dir results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input', help='Stock symbol (e.g., AAPL) or path to CSV file')
    parser.add_argument('--output-dir', '-o', default='output', 
                       help='Output directory for results (default: output)')
    
    args = parser.parse_args()
    
    # --- Define Output Directory ---
    OUTPUT_DIR = args.output_dir
    
    # --- Resolve Input File ---
    try:
        data_file = resolve_input_file(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- Data Loading ---
    try:
        # Load the CSV file from the specified path
        df = pd.read_csv(data_file)
        print(f"Successfully loaded data from '{data_file}'")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_file}'")
        print("Please make sure the file exists in that location.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Create Output Directory ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output will be saved to '{OUTPUT_DIR}' directory.")
    except Exception as e:
        print(f"Error creating output directory '{OUTPUT_DIR}': {e}")
        return

    # --- Data Preparation ---
    # Convert 'Time' (or 'Date') column to datetime objects
    # This is crucial for time-series analysis
    if 'Time' in df.columns:
        df['Date'] = pd.to_datetime(df['Time'])
        print("Using 'Time' column for date information.")
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        print("Using 'Date' column for date information.")
    else:
        print("Error: Neither 'Time' nor 'Date' column found in the data.")
        return

    # Drop any rows where the date could not be parsed (NaT)
    original_rows = len(df)
    df = df.dropna(subset=['Date'])
    new_rows = len(df)
    if new_rows < original_rows:
        print(f"Dropped {original_rows - new_rows} rows with invalid dates.")

    # Check for required columns
    if 'Symbol' not in df.columns or 'Close' not in df.columns:
        print("Error: Data must contain 'Symbol' and 'Close' columns.")
        return

    # Get a list of all unique symbols in the data
    unique_symbols = df['Symbol'].unique()
    print(f"Found {len(unique_symbols)} symbols: {unique_symbols}")

    # --- Analysis Loop ---
    # Loop through each symbol and run the analysis
    for symbol in unique_symbols:
        # Create a separate DataFrame for each symbol
        symbol_df = df[df['Symbol'] == symbol].copy()

        # Pass this symbol-specific data and output dir to the analysis function
        analyze_symbol(symbol, symbol_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()