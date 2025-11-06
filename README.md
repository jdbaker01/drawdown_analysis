# Drawdown Analysis Dashboard

A comprehensive Python tool for analyzing stock drawdowns with professional HTML reporting and automated data management.

## Features

### 📊 Complete Drawdown Analysis
- **Max Drawdown Analysis**: Peak-to-trough analysis with dates and values
- **Local Drawdowns**: Individual drawdown events with duration and magnitude statistics
- **Cumulative Returns**: Investment performance tracking over time
- **Statistical Summaries**: Average, median, and distribution metrics

### 📈 Professional Visualizations
- Cumulative return charts
- Drawdown series area plots
- Local drawdowns histogram
- Duration vs magnitude analysis

### 📋 HTML Reporting
- Single comprehensive HTML report per symbol
- Summary statistics at the top for quick insights
- Interactive charts and detailed data tables
- Professional styling with responsive design

### 🔄 Automated Data Management
- Download 5 years of daily stock data from Yahoo Finance
- Automatic archiving of previous data files
- Clean CSV format with standard financial columns
- Built-in data validation and error handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jdbaker01/drawdown_analysis.git
cd drawdown_analysis
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Download Stock Data
```bash
# Download 5 years of data for any symbol
python download_stock_data.py AAPL
python download_stock_data.py MSFT
python download_stock_data.py TSLA
```

### Run Drawdown Analysis
```bash
# Analyze downloaded data
python drawdown_analysis.py data/AAPL-2020-11-08-2025-11-07.csv

# Specify custom output directory
python drawdown_analysis.py data/MSFT-2020-11-08-2025-11-07.csv --output-dir results
```

## Output Files

### Data Files
- `data/[SYMBOL]-[START-DATE]-[END-DATE].csv` - Latest stock data
- `data/archive/` - Previous versions of data files

### Analysis Reports
- `output/[SYMBOL]_drawdown_report.html` - Complete HTML report
- `output/[SYMBOL]_*.png` - Individual chart files

## Report Structure

The HTML report includes:

1. **Summary Information** - Analysis date, data period, record count
2. **Max Drawdown Analysis** - Peak/trough dates, values, current drawdown
3. **Local Drawdowns Summary** - Key statistics including:
   - Number of drawdown events
   - Average and median drawdown percentages
   - Average and median duration in days
4. **Detailed Statistics** - Full statistical breakdown
5. **Charts** - Visual analysis with professional styling
6. **Detailed Data Series** - Complete time series with drawdown period labels

## Key Metrics

- **Max Drawdown**: Largest peak-to-trough decline
- **Current Drawdown**: Present decline from recent peak
- **Local Drawdowns**: Individual drawdown events
- **Duration Analysis**: How long drawdowns typically last
- **Recovery Patterns**: Historical drawdown and recovery cycles

## Data Columns

### Input CSV Format
- Symbol, Time, Open, High, Low, Close, Change, %Change, Volume

### Analysis Output
- Date, CumulativeReturn, Draw Period # (identifies which drawdown period each day belongs to)

## Requirements

- Python 3.8+
- pandas
- matplotlib
- numpy
- yfinance
- tabulate

## License

MIT License - feel free to use and modify for your own analysis needs.

## Contributing

Contributions welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.