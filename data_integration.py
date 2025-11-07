"""
Data integration layer for the Stock Drawdown Dashboard.
Provides wrapper functions to integrate with existing modules.
"""

import os
import re
import glob
import logging
import sys
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
import pandas as pd

# Import existing modules
from download_stock_data import download_stock_data
from drawdown_analysis import find_latest_csv_for_symbol, analyze_symbol


# Configure logging
def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration for the data integration layer.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dashboard.log', mode='a')
        ]
    )


# Error handling classes
class DataIntegrationError(Exception):
    """Base exception for data integration errors."""
    pass


class SymbolValidationError(DataIntegrationError):
    """Exception raised for invalid stock symbols."""
    pass


class DataDownloadError(DataIntegrationError):
    """Exception raised for data download failures."""
    pass


class AnalysisError(DataIntegrationError):
    """Exception raised for analysis processing failures."""
    pass


def get_user_friendly_error_message(error: Exception, symbol: str = "") -> str:
    """
    Convert technical errors into user-friendly messages with actionable guidance.
    
    Args:
        error (Exception): The original error
        symbol (str): Stock symbol being processed
        
    Returns:
        str: User-friendly error message with guidance
    """
    error_str = str(error).lower()
    
    # Network-related errors
    if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'unreachable']):
        return (
            f"Network connection issue while processing {symbol}.\n\n"
            "Troubleshooting steps:\n"
            "• Check your internet connection\n"
            "• Try again in a few minutes\n"
            "• If the problem persists, the data provider may be temporarily unavailable"
        )
    
    # Invalid symbol errors
    if any(keyword in error_str for keyword in ['no data', 'not found', 'invalid symbol', 'delisted']):
        return (
            f"Unable to find data for symbol '{symbol}'.\n\n"
            "This could mean:\n"
            "• The symbol is incorrect or misspelled\n"
            "• The stock is delisted or no longer traded\n"
            "• The symbol format is not recognized\n\n"
            "Please verify the symbol and try again with a valid ticker (e.g., AAPL, MSFT, SPY)"
        )
    
    # File system errors
    if any(keyword in error_str for keyword in ['permission', 'access denied', 'file not found']):
        return (
            f"File system access issue while processing {symbol}.\n\n"
            "Troubleshooting steps:\n"
            "• Ensure you have write permissions in the current directory\n"
            "• Check that the data and output directories are accessible\n"
            "• Try running from a different location if needed"
        )
    
    # Data format errors
    if any(keyword in error_str for keyword in ['parse', 'format', 'column', 'csv']):
        return (
            f"Data format issue while processing {symbol}.\n\n"
            "This could indicate:\n"
            "• Corrupted data file\n"
            "• Unexpected data format from provider\n"
            "• Missing required columns\n\n"
            "Try downloading fresh data for this symbol"
        )
    
    # Memory/resource errors
    if any(keyword in error_str for keyword in ['memory', 'resource', 'space']):
        return (
            f"System resource issue while processing {symbol}.\n\n"
            "Troubleshooting steps:\n"
            "• Close other applications to free up memory\n"
            "• Ensure sufficient disk space is available\n"
            "• Try processing a shorter time period of data"
        )
    
    # Rate limiting errors
    if any(keyword in error_str for keyword in ['rate limit', 'too many requests', 'quota']):
        return (
            f"API rate limit reached while downloading {symbol}.\n\n"
            "Please wait a few minutes before trying again.\n"
            "The data provider limits how frequently data can be requested."
        )
    
    # Generic error with technical details
    return (
        f"An unexpected error occurred while processing {symbol}.\n\n"
        f"Technical details: {str(error)}\n\n"
        "If this problem persists, please check the application logs for more information."
    )


def validate_stock_symbol(symbol: str) -> Tuple[bool, str]:
    """
    Validates stock symbol format and basic requirements.
    
    Args:
        symbol (str): Stock symbol to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not symbol:
        return False, "Stock symbol cannot be empty"
    
    # Convert to uppercase for consistency
    symbol = symbol.upper().strip()
    
    # Basic format validation - alphanumeric characters, dots, and hyphens
    if not re.match(r'^[A-Z0-9.-]+$', symbol):
        return False, "Stock symbol can only contain letters, numbers, dots, and hyphens"
    
    # Length validation - most symbols are 1-5 characters
    if len(symbol) < 1 or len(symbol) > 10:
        return False, "Stock symbol must be between 1 and 10 characters"
    
    # Additional format checks
    if symbol.startswith('.') or symbol.endswith('.'):
        return False, "Stock symbol cannot start or end with a dot"
    
    if symbol.startswith('-') or symbol.endswith('-'):
        return False, "Stock symbol cannot start or end with a hyphen"
    
    return True, ""


def ensure_stock_data(symbol: str) -> Tuple[bool, str, str]:
    """
    Ensures stock data exists for the given symbol.
    Checks for existing data and triggers download if needed.
    
    Args:
        symbol (str): Stock symbol to ensure data for
        
    Returns:
        tuple: (success, file_path, message)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate symbol first
        is_valid, error_msg = validate_stock_symbol(symbol)
        if not is_valid:
            logger.warning(f"Symbol validation failed for '{symbol}': {error_msg}")
            raise SymbolValidationError(error_msg)
        
        # Convert to uppercase for consistency
        symbol = symbol.upper().strip()
        logger.info(f"Processing data request for symbol: {symbol}")
        
        # Check if data already exists
        existing_file = find_latest_csv_for_symbol(symbol)
        
        if existing_file and os.path.exists(existing_file):
            logger.info(f"Found existing data file for {symbol}: {existing_file}")
            return True, existing_file, f"Using existing data file: {os.path.basename(existing_file)}"
        
        # No existing data found, attempt to download
        logger.info(f"No existing data found for {symbol}, attempting download...")
        
        try:
            downloaded_file = download_stock_data(symbol)
            
            if downloaded_file and os.path.exists(downloaded_file):
                logger.info(f"Successfully downloaded data for {symbol}: {downloaded_file}")
                return True, downloaded_file, f"Successfully downloaded data: {os.path.basename(downloaded_file)}"
            else:
                raise DataDownloadError(f"Download function returned no file for {symbol}")
                
        except Exception as download_error:
            logger.error(f"Download failed for {symbol}: {str(download_error)}")
            raise DataDownloadError(f"Failed to download data for {symbol}") from download_error
            
    except (SymbolValidationError, DataDownloadError) as e:
        # These are expected errors with user-friendly messages
        user_message = get_user_friendly_error_message(e, symbol)
        logger.error(f"Data integration error for {symbol}: {str(e)}")
        return False, "", user_message
        
    except Exception as e:
        # Unexpected errors
        user_message = get_user_friendly_error_message(e, symbol)
        logger.error(f"Unexpected error ensuring data for {symbol}: {str(e)}", exc_info=True)
        return False, "", user_message


def run_drawdown_analysis(symbol: str, data_file: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Executes drawdown analysis and returns structured results.
    
    Args:
        symbol (str): Stock symbol being analyzed
        data_file (str): Path to the CSV data file
        
    Returns:
        tuple: (success, analysis_results, message)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if not symbol:
            raise AnalysisError("Symbol cannot be empty")
        
        if not data_file or not os.path.exists(data_file):
            raise AnalysisError(f"Data file not found: {data_file}")
        
        # Convert symbol to uppercase for consistency
        symbol = symbol.upper().strip()
        logger.info(f"Starting drawdown analysis for {symbol} using file: {data_file}")
        
        # Create output directory for analysis results
        output_dir = "output"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {str(e)}")
            raise AnalysisError(f"Cannot create output directory: {str(e)}")
        
        # Load the CSV data
        try:
            df = pd.read_csv(data_file)
            logger.info(f"Successfully loaded {len(df)} rows from {data_file}")
        except Exception as e:
            logger.error(f"Failed to load CSV file {data_file}: {str(e)}")
            raise AnalysisError(f"Cannot read data file") from e
        
        # Prepare data for analysis
        try:
            if 'Time' in df.columns:
                df['Date'] = pd.to_datetime(df['Time'])
                logger.debug("Using 'Time' column for date information")
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                logger.debug("Using 'Date' column for date information")
            else:
                raise AnalysisError("Data file must contain 'Time' or 'Date' column")
        except Exception as e:
            logger.error(f"Date parsing failed: {str(e)}")
            raise AnalysisError("Cannot parse date information from data file") from e
        
        # Drop rows with invalid dates
        original_rows = len(df)
        df = df.dropna(subset=['Date'])
        if len(df) < original_rows:
            dropped_rows = original_rows - len(df)
            logger.warning(f"Dropped {dropped_rows} rows with invalid dates")
            
            # If too many rows were dropped, this might indicate a data quality issue
            if dropped_rows > original_rows * 0.1:  # More than 10% dropped
                logger.warning(f"High number of invalid dates ({dropped_rows}/{original_rows})")
        
        # Check for required columns
        if 'Symbol' not in df.columns or 'Close' not in df.columns:
            missing_cols = []
            if 'Symbol' not in df.columns:
                missing_cols.append('Symbol')
            if 'Close' not in df.columns:
                missing_cols.append('Close')
            raise AnalysisError(f"Data file missing required columns: {', '.join(missing_cols)}")
        
        # Filter data for the specific symbol
        symbol_df = df[df['Symbol'] == symbol].copy()
        
        if symbol_df.empty:
            available_symbols = df['Symbol'].unique().tolist()
            logger.error(f"No data found for {symbol}. Available symbols: {available_symbols}")
            raise AnalysisError(f"No data found for symbol {symbol}. Available symbols: {', '.join(available_symbols[:10])}")
        
        logger.info(f"Found {len(symbol_df)} records for {symbol}")
        
        # Sort by date for analysis
        symbol_df = symbol_df.sort_values(by='Date')
        
        # Get price series for calculations
        prices = symbol_df.set_index('Date')['Close']
        
        if prices.empty or prices.isna().all():
            raise AnalysisError(f"No valid price data found for {symbol}")
        
        # Check for sufficient data
        if len(prices) < 10:
            logger.warning(f"Limited data for {symbol}: only {len(prices)} records")
        
        # Calculate key metrics with error handling
        try:
            daily_returns = prices.pct_change(fill_method=None)
            daily_returns.iloc[0] = 0
            cumulative_return = (1 + daily_returns).cumprod()
            
            # Calculate drawdown metrics
            running_max = prices.cummax()
            drawdown_series = (prices - running_max) / running_max
            
            current_drawdown = drawdown_series.iloc[-1]
            max_drawdown = drawdown_series.min()
            
            # Find peak and trough
            peak_date = prices.idxmax()
            peak_value = prices.max()
            trough_date = prices.idxmin()
            trough_value = prices.min()
            
            logger.debug(f"Calculated basic metrics - Max DD: {max_drawdown:.2%}, Current DD: {current_drawdown:.2%}")
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            raise AnalysisError("Failed to calculate drawdown metrics") from e
        
        # Calculate local drawdowns with error handling
        try:
            not_in_drawdown = (drawdown_series == 0)
            drawdown_starts = (not_in_drawdown.shift(1, fill_value=True) & ~not_in_drawdown)
            drawdown_start_dates = drawdown_series[drawdown_starts].index
            
            local_drawdowns_list = []
            duration_list = []
            
            if not drawdown_start_dates.empty:
                all_dates = prices.index
                
                for i, start_date in enumerate(drawdown_start_dates):
                    try:
                        if i < len(drawdown_start_dates) - 1:
                            next_start_date = drawdown_start_dates[i + 1]
                            end_date_index = all_dates.get_loc(next_start_date) - 1
                            end_date = all_dates[end_date_index]
                        else:
                            end_date = all_dates[-1]
                        
                        event_period = drawdown_series.loc[start_date:end_date]
                        if not event_period.empty:
                            local_drawdowns_list.append(event_period.min())
                            duration_list.append(len(event_period))
                    except Exception as e:
                        logger.warning(f"Error processing drawdown event {i}: {str(e)}")
                        continue
            
            logger.info(f"Identified {len(local_drawdowns_list)} local drawdown events")
            
        except Exception as e:
            logger.error(f"Error calculating local drawdowns: {str(e)}")
            # Continue with empty lists if local drawdown calculation fails
            local_drawdowns_list = []
            duration_list = []
        
        # Calculate statistics
        num_local_drawdowns = len(local_drawdowns_list)
        avg_drawdown_pct = sum(local_drawdowns_list) / len(local_drawdowns_list) if local_drawdowns_list else 0
        median_drawdown_pct = pd.Series(local_drawdowns_list).median() if local_drawdowns_list else 0
        avg_duration_days = sum(duration_list) / len(duration_list) if duration_list else 0
        median_duration_days = pd.Series(duration_list).median() if duration_list else 0
        
        # Run the full analysis to generate charts and HTML report
        try:
            analyze_symbol(symbol, symbol_df, output_dir)
            logger.info(f"Charts and HTML report generated for {symbol}")
        except Exception as e:
            logger.warning(f"Chart generation failed for {symbol}: {str(e)}")
            # Continue with structured results even if chart generation fails
        
        # Structure the results
        analysis_results = {
            'symbol': symbol,
            'analysis_date': datetime.now(),
            'data_period_start': symbol_df['Date'].min(),
            'data_period_end': symbol_df['Date'].max(),
            'total_records': len(symbol_df),
            
            # Core metrics
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'peak_date': peak_date,
            'peak_value': peak_value,
            'trough_date': trough_date,
            'trough_value': trough_value,
            
            # Local drawdowns statistics
            'num_local_drawdowns': num_local_drawdowns,
            'avg_drawdown_pct': avg_drawdown_pct,
            'median_drawdown_pct': median_drawdown_pct,
            'avg_duration_days': avg_duration_days,
            'median_duration_days': median_duration_days,
            
            # Series data
            'cumulative_return_series': cumulative_return,
            'drawdown_series': drawdown_series,
            'local_drawdowns_list': local_drawdowns_list,
            'duration_list': duration_list,
            
            # File paths
            'data_file_path': data_file,
            'output_directory': output_dir
        }
        
        success_msg = f"Analysis completed successfully for {symbol}. " \
                     f"Found {num_local_drawdowns} drawdown events. " \
                     f"Max drawdown: {max_drawdown:.2%}, Current: {current_drawdown:.2%}"
        
        logger.info(success_msg)
        return True, analysis_results, success_msg
        
    except AnalysisError as e:
        # Expected analysis errors with user-friendly messages
        user_message = get_user_friendly_error_message(e, symbol)
        logger.error(f"Analysis error for {symbol}: {str(e)}")
        return False, {}, user_message
        
    except Exception as e:
        # Unexpected errors
        user_message = get_user_friendly_error_message(e, symbol)
        logger.error(f"Unexpected error during analysis of {symbol}: {str(e)}", exc_info=True)
        return False, {}, user_message


def ensure_directories() -> Tuple[bool, str]:
    """
    Ensure required directories exist for the application.
    
    Returns:
        tuple: (success, message)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Required directories
        directories = ['data', 'output', 'data/archive']
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")
                return False, f"Cannot create required directory '{directory}': {str(e)}"
        
        logger.info("All required directories are available")
        return True, "All required directories are available"
        
    except Exception as e:
        logger.error(f"Unexpected error ensuring directories: {str(e)}")
        return False, f"Error setting up directories: {str(e)}"


def get_available_symbols() -> Tuple[bool, list, str]:
    """
    Get list of available symbols from existing data files.
    
    Returns:
        tuple: (success, symbol_list, message)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure data directory exists
        if not os.path.exists('data'):
            logger.info("Data directory does not exist")
            return True, [], "No data directory found - no symbols available"
        
        # Find all CSV files in data directory
        csv_files = glob.glob(os.path.join('data', '*.csv'))
        
        if not csv_files:
            logger.info("No CSV files found in data directory")
            return True, [], "No data files found"
        
        # Extract symbols from filenames
        symbols = set()
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            # Extract symbol from filename pattern: SYMBOL-YYYY-MM-DD-YYYY-MM-DD.csv
            if '-' in filename:
                symbol = filename.split('-')[0]
                if symbol and symbol.upper() not in ['DATA']:  # Exclude generic filenames
                    symbols.add(symbol.upper())
        
        symbol_list = sorted(list(symbols))
        logger.info(f"Found {len(symbol_list)} symbols in data directory: {symbol_list}")
        
        message = f"Found {len(symbol_list)} symbols with existing data"
        return True, symbol_list, message
        
    except Exception as e:
        logger.error(f"Error getting available symbols: {str(e)}")
        return False, [], f"Error scanning data directory: {str(e)}"


def cleanup_old_files(days_old: int = 30) -> Tuple[bool, str]:
    """
    Clean up old analysis files from the output directory.
    
    Args:
        days_old (int): Remove files older than this many days
        
    Returns:
        tuple: (success, message)
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not os.path.exists('output'):
            return True, "No output directory to clean"
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        files_removed = 0
        
        for filename in os.listdir('output'):
            file_path = os.path.join('output', filename)
            
            if os.path.isfile(file_path):
                file_time = os.path.getmtime(file_path)
                
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        files_removed += 1
                        logger.debug(f"Removed old file: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not remove {filename}: {str(e)}")
        
        message = f"Cleaned up {files_removed} old files from output directory"
        logger.info(message)
        return True, message
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return False, f"Error cleaning up old files: {str(e)}"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a data file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        dict: File information including size, dates, record count
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not os.path.exists(file_path):
            return {'exists': False, 'error': 'File not found'}
        
        # Basic file stats
        stat = os.stat(file_path)
        file_info = {
            'exists': True,
            'path': file_path,
            'filename': os.path.basename(file_path),
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
        }
        
        # Try to get CSV-specific information
        try:
            df = pd.read_csv(file_path)
            file_info.update({
                'record_count': len(df),
                'columns': list(df.columns),
                'has_required_columns': all(col in df.columns for col in ['Symbol', 'Close']),
            })
            
            # Date range information if available
            if 'Time' in df.columns:
                dates = pd.to_datetime(df['Time'], errors='coerce')
                file_info.update({
                    'date_range_start': dates.min(),
                    'date_range_end': dates.max(),
                })
            elif 'Date' in df.columns:
                dates = pd.to_datetime(df['Date'], errors='coerce')
                file_info.update({
                    'date_range_start': dates.min(),
                    'date_range_end': dates.max(),
                })
                
        except Exception as csv_error:
            file_info['csv_error'] = str(csv_error)
            logger.debug(f"Could not parse CSV info for {file_path}: {csv_error}")
        
        return file_info
        
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        return {'exists': False, 'error': str(e)}


# Initialize logging when module is imported
setup_logging()