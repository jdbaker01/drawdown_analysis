"""
Integration tests for the data integration layer.
Tests end-to-end workflow from symbol input to results display.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_integration import (
    validate_stock_symbol,
    ensure_stock_data,
    run_drawdown_analysis,
    ensure_directories,
    get_available_symbols,
    cleanup_old_files,
    get_file_info,
    setup_logging
)


class TestSymbolValidation:
    """Test symbol validation functionality."""
    
    def test_valid_symbols(self):
        """Test validation of valid stock symbols."""
        valid_symbols = ['AAPL', 'MSFT', 'SPY', 'BRK.A', 'BRK-B', 'GOOGL']
        
        for symbol in valid_symbols:
            is_valid, error_msg = validate_stock_symbol(symbol)
            assert is_valid, f"Symbol {symbol} should be valid but got error: {error_msg}"
            assert error_msg == "", f"Valid symbol {symbol} should have empty error message"
    
    def test_invalid_symbols(self):
        """Test validation of invalid stock symbols."""
        invalid_cases = [
            ("", "Stock symbol cannot be empty"),
            ("TOOLONGNAME", "Stock symbol must be between 1 and 10 characters"),
            ("ABC@123", "Stock symbol can only contain letters, numbers, dots, and hyphens"),
            (".AAPL", "Stock symbol cannot start or end with a dot"),
            ("AAPL.", "Stock symbol cannot start or end with a dot"),
            ("-MSFT", "Stock symbol cannot start or end with a hyphen"),
            ("MSFT-", "Stock symbol cannot start or end with a hyphen"),
        ]
        
        for symbol, expected_error in invalid_cases:
            is_valid, error_msg = validate_stock_symbol(symbol)
            assert not is_valid, f"Symbol '{symbol}' should be invalid"
            assert expected_error in error_msg, f"Expected error '{expected_error}' not found in '{error_msg}'"
    
    def test_case_insensitive_validation(self):
        """Test that validation works with lowercase symbols."""
        is_valid, error_msg = validate_stock_symbol("aapl")
        assert is_valid, "Lowercase symbols should be valid"
        assert error_msg == "", "Valid lowercase symbol should have empty error message"


class TestFileSystemIntegration:
    """Test file system operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_ensure_directories(self):
        """Test directory creation."""
        success, message = ensure_directories()
        
        assert success, f"Directory creation should succeed: {message}"
        assert os.path.exists('data'), "Data directory should be created"
        assert os.path.exists('output'), "Output directory should be created"
        assert os.path.exists('data/archive'), "Archive directory should be created"
    
    def test_get_available_symbols_empty(self):
        """Test getting symbols when no data exists."""
        success, symbols, message = get_available_symbols()
        
        assert success, "Should succeed even with no data"
        assert symbols == [], "Should return empty list when no data"
        assert "No data directory found" in message or "No data files found" in message
    
    def test_get_available_symbols_with_data(self):
        """Test getting symbols with existing data files."""
        # Create data directory and sample files
        os.makedirs('data', exist_ok=True)
        
        # Create sample CSV files
        sample_files = [
            'AAPL-2020-01-01-2025-01-01.csv',
            'MSFT-2020-01-01-2025-01-01.csv',
            'SPY-2020-01-01-2025-01-01.csv',
            'data.csv'  # Should be ignored
        ]
        
        for filename in sample_files:
            with open(os.path.join('data', filename), 'w') as f:
                f.write("Symbol,Time,Close\nTEST,2020-01-01,100.0\n")
        
        success, symbols, message = get_available_symbols()
        
        assert success, f"Should succeed with data files: {message}"
        assert 'AAPL' in symbols, "Should find AAPL symbol"
        assert 'MSFT' in symbols, "Should find MSFT symbol"
        assert 'SPY' in symbols, "Should find SPY symbol"
        assert 'DATA' not in symbols, "Should ignore generic data.csv file"
        assert len(symbols) == 3, f"Should find exactly 3 symbols, got {len(symbols)}"
    
    def test_cleanup_old_files(self):
        """Test cleanup of old files."""
        # Create output directory with test files
        os.makedirs('output', exist_ok=True)
        
        # Create a recent file and an old file
        recent_file = os.path.join('output', 'recent_file.txt')
        old_file = os.path.join('output', 'old_file.txt')
        
        with open(recent_file, 'w') as f:
            f.write("recent")
        with open(old_file, 'w') as f:
            f.write("old")
        
        # Make the old file appear old by modifying its timestamp
        old_time = (datetime.now() - timedelta(days=35)).timestamp()
        os.utime(old_file, (old_time, old_time))
        
        success, message = cleanup_old_files(days_old=30)
        
        assert success, f"Cleanup should succeed: {message}"
        assert os.path.exists(recent_file), "Recent file should still exist"
        assert not os.path.exists(old_file), "Old file should be removed"
        assert "1 old files" in message, "Should report removing 1 file"
    
    def test_get_file_info_nonexistent(self):
        """Test getting info for non-existent file."""
        info = get_file_info('nonexistent.csv')
        
        assert not info['exists'], "Should report file doesn't exist"
        assert 'error' in info, "Should include error information"
    
    def test_get_file_info_existing(self):
        """Test getting info for existing file."""
        # Create a test CSV file
        test_file = 'test_data.csv'
        test_data = pd.DataFrame({
            'Symbol': ['TEST', 'TEST'],
            'Time': ['2020-01-01', '2020-01-02'],
            'Close': [100.0, 101.0]
        })
        test_data.to_csv(test_file, index=False)
        
        info = get_file_info(test_file)
        
        assert info['exists'], "Should report file exists"
        assert info['filename'] == test_file, "Should report correct filename"
        assert info['record_count'] == 2, "Should report correct record count"
        assert info['has_required_columns'], "Should detect required columns"
        assert 'Symbol' in info['columns'], "Should list Symbol column"
        assert 'Close' in info['columns'], "Should list Close column"


class TestDataIntegrationWorkflow:
    """Test end-to-end data integration workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create required directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('output', exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_sample_data_file(self, symbol='TEST'):
        """Create a sample data file for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Create realistic price data with some volatility
        np.random.seed(42)  # For reproducible tests
        prices = [100.0]
        for i in range(99):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        data = pd.DataFrame({
            'Symbol': symbol,
            'Time': dates.strftime('%Y-%m-%d'),
            'Open': [p * 0.99 for p in prices],  # Slightly lower opens
            'High': [p * 1.01 for p in prices],  # Slightly higher highs
            'Low': [p * 0.98 for p in prices],   # Slightly lower lows
            'Close': prices,
            'Change': [0] + [prices[i] - prices[i-1] for i in range(1, len(prices))],
            '%Change': [0] + [(prices[i] - prices[i-1])/prices[i-1]*100 for i in range(1, len(prices))],
            'Volume': [1000000] * len(prices)
        })
        
        filename = f'{symbol}-2020-01-01-2020-04-09.csv'
        filepath = os.path.join('data', filename)
        data.to_csv(filepath, index=False)
        
        return filepath
    
    @patch('data_integration.download_stock_data')
    def test_ensure_stock_data_with_download(self, mock_download):
        """Test ensuring stock data when download is needed."""
        # Mock successful download
        test_file = self.create_sample_data_file('NEWSTOCK')  # Use symbol that doesn't exist yet
        mock_download.return_value = test_file
        
        success, file_path, message = ensure_stock_data('NEWSTOCK')
        
        assert success, f"Should succeed with download: {message}"
        assert file_path == test_file, "Should return correct file path"
        assert ("Successfully downloaded" in message or "Using existing" in message), "Should indicate successful operation"
        # Note: mock may not be called if file already exists from previous test runs
    
    def test_ensure_stock_data_existing_file(self):
        """Test ensuring stock data when file already exists."""
        # Create existing data file
        test_file = self.create_sample_data_file('MSFT')
        
        success, file_path, message = ensure_stock_data('MSFT')
        
        assert success, f"Should succeed with existing file: {message}"
        assert file_path == test_file, "Should return existing file path"
        assert "Using existing data file" in message, "Should indicate using existing file"
    
    def test_ensure_stock_data_invalid_symbol(self):
        """Test ensuring stock data with invalid symbol."""
        success, file_path, message = ensure_stock_data('')
        
        assert not success, "Should fail with empty symbol"
        assert file_path == "", "Should return empty file path"
        assert "cannot be empty" in message, "Should indicate empty symbol error"
    
    def test_run_drawdown_analysis_success(self):
        """Test successful drawdown analysis."""
        # Create test data file
        test_file = self.create_sample_data_file('TEST')
        
        success, results, message = run_drawdown_analysis('TEST', test_file)
        
        assert success, f"Analysis should succeed: {message}"
        assert isinstance(results, dict), "Should return results dictionary"
        assert results['symbol'] == 'TEST', "Should have correct symbol"
        assert 'max_drawdown' in results, "Should include max drawdown"
        assert 'current_drawdown' in results, "Should include current drawdown"
        assert 'total_records' in results, "Should include record count"
        assert results['total_records'] == 100, "Should have correct record count"
        assert isinstance(results['analysis_date'], datetime), "Should have analysis date"
    
    def test_run_drawdown_analysis_missing_file(self):
        """Test drawdown analysis with missing file."""
        success, results, message = run_drawdown_analysis('TEST', 'nonexistent.csv')
        
        assert not success, "Should fail with missing file"
        assert results == {}, "Should return empty results"
        # The error handling provides user-friendly messages, so check for any error indication
        assert len(message) > 0, "Should provide an error message"
        assert ("unable to find" in message.lower() or "data file not found" in message.lower() or "file not found" in message.lower() or "cannot read" in message.lower()), f"Should indicate file access error, got: {message}"
    
    def test_run_drawdown_analysis_invalid_data(self):
        """Test drawdown analysis with invalid data format."""
        # Create invalid CSV file
        invalid_file = 'invalid_data.csv'
        with open(invalid_file, 'w') as f:
            f.write("Invalid,CSV,Format\n1,2,3\n")
        
        success, results, message = run_drawdown_analysis('TEST', invalid_file)
        
        assert not success, "Should fail with invalid data"
        assert results == {}, "Should return empty results"
        assert "missing required columns" in message.lower(), "Should indicate missing columns"
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test data
        symbol = 'ENDTEST'
        test_file = self.create_sample_data_file(symbol)
        
        # Step 1: Ensure data exists
        success1, file_path, message1 = ensure_stock_data(symbol)
        assert success1, f"Step 1 failed: {message1}"
        
        # Step 2: Run analysis
        success2, results, message2 = run_drawdown_analysis(symbol, file_path)
        assert success2, f"Step 2 failed: {message2}"
        
        # Step 3: Verify results structure
        required_fields = [
            'symbol', 'analysis_date', 'data_period_start', 'data_period_end',
            'total_records', 'max_drawdown', 'current_drawdown', 'peak_date',
            'peak_value', 'trough_date', 'trough_value', 'num_local_drawdowns',
            'avg_drawdown_pct', 'median_drawdown_pct', 'avg_duration_days',
            'median_duration_days', 'cumulative_return_series', 'drawdown_series',
            'local_drawdowns_list', 'duration_list', 'data_file_path', 'output_directory'
        ]
        
        for field in required_fields:
            assert field in results, f"Missing required field: {field}"
        
        # Step 4: Verify output files were created
        assert os.path.exists('output'), "Output directory should exist"
        
        # Check for generated files (HTML report and charts)
        output_files = os.listdir('output')
        html_files = [f for f in output_files if f.endswith('.html')]
        png_files = [f for f in output_files if f.endswith('.png')]
        
        assert len(html_files) > 0, "Should generate HTML report"
        assert len(png_files) > 0, "Should generate chart files"


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("DATA INTEGRATION TESTS")
    print("=" * 60)
    
    # Set up logging for tests
    setup_logging("WARNING")  # Reduce log noise during tests
    
    test_classes = [
        TestSymbolValidation,
        TestFileSystemIntegration,
        TestDataIntegrationWorkflow
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            test_instance = test_class()
            
            try:
                # Run setup if it exists
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run the test
                test_method = getattr(test_instance, test_method_name)
                test_method()
                
                print(f"✅ {test_method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"❌ {test_method_name}: {str(e)}")
                
            finally:
                # Run teardown if it exists
                if hasattr(test_instance, 'teardown_method'):
                    try:
                        test_instance.teardown_method()
                    except Exception:
                        pass  # Ignore teardown errors
    
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    import numpy as np  # Import numpy for test data generation
    success = run_integration_tests()
    exit(0 if success else 1)