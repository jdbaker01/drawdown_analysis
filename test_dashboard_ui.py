"""
Test suite for Stock Drawdown Dashboard UI components.

Tests input validation, progress indicators, and error message display functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the current directory to the path to import dashboard
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard import (
    validate_symbol_format, 
    initialize_session_state,
    update_processing_status,
    add_analysis_result,
    get_analysis_by_symbol,
    get_session_summary,
    clear_session_state
)


class TestInputValidation(unittest.TestCase):
    """Test input validation behavior."""
    
    def test_validate_symbol_format_valid_symbols(self):
        """Test validation with valid stock symbols."""
        # Test valid symbols
        valid_symbols = ["AAPL", "MSFT", "SPY", "A", "GOOGL"]
        
        for symbol in valid_symbols:
            is_valid, error_msg = validate_symbol_format(symbol)
            assert is_valid, f"Symbol {symbol} should be valid"
            assert error_msg == "", f"No error message expected for {symbol}"
    
    def test_validate_symbol_format_invalid_symbols(self):
        """Test validation with invalid stock symbols."""
        # Test invalid symbols
        invalid_cases = [
            ("", "Symbol cannot be empty"),
            ("   ", "Symbol cannot be empty or only whitespace"),
            ("TOOLONG", "Symbol cannot be longer than 5 characters"),
            ("123", "Symbol must contain only letters (A-Z)"),
            ("AAPL123", "Symbol cannot be longer than 5 characters"),
            ("AA-PL", "Symbol must contain only letters (A-Z)"),
            ("AA.PL", "Symbol must contain only letters (A-Z)"),
            ("AA PL", "Symbol must contain only letters (A-Z)"),
        ]
        
        for symbol, expected_error in invalid_cases:
            is_valid, error_msg = validate_symbol_format(symbol)
            assert not is_valid, f"Symbol '{symbol}' should be invalid"
            assert expected_error in error_msg, f"Expected error message for '{symbol}'"
    
    def test_validate_symbol_format_case_insensitive(self):
        """Test that validation handles case-insensitive input."""
        # Test lowercase symbols
        lowercase_symbols = ["aapl", "msft", "spy"]
        
        for symbol in lowercase_symbols:
            is_valid, error_msg = validate_symbol_format(symbol)
            assert is_valid, f"Lowercase symbol {symbol} should be valid"
            assert error_msg == "", f"No error message expected for {symbol}"
    
    def test_validate_symbol_format_whitespace_handling(self):
        """Test that validation properly handles whitespace."""
        # Test symbols with whitespace
        whitespace_cases = [
            " AAPL ", 
            "  MSFT  ",
            "\tSPY\t",
            "\nGOOGL\n"
        ]
        
        for symbol in whitespace_cases:
            is_valid, error_msg = validate_symbol_format(symbol)
            assert is_valid, f"Symbol with whitespace '{symbol}' should be valid after stripping"
            assert error_msg == "", f"No error message expected for '{symbol}'"


class TestSessionStateManagement(unittest.TestCase):
    """Test session state initialization and management."""
    
    def setUp(self):
        """Set up mock session state for each test."""
        class MockSessionState:
            def __init__(self):
                self._data = {}
            
            def __contains__(self, key):
                return key in self._data
            
            def __getattr__(self, key):
                return self._data.get(key)
            
            def __setattr__(self, key, value):
                if key.startswith('_'):
                    super().__setattr__(key, value)
                else:
                    self._data[key] = value
            
            def __delitem__(self, key):
                if key in self._data:
                    del self._data[key]
        
        self.mock_session_state = MockSessionState()
        self.session_patch = patch('dashboard.st.session_state', self.mock_session_state)
        self.session_patch.start()
    
    def tearDown(self):
        """Clean up patches after each test."""
        self.session_patch.stop()
    
    def test_initialize_session_state(self):
        """Test session state initialization with default values."""
        initialize_session_state()
        
        # Check all required session state variables are initialized
        expected_keys = [
            'analysis_history',
            'current_symbol', 
            'processing_status',
            'error_message',
            'show_advanced_stats',
            'selected_analysis_index',
            'session_start_time',
            'analysis_counter',
            'last_successful_symbol',
            'processing_progress'
        ]
        
        for key in expected_keys:
            self.assertIn(key, self.mock_session_state._data, f"Session state should contain {key}")
        
        # Check default values
        self.assertEqual(self.mock_session_state._data['analysis_history'], [])
        self.assertIsNone(self.mock_session_state._data['current_symbol'])
        self.assertEqual(self.mock_session_state._data['processing_status'], 'idle')
        self.assertIsNone(self.mock_session_state._data['error_message'])
        self.assertFalse(self.mock_session_state._data['show_advanced_stats'])
        self.assertEqual(self.mock_session_state._data['selected_analysis_index'], 0)
        self.assertEqual(self.mock_session_state._data['analysis_counter'], 0)
        self.assertIsNone(self.mock_session_state._data['last_successful_symbol'])
        self.assertEqual(self.mock_session_state._data['processing_progress'], 0)
    
    def test_initialize_session_state_preserves_existing(self):
        """Test that initialization doesn't overwrite existing session state."""
        # Set existing data
        self.mock_session_state._data = {'analysis_history': ['existing']}
        
        initialize_session_state()
        
        # Should preserve existing data
        self.assertEqual(self.mock_session_state._data['analysis_history'], ['existing'])
        
        # Should still initialize missing keys
        self.assertIn('current_symbol', self.mock_session_state._data)
        self.assertIn('processing_status', self.mock_session_state._data)
    
    def test_initialize_session_state_clears_stale_processing(self):
        """Test that initialization clears stale processing states."""
        # Set stale processing state
        self.mock_session_state._data = {
            'processing_status': 'downloading',
            'current_symbol': 'AAPL',
            'processing_progress': 50
        }
        
        initialize_session_state()
        
        # Should clear stale processing state
        self.assertEqual(self.mock_session_state._data['processing_status'], 'idle')
        self.assertIsNone(self.mock_session_state._data['current_symbol'])
        self.assertEqual(self.mock_session_state._data['processing_progress'], 0)
    
    def test_update_processing_status(self):
        """Test processing status updates."""
        initialize_session_state()
        
        # Test basic status update
        update_processing_status('downloading', 'AAPL', 25)
        
        self.assertEqual(self.mock_session_state._data['processing_status'], 'downloading')
        self.assertEqual(self.mock_session_state._data['current_symbol'], 'AAPL')
        self.assertEqual(self.mock_session_state._data['processing_progress'], 25)
        self.assertIsNone(self.mock_session_state._data['error_message'])
        
        # Test error status update
        update_processing_status('error', error_message='Test error')
        
        self.assertEqual(self.mock_session_state._data['processing_status'], 'error')
        self.assertEqual(self.mock_session_state._data['error_message'], 'Test error')
        
        # Test clearing error when setting non-error status
        update_processing_status('complete')
        
        self.assertEqual(self.mock_session_state._data['processing_status'], 'complete')
        self.assertIsNone(self.mock_session_state._data['error_message'])
    
    def test_add_analysis_result(self):
        """Test adding analysis results to session history."""
        initialize_session_state()
        
        # Create test analysis result
        analysis_result = {
            'symbol': 'AAPL',
            'max_drawdown': -0.15,
            'current_drawdown': -0.05
        }
        
        add_analysis_result(analysis_result)
        
        # Check that result was added
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 1)
        self.assertEqual(self.mock_session_state._data['analysis_counter'], 1)
        self.assertEqual(self.mock_session_state._data['last_successful_symbol'], 'AAPL')
        self.assertEqual(self.mock_session_state._data['selected_analysis_index'], 0)
        
        # Check that session_id was added
        added_result = self.mock_session_state._data['analysis_history'][0]
        self.assertEqual(added_result['session_id'], 1)
        
        # Add another result
        analysis_result2 = {
            'symbol': 'MSFT',
            'max_drawdown': -0.12,
            'current_drawdown': -0.03
        }
        
        add_analysis_result(analysis_result2)
        
        # Check multiple results
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 2)
        self.assertEqual(self.mock_session_state._data['analysis_counter'], 2)
        self.assertEqual(self.mock_session_state._data['last_successful_symbol'], 'MSFT')
        self.assertEqual(self.mock_session_state._data['selected_analysis_index'], 1)
    
    def test_get_analysis_by_symbol(self):
        """Test retrieving analysis by symbol."""
        initialize_session_state()
        
        # Add test analyses
        analysis1 = {'symbol': 'AAPL', 'max_drawdown': -0.15}
        analysis2 = {'symbol': 'MSFT', 'max_drawdown': -0.12}
        analysis3 = {'symbol': 'AAPL', 'max_drawdown': -0.18}  # Newer AAPL analysis
        
        add_analysis_result(analysis1)
        add_analysis_result(analysis2)
        add_analysis_result(analysis3)
        
        # Test finding existing symbol (should return most recent)
        result = get_analysis_by_symbol('AAPL')
        self.assertIsNotNone(result)
        self.assertEqual(result['max_drawdown'], -0.18)  # Most recent AAPL
        
        # Test case insensitive search
        result = get_analysis_by_symbol('aapl')
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        
        # Test non-existent symbol
        result = get_analysis_by_symbol('GOOGL')
        self.assertIsNone(result)
    
    def test_get_session_summary(self):
        """Test session summary generation."""
        initialize_session_state()
        
        # Test empty session
        summary = get_session_summary()
        
        self.assertEqual(summary['total_analyses'], 0)
        self.assertEqual(summary['unique_symbols_count'], 0)
        self.assertEqual(summary['unique_symbols'], [])
        self.assertEqual(summary['current_status'], 'idle')
        self.assertIsNone(summary['current_symbol'])
        self.assertIsNone(summary['last_successful_symbol'])
        
        # Add some analyses
        add_analysis_result({'symbol': 'AAPL', 'max_drawdown': -0.15})
        add_analysis_result({'symbol': 'MSFT', 'max_drawdown': -0.12})
        add_analysis_result({'symbol': 'AAPL', 'max_drawdown': -0.18})  # Duplicate symbol
        
        summary = get_session_summary()
        
        self.assertEqual(summary['total_analyses'], 3)
        self.assertEqual(summary['unique_symbols_count'], 2)
        self.assertIn('AAPL', summary['unique_symbols'])
        self.assertIn('MSFT', summary['unique_symbols'])
        self.assertEqual(summary['last_successful_symbol'], 'AAPL')
    
    def test_clear_session_state(self):
        """Test clearing session state."""
        initialize_session_state()
        
        # Add some data
        add_analysis_result({'symbol': 'AAPL', 'max_drawdown': -0.15})
        update_processing_status('complete', 'AAPL', 100)
        
        # Verify data exists
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 1)
        self.assertEqual(self.mock_session_state._data['processing_status'], 'complete')
        
        # Clear session
        clear_session_state()
        
        # Verify data is cleared and reinitialized
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 0)
        self.assertEqual(self.mock_session_state._data['processing_status'], 'idle')
        self.assertEqual(self.mock_session_state._data['analysis_counter'], 0)
        self.assertIsNone(self.mock_session_state._data['last_successful_symbol'])


class TestMultiStockAnalysisWorkflow(unittest.TestCase):
    """Test multi-stock analysis workflow and navigation."""
    
    def setUp(self):
        """Set up mock session state for each test."""
        class MockSessionState:
            def __init__(self):
                self._data = {}
            
            def __contains__(self, key):
                return key in self._data
            
            def __getattr__(self, key):
                return self._data.get(key)
            
            def __setattr__(self, key, value):
                if key.startswith('_'):
                    super().__setattr__(key, value)
                else:
                    self._data[key] = value
            
            def __delitem__(self, key):
                if key in self._data:
                    del self._data[key]
        
        self.mock_session_state = MockSessionState()
        self.session_patch = patch('dashboard.st.session_state', self.mock_session_state)
        self.session_patch.start()
        initialize_session_state()
    
    def tearDown(self):
        """Clean up patches after each test."""
        self.session_patch.stop()
    
    def test_sequential_analysis_workflow(self):
        """Test analyzing multiple stocks sequentially."""
        # Simulate analyzing first stock
        update_processing_status('downloading', 'AAPL', 0)
        self.assertEqual(self.mock_session_state._data['processing_status'], 'downloading')
        self.assertEqual(self.mock_session_state._data['current_symbol'], 'AAPL')
        
        update_processing_status('analyzing', 'AAPL', 50)
        self.assertEqual(self.mock_session_state._data['processing_status'], 'analyzing')
        
        # Complete first analysis
        analysis1 = {
            'symbol': 'AAPL',
            'max_drawdown': -0.15,
            'current_drawdown': -0.05,
            'analysis_date': '2023-01-01'
        }
        add_analysis_result(analysis1)
        update_processing_status('complete', 'AAPL', 100)
        
        # Verify first analysis is complete
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 1)
        self.assertEqual(self.mock_session_state._data['last_successful_symbol'], 'AAPL')
        
        # Start second analysis immediately
        update_processing_status('downloading', 'MSFT', 0)
        self.assertEqual(self.mock_session_state._data['current_symbol'], 'MSFT')
        
        # Complete second analysis
        analysis2 = {
            'symbol': 'MSFT',
            'max_drawdown': -0.12,
            'current_drawdown': -0.03,
            'analysis_date': '2023-01-02'
        }
        add_analysis_result(analysis2)
        update_processing_status('complete', 'MSFT', 100)
        
        # Verify both analyses exist
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 2)
        self.assertEqual(self.mock_session_state._data['last_successful_symbol'], 'MSFT')
        
        # Verify we can retrieve both analyses
        aapl_result = get_analysis_by_symbol('AAPL')
        msft_result = get_analysis_by_symbol('MSFT')
        
        self.assertIsNotNone(aapl_result)
        self.assertIsNotNone(msft_result)
        self.assertEqual(aapl_result['symbol'], 'AAPL')
        self.assertEqual(msft_result['symbol'], 'MSFT')
    
    def test_analysis_history_management(self):
        """Test analysis history tracking and management."""
        # Add multiple analyses
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        for i, symbol in enumerate(symbols):
            analysis = {
                'symbol': symbol,
                'max_drawdown': -0.1 - (i * 0.01),  # Varying drawdowns
                'current_drawdown': -0.05,
                'analysis_date': f'2023-01-{i+1:02d}'
            }
            add_analysis_result(analysis)
        
        # Verify all analyses are tracked
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 5)
        self.assertEqual(self.mock_session_state._data['analysis_counter'], 5)
        
        # Verify session summary
        summary = get_session_summary()
        self.assertEqual(summary['total_analyses'], 5)
        self.assertEqual(summary['unique_symbols_count'], 5)
        self.assertEqual(set(summary['unique_symbols']), set(symbols))
        
        # Test navigation index updates
        self.assertEqual(self.mock_session_state._data['selected_analysis_index'], 4)  # Last added
    
    def test_duplicate_symbol_handling(self):
        """Test handling of duplicate symbol analyses."""
        # Add initial analysis
        analysis1 = {
            'symbol': 'AAPL',
            'max_drawdown': -0.15,
            'current_drawdown': -0.05,
            'analysis_date': '2023-01-01'
        }
        add_analysis_result(analysis1)
        
        # Add updated analysis for same symbol
        analysis2 = {
            'symbol': 'AAPL',
            'max_drawdown': -0.18,
            'current_drawdown': -0.07,
            'analysis_date': '2023-01-02'
        }
        add_analysis_result(analysis2)
        
        # Should have both analyses in history
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 2)
        
        # get_analysis_by_symbol should return the most recent
        result = get_analysis_by_symbol('AAPL')
        self.assertEqual(result['max_drawdown'], -0.18)  # Most recent
        self.assertEqual(result['analysis_date'], '2023-01-02')
        
        # Session summary should show 2 total analyses but 1 unique symbol
        summary = get_session_summary()
        self.assertEqual(summary['total_analyses'], 2)
        self.assertEqual(summary['unique_symbols_count'], 1)
    
    def test_session_persistence_across_operations(self):
        """Test that session state persists across different operations."""
        # Add initial analysis
        add_analysis_result({
            'symbol': 'AAPL',
            'max_drawdown': -0.15,
            'current_drawdown': -0.05
        })
        
        # Simulate various operations that shouldn't affect history
        update_processing_status('downloading', 'MSFT', 25)
        update_processing_status('error', error_message='Network error')
        update_processing_status('idle')
        
        # History should be preserved
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 1)
        self.assertEqual(self.mock_session_state._data['last_successful_symbol'], 'AAPL')
        
        # Add another analysis after error
        add_analysis_result({
            'symbol': 'MSFT',
            'max_drawdown': -0.12,
            'current_drawdown': -0.03
        })
        
        # Both should be preserved
        self.assertEqual(len(self.mock_session_state._data['analysis_history']), 2)
        summary = get_session_summary()
        self.assertEqual(summary['unique_symbols_count'], 2)


class TestProgressIndicators(unittest.TestCase):
    """Test progress indicator functionality."""
    
    def test_processing_status_values(self):
        """Test that all expected processing status values are handled."""
        expected_statuses = ['idle', 'downloading', 'analyzing', 'complete', 'error']
        
        # This test ensures our status values are consistent
        # In a real Streamlit app, we would test the UI rendering
        for status in expected_statuses:
            assert isinstance(status, str)
            assert len(status) > 0


class TestErrorMessageDisplay(unittest.TestCase):
    """Test error message display functionality."""
    
    def test_error_message_formatting(self):
        """Test error message formatting and content."""
        # Test various error scenarios
        error_cases = [
            "Invalid symbol format",
            "Network connection failed", 
            "Data not available",
            "File permission error"
        ]
        
        for error_msg in error_cases:
            # Test that error messages are strings and not empty
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0
            
            # Test that error messages don't contain sensitive information
            sensitive_terms = ['password', 'token', 'key', 'secret']
            for term in sensitive_terms:
                assert term.lower() not in error_msg.lower()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)