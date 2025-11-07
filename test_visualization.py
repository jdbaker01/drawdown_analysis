"""
Test suite for visualization components of the Stock Drawdown Dashboard.

Tests chart rendering functionality, table formatting and display, and data presentation accuracy.
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_display import (
    display_analysis_summary,
    display_current_drawdown_status,
    display_key_metrics,
    display_statistics_tables,
    display_expandable_data_sections,
    render_analysis_results
)

try:
    from chart_renderer import (
        render_cumulative_return_chart,
        render_drawdown_series_chart,
        render_local_drawdowns_histogram,
        render_duration_by_magnitude_chart,
        render_all_charts
    )
    CHART_RENDERER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Chart renderer not available: {e}")
    CHART_RENDERER_AVAILABLE = False


class TestResultsDisplay(unittest.TestCase):
    """Test results display functionality."""
    
    def setUp(self):
        """Set up test data for results display tests."""
        # Create sample analysis results
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)  # Random walk
        
        # Calculate cumulative returns
        returns = pd.Series(prices / prices[0], index=dates)
        
        # Calculate drawdown series
        running_max = returns.cummax()
        drawdown_series = (returns - running_max) / running_max
        
        self.sample_results = {
            'symbol': 'TEST',
            'analysis_date': datetime(2025, 1, 1),
            'data_period_start': dates[0],
            'data_period_end': dates[-1],
            'total_records': 100,
            'max_drawdown': -0.15,
            'current_drawdown': -0.05,
            'peak_date': dates[50],
            'peak_value': 120.0,
            'trough_date': dates[75],
            'trough_value': 95.0,
            'num_local_drawdowns': 5,
            'avg_drawdown_pct': -0.08,
            'median_drawdown_pct': -0.06,
            'avg_duration_days': 15.0,
            'median_duration_days': 12.0,
            'cumulative_return_series': returns,
            'drawdown_series': drawdown_series,
            'local_drawdowns_list': [-0.05, -0.08, -0.12, -0.15, -0.03],
            'duration_list': [10, 15, 20, 25, 8],
            'data_file_path': 'data/TEST-2020-01-01-2025-01-01.csv',
            'output_directory': 'output'
        }
    
    @patch('results_display.st')
    def test_display_analysis_summary(self, mock_st):
        """Test analysis summary display."""
        # Mock streamlit components
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.metric = Mock()
        
        display_analysis_summary(self.sample_results)
        
        # Verify streamlit components were called
        mock_st.subheader.assert_called()
        mock_st.columns.assert_called_with(3)
        
        # Check that metrics were displayed
        self.assertTrue(mock_st.metric.called)
        
        # Verify correct symbol was used
        call_args = mock_st.subheader.call_args[0][0]
        self.assertIn('TEST', call_args)
    
    @patch('results_display.st')
    def test_display_current_drawdown_status(self, mock_st):
        """Test current drawdown status display."""
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock()])
        mock_st.metric = Mock()
        mock_st.success = Mock()
        mock_st.error = Mock()
        mock_st.progress = Mock()
        mock_st.write = Mock()
        mock_st.caption = Mock()
        
        display_current_drawdown_status(self.sample_results)
        
        # Verify components were called
        mock_st.subheader.assert_called()
        mock_st.columns.assert_called_with(2)
        mock_st.metric.assert_called()
        
        # For -5% drawdown, should show success (minor drawdown)
        mock_st.success.assert_called()
        
        # Test severe drawdown case
        severe_results = self.sample_results.copy()
        severe_results['current_drawdown'] = -0.35  # 35% drawdown
        
        mock_st.reset_mock()
        display_current_drawdown_status(severe_results)
        mock_st.error.assert_called()
    
    @patch('results_display.st')
    def test_display_key_metrics(self, mock_st):
        """Test key metrics display."""
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        mock_st.metric = Mock()
        mock_st.write = Mock()
        
        display_key_metrics(self.sample_results)
        
        # Verify components were called
        mock_st.subheader.assert_called()
        mock_st.columns.assert_called_with(4)
        
        # Check that metrics were called multiple times (for different metrics)
        self.assertTrue(mock_st.metric.call_count >= 4)
    
    @patch('results_display.st')
    def test_display_statistics_tables(self, mock_st):
        """Test statistics tables display."""
        mock_st.subheader = Mock()
        mock_st.tabs = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.write = Mock()
        mock_st.dataframe = Mock()
        
        display_statistics_tables(self.sample_results)
        
        # Verify tabs were created
        mock_st.tabs.assert_called()
        
        # Verify dataframes were displayed
        self.assertTrue(mock_st.dataframe.called)
    
    @patch('results_display.st')
    def test_display_expandable_data_sections(self, mock_st):
        """Test expandable data sections display."""
        mock_st.subheader = Mock()
        mock_st.expander = Mock()
        mock_st.dataframe = Mock()
        mock_st.download_button = Mock()
        mock_st.warning = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.metric = Mock()
        
        # Create mock expander context managers
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_st.expander.return_value = mock_expander
        
        display_expandable_data_sections(self.sample_results)
        
        # Verify expanders were created
        self.assertTrue(mock_st.expander.called)
        
        # Verify dataframes were displayed
        self.assertTrue(mock_st.dataframe.called)
        
        # Verify download buttons were created
        self.assertTrue(mock_st.download_button.called)
    
    def test_empty_results_handling(self):
        """Test handling of empty or invalid results."""
        with patch('results_display.st') as mock_st:
            mock_st.error = Mock()
            
            # Test with None results
            render_analysis_results(None)
            mock_st.error.assert_called()
            
            # Test with empty dict
            mock_st.reset_mock()
            render_analysis_results({})
            mock_st.error.assert_called()


@unittest.skipUnless(CHART_RENDERER_AVAILABLE, "Chart renderer dependencies not available")
class TestChartRenderer(unittest.TestCase):
    """Test chart rendering functionality."""
    
    def setUp(self):
        """Set up test data for chart rendering tests."""
        # Create sample time series data
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = 100 + np.cumsum(np.random.randn(50) * 0.02)
        
        # Calculate series
        cumulative_returns = pd.Series(prices / prices[0], index=dates)
        running_max = cumulative_returns.cummax()
        drawdown_series = (cumulative_returns - running_max) / running_max
        
        self.chart_data = {
            'symbol': 'CHART_TEST',
            'cumulative_return_series': cumulative_returns,
            'drawdown_series': drawdown_series,
            'local_drawdowns_list': [-0.05, -0.08, -0.12, -0.15, -0.03],
            'duration_list': [10, 15, 20, 25, 8],
            'current_drawdown': -0.05,
            'max_drawdown': -0.15,
            'avg_duration_days': 15.6,
            'median_duration_days': 15.0
        }
    
    @patch('chart_renderer.st')
    @patch('chart_renderer.go')
    def test_render_cumulative_return_chart(self, mock_go, mock_st):
        """Test cumulative return chart rendering."""
        mock_st.subheader = Mock()
        mock_st.plotly_chart = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.metric = Mock()
        mock_st.error = Mock()
        
        # Mock plotly Figure
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scatter = Mock()
        
        render_cumulative_return_chart(self.chart_data)
        
        # Verify chart components were called
        mock_st.subheader.assert_called()
        mock_st.plotly_chart.assert_called()
        mock_go.Figure.assert_called()
        
        # Verify metrics were displayed
        self.assertTrue(mock_st.metric.called)
    
    @patch('chart_renderer.st')
    @patch('chart_renderer.go')
    def test_render_drawdown_series_chart(self, mock_go, mock_st):
        """Test drawdown series chart rendering."""
        mock_st.subheader = Mock()
        mock_st.plotly_chart = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.metric = Mock()
        mock_st.error = Mock()
        
        # Mock plotly components
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scatter = Mock()
        
        render_drawdown_series_chart(self.chart_data)
        
        # Verify chart was rendered
        mock_st.subheader.assert_called()
        mock_st.plotly_chart.assert_called()
        mock_go.Figure.assert_called()
        
        # Verify metrics were displayed
        self.assertTrue(mock_st.metric.called)
    
    @patch('chart_renderer.st')
    @patch('chart_renderer.go')
    def test_render_local_drawdowns_histogram(self, mock_go, mock_st):
        """Test local drawdowns histogram rendering."""
        mock_st.subheader = Mock()
        mock_st.plotly_chart = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        mock_st.metric = Mock()
        mock_st.warning = Mock()
        
        # Mock plotly components
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Histogram = Mock()
        
        render_local_drawdowns_histogram(self.chart_data)
        
        # Verify chart was rendered
        mock_st.subheader.assert_called()
        mock_st.plotly_chart.assert_called()
        mock_go.Figure.assert_called()
        
        # Verify metrics were displayed
        self.assertTrue(mock_st.metric.called)
    
    @patch('chart_renderer.st')
    @patch('chart_renderer.go')
    def test_render_duration_by_magnitude_chart(self, mock_go, mock_st):
        """Test duration by magnitude chart rendering."""
        mock_st.subheader = Mock()
        mock_st.plotly_chart = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.metric = Mock()
        mock_st.warning = Mock()
        
        # Mock plotly components
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Bar = Mock()
        mock_go.Scatter = Mock()
        
        render_duration_by_magnitude_chart(self.chart_data)
        
        # Verify chart was rendered
        mock_st.subheader.assert_called()
        mock_st.plotly_chart.assert_called()
        mock_go.Figure.assert_called()
        
        # Verify metrics were displayed
        self.assertTrue(mock_st.metric.called)
    
    @patch('chart_renderer.st')
    def test_render_all_charts(self, mock_st):
        """Test rendering all charts together."""
        mock_st.header = Mock()
        mock_st.tabs = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        
        # Mock tab context managers
        mock_tab = Mock()
        mock_tab.__enter__ = Mock(return_value=mock_tab)
        mock_tab.__exit__ = Mock(return_value=None)
        mock_st.tabs.return_value = [mock_tab, mock_tab, mock_tab, mock_tab]
        
        with patch('chart_renderer.render_cumulative_return_chart') as mock_cum_chart, \
             patch('chart_renderer.render_drawdown_series_chart') as mock_dd_chart, \
             patch('chart_renderer.render_local_drawdowns_histogram') as mock_hist_chart, \
             patch('chart_renderer.render_duration_by_magnitude_chart') as mock_dur_chart:
            
            render_all_charts(self.chart_data)
            
            # Verify all chart functions were called
            mock_cum_chart.assert_called_once()
            mock_dd_chart.assert_called_once()
            mock_hist_chart.assert_called_once()
            mock_dur_chart.assert_called_once()
    
    @patch('chart_renderer.st')
    def test_empty_data_handling(self, mock_st):
        """Test handling of empty or invalid chart data."""
        mock_st.error = Mock()
        mock_st.warning = Mock()
        
        # Test with None data
        render_cumulative_return_chart({'symbol': 'TEST', 'cumulative_return_series': None})
        mock_st.error.assert_called()
        
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        mock_st.reset_mock()
        render_cumulative_return_chart({'symbol': 'TEST', 'cumulative_return_series': empty_series})
        mock_st.error.assert_called()
        
        # Test with empty drawdown list
        mock_st.reset_mock()
        render_local_drawdowns_histogram({'symbol': 'TEST', 'local_drawdowns_list': []})
        mock_st.warning.assert_called()


class TestDataPresentationAccuracy(unittest.TestCase):
    """Test data presentation accuracy and formatting."""
    
    def setUp(self):
        """Set up test data for accuracy tests."""
        self.test_data = {
            'symbol': 'ACCURACY_TEST',
            'max_drawdown': -0.1234,  # -12.34%
            'current_drawdown': -0.0567,  # -5.67%
            'avg_drawdown_pct': -0.0789,  # -7.89%
            'median_drawdown_pct': -0.0456,  # -4.56%
            'avg_duration_days': 15.678,
            'median_duration_days': 12.345,
            'num_local_drawdowns': 8,
            'total_records': 1000,
            'peak_value': 123.45,
            'trough_value': 98.76,
            'local_drawdowns_list': [-0.05, -0.08, -0.12, -0.15, -0.03, -0.07, -0.09, -0.11],
            'duration_list': [10, 15, 20, 25, 8, 12, 18, 22]
        }
    
    def test_percentage_formatting(self):
        """Test that percentages are formatted correctly."""
        # Test drawdown percentage formatting
        max_dd = self.test_data['max_drawdown']
        formatted_max_dd = f"{max_dd:.2%}"
        self.assertEqual(formatted_max_dd, "-12.34%")
        
        current_dd = self.test_data['current_drawdown']
        formatted_current_dd = f"{current_dd:.2%}"
        self.assertEqual(formatted_current_dd, "-5.67%")
    
    def test_number_formatting(self):
        """Test that numbers are formatted correctly."""
        # Test duration formatting
        avg_duration = self.test_data['avg_duration_days']
        formatted_avg_duration = f"{avg_duration:.1f}"
        self.assertEqual(formatted_avg_duration, "15.7")
        
        # Test price formatting
        peak_value = self.test_data['peak_value']
        formatted_peak = f"${peak_value:.2f}"
        self.assertEqual(formatted_peak, "$123.45")
    
    def test_count_formatting(self):
        """Test that counts are formatted correctly."""
        # Test integer formatting with commas
        total_records = self.test_data['total_records']
        formatted_records = f"{total_records:,}"
        self.assertEqual(formatted_records, "1,000")
        
        # Test simple integer
        num_drawdowns = self.test_data['num_local_drawdowns']
        formatted_count = f"{num_drawdowns}"
        self.assertEqual(formatted_count, "8")
    
    def test_statistical_calculations(self):
        """Test that statistical calculations are accurate."""
        drawdowns = self.test_data['local_drawdowns_list']
        durations = self.test_data['duration_list']
        
        # Test mean calculations
        calculated_avg_dd = sum(drawdowns) / len(drawdowns)
        expected_avg_dd = -0.075  # Average of the test data
        self.assertAlmostEqual(calculated_avg_dd, expected_avg_dd, places=3)
        
        calculated_avg_duration = sum(durations) / len(durations)
        expected_avg_duration = 16.25  # Average of the test data
        self.assertAlmostEqual(calculated_avg_duration, expected_avg_duration, places=2)
        
        # Test median calculations
        sorted_drawdowns = sorted(drawdowns)
        n = len(sorted_drawdowns)
        if n % 2 == 0:
            calculated_median_dd = (sorted_drawdowns[n//2-1] + sorted_drawdowns[n//2]) / 2
        else:
            calculated_median_dd = sorted_drawdowns[n//2]
        
        # For our test data: [-0.15, -0.12, -0.11, -0.09, -0.08, -0.07, -0.05, -0.03]
        # Median should be (-0.09 + -0.08) / 2 = -0.085
        expected_median_dd = -0.085
        self.assertAlmostEqual(calculated_median_dd, expected_median_dd, places=3)
    
    def test_data_consistency(self):
        """Test that displayed data is consistent across components."""
        # Verify that the same values appear consistently
        symbol = self.test_data['symbol']
        self.assertEqual(symbol, 'ACCURACY_TEST')
        
        # Verify that calculated statistics match expected ranges
        max_dd = abs(self.test_data['max_drawdown'])
        current_dd = abs(self.test_data['current_drawdown'])
        
        # Current drawdown should not exceed max drawdown
        self.assertLessEqual(current_dd, max_dd)
        
        # Duration values should be positive
        avg_duration = self.test_data['avg_duration_days']
        median_duration = self.test_data['median_duration_days']
        self.assertGreater(avg_duration, 0)
        self.assertGreater(median_duration, 0)


class TestTableFormatting(unittest.TestCase):
    """Test table formatting and display functionality."""
    
    def test_dataframe_creation(self):
        """Test that DataFrames are created correctly for tables."""
        # Test summary statistics table creation
        test_metrics = {
            "Metric": ["Max Drawdown", "Current Drawdown", "Avg Duration"],
            "Value": ["-12.34%", "-5.67%", "15.7 days"]
        }
        
        df = pd.DataFrame(test_metrics)
        
        # Verify DataFrame structure
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ["Metric", "Value"])
        self.assertEqual(df.iloc[0]["Metric"], "Max Drawdown")
        self.assertEqual(df.iloc[0]["Value"], "-12.34%")
    
    def test_table_data_types(self):
        """Test that table data has correct types."""
        # Create test data for statistics table
        drawdown_pct = [5.0, 8.0, 12.0, 15.0, 3.0]
        drawdown_series = pd.Series(drawdown_pct)
        
        # Test statistical calculations
        mean_val = drawdown_series.mean()
        std_val = drawdown_series.std()
        min_val = drawdown_series.min()
        max_val = drawdown_series.max()
        
        # Verify types
        self.assertIsInstance(mean_val, (float, np.floating))
        self.assertIsInstance(std_val, (float, np.floating))
        self.assertIsInstance(min_val, (float, np.floating))
        self.assertIsInstance(max_val, (float, np.floating))
        
        # Verify values are reasonable
        self.assertAlmostEqual(mean_val, 8.6, places=1)
        self.assertEqual(min_val, 3.0)
        self.assertEqual(max_val, 15.0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)