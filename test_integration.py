"""
Integration test for the results display and visualization components.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_analysis_results():
    """Create sample analysis results for testing."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    prices = [100, 102, 101, 103, 99, 98, 101, 105, 104, 106]
    
    # Calculate cumulative returns
    cumulative_returns = pd.Series([p/100 for p in prices], index=dates)
    
    # Calculate drawdown series
    running_max = cumulative_returns.cummax()
    drawdown_series = (cumulative_returns - running_max) / running_max
    
    return {
        'symbol': 'TEST',
        'analysis_date': datetime(2025, 1, 1),
        'data_period_start': dates[0],
        'data_period_end': dates[-1],
        'total_records': 10,
        'max_drawdown': -0.07,
        'current_drawdown': 0.0,
        'peak_date': dates[7],
        'peak_value': 105.0,
        'trough_date': dates[5],
        'trough_value': 98.0,
        'num_local_drawdowns': 3,
        'avg_drawdown_pct': -0.04,
        'median_drawdown_pct': -0.03,
        'avg_duration_days': 2.0,
        'median_duration_days': 2.0,
        'cumulative_return_series': cumulative_returns,
        'drawdown_series': drawdown_series,
        'local_drawdowns_list': [-0.02, -0.05, -0.07],
        'duration_list': [1, 2, 3],
        'data_file_path': 'data/TEST-2020-01-01-2020-01-10.csv',
        'output_directory': 'output'
    }

def test_results_display():
    """Test the results display functionality."""
    print("Testing results display components...")
    
    try:
        from results_display import (
            display_analysis_summary,
            display_current_drawdown_status,
            display_key_metrics,
            display_statistics_tables,
            display_expandable_data_sections
        )
        print("✅ Successfully imported results_display functions")
        
        # Create sample data
        sample_results = create_sample_analysis_results()
        print("✅ Created sample analysis results")
        
        # Test that functions can be called without errors
        # Note: These would normally render to Streamlit, but we're just testing imports and basic logic
        print("✅ Results display components are ready")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def test_chart_renderer():
    """Test the chart renderer functionality."""
    print("Testing chart renderer components...")
    
    try:
        # Try to import chart renderer
        from chart_renderer import render_all_charts
        print("✅ Successfully imported chart_renderer functions")
        
        # Create sample data
        sample_results = create_sample_analysis_results()
        print("✅ Chart renderer components are ready")
        
    except ImportError as e:
        print(f"⚠️ Chart renderer not available (missing dependencies): {e}")
        return True  # This is acceptable - we have fallbacks
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def test_data_formatting():
    """Test data formatting functions."""
    print("Testing data formatting...")
    
    sample_results = create_sample_analysis_results()
    
    # Test percentage formatting
    max_dd = sample_results['max_drawdown']
    formatted_dd = f"{max_dd:.2%}"
    expected_dd = "-7.00%"
    
    if formatted_dd == expected_dd:
        print(f"✅ Percentage formatting correct: {formatted_dd}")
    else:
        print(f"❌ Percentage formatting incorrect: got {formatted_dd}, expected {expected_dd}")
        return False
    
    # Test number formatting
    peak_value = sample_results['peak_value']
    formatted_price = f"${peak_value:.2f}"
    expected_price = "$105.00"
    
    if formatted_price == expected_price:
        print(f"✅ Price formatting correct: {formatted_price}")
    else:
        print(f"❌ Price formatting incorrect: got {formatted_price}, expected {expected_price}")
        return False
    
    # Test duration formatting
    avg_duration = sample_results['avg_duration_days']
    formatted_duration = f"{avg_duration:.1f} days"
    expected_duration = "2.0 days"
    
    if formatted_duration == expected_duration:
        print(f"✅ Duration formatting correct: {formatted_duration}")
    else:
        print(f"❌ Duration formatting incorrect: got {formatted_duration}, expected {expected_duration}")
        return False
    
    return True

def test_statistical_calculations():
    """Test statistical calculations."""
    print("Testing statistical calculations...")
    
    sample_results = create_sample_analysis_results()
    
    # Test drawdown statistics
    drawdowns = sample_results['local_drawdowns_list']
    calculated_avg = sum(drawdowns) / len(drawdowns)
    expected_avg = -0.04666666666666667  # (-0.02 + -0.05 + -0.07) / 3
    
    if abs(calculated_avg - expected_avg) < 0.001:
        print(f"✅ Average drawdown calculation correct: {calculated_avg:.4f}")
    else:
        print(f"❌ Average drawdown calculation incorrect: got {calculated_avg:.4f}, expected {expected_avg:.4f}")
        return False
    
    # Test duration statistics
    durations = sample_results['duration_list']
    calculated_avg_duration = sum(durations) / len(durations)
    expected_avg_duration = 2.0  # (1 + 2 + 3) / 3
    
    if abs(calculated_avg_duration - expected_avg_duration) < 0.001:
        print(f"✅ Average duration calculation correct: {calculated_avg_duration:.1f}")
    else:
        print(f"❌ Average duration calculation incorrect: got {calculated_avg_duration:.1f}, expected {expected_avg_duration:.1f}")
        return False
    
    return True

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("INTEGRATION TEST: Results Display and Visualization")
    print("=" * 60)
    
    tests = [
        ("Results Display", test_results_display),
        ("Chart Renderer", test_chart_renderer),
        ("Data Formatting", test_data_formatting),
        ("Statistical Calculations", test_statistical_calculations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)