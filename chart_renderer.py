"""
Chart rendering component for the Stock Drawdown Dashboard.
Converts analysis data to Streamlit-compatible visualizations.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, Any, List, Optional

# Configure matplotlib for Streamlit
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('default')  # Use default style

# Try to import plotly, but fall back to matplotlib if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_cumulative_return_chart(analysis_results: Dict[str, Any]) -> None:
    """
    Create and display cumulative return chart with proper scaling.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    # Always use matplotlib for now
    render_cumulative_return_matplotlib(analysis_results)


def render_drawdown_series_chart(analysis_results: Dict[str, Any]) -> None:
    """
    Create and display drawdown series as an area chart.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    # Always use matplotlib for now
    render_drawdown_series_matplotlib(analysis_results)


def render_local_drawdowns_histogram(analysis_results: Dict[str, Any]) -> None:
    """
    Create and display local drawdowns histogram with current drawdown indicator.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    # Always use matplotlib for now
    render_local_drawdowns_histogram_matplotlib(analysis_results)


def render_duration_by_magnitude_chart(analysis_results: Dict[str, Any]) -> None:
    """
    Create and display duration by magnitude chart.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    # Always use matplotlib for now
    render_duration_by_magnitude_matplotlib(analysis_results)


def render_all_charts(analysis_results: Dict[str, Any]) -> None:
    """
    Render all charts for the analysis results.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    if not analysis_results:
        st.error("❌ No analysis results available for charts")
        return
    
    st.header("📊 Visualizations")
    
    create_matplotlib_fallback_charts(analysis_results)


def create_matplotlib_fallback_charts(analysis_results: Dict[str, Any]) -> None:
    """
    Create matplotlib charts as fallback if Plotly fails.
    Uses st.pyplot() for display.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    # Render charts in tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Cumulative Return", 
        "📉 Drawdown Series", 
        "📊 Drawdown Distribution", 
        "⏱️ Duration Analysis"
    ])
    
    with tab1:
        render_cumulative_return_matplotlib(analysis_results)
    
    with tab2:
        render_drawdown_series_matplotlib(analysis_results)
        # Add detailed drawdown statistics to the drawdown series tab
        render_drawdown_details_table(analysis_results)
    
    with tab3:
        render_local_drawdowns_histogram_matplotlib(analysis_results)
    
    with tab4:
        render_duration_by_magnitude_matplotlib(analysis_results)
        # Add detailed duration statistics to the duration analysis tab
        render_duration_statistics_table(analysis_results)


def render_cumulative_return_matplotlib(analysis_results: Dict[str, Any]) -> None:
    """Matplotlib chart with integrated statistics for cumulative return."""
    symbol = analysis_results.get('symbol', 'Unknown')
    cumulative_return_series = analysis_results.get('cumulative_return_series')
    
    if cumulative_return_series is None or cumulative_return_series.empty:
        st.error("❌ No cumulative return data available for chart")
        return
    
    st.subheader(f"📈 Cumulative Return - {symbol}")
    
    # Display the chart
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        cumulative_return_series.plot(ax=ax, title=f'Cumulative Return for {symbol}', color='#1f77b4', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Initial Investment Level')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (1.0 = Initial Investment)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    finally:
        plt.close(fig)
    
    # Calculate all metrics
    final_return = cumulative_return_series.iloc[-1]
    max_return = cumulative_return_series.max()
    min_return = cumulative_return_series.min()
    returns = cumulative_return_series.pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
    total_return = (final_return - 1) * 100
    
    # Calculate annualized return
    data_start = analysis_results.get('data_period_start')
    data_end = analysis_results.get('data_period_end')
    if data_start and data_end and hasattr(data_start, 'year') and hasattr(data_end, 'year'):
        years = (data_end - data_start).days / 365.25
        annualized_return = ((final_return ** (1/years)) - 1) * 100 if years > 0 else 0
        annualized_return_str = f"{annualized_return:.1f}%"
    else:
        annualized_return_str = "N/A"
    
    # Calculate Sharpe ratio
    if volatility > 0:
        sharpe_ratio = (total_return / 100) / volatility
        sharpe_ratio_str = f"{sharpe_ratio:.2f}"
    else:
        sharpe_ratio_str = "N/A"
    
    # Display ALL return statistics as ONE comprehensive table
    st.subheader("📊 Return Statistics & Analysis")
    
    return_stats = {
        "Metric": [
            "Final Return (Multiplier)",
            "Final Return (Percentage)",
            "Peak Return (Multiplier)",
            "Peak Return (Percentage)",
            "Lowest Return (Multiplier)",
            "Lowest Return (Percentage)",
            "Total Return",
            "Annualized Return",
            "Volatility (Annual)",
            "Sharpe Ratio"
        ],
        "Value": [
            f"{final_return:.2f}x",
            f"{(final_return-1)*100:.1f}%",
            f"{max_return:.2f}x",
            f"{(max_return-1)*100:.1f}%",
            f"{min_return:.2f}x",
            f"{(min_return-1)*100:.1f}%",
            f"{total_return:.1f}%",
            annualized_return_str,
            f"{volatility:.1%}",
            sharpe_ratio_str
        ]
    }
    
    return_stats_df = pd.DataFrame(return_stats)
    st.dataframe(return_stats_df, use_container_width=True, hide_index=True)


def render_drawdown_series_matplotlib(analysis_results: Dict[str, Any]) -> None:
    """Matplotlib chart with integrated statistics for drawdown series."""
    symbol = analysis_results.get('symbol', 'Unknown')
    drawdown_series = analysis_results.get('drawdown_series')
    current_drawdown = analysis_results.get('current_drawdown', 0)
    
    if drawdown_series is None or drawdown_series.empty:
        st.error("❌ No drawdown series data available for chart")
        return
    
    st.subheader(f"📉 Drawdown Series - {symbol}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        drawdown_series.plot(kind='area', ax=ax, color='red', alpha=0.3, title=f'Drawdown Series for {symbol}')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Highlight current drawdown if significant
        if current_drawdown < -0.01:  # More than 1% drawdown
            ax.axhline(y=current_drawdown, color='orange', linestyle='--', linewidth=2, 
                      label=f'Current: {current_drawdown:.2%}')
            ax.legend()
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.tight_layout()
        st.pyplot(fig)
    finally:
        plt.close(fig)
    
    # Display comprehensive drawdown statistics
    st.subheader("📊 Drawdown Statistics")
    
    # Core metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_drawdown = analysis_results.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    
    with col2:
        st.metric("Current Drawdown", f"{current_drawdown:.2%}")
    
    with col3:
        # Calculate time in drawdown
        in_drawdown = (drawdown_series < 0).sum()
        total_periods = len(drawdown_series)
        pct_in_drawdown = (in_drawdown / total_periods) * 100 if total_periods > 0 else 0
        st.metric("Time in Drawdown", f"{pct_in_drawdown:.1f}%")
    
    with col4:
        num_drawdowns = analysis_results.get('num_local_drawdowns', 0)
        st.metric("Total Drawdown Events", f"{num_drawdowns}")
    
    # Peak and Trough Analysis
    st.subheader("🏔️ Peak & Trough Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Historical Peak**")
        peak_date = analysis_results.get('peak_date')
        peak_value = analysis_results.get('peak_value', 0)
        
        if peak_date and hasattr(peak_date, 'strftime'):
            st.write(f"📅 Date: {peak_date.strftime('%m/%d/%Y')}")
        else:
            st.write(f"📅 Date: {peak_date}")
        st.write(f"💰 Price: ${peak_value:.2f}")
    
    with col2:
        st.write("**Historical Trough**")
        trough_date = analysis_results.get('trough_date')
        trough_value = analysis_results.get('trough_value', 0)
        
        if trough_date and hasattr(trough_date, 'strftime'):
            st.write(f"📅 Date: {trough_date.strftime('%m/%d/%Y')}")
        else:
            st.write(f"📅 Date: {trough_date}")
        st.write(f"💰 Price: ${trough_value:.2f}")
    
    # Current Status
    st.subheader("🎯 Current Status")
    
    # Determine status color and message
    if current_drawdown == 0:
        status_message = "✅ At Peak - No Current Drawdown"
        status_description = "The stock is currently at or near its all-time high."
        st.success(f"{status_message}\n\n{status_description}")
    elif current_drawdown > -0.05:  # Less than 5% drawdown
        status_message = "🟢 Minor Drawdown"
        status_description = "The stock is experiencing a small pullback from recent highs."
        st.success(f"{status_message}\n\n{status_description}")
    elif current_drawdown > -0.15:  # 5-15% drawdown
        status_message = "🟡 Moderate Drawdown"
        status_description = "The stock is in a moderate correction phase."
        st.warning(f"{status_message}\n\n{status_description}")
    elif current_drawdown > -0.30:  # 15-30% drawdown
        status_message = "🟠 Significant Drawdown"
        status_description = "The stock is experiencing a significant decline from recent highs."
        st.error(f"{status_message}\n\n{status_description}")
    else:  # Greater than 30% drawdown
        status_message = "🔴 Severe Drawdown"
        status_description = "The stock is in a severe decline phase."
        st.error(f"{status_message}\n\n{status_description}")
    
    # Progress bar showing current drawdown relative to max drawdown
    if max_drawdown < 0:  # Only show if we have a valid max drawdown
        progress_value = abs(current_drawdown) / abs(max_drawdown) if max_drawdown != 0 else 0
        progress_value = min(progress_value, 1.0)  # Cap at 100%
        
        st.write("**Current vs Historical Max Drawdown:**")
        st.progress(progress_value)
        st.caption(f"Current drawdown is {progress_value:.1%} of the historical maximum")


def render_local_drawdowns_histogram_matplotlib(analysis_results: Dict[str, Any]) -> None:
    """Matplotlib chart with integrated statistics for local drawdowns distribution."""
    symbol = analysis_results.get('symbol', 'Unknown')
    local_drawdowns_list = analysis_results.get('local_drawdowns_list', [])
    current_drawdown = analysis_results.get('current_drawdown', 0)
    
    if not local_drawdowns_list:
        st.warning("⚠️ No local drawdown events found for histogram")
        return
    
    st.subheader(f"📊 Local Drawdowns Distribution - {symbol}")
    
    # Convert to positive percentages for better visualization
    drawdown_pct = [abs(dd) * 100 for dd in local_drawdowns_list]
    current_dd_pct = abs(current_drawdown) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.hist(drawdown_pct, bins=min(20, len(set(drawdown_pct))), 
               edgecolor='black', alpha=0.7, color='skyblue')
        
        # Add vertical line for current drawdown if significant
        if current_drawdown < -0.005:  # More than 0.5% drawdown
            ax.axvline(current_dd_pct, color='red', linestyle='--', linewidth=2,
                      label=f'Current: {current_drawdown:.2%}')
            ax.legend()
        
        ax.set_title(f'Distribution of Local Drawdowns for {symbol}')
        ax.set_xlabel('Drawdown Magnitude (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    finally:
        plt.close(fig)
    
    # Display comprehensive distribution statistics
    st.subheader("📊 Distribution Statistics")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    drawdown_series = pd.Series(drawdown_pct)
    
    with col1:
        st.metric("Total Events", f"{len(local_drawdowns_list)}")
    
    with col2:
        st.metric("Average Magnitude", f"{drawdown_series.mean():.2f}%")
    
    with col3:
        st.metric("Median Magnitude", f"{drawdown_series.median():.2f}%")
    
    with col4:
        st.metric("Max Magnitude", f"{drawdown_series.max():.2f}%")
    
    # Detailed statistics table
    st.subheader("📈 Detailed Statistics")
    
    stats_data = {
        "Statistic": [
            "Count",
            "Mean",
            "Standard Deviation",
            "Minimum",
            "10th Percentile",
            "25th Percentile (Q1)",
            "50th Percentile (Median)",
            "75th Percentile (Q3)",
            "90th Percentile",
            "Maximum"
        ],
        "Value": [
            f"{len(drawdown_pct):,}",
            f"{drawdown_series.mean():.2f}%",
            f"{drawdown_series.std():.2f}%",
            f"{drawdown_series.min():.2f}%",
            f"{drawdown_series.quantile(0.10):.2f}%",
            f"{drawdown_series.quantile(0.25):.2f}%",
            f"{drawdown_series.quantile(0.50):.2f}%",
            f"{drawdown_series.quantile(0.75):.2f}%",
            f"{drawdown_series.quantile(0.90):.2f}%",
            f"{drawdown_series.max():.2f}%"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Severity classification
    st.subheader("⚖️ Severity Classification")
    
    # Classify drawdowns by severity
    minor = sum(1 for dd in drawdown_pct if dd < 5)
    moderate = sum(1 for dd in drawdown_pct if 5 <= dd < 15)
    significant = sum(1 for dd in drawdown_pct if 15 <= dd < 30)
    severe = sum(1 for dd in drawdown_pct if dd >= 30)
    
    total = len(drawdown_pct)
    
    severity_data = {
        "Severity": ["Minor (<5%)", "Moderate (5-15%)", "Significant (15-30%)", "Severe (≥30%)"],
        "Count": [minor, moderate, significant, severe],
        "Percentage": [
            f"{(minor/total)*100:.1f}%" if total > 0 else "0%",
            f"{(moderate/total)*100:.1f}%" if total > 0 else "0%",
            f"{(significant/total)*100:.1f}%" if total > 0 else "0%",
            f"{(severe/total)*100:.1f}%" if total > 0 else "0%"
        ]
    }
    
    severity_df = pd.DataFrame(severity_data)
    st.dataframe(severity_df, use_container_width=True, hide_index=True)


def render_duration_by_magnitude_matplotlib(analysis_results: Dict[str, Any]) -> None:
    """Matplotlib chart with integrated statistics for duration by magnitude analysis."""
    symbol = analysis_results.get('symbol', 'Unknown')
    local_drawdowns_list = analysis_results.get('local_drawdowns_list', [])
    duration_list = analysis_results.get('duration_list', [])
    
    if not local_drawdowns_list or not duration_list or len(local_drawdowns_list) != len(duration_list):
        st.warning("⚠️ Insufficient data for duration by magnitude analysis")
        return
    
    st.subheader(f"⏱️ Drawdown Duration by Magnitude - {symbol}")
    
    # Convert to positive percentages
    drawdown_pct = [abs(dd) * 100 for dd in local_drawdowns_list]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # Create scatter plot
        scatter = ax.scatter(drawdown_pct, duration_list, alpha=0.7, color='skyblue', edgecolors='black', s=50)
        
        # Add trend line if we have enough data points
        if len(drawdown_pct) > 2:
            z = np.polyfit(drawdown_pct, duration_list, 1)
            p = np.poly1d(z)
            ax.plot(sorted(drawdown_pct), p(sorted(drawdown_pct)), "r--", alpha=0.8, linewidth=2, label='Trend Line')
            ax.legend()
        
        ax.set_title(f'Drawdown Duration vs Magnitude for {symbol}')
        ax.set_xlabel('Drawdown Magnitude (%)')
        ax.set_ylabel('Duration (Days)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    finally:
        plt.close(fig)
    
    # Display comprehensive duration statistics
    st.subheader("📊 Duration Statistics")
    
    # Basic duration metrics
    col1, col2, col3, col4 = st.columns(4)
    
    duration_series = pd.Series(duration_list)
    
    with col1:
        avg_duration = analysis_results.get('avg_duration_days', 0)
        st.metric("Average Duration", f"{avg_duration:.1f} days")
    
    with col2:
        median_duration = analysis_results.get('median_duration_days', 0)
        st.metric("Median Duration", f"{median_duration:.1f} days")
    
    with col3:
        max_duration = duration_series.max() if not duration_series.empty else 0
        st.metric("Max Duration", f"{max_duration} days")
    
    with col4:
        min_duration = duration_series.min() if not duration_series.empty else 0
        st.metric("Min Duration", f"{min_duration} days")
    
    # Detailed duration statistics
    st.subheader("📈 Detailed Duration Analysis")
    
    duration_stats = {
        "Statistic": [
            "Count",
            "Mean",
            "Standard Deviation",
            "Minimum",
            "25th Percentile (Q1)",
            "50th Percentile (Median)",
            "75th Percentile (Q3)",
            "Maximum",
            "Total Days in Drawdown",
            "Average % of Time in Drawdown"
        ],
        "Value": [
            f"{len(duration_list):,}",
            f"{duration_series.mean():.1f}",
            f"{duration_series.std():.1f}",
            f"{duration_series.min()}",
            f"{duration_series.quantile(0.25):.1f}",
            f"{duration_series.quantile(0.50):.1f}",
            f"{duration_series.quantile(0.75):.1f}",
            f"{duration_series.max()}",
            f"{duration_series.sum():,}",
            f"{(duration_series.sum() / analysis_results.get('total_records', 1)) * 100:.1f}%"
        ]
    }
    
    duration_df = pd.DataFrame(duration_stats)
    st.dataframe(duration_df, use_container_width=True, hide_index=True)
    
    # Duration classification
    st.subheader("⏱️ Duration Classification")
    
    # Classify by duration ranges
    short = sum(1 for d in duration_list if d <= 30)  # 1 month or less
    medium = sum(1 for d in duration_list if 30 < d <= 90)  # 1-3 months
    long = sum(1 for d in duration_list if 90 < d <= 365)  # 3 months - 1 year
    extended = sum(1 for d in duration_list if d > 365)  # More than 1 year
    
    total = len(duration_list)
    
    duration_class_data = {
        "Duration Range": ["Short (≤30 days)", "Medium (31-90 days)", "Long (91-365 days)", "Extended (>365 days)"],
        "Count": [short, medium, long, extended],
        "Percentage": [
            f"{(short/total)*100:.1f}%" if total > 0 else "0%",
            f"{(medium/total)*100:.1f}%" if total > 0 else "0%",
            f"{(long/total)*100:.1f}%" if total > 0 else "0%",
            f"{(extended/total)*100:.1f}%" if total > 0 else "0%"
        ]
    }
    
    duration_class_df = pd.DataFrame(duration_class_data)
    st.dataframe(duration_class_df, use_container_width=True, hide_index=True)
    
    # Correlation analysis
    if len(drawdown_pct) > 2:
        correlation = np.corrcoef(drawdown_pct, duration_list)[0, 1]
        st.subheader("🔗 Correlation Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Magnitude-Duration Correlation", f"{correlation:.3f}")
        
        with col2:
            if abs(correlation) > 0.7:
                strength = "Strong"
            elif abs(correlation) > 0.4:
                strength = "Moderate"
            elif abs(correlation) > 0.2:
                strength = "Weak"
            else:
                strength = "Very Weak"
            
            direction = "Positive" if correlation > 0 else "Negative"
            st.metric("Correlation Strength", f"{strength} {direction}")


def render_drawdown_details_table(analysis_results: Dict[str, Any]) -> None:
    """
    Render detailed drawdown statistics table in the drawdown series tab.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    local_drawdowns_list = analysis_results.get('local_drawdowns_list', [])
    
    if not local_drawdowns_list:
        st.warning("⚠️ No local drawdown data available for detailed analysis")
        return
    
    st.subheader("📈 Advanced Drawdown Statistics")
    
    # Convert to positive percentages for analysis
    drawdown_pct = [abs(dd) * 100 for dd in local_drawdowns_list]
    drawdown_series = pd.Series(drawdown_pct)
    
    # Calculate comprehensive statistics
    stats_data = {
        "Statistic": [
            "Count",
            "Mean",
            "Standard Deviation",
            "Minimum",
            "10th Percentile",
            "25th Percentile (Q1)",
            "50th Percentile (Median)",
            "75th Percentile (Q3)",
            "90th Percentile",
            "Maximum",
            "Skewness",
            "Kurtosis"
        ],
        "Value": [
            f"{len(drawdown_pct):,}",
            f"{drawdown_series.mean():.2f}%",
            f"{drawdown_series.std():.2f}%",
            f"{drawdown_series.min():.2f}%",
            f"{drawdown_series.quantile(0.10):.2f}%",
            f"{drawdown_series.quantile(0.25):.2f}%",
            f"{drawdown_series.quantile(0.50):.2f}%",
            f"{drawdown_series.quantile(0.75):.2f}%",
            f"{drawdown_series.quantile(0.90):.2f}%",
            f"{drawdown_series.max():.2f}%",
            f"{drawdown_series.skew():.2f}",
            f"{drawdown_series.kurtosis():.2f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Drawdown severity classification
    st.write("**Drawdown Severity Classification**")
    
    # Classify drawdowns by severity
    minor = sum(1 for dd in drawdown_pct if dd < 5)
    moderate = sum(1 for dd in drawdown_pct if 5 <= dd < 15)
    significant = sum(1 for dd in drawdown_pct if 15 <= dd < 30)
    severe = sum(1 for dd in drawdown_pct if dd >= 30)
    
    total = len(drawdown_pct)
    
    severity_data = {
        "Severity": ["Minor (<5%)", "Moderate (5-15%)", "Significant (15-30%)", "Severe (≥30%)"],
        "Count": [minor, moderate, significant, severe],
        "Percentage": [
            f"{(minor/total)*100:.1f}%" if total > 0 else "0%",
            f"{(moderate/total)*100:.1f}%" if total > 0 else "0%",
            f"{(significant/total)*100:.1f}%" if total > 0 else "0%",
            f"{(severe/total)*100:.1f}%" if total > 0 else "0%"
        ]
    }
    
    severity_df = pd.DataFrame(severity_data)
    st.dataframe(severity_df, use_container_width=True, hide_index=True)


def render_duration_statistics_table(analysis_results: Dict[str, Any]) -> None:
    """
    Render detailed duration statistics table in the duration analysis tab.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    duration_list = analysis_results.get('duration_list', [])
    
    if not duration_list:
        st.warning("⚠️ No duration data available for analysis")
        return
    
    st.subheader("📊 Advanced Duration Statistics")
    
    duration_series = pd.Series(duration_list)
    
    # Duration statistics
    st.write("**Duration Statistics (Days)**")
    
    duration_stats = {
        "Statistic": [
            "Count",
            "Mean",
            "Standard Deviation",
            "Minimum",
            "25th Percentile (Q1)",
            "50th Percentile (Median)",
            "75th Percentile (Q3)",
            "Maximum",
            "Total Days in Drawdown",
            "Average % of Time in Drawdown"
        ],
        "Value": [
            f"{len(duration_list):,}",
            f"{duration_series.mean():.1f}",
            f"{duration_series.std():.1f}",
            f"{duration_series.min()}",
            f"{duration_series.quantile(0.25):.1f}",
            f"{duration_series.quantile(0.50):.1f}",
            f"{duration_series.quantile(0.75):.1f}",
            f"{duration_series.max()}",
            f"{duration_series.sum():,}",
            f"{(duration_series.sum() / analysis_results.get('total_records', 1)) * 100:.1f}%"
        ]
    }
    
    duration_df = pd.DataFrame(duration_stats)
    st.dataframe(duration_df, use_container_width=True, hide_index=True)
    
    # Duration classification
    st.write("**Duration Classification**")
    
    # Classify by duration ranges
    short = sum(1 for d in duration_list if d <= 30)  # 1 month or less
    medium = sum(1 for d in duration_list if 30 < d <= 90)  # 1-3 months
    long = sum(1 for d in duration_list if 90 < d <= 365)  # 3 months - 1 year
    extended = sum(1 for d in duration_list if d > 365)  # More than 1 year
    
    total = len(duration_list)
    
    duration_class_data = {
        "Duration Range": ["Short (≤30 days)", "Medium (31-90 days)", "Long (91-365 days)", "Extended (>365 days)"],
        "Count": [short, medium, long, extended],
        "Percentage": [
            f"{(short/total)*100:.1f}%" if total > 0 else "0%",
            f"{(medium/total)*100:.1f}%" if total > 0 else "0%",
            f"{(long/total)*100:.1f}%" if total > 0 else "0%",
            f"{(extended/total)*100:.1f}%" if total > 0 else "0%"
        ]
    }
    
    duration_class_df = pd.DataFrame(duration_class_data)
    st.dataframe(duration_class_df, use_container_width=True, hide_index=True)