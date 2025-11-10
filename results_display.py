"""
Results display component for the Stock Drawdown Dashboard.
Handles presentation of analysis results in Streamlit format.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional


def display_analysis_summary(analysis_results: Dict[str, Any]) -> None:
    """
    Display key drawdown metrics and summary information.
    
    Args:
        analysis_results: Dictionary containing analysis results from data_integration
    """
    symbol = analysis_results.get('symbol', 'Unknown')
    
    # Create header section
    st.subheader(f"Analysis Results for {symbol}")
    
    # Display analysis metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_date = analysis_results.get('analysis_date', datetime.now())
        if isinstance(analysis_date, datetime):
            st.metric("Analysis Date", analysis_date.strftime('%m/%d/%Y'))
        else:
            st.metric("Analysis Date", str(analysis_date))
    
    with col2:
        total_records = analysis_results.get('total_records', 0)
        st.metric("Data Points", f"{total_records:,}")
    
    with col3:
        data_start = analysis_results.get('data_period_start')
        data_end = analysis_results.get('data_period_end')
        if data_start and data_end:
            if hasattr(data_start, 'strftime') and hasattr(data_end, 'strftime'):
                period_text = f"{data_start.strftime('%m/%d/%Y')} - {data_end.strftime('%m/%d/%Y')}"
            else:
                period_text = f"{data_start} - {data_end}"
            st.metric("Data Period", period_text)
        else:
            st.metric("Data Period", "N/A")


def display_current_drawdown_status(analysis_results: Dict[str, Any]) -> None:
    """
    Display current drawdown status with highlighting.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    current_drawdown = analysis_results.get('current_drawdown', 0)
    max_drawdown = analysis_results.get('max_drawdown', 0)
    
    # Create prominent display for current drawdown status
    st.subheader("Current Drawdown Status")
    
    # Determine status color and message
    if current_drawdown == 0:
        status_color = "normal"
        status_message = "✅ At Peak - No Current Drawdown"
        status_description = "The stock is currently at or near its all-time high."
    elif current_drawdown > -0.05:  # Less than 5% drawdown
        status_color = "normal"
        status_message = "🟢 Minor Drawdown"
        status_description = "The stock is experiencing a small pullback from recent highs."
    elif current_drawdown > -0.15:  # 5-15% drawdown
        status_color = "normal"
        status_message = "🟡 Moderate Drawdown"
        status_description = "The stock is in a moderate correction phase."
    elif current_drawdown > -0.30:  # 15-30% drawdown
        status_color = "inverse"
        status_message = "🟠 Significant Drawdown"
        status_description = "The stock is experiencing a significant decline from recent highs."
    else:  # Greater than 30% drawdown
        status_color = "inverse"
        status_message = "🔴 Severe Drawdown"
        status_description = "The stock is in a severe decline phase."
    
    # Display current drawdown with appropriate styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Current Drawdown",
            value=f"{current_drawdown:.2%}",
            delta=None
        )
        
    with col2:
        st.metric(
            label="Max Drawdown (Historical)",
            value=f"{max_drawdown:.2%}",
            delta=None
        )
    
    # Status message with color coding
    if status_color == "inverse":
        st.error(f"{status_message}\n\n{status_description}")
    else:
        st.success(f"{status_message}\n\n{status_description}")
    
    # Progress bar showing current drawdown relative to max drawdown
    if max_drawdown < 0:  # Only show if we have a valid max drawdown
        progress_value = abs(current_drawdown) / abs(max_drawdown) if max_drawdown != 0 else 0
        progress_value = min(progress_value, 1.0)  # Cap at 100%
        
        st.write("**Current vs Historical Max Drawdown:**")
        st.progress(progress_value)
        st.caption(f"Current drawdown is {progress_value:.1%} of the historical maximum")


def display_key_metrics(analysis_results: Dict[str, Any]) -> None:
    """
    Display key drawdown statistics in a structured format.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    st.subheader("Key Metrics")
    
    # Core drawdown metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_drawdown = analysis_results.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    
    with col2:
        num_drawdowns = analysis_results.get('num_local_drawdowns', 0)
        st.metric("Total Drawdowns", f"{num_drawdowns}")
    
    with col3:
        avg_drawdown = analysis_results.get('avg_drawdown_pct', 0)
        st.metric("Avg Drawdown", f"{avg_drawdown:.2%}")
    
    with col4:
        avg_duration = analysis_results.get('avg_duration_days', 0)
        st.metric("Avg Duration", f"{avg_duration:.0f} days")
    
    # Peak and trough information
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


def display_analysis_metadata(analysis_results: Dict[str, Any]) -> None:
    """
    Display analysis metadata and data information.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    with st.expander("📋 Analysis Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Information**")
            
            # Data file information
            data_file = analysis_results.get('data_file_path', 'N/A')
            if data_file != 'N/A':
                import os
                st.write(f"📁 Source File: {os.path.basename(data_file)}")
            else:
                st.write("📁 Source File: N/A")
            
            # Data period
            data_start = analysis_results.get('data_period_start')
            data_end = analysis_results.get('data_period_end')
            if data_start and data_end:
                if hasattr(data_start, 'strftime') and hasattr(data_end, 'strftime'):
                    st.write(f"📅 Period: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")
                    
                    # Calculate period length
                    period_days = (data_end - data_start).days
                    st.write(f"⏱️ Duration: {period_days:,} days ({period_days/365.25:.1f} years)")
                else:
                    st.write(f"📅 Period: {data_start} to {data_end}")
            
            # Record count
            total_records = analysis_results.get('total_records', 0)
            st.write(f"📊 Records: {total_records:,} data points")
        
        with col2:
            st.write("**Analysis Information**")
            
            # Analysis timestamp
            analysis_date = analysis_results.get('analysis_date', datetime.now())
            if isinstance(analysis_date, datetime):
                st.write(f"🕐 Generated: {analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.write(f"🕐 Generated: {analysis_date}")
            
            # Output directory
            output_dir = analysis_results.get('output_directory', 'N/A')
            st.write(f"📂 Output Dir: {output_dir}")
            
            # Symbol
            symbol = analysis_results.get('symbol', 'N/A')
            st.write(f"🏷️ Symbol: {symbol}")


def render_analysis_results(analysis_results: Dict[str, Any]) -> None:
    """
    Main function to render complete analysis results.
    
    Args:
        analysis_results: Dictionary containing analysis results from data_integration
    """
    if not analysis_results:
        st.error("❌ No analysis results to display")
        return
    
    # Display all components in order
    display_analysis_summary(analysis_results)
    st.divider()
    
    display_current_drawdown_status(analysis_results)
    st.divider()
    
    display_key_metrics(analysis_results)
    st.divider()
    
    # Import and display charts with integrated statistics (single tab bar)
    try:
        from chart_renderer import render_all_charts
        render_all_charts(analysis_results)
        st.divider()
    except ImportError as e:
        st.warning(f"⚠️ Chart rendering unavailable: {e}")
    except Exception as e:
        st.error(f"❌ Error rendering charts: {e}")
    
    # Display expandable data sections
    display_expandable_data_sections(analysis_results)
    st.divider()
    
    display_analysis_metadata(analysis_results)


def display_statistics_tables(analysis_results: Dict[str, Any]) -> None:
    """
    Display comprehensive statistics tables with proper formatting.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    st.subheader("📋 Detailed Statistics")
    
    # Create tabs for different statistical views
    tab1, tab2, tab3 = st.tabs(["📊 Summary Stats", "📈 Drawdown Details", "⏱️ Duration Analysis"])
    
    with tab1:
        display_summary_statistics_table(analysis_results)
    
    with tab2:
        display_drawdown_details_table(analysis_results)
    
    with tab3:
        display_duration_statistics_table(analysis_results)


def display_summary_statistics_table(analysis_results: Dict[str, Any]) -> None:
    """
    Display summary statistics in a formatted table.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    # Core metrics table
    st.write("**Core Drawdown Metrics**")
    
    core_metrics = {
        "Metric": [
            "Maximum Drawdown",
            "Current Drawdown", 
            "Number of Drawdown Events",
            "Average Drawdown",
            "Median Drawdown",
            "Average Duration",
            "Median Duration"
        ],
        "Value": [
            f"{analysis_results.get('max_drawdown', 0):.2%}",
            f"{analysis_results.get('current_drawdown', 0):.2%}",
            f"{analysis_results.get('num_local_drawdowns', 0):,}",
            f"{analysis_results.get('avg_drawdown_pct', 0):.2%}",
            f"{analysis_results.get('median_drawdown_pct', 0):.2%}",
            f"{analysis_results.get('avg_duration_days', 0):.1f} days",
            f"{analysis_results.get('median_duration_days', 0):.1f} days"
        ]
    }
    
    core_df = pd.DataFrame(core_metrics)
    st.dataframe(core_df, use_container_width=True, hide_index=True)
    
    # Peak and Trough details
    st.write("**Peak & Trough Information**")
    
    peak_trough_data = {
        "Type": ["Historical Peak", "Historical Trough"],
        "Date": [
            analysis_results.get('peak_date', 'N/A'),
            analysis_results.get('trough_date', 'N/A')
        ],
        "Price": [
            f"${analysis_results.get('peak_value', 0):.2f}",
            f"${analysis_results.get('trough_value', 0):.2f}"
        ]
    }
    
    # Format dates if they are datetime objects
    for i, date_val in enumerate(peak_trough_data["Date"]):
        if hasattr(date_val, 'strftime'):
            peak_trough_data["Date"][i] = date_val.strftime('%Y-%m-%d')
        else:
            peak_trough_data["Date"][i] = str(date_val)
    
    peak_trough_df = pd.DataFrame(peak_trough_data)
    st.dataframe(peak_trough_df, use_container_width=True, hide_index=True)


def display_drawdown_details_table(analysis_results: Dict[str, Any]) -> None:
    """
    Display detailed drawdown statistics and distribution.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    local_drawdowns_list = analysis_results.get('local_drawdowns_list', [])
    
    if not local_drawdowns_list:
        st.warning("⚠️ No local drawdown data available for detailed analysis")
        return
    
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


def display_duration_statistics_table(analysis_results: Dict[str, Any]) -> None:
    """
    Display duration statistics and analysis.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    duration_list = analysis_results.get('duration_list', [])
    
    if not duration_list:
        st.warning("⚠️ No duration data available for analysis")
        return
    
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


def display_expandable_data_sections(analysis_results: Dict[str, Any]) -> None:
    """
    Create expandable sections for detailed data exploration.
    
    Args:
        analysis_results: Dictionary containing analysis results
    """
    st.subheader("Detailed Data Exploration")
    
    # Raw data series in expandable sections
    with st.expander("📈 Cumulative Return Series", expanded=False):
        cumulative_return_series = analysis_results.get('cumulative_return_series')
        if cumulative_return_series is not None and not cumulative_return_series.empty:
            # Convert to DataFrame for better display
            cum_return_df = cumulative_return_series.reset_index()
            cum_return_df.columns = ['Date', 'Cumulative_Return']
            cum_return_df['Return_Percentage'] = (cum_return_df['Cumulative_Return'] - 1) * 100
            
            # Format date column
            if 'Date' in cum_return_df.columns:
                cum_return_df['Date'] = pd.to_datetime(cum_return_df['Date']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                cum_return_df.style.format({
                    'Cumulative_Return': '{:.4f}',
                    'Return_Percentage': '{:.2f}%'
                }),
                use_container_width=True,
                height=300
            )
            
            # Download button for the data
            csv = cum_return_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cumulative Return Data",
                data=csv,
                file_name=f"{analysis_results.get('symbol', 'stock')}_cumulative_returns.csv",
                mime='text/csv'
            )
        else:
            st.warning("No cumulative return data available")
    
    with st.expander("📉 Drawdown Series", expanded=False):
        drawdown_series = analysis_results.get('drawdown_series')
        if drawdown_series is not None and not drawdown_series.empty:
            # Convert to DataFrame
            drawdown_df = drawdown_series.reset_index()
            drawdown_df.columns = ['Date', 'Drawdown']
            drawdown_df['Drawdown_Percentage'] = drawdown_df['Drawdown'] * 100
            
            # Format date column
            if 'Date' in drawdown_df.columns:
                drawdown_df['Date'] = pd.to_datetime(drawdown_df['Date']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                drawdown_df.style.format({
                    'Drawdown': '{:.4f}',
                    'Drawdown_Percentage': '{:.2f}%'
                }),
                use_container_width=True,
                height=300
            )
            
            # Download button
            csv = drawdown_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Drawdown Series Data",
                data=csv,
                file_name=f"{analysis_results.get('symbol', 'stock')}_drawdown_series.csv",
                mime='text/csv'
            )
        else:
            st.warning("No drawdown series data available")
    
    with st.expander("📊 Local Drawdown Events", expanded=False):
        local_drawdowns_list = analysis_results.get('local_drawdowns_list', [])
        duration_list = analysis_results.get('duration_list', [])
        
        if local_drawdowns_list and duration_list and len(local_drawdowns_list) == len(duration_list):
            # Create detailed events table
            events_data = {
                'Event_Number': range(1, len(local_drawdowns_list) + 1),
                'Drawdown_Magnitude': [f"{abs(dd):.2%}" for dd in local_drawdowns_list],
                'Duration_Days': duration_list,
                'Severity': []
            }
            
            # Classify severity
            for dd in local_drawdowns_list:
                dd_pct = abs(dd) * 100
                if dd_pct < 5:
                    events_data['Severity'].append('Minor')
                elif dd_pct < 15:
                    events_data['Severity'].append('Moderate')
                elif dd_pct < 30:
                    events_data['Severity'].append('Significant')
                else:
                    events_data['Severity'].append('Severe')
            
            events_df = pd.DataFrame(events_data)
            
            st.dataframe(events_df, use_container_width=True, height=300)
            
            # Summary statistics for the table
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Events", len(local_drawdowns_list))
            with col2:
                avg_magnitude = sum([abs(dd) for dd in local_drawdowns_list]) / len(local_drawdowns_list)
                st.metric("Avg Magnitude", f"{avg_magnitude:.2%}")
            with col3:
                avg_duration = sum(duration_list) / len(duration_list)
                st.metric("Avg Duration", f"{avg_duration:.1f} days")
            
            # Download button
            csv = events_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Drawdown Events Data",
                data=csv,
                file_name=f"{analysis_results.get('symbol', 'stock')}_drawdown_events.csv",
                mime='text/csv'
            )
        else:
            st.warning("No local drawdown events data available")


def render_multiple_analysis_results(analysis_history: list) -> None:
    """
    Render results for multiple analyses with navigation.
    
    Args:
        analysis_history: List of analysis result dictionaries
    """
    if not analysis_history:
        st.info("💡 No analysis results available yet.")
        return
    
    # Create tabs for multiple analyses
    if len(analysis_history) == 1:
        # Single analysis - display directly
        render_analysis_results(analysis_history[0])
    else:
        # Multiple analyses - create tabs
        tab_labels = []
        for i, result in enumerate(analysis_history):
            symbol = result.get('symbol', f'Analysis {i+1}')
            analysis_date = result.get('analysis_date', datetime.now())
            if isinstance(analysis_date, datetime):
                date_str = analysis_date.strftime('%m/%d')
            else:
                date_str = str(analysis_date)[:10] if len(str(analysis_date)) > 10 else str(analysis_date)
            tab_labels.append(f"{symbol} ({date_str})")
        
        tabs = st.tabs(tab_labels)
        
        for i, (tab, result) in enumerate(zip(tabs, analysis_history)):
            with tab:
                render_analysis_results(result)