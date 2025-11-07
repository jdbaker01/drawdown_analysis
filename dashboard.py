"""
Stock Drawdown Dashboard - Streamlit Application

A web-based dashboard for stock drawdown analysis that integrates with existing
stock data download and analysis functionality.
"""

import streamlit as st
from typing import Optional, Tuple
import re


def main():
    """
    Main Streamlit application entry point.
    Configures the page and orchestrates the dashboard layout.
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="Stock Drawdown Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"  # Start with sidebar collapsed for clean look
    )
    
    # Add custom CSS for sky blue Analyze button only
    st.markdown("""
    <style>
    /* Target only the main Analyze form submit button */
    div[data-testid="stFormSubmitButton"] button[type="submit"] {
        background-color: #87CEEB !important;
        border-color: #87CEEB !important;
        color: white !important;
        border-radius: 0.5rem !important;
    }
    
    div[data-testid="stFormSubmitButton"] button[type="submit"]:hover {
        background-color: #6BB6E0 !important;
        border-color: #6BB6E0 !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(135, 206, 235, 0.3) !important;
    }
    
        background-color: #5AA5D4 !important;
        border-color: #5AA5D4 !important;
        color: white !important;
        transform: translateY(0px) !important;
    }
    
    div[data-testid="stFormSubmitButton"] button[type="submit"]:focus {
        background-color: #87CEEB !important;
        border-color: #87CEEB !important;
        color: white !important;
        box-shadow: 0 0 0 2px rgba(135, 206, 235, 0.5) !important;
        outline: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Handle URL parameters for deep linking
    handle_url_parameters()
    
    # Render sidebar with analysis history (only if there's history)
    if st.session_state.analysis_history:
        render_analysis_history_sidebar()
    
    # Render main dashboard components - Google-like layout
    render_header()
    render_input_section()
    render_results_section()


def handle_url_parameters():
    """
    Handle URL parameters for deep linking to specific stock analyses.
    """
    # Get query parameters from URL
    query_params = st.query_params
    
    # Check if there's a symbol parameter in the URL
    if "symbol" in query_params:
        symbol = query_params["symbol"].upper().strip()
        
        # Validate the symbol
        is_valid, error_msg = validate_symbol_format(symbol)
        
        if is_valid:
            # Check if we already have analysis for this symbol
            existing_analysis = get_analysis_by_symbol(symbol)
            
            if not existing_analysis:
                # If we don't have analysis for this symbol, trigger analysis
                # But only if we're not already processing something
                if st.session_state.processing_status == 'idle':
                    # Set the symbol in session state so the input shows it
                    if 'url_symbol' not in st.session_state:
                        st.session_state.url_symbol = symbol
                        # Trigger analysis for the URL symbol
                        handle_analysis_request(symbol)
            else:
                # We have existing analysis, make sure it's selected
                for i, analysis in enumerate(st.session_state.analysis_history):
                    if analysis.get('symbol', '').upper() == symbol:
                        st.session_state.selected_analysis_index = i
                        break


def update_url_with_symbol(symbol: str):
    """
    Update the URL to include the current symbol for deep linking.
    
    Args:
        symbol: Stock symbol to add to URL
    """
    # Update query parameters
    st.query_params["symbol"] = symbol.upper()


def validate_symbol_format(symbol: str) -> Tuple[bool, str]:
    """
    Validate stock symbol format with detailed error messages.
    
    Args:
        symbol: The stock symbol to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not symbol:
        return False, "Symbol cannot be empty"
    
    # Remove whitespace and convert to uppercase for validation
    clean_symbol = symbol.strip().upper()
    
    if not clean_symbol:
        return False, "Symbol cannot be empty or only whitespace"
    
    if len(clean_symbol) < 1:
        return False, "Symbol must be at least 1 character long"
    
    if len(clean_symbol) > 5:
        return False, "Symbol cannot be longer than 5 characters"
    
    # Check if symbol contains only letters
    if not re.match(r'^[A-Z]+$', clean_symbol):
        return False, "Symbol must contain only letters (A-Z)"
    
    return True, ""


def initialize_session_state():
    """
    Initialize Streamlit session state variables for analysis history and processing status.
    
    This function sets up all necessary session state variables according to the design
    document schema for managing multi-stock analysis sessions.
    """
    # Analysis history - List of AnalysisResult dictionaries
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Current processing state
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = None
    
    # Processing status tracking
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = 'idle'  # 'idle', 'downloading', 'analyzing', 'complete', 'error'
    
    # Error message state management
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
    # UI state management
    if 'show_advanced_stats' not in st.session_state:
        st.session_state.show_advanced_stats = False
    
    # Navigation state for multiple analyses
    if 'selected_analysis_index' not in st.session_state:
        st.session_state.selected_analysis_index = 0
    
    # Session metadata
    if 'session_start_time' not in st.session_state:
        from datetime import datetime
        st.session_state.session_start_time = datetime.now()
    
    # Analysis counter for unique identification
    if 'analysis_counter' not in st.session_state:
        st.session_state.analysis_counter = 0
    
    # Last successful analysis symbol for quick reference
    if 'last_successful_symbol' not in st.session_state:
        st.session_state.last_successful_symbol = None
    
    # Processing progress tracking
    if 'processing_progress' not in st.session_state:
        st.session_state.processing_progress = 0
    
    # Clear any stale processing state on initialization
    if st.session_state.processing_status in ['downloading', 'analyzing']:
        st.session_state.processing_status = 'idle'
        st.session_state.current_symbol = None
        st.session_state.processing_progress = 0


def update_processing_status(status: str, symbol: Optional[str] = None, progress: int = 0, error_message: Optional[str] = None):
    """
    Update processing status and related session state variables.
    
    Args:
        status: New processing status ('idle', 'downloading', 'analyzing', 'complete', 'error')
        symbol: Current symbol being processed (optional)
        progress: Processing progress percentage (0-100)
        error_message: Error message if status is 'error' (optional)
    """
    st.session_state.processing_status = status
    st.session_state.processing_progress = progress
    
    if symbol is not None:
        st.session_state.current_symbol = symbol
    
    if error_message is not None:
        st.session_state.error_message = error_message
    elif status != 'error':
        # Clear error message if not setting error status
        st.session_state.error_message = None


def add_analysis_result(analysis_result: dict):
    """
    Add a new analysis result to the session history.
    
    Args:
        analysis_result: Dictionary containing analysis results
    """
    # Increment analysis counter
    st.session_state.analysis_counter += 1
    
    # Add unique ID to the result
    analysis_result['session_id'] = st.session_state.analysis_counter
    
    # Add to history
    st.session_state.analysis_history.append(analysis_result)
    
    # Update last successful symbol
    if 'symbol' in analysis_result:
        st.session_state.last_successful_symbol = analysis_result['symbol']
    
    # Set selected index to the newest analysis
    st.session_state.selected_analysis_index = len(st.session_state.analysis_history) - 1


def get_analysis_by_symbol(symbol: str) -> Optional[dict]:
    """
    Retrieve the most recent analysis result for a given symbol.
    
    Args:
        symbol: Stock symbol to search for
        
    Returns:
        Most recent analysis result dictionary or None if not found
    """
    symbol = symbol.upper().strip()
    
    # Search from most recent to oldest
    for analysis in reversed(st.session_state.analysis_history):
        if analysis.get('symbol', '').upper() == symbol:
            return analysis
    
    return None


def get_session_summary() -> dict:
    """
    Get a summary of the current session state.
    
    Returns:
        Dictionary containing session summary information
    """
    from datetime import datetime
    
    total_analyses = len(st.session_state.analysis_history)
    unique_symbols = set()
    
    for analysis in st.session_state.analysis_history:
        if 'symbol' in analysis:
            unique_symbols.add(analysis['symbol'])
    
    session_duration = datetime.now() - st.session_state.session_start_time
    
    return {
        'session_start_time': st.session_state.session_start_time,
        'session_duration': session_duration,
        'total_analyses': total_analyses,
        'unique_symbols_count': len(unique_symbols),
        'unique_symbols': list(unique_symbols),
        'current_status': st.session_state.processing_status,
        'current_symbol': st.session_state.current_symbol,
        'last_successful_symbol': st.session_state.last_successful_symbol
    }


def clear_session_state():
    """
    Clear all session state variables (useful for testing or reset functionality).
    """
    keys_to_clear = [
        'analysis_history', 'current_symbol', 'processing_status', 'error_message',
        'show_advanced_stats', 'selected_analysis_index', 'session_start_time',
        'analysis_counter', 'last_successful_symbol', 'processing_progress'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear URL parameters
    st.query_params.clear()
    
    # Reinitialize
    initialize_session_state()


def render_header():
    """Render the application header with centered title."""
    # Add spacing for better vertical centering
    st.write("")
    st.write("")
    
    # Center the title using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>📈 Stock Drawdown Dashboard</h1>", unsafe_allow_html=True)


def render_input_section():
    """Render the stock symbol input section with Google-like clean interface and Enter key support."""
    # Minimal spacing since title is already centered above
    st.write("")
    
    # Create centered columns for Google-like layout (same width as title)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Use a form to enable Enter key functionality
        with st.form(key="symbol_form", clear_on_submit=False):
            # Get symbol from URL parameters if available
            query_params = st.query_params
            default_symbol = ""
            if "symbol" in query_params:
                default_symbol = query_params["symbol"].upper()
            
            # Clean input without label
            symbol_input = st.text_input(
                label="",
                placeholder="Enter stock symbol (e.g., AAPL, MSFT, SPY) and press Enter",
                value=default_symbol,
                key="symbol_input_form",
                label_visibility="collapsed"
            )
            
            # Center the analyze button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                # Disable analyze button during processing
                button_disabled = st.session_state.processing_status in ['downloading', 'analyzing']
                
                if button_disabled:
                    analyze_button = st.form_submit_button("Processing...", disabled=True, use_container_width=True, type="primary")
                else:
                    analyze_button = st.form_submit_button("Analyze", use_container_width=True, type="primary")
        
        # Minimal validation feedback (outside the form to avoid form resubmission)
        if symbol_input:
            validation_result, error_msg = validate_symbol_format(symbol_input)
            if not validation_result:
                st.error(f"{error_msg}")
    
    st.write("")
    
    # Show session summary if there are previous analyses (below input)
    if st.session_state.analysis_history:
        st.divider()
        render_session_summary()
    
    # Handle analysis request (triggered by Enter key or button click)
    if analyze_button and not button_disabled:
        if not symbol_input:
            st.error("Please enter a stock symbol")
        else:
            validation_result, error_msg = validate_symbol_format(symbol_input)
            if not validation_result:
                st.error(f"{error_msg}")
            else:
                handle_analysis_request(symbol_input.upper().strip())


def render_session_summary():
    """Render a summary of the current analysis session."""
    session_summary = get_session_summary()
    
    # Single metric row - just show the current symbol
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if session_summary['last_successful_symbol']:
            st.metric("Symbol", session_summary['last_successful_symbol'])
        else:
            st.metric("Status", session_summary['current_status'].title())
    
    with col2:
        # Session management controls
        if st.button("🗑️ Clear Session", help="Clear all analysis history"):
            clear_session_state()
            st.rerun()


def render_analysis_history_sidebar():
    """Render analysis history in the sidebar for easy navigation."""
    if not st.session_state.analysis_history:
        return
    
    with st.sidebar:
        st.subheader("📚 Analysis History")
        
        for i, analysis in enumerate(st.session_state.analysis_history):
            symbol = analysis.get('symbol', f'Analysis {i+1}')
            analysis_date = analysis.get('analysis_date')
            
            if analysis_date and hasattr(analysis_date, 'strftime'):
                date_str = analysis_date.strftime('%m/%d %H:%M')
            else:
                date_str = f"#{i+1}"
            
            # Create clickable button for each analysis
            if st.button(f"{symbol} ({date_str})", key=f"history_{i}"):
                st.session_state.selected_analysis_index = i
                # Update URL when switching to different analysis
                update_url_with_symbol(symbol)
                st.rerun()
        
        st.divider()
        
        # Session controls
        if st.button("🔄 New Session"):
            clear_session_state()
            st.rerun()
        
        # Export session data
        if st.button("📥 Export Session"):
            export_session_data()


def export_session_data():
    """Export session analysis data as JSON."""
    import json
    from datetime import datetime
    
    if not st.session_state.analysis_history:
        st.warning("No analysis data to export")
        return
    
    # Prepare export data
    export_data = {
        'session_info': get_session_summary(),
        'analyses': []
    }
    
    # Convert analysis results to JSON-serializable format
    for analysis in st.session_state.analysis_history:
        export_analysis = {}
        for key, value in analysis.items():
            if hasattr(value, 'strftime'):  # datetime objects
                export_analysis[key] = value.isoformat()
            elif hasattr(value, 'to_dict'):  # pandas objects
                export_analysis[key] = value.to_dict()
            elif hasattr(value, 'tolist'):  # numpy arrays
                export_analysis[key] = value.tolist()
            else:
                export_analysis[key] = value
        
        export_data['analyses'].append(export_analysis)
    
    # Convert session info datetime objects
    session_info = export_data['session_info']
    if 'session_start_time' in session_info and hasattr(session_info['session_start_time'], 'isoformat'):
        session_info['session_start_time'] = session_info['session_start_time'].isoformat()
    if 'session_duration' in session_info:
        session_info['session_duration'] = str(session_info['session_duration'])
    
    # Create JSON string
    json_str = json.dumps(export_data, indent=2, default=str)
    
    # Create download button
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"stock_analysis_session_{timestamp}.json"
    
    st.download_button(
        label="📥 Download Session Data",
        data=json_str,
        file_name=filename,
        mime="application/json"
    )


def render_results_section():
    """Render the analysis results section with multi-stock support."""
    if st.session_state.processing_status == 'idle' and not st.session_state.analysis_history:
        return  # Show nothing on initial load for clean interface
    
    elif st.session_state.processing_status in ['downloading', 'analyzing']:
        render_processing_status()
        
        # Show previous results below processing status if available
        if st.session_state.analysis_history:
            st.divider()
            st.subheader("📚 Previous Analysis Results")
            render_analysis_navigation()
            render_analysis_results()
    
    elif st.session_state.processing_status == 'complete':
        render_processing_status()
        
        # Show results if available
        if st.session_state.analysis_history:
            st.divider()
            render_analysis_navigation()
            render_analysis_results()
    
    elif st.session_state.processing_status == 'error':
        render_error_message()
        
        # Show previous results if available
        if st.session_state.analysis_history:
            st.divider()
            st.subheader("📚 Previous Analysis Results")
            render_analysis_navigation()
            render_analysis_results()
    
    elif st.session_state.analysis_history:
        render_analysis_navigation()
        render_analysis_results()


def render_analysis_navigation():
    """Render navigation controls for multiple analyses."""
    if len(st.session_state.analysis_history) <= 1:
        return
    
    st.subheader("📊 Analysis Results")
    
    # Create tabs for different analyses
    tab_labels = []
    for i, analysis in enumerate(st.session_state.analysis_history):
        symbol = analysis.get('symbol', f'Analysis {i+1}')
        analysis_date = analysis.get('analysis_date')
        
        if analysis_date and hasattr(analysis_date, 'strftime'):
            date_str = analysis_date.strftime('%m/%d %H:%M')
        else:
            date_str = f"#{i+1}"
        
        tab_labels.append(f"{symbol} ({date_str})")
    
    # Create tabs
    tabs = st.tabs(tab_labels)
    
    # Store the selected tab index
    for i, tab in enumerate(tabs):
        with tab:
            st.session_state.selected_analysis_index = i
            break  # Only process the active tab
    
    return tabs


def render_analysis_results():
    """Render analysis results with navigation support."""
    if not st.session_state.analysis_history:
        st.info("💡 No analysis results available yet.")
        return
    
    # Handle single vs multiple analyses
    if len(st.session_state.analysis_history) == 1:
        # Single analysis - display directly
        try:
            from results_display import render_analysis_results as display_results
            display_results(st.session_state.analysis_history[0])
        except ImportError as e:
            st.error(f"❌ Results display module not available: {e}")
            render_fallback_results(st.session_state.analysis_history[0])
        except Exception as e:
            st.error(f"❌ Error displaying results: {e}")
            render_fallback_results(st.session_state.analysis_history[0])
    
    else:
        # Multiple analyses - use tabs
        tab_labels = []
        for i, analysis in enumerate(st.session_state.analysis_history):
            symbol = analysis.get('symbol', f'Analysis {i+1}')
            analysis_date = analysis.get('analysis_date')
            
            if analysis_date and hasattr(analysis_date, 'strftime'):
                date_str = analysis_date.strftime('%m/%d %H:%M')
            else:
                date_str = f"#{i+1}"
            
            tab_labels.append(f"{symbol} ({date_str})")
        
        tabs = st.tabs(tab_labels)
        
        for i, (tab, analysis) in enumerate(zip(tabs, st.session_state.analysis_history)):
            with tab:
                try:
                    from results_display import render_analysis_results as display_results
                    display_results(analysis)
                except ImportError as e:
                    st.error(f"❌ Results display module not available: {e}")
                    render_fallback_results(analysis)
                except Exception as e:
                    st.error(f"❌ Error displaying results: {e}")
                    render_fallback_results(analysis)


def render_fallback_results(analysis_result: dict):
    """Render basic analysis results when the full display module is not available."""
    symbol = analysis_result.get('symbol', 'Unknown')
    st.subheader(f"📊 Analysis Results for {symbol}")
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_dd = analysis_result.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_dd:.2%}")
    
    with col2:
        current_dd = analysis_result.get('current_drawdown', 0)
        st.metric("Current Drawdown", f"{current_dd:.2%}")
    
    with col3:
        num_events = analysis_result.get('num_local_drawdowns', 0)
        st.metric("Drawdown Events", f"{num_events}")
    
    # Analysis metadata
    analysis_date = analysis_result.get('analysis_date')
    if analysis_date and hasattr(analysis_date, 'strftime'):
        st.write(f"**Analysis Date:** {analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_records = analysis_result.get('total_records', 0)
    st.write(f"**Data Points:** {total_records:,}")
    
    if analysis_result.get('status') == 'placeholder':
        st.info("📝 This is placeholder data. Full analysis will be available when integrated with analysis modules.")


def render_processing_status():
    """Render processing status indicators with detailed progress messages."""
    current_symbol = st.session_state.current_symbol or "symbol"
    
    if st.session_state.processing_status == 'downloading':
        st.subheader("📥 Downloading Data")
        
        # Create progress container
        progress_container = st.container()
        with progress_container:
            # Progress bar (indeterminate for now)
            progress_bar = st.progress(0)
            
            # Status messages
            with st.spinner(f"Downloading stock data for {current_symbol}..."):
                st.info(f"🔄 Fetching historical data for **{current_symbol}**")
                st.markdown("""
                **Progress:**
                - ✅ Validating symbol format
                - 🔄 Connecting to data source
                - ⏳ Downloading historical prices
                - ⏳ Saving data to local storage
                """)
                
                # Simulate progress updates
                import time
                for i in range(0, 101, 25):
                    progress_bar.progress(i)
                    time.sleep(0.1)
    
    elif st.session_state.processing_status == 'analyzing':
        st.subheader("📊 Analyzing Drawdowns")
        
        # Create progress container
        progress_container = st.container()
        with progress_container:
            # Progress bar
            progress_bar = st.progress(0)
            
            # Status messages
            with st.spinner(f"Computing drawdown analysis for {current_symbol}..."):
                st.info(f"🧮 Processing **{current_symbol}** data for drawdown patterns")
                st.markdown("""
                **Analysis Steps:**
                - ✅ Loading price data
                - 🔄 Calculating cumulative returns
                - ⏳ Computing drawdown series
                - ⏳ Identifying local drawdowns
                - ⏳ Generating statistics and charts
                """)
                
                # Simulate progress updates
                import time
                for i in range(0, 101, 20):
                    progress_bar.progress(i)
                    time.sleep(0.1)
    
    elif st.session_state.processing_status == 'complete':
        st.success(f"✅ Analysis complete for **{current_symbol}**!")
        
        # Show completion summary
        with st.expander("📋 Processing Summary", expanded=False):
            st.markdown(f"""
            **Completed Steps:**
            - ✅ Downloaded data for {current_symbol}
            - ✅ Computed drawdown analysis
            - ✅ Generated visualizations
            - ✅ Calculated statistics
            
            **Ready to view results below** 👇
            """)


def render_error_message():
    """Render error messages with troubleshooting guidance."""
    if st.session_state.error_message:
        st.error(f"❌ {st.session_state.error_message}")
        
        with st.expander("Troubleshooting Tips"):
            st.markdown("""
            **Common issues and solutions:**
            - **Invalid symbol**: Ensure the symbol exists and is correctly spelled
            - **Network issues**: Check your internet connection
            - **Data unavailable**: Try a different symbol or check if markets are open
            - **File permissions**: Ensure the application can write to the data directory
            """)


def render_analysis_results():
    """Render analysis results and history."""
    if st.session_state.analysis_history:
        try:
            from results_display import render_multiple_analysis_results
            render_multiple_analysis_results(st.session_state.analysis_history)
        except ImportError as e:
            st.error(f"❌ Results display module not available: {e}")
            # Fallback to basic display
            st.subheader("Analysis Results")
            st.success(f"✅ Analysis complete for {len(st.session_state.analysis_history)} symbol(s)")
            
            for i, result in enumerate(st.session_state.analysis_history):
                with st.expander(f"Analysis {i+1}: {result.get('symbol', 'Unknown')}", expanded=(i == 0)):
                    st.write("📊 Detailed results will be displayed here")
        except Exception as e:
            st.error(f"❌ Error displaying results: {e}")
            st.write("Please check the application logs for more details.")
    else:
        st.info("💡 No analysis results available yet.")


def handle_analysis_request(symbol: str):
    """
    Handle analysis request for a stock symbol using the real data integration.
    """
    try:
        # Update URL with the symbol for deep linking
        update_url_with_symbol(symbol)
        
        # Import the data integration module
        from data_integration import ensure_stock_data, run_drawdown_analysis
        
        # Update processing status to downloading
        update_processing_status('downloading', symbol, 0)
        
        # Step 1: Ensure stock data exists (download if needed)
        success, file_path, message = ensure_stock_data(symbol)
        
        if not success:
            update_processing_status('error', symbol, 0, message)
            st.rerun()
            return
        
        # Update processing status to analyzing
        update_processing_status('analyzing', symbol, 50)
        
        # Step 2: Run drawdown analysis
        success, analysis_results, message = run_drawdown_analysis(symbol, file_path)
        
        if not success:
            update_processing_status('error', symbol, 0, message)
            st.rerun()
            return
        
        # Add result to session history
        add_analysis_result(analysis_results)
        
        # Update status to complete
        update_processing_status('complete', symbol, 100)
        
        st.rerun()
        
    except ImportError as e:
        error_msg = f"Data integration module not available: {str(e)}"
        update_processing_status('error', symbol, 0, error_msg)
        st.rerun()
        
    except Exception as e:
        error_msg = f"Unexpected error during analysis: {str(e)}"
        update_processing_status('error', symbol, 0, error_msg)
        st.rerun()


if __name__ == "__main__":
    main()