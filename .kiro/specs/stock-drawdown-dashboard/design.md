# Design Document

## Overview

The Stock Drawdown Dashboard is a Streamlit-based web application that provides an intuitive interface for stock drawdown analysis. The design leverages existing functionality from `download_stock_data.py` and `drawdown_analysis.py` modules, wrapping them in a user-friendly web interface. The application follows a single-page design with progressive disclosure of information and real-time feedback during processing.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Integration    │    │   Existing      │
│   Dashboard     │◄──►│   Layer          │◄──►│   Modules       │
│   (UI Layer)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌────▼────┐             ┌────▼────┐             ┌────▼────┐
    │ Input   │             │ Data    │             │ Stock   │
    │ Widget  │             │ Manager │             │ Data    │
    │ State   │             │         │             │ Files   │
    └─────────┘             └─────────┘             └─────────┘
```

### Component Architecture

1. **Streamlit Dashboard (UI Layer)**
   - User input handling and validation
   - Progress indication and status updates
   - Results presentation and visualization
   - Session state management

2. **Integration Layer**
   - Wrapper functions for existing modules
   - Data flow coordination
   - Error handling and user feedback
   - File system interaction

3. **Existing Modules (Unchanged)**
   - `download_stock_data.py`: Data acquisition
   - `drawdown_analysis.py`: Analysis computation

## Components and Interfaces

### 1. Main Dashboard Component (`dashboard.py`)

**Purpose**: Primary Streamlit application entry point

**Key Functions**:
- `main()`: Application entry point and layout orchestration
- `render_header()`: Application title and description
- `render_input_section()`: Stock symbol input and validation
- `render_results_section()`: Analysis results display
- `handle_analysis_request()`: Coordinate data download and analysis

**State Management**:
- `st.session_state.analysis_history`: List of completed analyses
- `st.session_state.current_symbol`: Currently processing symbol
- `st.session_state.processing_status`: Current operation status

### 2. Data Integration Component (`data_integration.py`)

**Purpose**: Bridge between Streamlit UI and existing analysis modules

**Key Functions**:
```python
def ensure_stock_data(symbol: str) -> tuple[bool, str, str]:
    """
    Ensures stock data exists for the given symbol.
    Returns: (success, file_path, message)
    """

def run_drawdown_analysis(symbol: str, data_file: str) -> tuple[bool, dict, str]:
    """
    Executes drawdown analysis and returns structured results.
    Returns: (success, analysis_results, message)
    """

def validate_stock_symbol(symbol: str) -> tuple[bool, str]:
    """
    Validates stock symbol format and availability.
    Returns: (is_valid, error_message)
    """
```

### 3. Results Display Component (`results_display.py`)

**Purpose**: Render analysis results in Streamlit format

**Key Functions**:
- `display_analysis_summary()`: Key metrics overview
- `display_charts()`: Interactive chart rendering
- `display_statistics_tables()`: Formatted data tables
- `display_analysis_metadata()`: Analysis date, period, etc.

### 4. Chart Rendering Component (`chart_renderer.py`)

**Purpose**: Convert matplotlib figures to Streamlit-compatible format

**Key Functions**:
- `render_cumulative_return_chart()`: Streamlit line chart
- `render_drawdown_series_chart()`: Streamlit area chart
- `render_local_drawdowns_histogram()`: Matplotlib integration
- `render_duration_by_magnitude_chart()`: Custom bar chart

## Data Models

### Analysis Result Structure

```python
@dataclass
class AnalysisResult:
    symbol: str
    analysis_date: datetime
    data_period_start: datetime
    data_period_end: datetime
    total_records: int
    
    # Core metrics
    max_drawdown: float
    current_drawdown: float
    peak_date: datetime
    peak_value: float
    trough_date: datetime
    trough_value: float
    
    # Local drawdowns statistics
    num_local_drawdowns: int
    avg_drawdown_pct: float
    median_drawdown_pct: float
    avg_duration_days: float
    median_duration_days: float
    
    # Chart data
    cumulative_return_series: pd.Series
    drawdown_series: pd.Series
    local_drawdowns_list: List[float]
    duration_list: List[int]
    
    # File paths
    data_file_path: str
    output_directory: str
```

### Session State Schema

```python
# Streamlit session state structure
{
    'analysis_history': List[AnalysisResult],
    'current_symbol': Optional[str],
    'processing_status': str,  # 'idle', 'downloading', 'analyzing', 'complete', 'error'
    'error_message': Optional[str],
    'show_advanced_stats': bool,
    'selected_analysis_index': int
}
```

## Error Handling

### Error Categories and Responses

1. **Input Validation Errors**
   - Invalid symbol format → Clear format guidance
   - Empty input → Prompt for symbol entry
   - Special characters → Symbol format requirements

2. **Data Download Errors**
   - Network connectivity → Retry suggestion with offline mode
   - Invalid symbol → Symbol verification and alternatives
   - API rate limits → Wait time indication

3. **Analysis Processing Errors**
   - Insufficient data → Data requirements explanation
   - File corruption → Re-download suggestion
   - Memory issues → Data size limitations

4. **System Errors**
   - File permissions → Directory access guidance
   - Disk space → Storage requirements
   - Module import → Dependency installation

### Error Display Strategy

- **Toast Notifications**: For temporary status updates
- **Error Containers**: For detailed error information with solutions
- **Inline Validation**: Real-time input feedback
- **Progress Indicators**: Clear status during long operations

## Testing Strategy

### Unit Testing Approach

1. **Component Testing**
   - Input validation functions
   - Data integration wrapper functions
   - Chart rendering utilities
   - Error handling mechanisms

2. **Integration Testing**
   - End-to-end analysis workflow
   - File system interactions
   - Module integration points
   - Session state management

3. **UI Testing**
   - Widget behavior validation
   - Chart rendering verification
   - Error message display
   - Navigation flow testing

### Test Data Strategy

- **Mock Data**: Synthetic stock data for consistent testing
- **Sample Files**: Known-good CSV files for integration tests
- **Error Scenarios**: Malformed data for error handling tests
- **Performance Data**: Large datasets for performance validation

### Testing Tools

- **pytest**: Core testing framework
- **streamlit-testing**: UI component testing
- **unittest.mock**: Module mocking for isolation
- **pandas.testing**: DataFrame comparison utilities

## Performance Considerations

### Optimization Strategies

1. **Caching Implementation**
   - `@st.cache_data` for analysis results
   - `@st.cache_resource` for chart objects
   - File-based caching for downloaded data

2. **Progressive Loading**
   - Lazy loading of chart components
   - Incremental results display
   - Background processing indicators

3. **Memory Management**
   - Efficient DataFrame operations
   - Chart object cleanup
   - Session state optimization

### Scalability Considerations

- **Data Size Limits**: Handle large datasets gracefully
- **Concurrent Users**: Session isolation and resource management
- **Chart Complexity**: Optimize rendering for large time series
- **Storage Management**: Automatic cleanup of old analysis files

## Security Considerations

### Input Sanitization

- Stock symbol validation against known patterns
- File path sanitization for output directories
- Prevention of code injection through symbol input

### File System Security

- Restricted file access to designated directories
- Validation of file paths and extensions
- Secure temporary file handling

### Data Privacy

- No persistent storage of sensitive data
- Session-based data isolation
- Automatic cleanup of temporary files

## Deployment Architecture

### Local Development

```
streamlit run dashboard.py
```

### Production Deployment Options

1. **Streamlit Cloud**: Direct GitHub integration
2. **Docker Container**: Containerized deployment
3. **Cloud Platforms**: AWS/GCP/Azure hosting
4. **Local Network**: Internal company deployment

### Configuration Management

- Environment-specific settings
- API key management (if needed)
- Output directory configuration
- Logging level configuration