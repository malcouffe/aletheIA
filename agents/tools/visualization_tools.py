"""
Visualization Tools for Data Analysis Agents
Handles matplotlib and plotly figure display in Streamlit.
"""

import streamlit as st
import pandas as pd
import os
from smolagents import tool
from ..config.agent_config import VISUALIZATION_CONFIG


@tool
def load_csv_data(file_context: str) -> str:
    """
    Load CSV data for analysis from available files or directories.
    
    Discovers and loads CSV files from common locations with comprehensive data summaries.
    Provides detailed error reporting for troubleshooting missing or inaccessible files.
    
    Args:
        file_context: Filename (e.g., "data.csv") or descriptive text about needed data.
                     Searches in current directory, data/, and available/ folders.
                     Use exact filenames for faster loading: "sales_2024.csv"
        
    Returns:
        Detailed data summary with shape, columns, preview, and loading instructions.
        Includes missing values analysis and usage guidance for visualization tools.
        
    Error Handling:
        - "File not found": Check filename spelling and file location
        - "Permission denied": Verify file access rights
        - "Invalid CSV format": Ensure file is properly formatted CSV
        - "Memory error": File too large, consider using pd.read_csv() with chunksize
        
    Usage Examples:
        - load_csv_data("titanic.csv")  # Load specific file
        - load_csv_data("sales data")   # Search for sales-related files
        - load_csv_data("any CSV files")  # Find any available CSV
    """

    
    # Try to discover CSV files in common locations
    possible_paths = []
    
    # If file_context looks like a filename, try different locations
    if file_context and file_context.endswith('.csv'):
        possible_paths = [
            file_context,  # Direct path
            f"data/{file_context}",  # data folder
            f"available/{file_context}",  # available folder  
            f"./data/{file_context}",  # relative data folder
            f"./{file_context}",  # current directory
        ]
        print(f"📁 Searching for specific file: {file_context}")
    else:
        # Search for CSV files in common directories
        search_dirs = ['.', 'data', 'available', './data', './available']
        print(f"📁 Searching for CSV files in directories: {search_dirs}")
        for directory in search_dirs:
            if os.path.exists(directory):
                try:
                    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                    for csv_file in files:
                        possible_paths.append(os.path.join(directory, csv_file))
                    print(f"   Found {len(files)} CSV files in {directory}")
                except PermissionError:
                    print(f"   ⚠️ Permission denied for directory: {directory}")
                    continue
            else:
                print(f"   ❌ Directory not found: {directory}")
    
    print(f"📋 Total candidate paths: {len(possible_paths)}")
    
    # Try to load the first available CSV file
    loaded_file = None
    df = None
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"   ✅ Attempting to load: {path}")
                df = pd.read_csv(path)
                loaded_file = path
                print(f"   🎉 Successfully loaded: {path}")
                break
            else:
                print(f"   ❌ File not found: {path}")
        except Exception as e:
            print(f"   ⚠️ Error loading {path}: {str(e)}")
            continue
    
    if df is not None:
        # Return comprehensive data summary
        summary = f"""✅ SUCCESS: CSV file loaded from '{loaded_file}'

📊 DATA SUMMARY:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
- Columns: {list(df.columns)}

🔍 DATA PREVIEW (first 5 rows):
{df.head().to_string()}

⚠️ MISSING VALUES:
{df.isnull().sum().to_string()}

💡 DATA TYPES:
{df.dtypes.to_string()}

📈 NUMERIC SUMMARY:
{df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else 'No numeric columns found'}

📝 TO USE THIS DATA IN PYTHON:
df = pd.read_csv('{loaded_file}')

🎨 VISUALIZATION REMINDER:
After creating ANY chart/figure, you MUST immediately use:
- display_matplotlib_figures({{"chart_name": fig}}) for matplotlib/seaborn charts  
- display_plotly_figures({{"chart_name": plotly_fig}}) for plotly charts

⚠️ IMPORTANT: Create ONE chart at a time and display it IMMEDIATELY!"""
        
        print("✅ Data loading completed successfully")
        return summary
    else:
        # No file found - provide helpful debugging info
        error_report = f"""❌ NO CSV FILE FOUND

🔍 SEARCHED LOCATIONS ({len(possible_paths)} total):
{chr(10).join(f"- {path} {'✅ exists' if os.path.exists(path) else '❌ not found'}" for path in possible_paths)}

📁 CURRENT DIRECTORY CONTENTS:
{', '.join(os.listdir('.')) if os.path.exists('.') else 'Cannot access current directory'}

💡 TROUBLESHOOTING SOLUTIONS:
1. Make sure the CSV file is in the current directory
2. Use the full file path: df = pd.read_csv('full/path/to/file.csv')
3. Check if the file exists: os.path.exists('filename.csv')
4. Verify file permissions and access rights
5. Try using discover_data_files() tool to see all available CSV files

🎨 VISUALIZATION REMINDER:
After creating ANY chart/figure, you MUST immediately use:
- display_matplotlib_figures({{"chart_name": fig}}) for matplotlib/seaborn charts
- display_plotly_figures({{"chart_name": plotly_fig}}) for plotly charts"""
        
        print("❌ Data loading failed - no valid CSV files found")
        return error_report


@tool
def display_matplotlib_figures(figures_dict: dict) -> str:
    """
    Display matplotlib/seaborn figures in Streamlit with memory management.
    
    Safely displays multiple matplotlib figures while managing memory and providing
    detailed feedback. Call IMMEDIATELY after creating any chart for proper display.
    
    Args:
        figures_dict: Dictionary mapping figure names to matplotlib figure objects.
                     Maximum 10 figures per call to prevent memory issues.
                     Use descriptive names: {"sales_trend": fig1, "correlation_heatmap": fig2}
        
    Returns:
        Status message about successfully displayed figures or specific error information
        with troubleshooting guidance for failed displays.
        
    Error Handling:
        - "Invalid input": Pass dictionary with figure objects: {"name": matplotlib_figure}
        - "Too many figures": Split into multiple calls (≤10 figures each)
        - "Figure conversion failed": Ensure figure objects are valid matplotlib figures
        - "Display error": Check Streamlit environment and figure validity
        
    Usage Examples:
        # Single figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        display_matplotlib_figures({"line_plot": fig})
        
        # Multiple seaborn figures
        fig1 = sns.heatmap(data).get_figure()
        display_matplotlib_figures({"heatmap": fig1})
    """
    import matplotlib.pyplot as plt
    
    print(f"🎨 display_matplotlib_figures called with {len(figures_dict) if isinstance(figures_dict, dict) else 'invalid'} figures")
    
    # Validate input
    if not isinstance(figures_dict, dict):
        error_msg = "❌ ERROR: Expected dictionary of figures, received: " + str(type(figures_dict))
        print(error_msg)
        return error_msg + "\n\n💡 CORRECT FORMAT: {\"figure_name\": matplotlib_figure_object}"
    
    if not figures_dict:
        warning_msg = "⚠️ WARNING: Empty figures dictionary provided"
        print(warning_msg)
        return warning_msg + "\n\n💡 TIP: Make sure to create figures before displaying them"
    
    # Check figure limit
    max_figures = VISUALIZATION_CONFIG["max_figures_per_call"]
    if len(figures_dict) > max_figures:
        error_msg = f"❌ ERROR: Too many figures ({len(figures_dict)}). Maximum {max_figures} allowed per call."
        print(error_msg)
        return error_msg + f"\n\n💡 SOLUTION: Split into multiple calls with ≤{max_figures} figures each"
    
    displayed_count = 0
    failed_figures = []
    
    for fig_name, fig_obj in figures_dict.items():
        try:
            print(f"   🖼️ Processing figure: {fig_name}")
            
            # Handle axes objects by getting their figure
            if hasattr(fig_obj, 'get_figure'):
                print(f"   🔄 Converting axes to figure for: {fig_name}")
                fig_obj = fig_obj.get_figure()
            
            # Verify it's a matplotlib figure
            if hasattr(fig_obj, 'savefig'):
                st.pyplot(fig_obj)
                plt.close(fig_obj)  # Close to free memory
                displayed_count += 1
                print(f"   ✅ Successfully displayed: {fig_name}")
            else:
                failed_figures.append(f"{fig_name}: Not a valid matplotlib figure object")
                print(f"   ❌ Invalid figure object: {fig_name}")
                
        except Exception as e:
            error_details = f"{fig_name}: {str(e)}"
            failed_figures.append(error_details)
            print(f"   ❌ Error displaying {fig_name}: {str(e)}")
            st.error(f"Error displaying figure {fig_name}: {str(e)}")
    
    # Prepare comprehensive result message
    result_parts = [f"🎨 MATPLOTLIB DISPLAY RESULTS:"]
    result_parts.append(f"✅ Successfully displayed: {displayed_count} figures")
    
    if failed_figures:
        result_parts.append(f"❌ Failed to display: {len(failed_figures)} figures")
        result_parts.append("📋 FAILURE DETAILS:")
        for failure in failed_figures:
            result_parts.append(f"   - {failure}")
        result_parts.append("\n💡 TROUBLESHOOTING:")
        result_parts.append("   - Ensure figures are created with plt.figure() or plt.subplots()")
        result_parts.append("   - For seaborn: use .get_figure() method")
        result_parts.append("   - Check that figure objects are not None")
    
    if displayed_count > 0:
        result_parts.append(f"\n🧹 MEMORY: Closed {displayed_count} figures to free memory")
    
    final_result = "\n".join(result_parts)
    print(f"📊 Final result: {displayed_count} displayed, {len(failed_figures)} failed")
    return final_result


@tool
def display_plotly_figures(figures_dict: dict) -> str:
    """
    Display interactive Plotly figures in Streamlit interface.
    
    Renders Plotly figures with full interactivity and proper error handling.
    Call IMMEDIATELY after creating any Plotly chart for proper display.
    
    Args:
        figures_dict: Dictionary mapping figure names to Plotly figure objects.
                     Maximum 10 figures per call to maintain performance.
                     Use descriptive names: {"interactive_scatter": fig, "3d_surface": fig2}
        
    Returns:
        Status message about successfully displayed figures or detailed error information
        with troubleshooting steps for failed displays.
        
    Error Handling:
        - "Invalid input": Pass dictionary with Plotly figure objects
        - "Too many figures": Split into multiple calls (≤10 figures each)  
        - "Figure rendering failed": Verify Plotly figure is properly constructed
        - "Streamlit error": Check Streamlit version compatibility
        
    Usage Examples:
        # Single interactive plot
        fig = px.scatter(df, x='col1', y='col2', title='Scatter Plot')
        display_plotly_figures({"scatter": fig})
        
        # Multiple plots
        fig1 = px.line(df, x='date', y='value')
        fig2 = px.bar(df, x='category', y='count')
        display_plotly_figures({"trend": fig1, "distribution": fig2})
    """
    print(f"📈 display_plotly_figures called with {len(figures_dict) if isinstance(figures_dict, dict) else 'invalid'} figures")
    
    # Validate input type
    if not isinstance(figures_dict, dict):
        error_msg = "❌ ERROR: Expected dictionary of figures, received: " + str(type(figures_dict))
        print(error_msg)
        return error_msg + "\n\n💡 CORRECT FORMAT: {\"figure_name\": plotly_figure_object}"
    
    if not figures_dict:
        warning_msg = "⚠️ WARNING: Empty figures dictionary provided"
        print(warning_msg)
        return warning_msg + "\n\n💡 TIP: Create plotly figures before displaying them"
    
    # Check figure limit
    max_figures = VISUALIZATION_CONFIG["max_figures_per_call"]
    if len(figures_dict) > max_figures:
        error_msg = f"❌ ERROR: Too many figures ({len(figures_dict)}). Maximum {max_figures} allowed per call."
        print(error_msg)
        return error_msg + f"\n\n💡 SOLUTION: Split into multiple calls with ≤{max_figures} figures each"
    
    displayed_count = 0
    failed_figures = []
    width_setting = VISUALIZATION_CONFIG.get("default_plotly_width")
    if width_setting is None:
        width_setting = True  # Default to use container width
    
    for fig_name, fig_obj in figures_dict.items():
        try:
            print(f"   📊 Processing plotly figure: {fig_name}")
            
            # Verify it's a plotly figure
            if hasattr(fig_obj, 'show') and hasattr(fig_obj, 'data'):
                st.plotly_chart(fig_obj, use_container_width=width_setting)
                displayed_count += 1
                print(f"   ✅ Successfully displayed: {fig_name}")
            else:
                failed_figures.append(f"{fig_name}: Not a valid plotly figure object")
                print(f"   ❌ Invalid plotly figure: {fig_name}")
                
        except Exception as e:
            error_details = f"{fig_name}: {str(e)}"
            failed_figures.append(error_details)
            print(f"   ❌ Error displaying {fig_name}: {str(e)}")
            st.error(f"Error displaying figure {fig_name}: {str(e)}")
    
    # Prepare comprehensive result message
    result_parts = [f"📈 PLOTLY DISPLAY RESULTS:"]
    result_parts.append(f"✅ Successfully displayed: {displayed_count} interactive figures")
    
    if failed_figures:
        result_parts.append(f"❌ Failed to display: {len(failed_figures)} figures")
        result_parts.append("📋 FAILURE DETAILS:")
        for failure in failed_figures:
            result_parts.append(f"   - {failure}")
        result_parts.append("\n💡 TROUBLESHOOTING:")
        result_parts.append("   - Use plotly.express: fig = px.scatter(df, x='col1', y='col2')")
        result_parts.append("   - Use plotly.graph_objects: fig = go.Figure(data=...)")
        result_parts.append("   - Ensure figures have .show() and .data attributes")
        result_parts.append("   - Check that figure objects are not None")
    
    if displayed_count > 0:
        result_parts.append(f"\n🎛️ SETTINGS: Used container width = {width_setting}")
    
    final_result = "\n".join(result_parts)
    print(f"📈 Final result: {displayed_count} displayed, {len(failed_figures)} failed")
    return final_result


@tool
def discover_data_files() -> str:
    """
    Discover and list all available CSV and Excel files for analysis.
    
    Searches common directories for data files and provides detailed information
    about each file including size, modification date, and loading suggestions.
    
    Args:
        None required - automatically searches standard data directories.
        
    Returns:
        Comprehensive list of available data files with metadata and loading instructions.
        Includes file paths, sizes, and recommendations for analysis workflow.
        
    Error Handling:
        - "No files found": Check if data files exist in current/data/available directories
        - "Permission denied": Verify read access to directories
        - "Directory not found": Ensure data directories exist
        
    Usage Examples:
        - discover_data_files()  # Find all available data files
        
    Tip: Use this tool first to see what data is available before analysis.
    """
    import os
    import glob
    
    print("🔍 discover_data_files: Starting comprehensive CSV file search...")
    
    # Search for CSV files in multiple locations with detailed logging
    search_patterns = [
        "*.csv",           # Current directory
        "data/*.csv",      # Data folder
        "available/*.csv", # Available folder
        "./data/*.csv",    # Relative data folder
        "./available/*.csv", # Relative available folder
        "**/*.csv"         # Recursive search (limited depth for performance)
    ]
    
    found_files = []
    
    for pattern in search_patterns:
        try:
            print(f"   🔍 Searching pattern: {pattern}")
            files = glob.glob(pattern, recursive=True)
            found_files.extend(files)
            print(f"   📁 Found {len(files)} files with pattern {pattern}")
        except Exception as e:
            print(f"   ⚠️ Error searching pattern {pattern}: {str(e)}")
            continue
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))
    print(f"📊 Total unique CSV files found: {len(found_files)}")
    
    if found_files:
        result = ["🔍 COMPREHENSIVE CSV FILE DISCOVERY", "=" * 50, ""]
        
        for i, file_path in enumerate(found_files, 1):
            try:
                print(f"   📋 Analyzing file {i}/{len(found_files)}: {file_path}")
                
                # Get basic file info
                size_kb = os.path.getsize(file_path) / 1024
                
                # Try to get column info quickly (just headers)
                df_sample = pd.read_csv(file_path, nrows=0)
                cols = len(df_sample.columns)
                headers = list(df_sample.columns)
                
                # Try to get row count efficiently
                try:
                    row_count = sum(1 for _ in open(file_path)) - 1  # Subtract header
                except:
                    row_count = "Unknown"
                
                result.append(f"{i}. 📄 {file_path}")
                result.append(f"   📊 Size: {size_kb:.1f} KB")
                result.append(f"   📈 Rows: {row_count}, Columns: {cols}")
                result.append(f"   📝 Headers: {headers[:5]}{'...' if cols > 5 else ''}")
                
                # Add quick data type info if possible
                try:
                    sample_data = pd.read_csv(file_path, nrows=1)
                    numeric_cols = len(sample_data.select_dtypes(include=['number']).columns)
                    text_cols = len(sample_data.select_dtypes(include=['object']).columns)
                    result.append(f"   🔢 Data types: {numeric_cols} numeric, {text_cols} text")
                except:
                    result.append(f"   🔢 Data types: Could not determine")
                
                result.append("")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {file_path}: {str(e)}")
                result.append(f"{i}. ⚠️ {file_path}")
                result.append(f"   ❌ Error reading file: {str(e)[:50]}...")
                result.append("")
        
        result.extend([
            "💡 TO LOAD ANY FILE:",
            "   df = pd.read_csv('path_from_above')",
            "   # Then use load_csv_data('filename.csv') for detailed analysis",
            "",
            "🎨 VISUALIZATION WORKFLOW:",
            "   1. Load data with load_csv_data()",
            "   2. Create visualization with matplotlib/plotly",
            "   3. Display with display_matplotlib_figures() or display_plotly_figures()",
        ])
        
        final_result = "\n".join(result)
        print("✅ CSV discovery completed successfully")
        return final_result
    else:
        # No files found - provide comprehensive troubleshooting
        troubleshooting = f"""❌ NO CSV FILES DISCOVERED

🔍 SEARCHED LOCATIONS:
{chr(10).join(f"   - {pattern}" for pattern in search_patterns)}

📁 CURRENT DIRECTORY CONTENTS:
{', '.join([f for f in os.listdir('.') if '.' in f][:10]) if os.path.exists('.') else 'Cannot access directory'}{'...' if len([f for f in os.listdir('.') if '.' in f]) > 10 else ''}

💡 TROUBLESHOOTING SOLUTIONS:
1. 📥 Upload CSV files to the current directory
2. 📁 Create a 'data' folder and place CSV files there
3. 🔗 Use full file paths: df = pd.read_csv('/full/path/to/file.csv')
4. ✅ Verify file extensions are exactly '.csv' (not .CSV or .txt)
5. 🔐 Check file permissions and access rights
6. 📋 Ensure files are properly formatted CSV files

🎯 NEXT STEPS:
   - Place CSV files in the current directory or 'data' folder
   - Use load_csv_data('filename.csv') once files are available
   - Run discover_data_files() again to verify file detection"""
        
        print("❌ No CSV files found - providing troubleshooting guidance")
        return troubleshooting 


@tool
def check_undisplayed_figures() -> str:
    """
    Check if matplotlib figures exist but haven't been displayed in Streamlit.
    
    Automatically detects forgotten figure displays and provides corrective guidance.
    Call this tool if you suspect figures were created but not shown to the user.
    
    Returns:
        Status of undisplayed figures with specific guidance to display them properly.
        
    Usage Examples:
        - check_undisplayed_figures()  # Check for forgotten displays
    """
    import matplotlib.pyplot as plt
    
    # Get all current figures
    figs = plt.get_fignums()
    
    if not figs:
        return "✅ No matplotlib figures detected. All good!"
    
    warning_message = f"""⚠️ WARNING: {len(figs)} matplotlib figure(s) detected but not displayed!

🔧 SOLUTION: You must call display_matplotlib_figures() to show your charts:

```python
# Get current figures and display them
import matplotlib.pyplot as plt
current_figs = {{}}
for i, fig_num in enumerate(plt.get_fignums()):
    fig = plt.figure(fig_num)
    current_figs[f"chart_{i+1}"] = fig

# Display all figures
display_matplotlib_figures(current_figs)
```

💡 PREVENTION: Always call display_matplotlib_figures() immediately after creating each chart!"""
    
    print(f"📊 Validation: Found {len(figs)} undisplayed figures")
    return warning_message 