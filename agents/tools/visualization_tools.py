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
    Load CSV data for analysis from the available context.
    
    This tool discovers and loads CSV files from common locations, providing
    comprehensive data summaries and usage guidance for effective data analysis.
    Following smolagents best practices: Tools should be functional with detailed logging.
    
    Args:
        file_context: Context information about available CSV files or direct file path.
                     Can be a filename (e.g., "data.csv") or descriptive text about the data needed.
        
    Returns:
        Detailed data summary including shape, columns, preview, and loading instructions,
        or comprehensive error information with suggested solutions.
        
    Usage Examples:
        - load_csv_data("sales_data.csv")
        - load_csv_data("financial data")
        - load_csv_data("any available CSV files")
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
    Display matplotlib/seaborn figures in Streamlit interface with comprehensive error handling.
    
    This tool safely displays multiple matplotlib figures while managing memory and
    providing detailed feedback. Following smolagents best practices for clear error reporting.
    
    Args:
        figures_dict: Dictionary mapping figure names to matplotlib figure objects.
                     Example: {"correlation_plot": fig1, "histogram": fig2, "scatter_analysis": fig3}
                     Supports both figure objects and axes objects (automatically converted).
        
    Returns:
        Detailed status message about successfully displayed figures or specific error information
        with troubleshooting guidance for failed displays.
        
    Usage Examples:
        # Single figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        display_matplotlib_figures({"line_plot": fig})
        
        # Multiple seaborn figures
        fig1 = sns.heatmap(data).get_figure()
        fig2 = sns.boxplot(data).get_figure() 
        display_matplotlib_figures({"heatmap": fig1, "boxplot": fig2})
        
        # From axes objects
        ax = sns.scatterplot(data=df, x='col1', y='col2')
        display_matplotlib_figures({"scatter": ax.get_figure()})
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
    Display interactive plotly figures in Streamlit interface with comprehensive validation.
    
    This tool handles both plotly express and graph objects figures, providing detailed
    error reporting and usage guidance. Following smolagents best practices for informative tools.
    
    Args:
        figures_dict: Dictionary mapping figure names to plotly figure objects.
                     Example: {"dashboard": plotly_fig, "3d_plot": scatter3d, "timeline": line_chart}
                     Supports both plotly.express and plotly.graph_objects figures.
        
    Returns:
        Detailed status message about successfully displayed figures or comprehensive error
        information with specific troubleshooting steps for each failure.
        
    Usage Examples:
        # Plotly Express figures
        import plotly.express as px
        fig = px.scatter(df, x='col1', y='col2', title='Analysis')
        display_plotly_figures({"scatter": fig})
        
        # Plotly Graph Objects
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Bar(x=names, y=values))
        display_plotly_figures({"bar_chart": fig})
        
        # Multiple interactive charts
        display_plotly_figures({
            "overview": dashboard_fig,
            "details": detail_fig,
            "comparison": comparison_fig
        })
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
    width_setting = VISUALIZATION_CONFIG.get("default_plotly_width", True)
    
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
    Discover all available CSV files in the current environment with detailed metadata.
    
    This tool performs a comprehensive search across multiple directories and provides
    actionable information about each discovered file. Following smolagents best practice
    of providing functional, informative tools with clear guidance.
    
    Returns:
        Comprehensive list of discovered CSV files with metadata including size,
        column count, headers preview, and loading instructions, or detailed
        explanation if no files are found with troubleshooting suggestions.
        
    Usage Examples:
        - discover_data_files()  # Finds all CSV files in common locations
        - Use before load_csv_data() to see what files are available
        - Helpful for debugging when specific files cannot be found
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