"""
Unified Data Tools for Data Analysis Agents
Combines data loading, discovery, and visualization capabilities following smolagents best practices.
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from smolagents import tool
from ..config.agent_config import VISUALIZATION_CONFIG


@tool
def data_loader(file_context: str, mode: str = "auto") -> Union[str, pd.DataFrame]:
    """
    Unified data loading and discovery tool following smolagents best practices.
    
    Combines file discovery and loading capabilities with intelligent mode selection
    based on the input context. Provides comprehensive data summaries and error handling.
    
    Args:
        file_context: Filename (e.g., "data.csv") or descriptive text about needed data.
                     Searches in current directory, data/, and available/ folders.
                     Use exact filenames for faster loading: "sales_2024.csv"
        mode: Operation mode - "load" (load specific file), "discover" (list all files),
              or "auto" (default, intelligently selects mode based on input)
    
    Returns:
        For mode="load" or auto-detected load:
            Returns the pandas DataFrame containing the data
        For mode="discover" or auto-detected discovery:
            Returns a string containing list of available CSV files with their locations and basic information.
    
    Error Handling:
        - "File not found": Check filename spelling and file location
        - "Permission denied": Verify file access rights
        - "Invalid CSV format": Ensure file is properly formatted CSV
        - "Memory error": File too large, consider using pd.read_csv() with chunksize
    
    Usage Examples:
        # Load specific file
        data_loader("titanic.csv")
        
        # Discover available files
        data_loader("any CSV files", mode="discover")
        
        # Auto mode (intelligent selection)
        data_loader("sales data")  # Will load if specific file found, otherwise discover
    """
    # Determine mode if auto
    if mode == "auto":
        mode = "load" if file_context.endswith('.csv') else "discover"
    
    # Common search paths
    search_dirs = ['.', 'data', 'available', './data', './available']
    
    if mode == "discover":
        return _discover_files(search_dirs)
    else:  # mode == "load"
        return _load_file(file_context, search_dirs)


def _discover_files(search_dirs: list) -> str:
    """Helper function to discover CSV files in specified directories."""
    found_files = []
    
    for directory in search_dirs:
        if os.path.exists(directory):
            try:
                files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                for csv_file in files:
                    path = os.path.join(directory, csv_file)
                    try:
                        # Get basic file info
                        size = os.path.getsize(path) / 1024  # KB
                        found_files.append({
                            'path': path,
                            'size': f"{size:.1f} KB",
                            'directory': directory
                        })
                    except Exception as e:
                        print(f"⚠️ Error getting info for {path}: {str(e)}")
            except PermissionError:
                print(f"⚠️ Permission denied for directory: {directory}")
    
    if not found_files:
        return """❌ NO CSV FILES FOUND

💡 TROUBLESHOOTING:
1. Check if CSV files exist in the workspace
2. Verify file permissions
3. Try using absolute paths
4. Ensure files have .csv extension"""

    # Format the response
    response = ["📁 AVAILABLE CSV FILES:"]
    for file in found_files:
        response.append(f"\n📄 {os.path.basename(file['path'])}")
        response.append(f"   📂 Location: {file['directory']}")
        response.append(f"   📊 Size: {file['size']}")
    
    response.append("\n💡 TIP: Use data_loader('filename.csv') to load a specific file")
    return "\n".join(response)


def _load_file(file_context: str, search_dirs: list) -> Union[str, pd.DataFrame]:
    """Helper function to load a specific CSV file."""
    possible_paths = []
    
    # If file_context looks like a filename, try different locations
    if file_context.endswith('.csv'):
        possible_paths = [
            file_context,  # Direct path
            f"data/{file_context}",  # data folder
            f"available/{file_context}",  # available folder  
            f"./data/{file_context}",  # relative data folder
            f"./{file_context}",  # current directory
        ]
    else:
        # Search for matching files in all directories
        for directory in search_dirs:
            if os.path.exists(directory):
                try:
                    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                    for csv_file in files:
                        if file_context.lower() in csv_file.lower():
                            possible_paths.append(os.path.join(directory, csv_file))
                except PermissionError:
                    continue
    
    # Try to load the first available CSV file
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(_generate_data_summary(df, path))  # Print the summary but return the DataFrame
                return df
        except Exception as e:
            print(f"⚠️ Error loading {path}: {str(e)}")
            continue
    
    # If no file found, return discovery mode
    return _discover_files(search_dirs)


def _generate_data_summary(df: pd.DataFrame, file_path: str) -> str:
    """Helper function to generate comprehensive data summary."""
    return f"""✅ SUCCESS: CSV file loaded from '{file_path}'

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
df = pd.read_csv('{file_path}')

🎨 VISUALIZATION REMINDER:
After creating ANY chart/figure, you MUST immediately use:
display_figures({{"chart_name": fig}}, figure_type="matplotlib")  # for matplotlib/seaborn
display_figures({{"chart_name": fig}}, figure_type="plotly")      # for plotly"""


@tool
def display_figures(figures_dict: Dict[str, Any], figure_type: str = "auto") -> str:
    """
    Unified figure display tool for both matplotlib and plotly figures.
    Follows smolagents best practices for tool implementation.
    
    Safely displays multiple figures while managing memory and providing
    detailed feedback. Call IMMEDIATELY after creating any chart for proper display.
    
    Args:
        figures_dict: Dictionary mapping figure names to figure objects.
                     Maximum 10 figures per call to prevent memory issues.
                     Use descriptive names: {"sales_trend": fig1, "correlation_heatmap": fig2}
                     
                     CORRECT USAGE:
                     - Matplotlib: fig, ax = plt.subplots() -> {"name": fig}
                     - Seaborn: plot = sns.histplot(data) -> {"name": plot.get_figure()}
                     - Plotly: fig = go.Figure() -> {"name": fig}
                     
                     INCORRECT USAGE:
                     - {"name": plt}  # Don't pass the library
                     - {"name": sns}  # Don't pass the library
                     - {"name": go}   # Don't pass the library
                     
        figure_type: Type of figures to display:
                    - "matplotlib": For matplotlib/seaborn figures
                    - "plotly": For plotly figures
                    - "auto": (default) Automatically detect figure type
    
    Returns:
        Status message about successfully displayed figures or specific error information
        with troubleshooting guidance for failed displays.
    
    Error Handling:
        - "Invalid input": Pass dictionary with figure objects: {"name": figure_object}
        - "Too many figures": Split into multiple calls (≤10 figures each)
        - "Figure conversion failed": Ensure figure objects are valid
        - "Display error": Check Streamlit environment and figure validity
    
    Usage Examples:
        # Matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        display_figures({"line_plot": fig}, figure_type="matplotlib")
        
        # Seaborn figure
        import seaborn as sns
        plot = sns.histplot(data=df, x='column')
        fig = plot.get_figure()  # Important: get the figure object
        display_figures({"histogram": fig}, figure_type="matplotlib")
        
        # Plotly figure
        fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
        display_figures({"bar_chart": fig}, figure_type="plotly")
        
        # Auto detection
        display_figures({"chart": fig})  # Will detect figure type automatically
    """
    print(f"🎨 display_figures called with {len(figures_dict) if isinstance(figures_dict, dict) else 'invalid'} figures")
    
    # Validate input
    if not isinstance(figures_dict, dict):
        error_msg = """❌ ERROR: Expected dictionary of figures, received: """ + str(type(figures_dict)) + """

💡 CORRECT USAGE:
1. For matplotlib:
   fig, ax = plt.subplots()
   ax.plot(data)
   display_figures({"chart_name": fig})

2. For seaborn:
   plot = sns.histplot(data)
   fig = plot.get_figure()
   display_figures({"chart_name": fig})

3. For plotly:
   fig = go.Figure()
   display_figures({"chart_name": fig})

❌ DO NOT pass the library itself (plt, sns, or go)"""
        print(error_msg)
        return error_msg
    
    if not figures_dict:
        warning_msg = """⚠️ WARNING: Empty figures dictionary provided

💡 TIP: Make sure to create figures before displaying them. Example:
fig, ax = plt.subplots()
ax.plot(data)
display_figures({"chart_name": fig})"""
        print(warning_msg)
        return warning_msg
    
    # Check figure limit
    max_figures = VISUALIZATION_CONFIG["max_figures_per_call"]
    if len(figures_dict) > max_figures:
        error_msg = f"""❌ ERROR: Too many figures ({len(figures_dict)}). Maximum {max_figures} allowed per call.

💡 SOLUTION: Split into multiple calls with ≤{max_figures} figures each. Example:
# First call
display_figures({{"chart1": fig1, "chart2": fig2}})

# Second call
display_figures({{"chart3": fig3, "chart4": fig4}})"""
        print(error_msg)
        return error_msg
    
    displayed_count = 0
    failed_figures = []
    
    for fig_name, fig_obj in figures_dict.items():
        try:
            print(f"   🖼️ Processing figure: {fig_name}")
            
            # Determine figure type if auto
            if figure_type == "auto":
                if isinstance(fig_obj, go.Figure):
                    figure_type = "plotly"
                else:
                    figure_type = "matplotlib"
            
            # Handle matplotlib figures
            if figure_type == "matplotlib":
                # Handle axes objects by getting their figure
                if hasattr(fig_obj, 'get_figure'):
                    print(f"   🔄 Converting axes to figure for: {fig_name}")
                    fig_obj = fig_obj.get_figure()
                
                if hasattr(fig_obj, 'savefig'):
                    st.pyplot(fig_obj)
                    plt.close(fig_obj)  # Close to free memory
                    displayed_count += 1
                    print(f"   ✅ Successfully displayed matplotlib figure: {fig_name}")
                else:
                    failed_figures.append(f"""{fig_name}: Not a valid matplotlib figure object

💡 TROUBLESHOOTING:
1. Make sure you're passing the figure object, not the library:
   fig, ax = plt.subplots()  # Correct
   display_figures({{"name": fig}})  # Correct
   display_figures({{"name": plt}})  # Incorrect

2. For seaborn plots, get the figure object:
   plot = sns.histplot(data)
   fig = plot.get_figure()  # Important!
   display_figures({{"name": fig}})""")
            
            # Handle plotly figures
            elif figure_type == "plotly":
                if isinstance(fig_obj, go.Figure):
                    st.plotly_chart(fig_obj, use_container_width=True)
                    displayed_count += 1
                    print(f"   ✅ Successfully displayed plotly figure: {fig_name}")
                else:
                    failed_figures.append(f"""{fig_name}: Not a valid plotly figure object

💡 TROUBLESHOOTING:
1. Make sure you're passing a go.Figure object:
   fig = go.Figure()  # Correct
   display_figures({{"name": fig}})  # Correct
   display_figures({{"name": go}})   # Incorrect""")
            
            else:
                failed_figures.append(f"""{fig_name}: Invalid figure_type '{figure_type}'

💡 TROUBLESHOOTING:
Valid figure types are:
- "matplotlib": For matplotlib/seaborn figures
- "plotly": For plotly figures
- "auto": (default) Automatically detect figure type""")
                
        except Exception as e:
            error_details = f"""{fig_name}: {str(e)}

💡 TROUBLESHOOTING:
1. Check that your figure object is valid
2. Make sure you're passing the figure object, not the library
3. For seaborn plots, use .get_figure() method
4. For plotly, ensure you're passing a go.Figure instance"""
            failed_figures.append(error_details)
            print(f"   ❌ Error displaying {fig_name}: {str(e)}")
            st.error(f"Error displaying figure {fig_name}: {str(e)}")
    
    # Prepare comprehensive result message
    result_parts = [f"🎨 FIGURE DISPLAY RESULTS:"]
    result_parts.append(f"✅ Successfully displayed: {displayed_count} figures")
    
    if failed_figures:
        result_parts.append(f"❌ Failed to display: {len(failed_figures)} figures")
        result_parts.append("📋 FAILURE DETAILS:")
        for failure in failed_figures:
            result_parts.append(f"   - {failure}")
        result_parts.append("\n💡 TROUBLESHOOTING:")
        result_parts.append("   - For matplotlib: ensure figures are created with plt.figure() or plt.subplots()")
        result_parts.append("   - For seaborn: use .get_figure() method")
        result_parts.append("   - For plotly: ensure objects are go.Figure instances")
        result_parts.append("   - Check that figure objects are not None")
        result_parts.append("   - Make sure you're passing the figure object, not the library itself")
    
    if displayed_count > 0:
        result_parts.append(f"\n🧹 MEMORY: Closed {displayed_count} figures to free memory")
    
    return "\n".join(result_parts) 