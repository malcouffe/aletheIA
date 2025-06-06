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
    According to smolagents best practices: Tools should be functional, not just informational.
    
    Args:
        file_context: Context information about available CSV files or file path
        
    Returns:
        Actual data summary and loading guidance with discovered file paths
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
    else:
        # Search for CSV files in common directories
        search_dirs = ['.', 'data', 'available', './data', './available']
        for directory in search_dirs:
            if os.path.exists(directory):
                try:
                    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                    for csv_file in files:
                        possible_paths.append(os.path.join(directory, csv_file))
                except PermissionError:
                    continue
    
    # Try to load the first available CSV file
    loaded_file = None
    df = None
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                loaded_file = path
                break
        except Exception as e:
            continue
    
    if df is not None:
        # Return comprehensive data summary
        summary = f"""✅ SUCCESS: CSV file loaded from '{loaded_file}'

📊 DATA SUMMARY:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {list(df.columns)}
- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB

🔍 DATA PREVIEW:
{df.head().to_string()}

⚠️ MISSING VALUES:
{df.isnull().sum().to_string()}

💡 DATA TYPES:
{df.dtypes.to_string()}

📝 TO USE THIS DATA:
df = pd.read_csv('{loaded_file}')

CRITICAL REMINDER: After creating ANY chart/figure, you MUST immediately use:
- display_matplotlib_figures({{"chart_name": fig}}) for matplotlib/seaborn charts  
- display_plotly_figures({{"chart_name": plotly_fig}}) for plotly charts

Create ONE chart at a time and display it IMMEDIATELY!"""
        
        return summary
    else:
        # No file found - provide helpful debugging info
        return f"""❌ NO CSV FILE FOUND

🔍 SEARCHED LOCATIONS:
{chr(10).join(f"- {path} {'✅ exists' if os.path.exists(path) else '❌ not found'}" for path in possible_paths)}

📁 CURRENT DIRECTORY CONTENTS:
{os.listdir('.') if os.path.exists('.') else 'Cannot access current directory'}

💡 SOLUTIONS:
1. Make sure the CSV file is in the current directory
2. Use the full file path: df = pd.read_csv('full/path/to/file.csv')
3. Check if the file exists: os.path.exists('filename.csv')

CRITICAL REMINDER: After creating ANY chart/figure, you MUST immediately use:
- display_matplotlib_figures({{"chart_name": fig}}) for matplotlib/seaborn charts
- display_plotly_figures({{"chart_name": plotly_fig}}) for plotly charts"""


@tool
def display_matplotlib_figures(figures_dict: dict) -> str:
    """
    Display matplotlib figures in Streamlit interface.
    
    Args:
        figures_dict: Dictionary containing matplotlib figure objects
        Example: {"correlation_plot": fig1, "histogram": fig2, "scatter_analysis": fig3}
        
    Returns:
        Status message about displayed figures
        
    Usage Examples:
        # Single figure
        figures = {"sales_trend": plt.figure()}
        display_matplotlib_figures(figures)
        
        # Multiple figures from seaborn
        fig1 = sns.heatmap(data).get_figure()
        fig2 = sns.boxplot(data).get_figure() 
        display_matplotlib_figures({"heatmap": fig1, "boxplot": fig2})
    """
    import matplotlib.pyplot as plt
    
    if not isinstance(figures_dict, dict):
        return "Error: Expected dictionary of figures"
    
    max_figures = VISUALIZATION_CONFIG["max_figures_per_call"]
    if len(figures_dict) > max_figures:
        return f"Error: Too many figures ({len(figures_dict)}). Maximum {max_figures} allowed."
    
    displayed_count = 0
    for fig_name, fig_obj in figures_dict.items():
        try:
            if hasattr(fig_obj, 'get_figure'):  # Handle axes objects
                fig_obj = fig_obj.get_figure()
            
            if hasattr(fig_obj, 'savefig'):  # Verify it's a matplotlib figure
                st.pyplot(fig_obj)
                plt.close(fig_obj)  # Close to free memory
                displayed_count += 1
        except Exception as e:
            st.error(f"Error displaying figure {fig_name}: {str(e)}")
    
    return f"Successfully displayed {displayed_count} matplotlib figures"


@tool
def display_plotly_figures(figures_dict: dict) -> str:
    """
    Display plotly figures in Streamlit interface.
    
    Args:
        figures_dict: Dictionary containing plotly figure objects
        Example: {"dashboard": plotly_fig, "3d_plot": scatter3d, "timeline": line_chart}
        
    Returns:
        Status message about displayed figures
        
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
    if not isinstance(figures_dict, dict):
        return "Error: Expected dictionary of figures"
    
    max_figures = VISUALIZATION_CONFIG["max_figures_per_call"]
    if len(figures_dict) > max_figures:
        return f"Error: Too many figures ({len(figures_dict)}). Maximum {max_figures} allowed."
    
    displayed_count = 0
    width_setting = VISUALIZATION_CONFIG.get("default_plotly_width", True)
    
    for fig_name, fig_obj in figures_dict.items():
        try:
            if hasattr(fig_obj, 'show'):  # Verify it's a plotly figure
                st.plotly_chart(fig_obj, use_container_width=width_setting)
                displayed_count += 1
        except Exception as e:
            st.error(f"Error displaying figure {fig_name}: {str(e)}")
    
    return f"Successfully displayed {displayed_count} plotly figures"


@tool
def discover_data_files() -> str:
    """
    Discover all available CSV files in the current environment.
    Following smolagents best practice: Provide functional, actionable tools.
    
    Returns:
        List of discovered CSV files with basic information
    """
    import os
    import glob
    
    # Search for CSV files in multiple locations
    search_patterns = [
        "*.csv",
        "data/*.csv", 
        "available/*.csv",
        "./data/*.csv",
        "./available/*.csv",
        "**/*.csv"  # Recursive search
    ]
    
    found_files = []
    
    for pattern in search_patterns:
        try:
            files = glob.glob(pattern, recursive=True)
            found_files.extend(files)
        except:
            continue
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))
    
    if found_files:
        result = "📁 DISCOVERED CSV FILES:\n\n"
        for i, file_path in enumerate(found_files, 1):
            try:
                # Get basic file info
                size = os.path.getsize(file_path) / 1024  # KB
                # Try to get column info quickly
                df_sample = pd.read_csv(file_path, nrows=0)  # Just headers
                cols = len(df_sample.columns)
                
                result += f"{i}. {file_path}\n"
                result += f"   📊 Size: {size:.1f} KB, Columns: {cols}\n"
                result += f"   📝 Headers: {list(df_sample.columns)[:5]}{'...' if cols > 5 else ''}\n\n"
            except Exception as e:
                result += f"{i}. {file_path}\n"
                result += f"   ⚠️ Error reading file: {str(e)[:50]}...\n\n"
        
        result += "💡 TO LOAD A FILE:\ndf = pd.read_csv('path_from_above')\n"
        return result
    else:
        return """❌ NO CSV FILES FOUND

🔍 SEARCHED PATTERNS:
- *.csv (current directory)
- data/*.csv (data folder)
- available/*.csv (available folder)
- **/*.csv (recursive search)

📁 CURRENT DIRECTORY CONTENTS:
""" + str(os.listdir('.') if os.path.exists('.') else 'Cannot access directory') 