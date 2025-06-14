"""
Visualization Tools for Data Analysis Agents
Handles undisplayed figures checking in Streamlit.
"""

import streamlit as st
from smolagents import tool


@tool
def check_undisplayed_figures() -> str:
    """
    Check for any undisplayed figures in the current session.
    Useful for debugging and ensuring all figures are properly displayed.
    
    Returns:
        Status message about undisplayed figures or confirmation that all figures
        have been properly displayed.
    
    Usage Example:
        check_undisplayed_figures()  # Check for any undisplayed figures
    """
    # Get the current figure count from Streamlit's session state
    current_figures = st.session_state.get('_figure_count', 0)
    
    if current_figures > 0:
        return f"""⚠️ WARNING: {current_figures} undisplayed figures detected

💡 RECOMMENDATIONS:
1. Make sure to call display_figures() IMMEDIATELY after creating each figure
2. Check that all figure objects are valid
3. Verify that the figure_type parameter is correct
4. Ensure figures are not being created in loops without display

🔍 DEBUGGING:
- Check your code for any plt.figure() or go.Figure() calls
- Look for any seaborn plotting functions that create figures
- Verify that all visualization code paths call display_figures()"""
    else:
        return "✅ All figures have been properly displayed" 