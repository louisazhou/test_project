"""
Plots for Closed-Lost Reason Over-indexing Hypothesis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Any, Tuple

from ..plot_styles import setup_style, STYLE
from ...registry import hypothesis_plotter, register_plotter

logger = logging.getLogger(__name__)

def plot_closed_lost_overindex(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    region: Optional[str] = None,
    metric_name: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    max_reasons: int = 10,
    value_col: str = "overindex",
    reason_col: str = "reason",
    **kwargs
) -> None:
    """Plot the overindexing of closed lost reasons for a region compared to global.
    
    Args:
        ax: The matplotlib axes to draw on
        df: DataFrame containing closed lost reason data
        region: The region to highlight
        metric_name: Name of the metric being analyzed
        title: Plot title (optional)
        subtitle: Plot subtitle (optional)
        max_reasons: Maximum number of reasons to display
        value_col: Column containing the value for sorting/plotting (default: "overindex")
        reason_col: Column containing the reason names (default: "reason")
        **kwargs: Additional parameters
    """
    # Return early if essential data is missing
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data available for closed lost reasons plot", ha='center', va='center')
        return
    
    # Ensure style consistency
    setup_style()
    
    # Set title
    if title:
        ax.set_title(title, fontsize=12)
    else:
        region_text = f"for {region}" if region else ""
        metric_text = f"in {metric_name}" if metric_name else ""
        ax.set_title(f"Closed Lost Reasons Overindex {region_text} {metric_text}", fontsize=12)
    
    if subtitle:
        ax.text(0.5, 0.95, subtitle, ha='center', va='center', transform=ax.transAxes, 
               fontsize=9, style='italic')
    
    # Process data
    try:
        # Verify required columns exist
        if reason_col not in df.columns:
            error_msg = f"Missing required column: {reason_col}"
            logger.error(error_msg)
            ax.text(0.5, 0.5, error_msg, ha='center', va='center')
            return
            
        if value_col not in df.columns:
            error_msg = f"Missing required column: {value_col}"
            logger.error(error_msg)
            ax.text(0.5, 0.5, error_msg, ha='center', va='center')
            return
            
        # If data is empty or has no valid rows
        if len(df) == 0 or df[reason_col].isna().all():
            ax.text(0.5, 0.5, "No valid data for closed lost reasons plot", ha='center', va='center')
            return
            
        # Ensure value column is numeric and non-null
        df = df.copy()  # Create a copy to avoid modifying the original
        if df[value_col].dtype == 'object':
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            
        # Drop rows with NaN values
        df = df.dropna(subset=[value_col])
        
        if len(df) == 0:
            ax.text(0.5, 0.5, "No valid data for plot after filtering", ha='center', va='center')
            return
            
        # Get the top reasons by value (typically overindex or excess_loss_$)
        top_reasons = df.sort_values(value_col, ascending=False).head(max_reasons)
        
        # Format value for display based on column type
        if value_col == 'excess_loss_$' or 'dollar' in value_col.lower() or '$' in value_col:
            # Format as dollar amount
            formatter = lambda x: f"${x:,.0f}"
            x_label = "Excess Loss ($)"
        elif 'ratio' in value_col.lower() or 'overindex' in value_col.lower() or 'index' in value_col.lower():
            # Format as ratio/percentage
            formatter = lambda x: f"{x:.1%}"
            x_label = "Overindex Ratio (Region % / Global %)"
        else:
            # Default format for unknown value types
            formatter = lambda x: f"{x:.2f}"
            x_label = value_col.replace('_', ' ').title()
        
        # Create the bar chart
        bars = ax.barh(top_reasons[reason_col], top_reasons[value_col], 
                     color=STYLE['colors']['primary'])
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = max(width + (width * 0.05), width + 0.1)  # Ensure label is visible even for very small bars
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   formatter(width), va='center')
        
        # Add percentage to global share if available
        if 'global_share' in top_reasons.columns:
            for i, (_, row) in enumerate(top_reasons.iterrows()):
                if pd.notna(row['global_share']):
                    ax.text(0.01, i, f"Global: {row['global_share']:.1%}", va='center',
                           color='gray', fontsize=8)
        
        # Add region percentage to the right if available
        if 'regional_share' in top_reasons.columns:
            for i, (_, row) in enumerate(top_reasons.iterrows()):
                if pd.notna(row['regional_share']):
                    ax.text(0.5, i, f"Region: {row['regional_share']:.1%}", va='center',
                           color=STYLE['colors']['highlight'], fontsize=8)
        
        # Set axis labels and format
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(reason_col.replace('_', ' ').title(), fontsize=10)
        
        # Format x-axis based on the value type
        if 'ratio' in value_col.lower() or 'overindex' in value_col.lower() or 'index' in value_col.lower():
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            # Add reference line at 1.0 (no overindex) if appropriate
            ax.axvline(1.0, color='gray', linestyle='--', linewidth=1)
        elif value_col == 'excess_loss_$' or 'dollar' in value_col.lower() or '$' in value_col:
            # Format as dollar amount
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        # Adjust layout
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
    except Exception as e:
        logger.error(f"Error plotting closed lost reasons: {e}")
        ax.text(0.5, 0.5, f"Error plotting closed lost reasons: {str(e)}", 
               ha='center', va='center')

@hypothesis_plotter
def plot_for_report(ax, hypo_result, focus_region=None, **kwargs):
    """Plot closed lost reasons overindex for a report.
    
    This function extracts data from a hypothesis result and calls the
    plot_closed_lost_overindex function.
    
    Args:
        ax: The matplotlib axes to draw on
        hypo_result: The hypothesis result object
        focus_region: The region to focus on
        **kwargs: Additional parameters
        
    Returns:
        True if plotting was successful, False otherwise
    """
    if not hasattr(hypo_result, 'plots') or not hypo_result.plots:
        ax.text(0.5, 0.5, "No plots defined for closed_lost_reason hypothesis", 
               ha='center', va='center')
        return False
        
    # Use the first plot specification
    for plot_spec in hypo_result.plots:
        if plot_spec.plot_key == 'plot_closed_lost_overindex':
            # Extract the necessary parameters from the plot context
            context = plot_spec.context or {}
            df = plot_spec.data
            
            # Use focus_region if provided, otherwise try to get from context
            region = focus_region or context.get('region')
            
            # Call the plotting function
            plot_closed_lost_overindex(ax=ax, df=df, region=region, **{**context, **kwargs})
            return True
    
    # If we get here, no suitable plot was found
    ax.text(0.5, 0.5, "No closed lost reason overindex plot found in hypothesis", 
           ha='center', va='center')
    return False

# Register the plotter
register_plotter('closed_lost_reason', plot_for_report) 