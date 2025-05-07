import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import pandas as pd
from ..core.data_registry import DataRegistry
from ..core.types import PlotSpec
import os
import matplotlib.gridspec as gridspec

# Import plot functions from specialized modules
from .metric_plots.anomaly_plots import metric_bar_anomaly
from .hypothesis_plots.single_dim import hypo_bar_scored
from ..reporting.report_plots.detailed import plot_summary_report

logger = logging.getLogger(__name__)

# Centralized style settings
STYLE = {
    'colors': {
        'anomaly_positive': '#2ecc71',  # Green
        'anomaly_negative': '#e74c3c',  # Red
        'global_line': '#34495e',       # Dark blue-gray
        'confidence_band': '#AED6F1',   # Lighter blue
        'default_bar': '#BDC3C7',       # Lighter gray
        'highlight': '#5DADE2',         # Blue highlight
        'text': '#2c3e50',             # Dark text
        'highlight_text': 'dodgerblue', # Highlight text for [selected] / [Anomaly Detected in KPI]
        'score_color': '#AF7AC5',       # Purple for score
        'score_components': {
            'direction_alignment': '#3498DB',  # Blue
            'consistency': '#4ECDC4',         # Teal
            'hypo_z_score_norm': '#FFC300',   # Yellow/Orange
            'explained_ratio': '#FF9F43'      # Orange
        }
    },
    'score_components': {
        'direction_alignment': {'weight': 0.3, 'name': 'Dir. Align'},
        'consistency': {'weight': 0.3, 'name': 'Consistency'},
        'hypo_z_score_norm': {'weight': 0.2, 'name': 'Hypo Z-Score'},
        'explained_ratio': {'weight': 0.2, 'name': 'Expl. Ratio'}
    },
    'score_component_order': [
        'direction_alignment',
        'consistency',
        'hypo_z_score_norm',
        'explained_ratio'
    ],
    'anomaly_band_alpha': 0.2
}

def setup_style():
    """Set consistent style for all visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.grid": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "font.family": "sans-serif",
        "font.size": 10
    })

# Helper functions used across all plot types
def _create_scorecard(ax: plt.Axes, x: float, y: float, value: float, component_name: str, color: str, show_text: bool = False, fontsize: Optional[float] = None, width_factor=1.0):
    """Create a colored scorecard on an axis."""
    transform = ax.transAxes
    base_card_width = 0.11 # Base width
    card_width = base_card_width * width_factor # Allow scaling
    card_height = 0.08
    # Prevent cards going off-axis slightly
    if x + card_width > 1.0: x = 1.0 - card_width - 0.005 
    if y + card_height > 1.0: y = 1.0 - card_height - 0.005

    rect = plt.Rectangle((x, y), card_width, card_height, facecolor=color, alpha=0.2, transform=transform)
    ax.add_artist(rect)
    text_content = component_name if show_text else f"{value:.2f}"
    ax.text(x + card_width/2, y + card_height/2, text_content, ha='center', va='center',
            transform=transform, fontweight='bold', color=color, fontsize=fontsize or 8)
    return card_width

def _add_score_components(ax: plt.Axes, start_x: float, y: float, components_dict: Dict[str, float], show_score: bool = True, show_text: bool = False, align_right: bool = False):
    """Add score components with operators to an axis. Can align right."""
    transform = ax.transAxes
    card_spacing = 0.02
    text_size = 11
    base_card_width = 0.11
    operator_spacing = 0.05
    score_card_height = 0.08
    x_pos = start_x
    ordered_components = [(comp, components_dict.get(comp, 0)) for comp in STYLE['score_component_order']]

    # --- Calculate total width needed for right alignment --- 
    total_width = 0
    num_components = len(ordered_components)
    if show_score:
        total_width += base_card_width * 1.15 + card_spacing # Score card width
    for i in range(num_components):
        if i > 0 or not show_score:
             total_width += operator_spacing # Plus sign space
        total_width += 0.04 # Weight space
        total_width += base_card_width + card_spacing # Component card
    total_width -= card_spacing # Remove last spacing

    if align_right:
         x_pos = 0.99 - total_width # Start from right edge minus total width
    # --- End Width Calculation ---

    score_value = components_dict.get('score', 0.0)

    if show_score:
        score_label_color = STYLE['colors'].get('score_color', '#AF7AC5')
        label_text = f"Score ({score_value:.2f})"
        width_factor = 1.15 # Estimate wider card
        card_width = _create_scorecard(ax, x_pos, y, 0, label_text, score_label_color, show_text=True, fontsize=text_size, width_factor=width_factor)
        x_pos += card_width + card_spacing 

    for i, (component, value) in enumerate(ordered_components):

        component_config = STYLE['score_components'][component]
        color = STYLE['colors']['score_components'][component]
        weight = component_config['weight']
        if i > 0 or not show_score:
            ax.text(x_pos, y + score_card_height/2, " + ", ha='center', va='center', color=color, fontsize=text_size, transform=transform)
            x_pos += operator_spacing
        ax.text(x_pos, y + score_card_height/2, f"{weight:.1f}×", ha='center', va='center', color=color, fontsize=text_size, transform=transform)
        x_pos += 0.04
        component_text_size = text_size * 0.9 if not show_text else text_size
        card_width = _create_scorecard(ax, x_pos, y, value, component_config['name'], color, show_text=show_text, fontsize=component_text_size)
        x_pos += card_width + card_spacing
    return x_pos

def _add_score_components_compact(ax: plt.Axes, start_x: float, y: float, components_dict: Dict[str, float], align_right: bool = False):
    """Add compact score component boxes without operators. Can align right."""
    card_spacing = 0.02 # Tighter spacing for compact
    card_width = 0.09 # Smaller cards for compact
    card_height = 0.07 # Smaller height
    text_size = 10
    ordered_components = [(comp, components_dict.get(comp, 0)) for comp in STYLE['score_component_order']]
    num_items = len(ordered_components) + 1 # +1 for score
    total_width = num_items * card_width + (num_items - 1) * card_spacing
    
    if align_right:
         x_pos = 0.99 - total_width # Align to right edge
    else:
         # Default: Center
         x_pos = 0.5 - (total_width / 2)
         x_pos = max(0.01, x_pos)

    score_value = components_dict.get('score', 0.0)
    score_color = STYLE['colors'].get('score_color')
    rect = plt.Rectangle((x_pos, y), card_width, card_height, facecolor=score_color, alpha=0.2, transform=ax.transAxes)
    ax.add_artist(rect)
    ax.text(x_pos + card_width/2, y + card_height/2, f"{score_value:.2f}", ha='center', va='center', transform=ax.transAxes, fontweight='bold', color=score_color, fontsize=text_size)
    x_pos += card_width + card_spacing

    for i, (component, value) in enumerate(ordered_components):
        if component not in STYLE['colors']['score_components']:
            continue
        color = STYLE['colors']['score_components'][component]
        rect = plt.Rectangle((x_pos, y), card_width, card_height, facecolor=color, alpha=0.2, transform=ax.transAxes)
        ax.add_artist(rect)
        ax.text(x_pos + card_width/2, y + card_height/2, f"{value:.2f}", ha='center', va='center', transform=ax.transAxes, fontweight='bold', color=color, fontsize=text_size)
        x_pos += card_width + card_spacing

def _apply_yaxis_percentage_formatting(ax: plt.Axes, df: pd.DataFrame, y_label: Optional[str]=None, 
                                    value_col: Optional[str]=None, 
                                    metric_name: Optional[str]=None, 
                                    hypothesis_name: Optional[str]=None, 
                                    metric_natural_name: Optional[str]=None,
                                    hypothesis_natural_name: Optional[str]=None,
                                    force_zero_base: bool=True,
                                    **kwargs):
    """Applies percentage formatting to the y-axis and sets appropriate y-axis limits.
    
    Args:
        ax: The matplotlib axis to format
        df: DataFrame with the data being plotted
        y_label: The y-axis label
        value_col: The column name containing values
        metric_name: The metric name
        hypothesis_name: The hypothesis name
        metric_natural_name: Natural name for metric
        hypothesis_natural_name: Natural name for hypothesis
        force_zero_base: Whether to force y-axis to start at 0 for percentages
    """
    if ax is None or not hasattr(ax, 'yaxis'):
        return False  # Can't format if axis is invalid
        
    # Determine if this is a percentage metric
    is_percent = False
    
    # Prioritize natural name if available
    label_to_check = metric_natural_name or hypothesis_natural_name or y_label or ""
    
    # Fallback to technical name if natural name missing
    col_name_to_check = value_col or metric_name or hypothesis_name or ""
    
    # 1. Check explicit indicators like % in label
    if '%' in label_to_check:
        is_percent = True
    
    # 2. Check for _pct suffix in column/metric name
    if not is_percent and "_pct" in col_name_to_check.lower():
        is_percent = True
    
    # Apply percentage formatting
    if is_percent:
        # Format y-axis ticks as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        
        # Ensure y-label indicates percentage if not already
        current_ylabel = ax.get_ylabel()
        if '%' not in current_ylabel:
            ax.set_ylabel(f"{current_ylabel} (%)")
        
        # Force y-axis to start at 0 for percentages (critical fix)
        if force_zero_base:
            ylim = ax.get_ylim()
            if value_col and value_col in df.columns:
                # Get actual data range
                data_max = df[value_col].max()
                # Add 20% padding to the top, but ensure we start at 0
                ax.set_ylim(0, max(ylim[1], data_max * 1.2))
            else:
                # Just ensure we start at 0
                ax.set_ylim(0, ylim[1])
    
    return is_percent

def _adjust_bars_for_annotations(ax, lower_by_percent=0.2):
    """Adjust the position of bars to lower them on the plot, making room for annotations.
    
    Args:
        ax: Matplotlib axis containing the bars
        lower_by_percent: How much to lower the bars (as percentage of the axis height)
    """
    # Get current y limits
    y_min, y_max = ax.get_ylim()
    
    # Calculate height of current axis
    height = y_max - y_min
    
    # Calculate new y limits
    # - Keep same y_min (bottom)
    # - Increase y_max to create more space for annotations
    new_max = y_max + (height * lower_by_percent)
    
    # Set the new limits
    ax.set_ylim(y_min, new_max)
    
    # Return new limits in case needed elsewhere
    return (y_min, new_max)

def create_score_formula_on_fig(fig, y_pos = 0.005):
    """Create a colored score formula at the bottom of the figure."""
    # Components dictionary with dummy values (0.0) just for structure/names
    components_dict = {comp: 0.0 for comp in STYLE['score_component_order']}
    
    # --- Recalculate start_x for centering (adapted logic) --- 
    card_spacing = 0.01 
    text_size = 14 # Increased font size
    base_card_width = 0.09 
    operator_spacing = 0.04
    transform = fig.transFigure # Use figure transform
    score_card_height = 0.025 # Figure-specific height

    approx_total_width = 0
    # Score label card
    approx_total_width += base_card_width + card_spacing 
    approx_total_width += operator_spacing # Space for '=' 
             
    # Components part
    num_components = len(STYLE['score_component_order'])
    for i in range(num_components):
        approx_total_width += operator_spacing # Minimal space for '+'
        approx_total_width += 0.03 # Minimal space for '0.Nx'
        approx_total_width += base_card_width + card_spacing
            
    if approx_total_width > 0: 
        approx_total_width -= card_spacing 
        
    start_x = (1.0 - approx_total_width) / 2
    start_x = max(0.01, start_x) 
    # --- End Recalculation --- 
    
    # Now draw using adapted logic similar to _add_score_components but on Figure
    x_pos = start_x
    score_label_color = STYLE['colors'].get('score_color')

    # Draw "Score" label card
    rect = plt.Rectangle((x_pos, y_pos), base_card_width, score_card_height, facecolor=score_label_color, alpha=0.2, transform=transform)
    fig.add_artist(rect)
    fig.text(x_pos + base_card_width/2, y_pos + score_card_height/2, "Score", ha='center', va='center',
             transform=transform, fontweight='bold', color=score_label_color, fontsize=text_size)
    x_pos += base_card_width + card_spacing/2
    
    # Draw "=" sign
    fig.text(x_pos, y_pos + score_card_height/2, " = ", ha='center', va='center', color=score_label_color, fontsize=text_size, transform=transform)
    x_pos += operator_spacing

    # Draw components
    ordered_components = [(comp, components_dict.get(comp, 0)) for comp in STYLE['score_component_order']]
    for i, (component, value) in enumerate(ordered_components):
        if component not in STYLE['score_components'] or component not in STYLE['colors']['score_components']:
            continue
        component_config = STYLE['score_components'][component]
        color = STYLE['colors']['score_components'][component]
        weight = component_config['weight']
        if i > 0:
            fig.text(x_pos, y_pos + score_card_height/2, " + ", ha='center', va='center', color=color, fontsize=text_size, transform=transform)
            x_pos += operator_spacing
        fig.text(x_pos, y_pos + score_card_height/2, f"{weight:.1f}×", ha='center', va='center', color=color, fontsize=text_size, transform=transform)
        x_pos += 0.04
        # Draw component name card
        rect = plt.Rectangle((x_pos, y_pos), base_card_width, score_card_height, facecolor=color, alpha=0.2, transform=transform)
        fig.add_artist(rect)
        fig.text(x_pos + base_card_width/2, y_pos + score_card_height/2, component_config['name'], ha='center', va='center',
                 transform=transform, fontweight='bold', color=color, fontsize=text_size * 0.9) # Slightly smaller text
        x_pos += base_card_width + card_spacing

# Main routing function
def route_plot(plot_spec: PlotSpec, plot_engine=None) -> Optional[str]:
    """Route a plot specification to the appropriate plotting function.
    
    Args:
        plot_spec: Plot specification containing data and context
        plot_engine: Optional PlotEngine instance for saving plots
        
    Returns:
        The path to the saved figure (batch mode) or Figure object (inline mode)
    """
        
    setup_style()
    
    # Get plot key and data
    plot_key = plot_spec.plot_key
    df = plot_spec.data
    context = plot_spec.context
    
    # Ensure we have data to plot
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.error(f"No data available for plot '{plot_key}'")
        return None
    
    # Define a mapping of plot_key prefixes to plotting functions
    plot_functions = {
        'metric_bar_anomaly': metric_bar_anomaly,
        'hypo_bar_scored': hypo_bar_scored,
        'summary_report': plot_summary_report
    }
    
    # Dynamic mapping for future extensions
    # Check if plot_key matches any of our prefixes (metric_* or hypo_*)
    if plot_key not in plot_functions:
        if plot_key.startswith('metric_'):
            plot_functions[plot_key] = metric_bar_anomaly  # Default for all metric_* plots
        elif plot_key.startswith('hypo_'):
            plot_functions[plot_key] = hypo_bar_scored  # Default for all hypo_* plots
    
    # Route to the appropriate plot function based on plot_key
    try:
        # Get the appropriate plotting function
        plot_func = plot_functions.get(plot_key)
        
        if not plot_func:
            logger.error(f"Unknown plot type: {plot_key}")
            return None
            
        # Handle special case for summary report
        if plot_func == plot_summary_report:
            return plot_summary_report(df=df, **context)
            
        # For standard plots that need a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a clean context without df to avoid duplicate parameter
        context_without_df = {k: v for k, v in context.items() if k != 'df'}
        
        # Call the appropriate plotting function
        if plot_key.startswith('metric_'):
            logger.info(f"{plot_key} called for {context.get('metric_name')}, value_col={context.get('value_col')}, columns in df: {list(df.columns)}")
        
        plot_func(ax=ax, df=df, **context_without_df)
            
        # Ensure tight layout
        fig.tight_layout()
            
        # In batch mode, save the figure using the plot engine
        if plot_engine:
            # Determine the filename based on the plot key and context
            if plot_key.startswith('metric_'):
                filename = f"metric_{context.get('metric_name')}.png"
            elif plot_key.startswith('hypo_'):
                filename = f"hypo_{context.get('hypothesis_name')}_for_{context.get('metric_name')}.png"
            else:
                # Fallback filename
                filename = f"{plot_key}_{hash(str(context))}.png"
                
            # Save the plot using the engine
            return plot_engine.save_plot(fig, filename)
        
        # Otherwise return the figure object directly (inline mode)
        return fig
            
    except Exception as e:
        logger.error(f"Error generating plot '{plot_key}': {e}")
        logger.exception(e)
        return None 