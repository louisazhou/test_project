"""
Single dimension hypothesis plots.

This module implements visualizations for single dimension hypotheses.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging

# Import shared style and utility functions
from .. import plot_router
from ..plot_styles import setup_style, STYLE
from ...registry import hypothesis_plotter, register_plotter
from ...core.types import MetricFormatting

logger = logging.getLogger(__name__)


def hypo_bar_scored(
    ax: plt.Axes,
    df: pd.DataFrame,
    *, # Make subsequent args keyword-only
    region_col: str,
    value_col: str,
    hypothesis_name: str,
    hypothesis_natural_name: Optional[str] = None, # Expect natural name
    explaining_region: Optional[str] = None,
    primary_region: Optional[str] = None,
    score_components: Optional[Dict[str, float]] = None,
    selected: bool = False,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    **kwargs # Accept extra kwargs
) -> None:
    """Plot hypothesis values, highlighting the explaining region and showing score.
    
    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame containing hypothesis values by region
        region_col: Column name for region identifier
        value_col: Column name for hypothesis values
        hypothesis_name: Technical name of the hypothesis
        hypothesis_natural_name: Human-readable name of the hypothesis
        explaining_region: Region being explained (will be highlighted)
        primary_region: Primary region of interest (usually same as explaining_region)
        score_components: Dictionary containing score components
        selected: Whether this hypothesis is selected (best hypothesis)
        title: Plot title (defaults to hypothesis name)
        y_label: Y-axis label (defaults to value_col)
    """
    plot_router.setup_style()
    
    if df.empty:
        logger.warning(f"No data provided for hypo_bar_scored: {title or hypothesis_name}")
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        ax.set_title(title or f"Hypothesis: {hypothesis_name} - No Data")
        return

    # --- Logic Adapted from original plot_hypothesis --- 
    # Ensure data is indexed by the specified region column
    if df.index.name != region_col:
        if region_col in df.columns:
            df = df.set_index(region_col)
        else:
            logger.error(f"Region column '{region_col}' not found for hypo plot: {title or hypothesis_name}")
            ax.text(0.5, 0.5, f"Error: '{region_col}' missing", ha='center', va='center')
            return

    # Check if value_col exists
    if value_col not in df.columns:
        logger.error(f"Value column '{value_col}' not found for hypo plot: {title or hypothesis_name}")
        ax.text(0.5, 0.5, f"Error: '{value_col}' missing", ha='center', va='center')
        return

    df = df.sort_index()
    regions = df.index.tolist()
    x_positions = np.arange(len(regions))
    values = df[value_col]

    # Get reference value (Global)
    ref_val = df.loc['Global', value_col] if 'Global' in df.index else values.mean()

    # Create bar colors
    bar_colors = [plot_router.STYLE['colors']['default_bar']] * len(regions)
    explaining_idx = regions.index(explaining_region) if explaining_region and explaining_region in regions else -1
    if explaining_idx != -1:
        bar_colors[explaining_idx] = plot_router.STYLE['colors']['highlight']

    # Plot bars
    bars = ax.bar(x_positions, values, color=bar_colors)

    # Set x-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regions, rotation=0, ha='center')
    plt.setp(ax.get_xticklabels(), fontsize=9)

    # Add reference line
    if not np.isnan(ref_val):
        ax.axhline(ref_val, color=plot_router.STYLE['colors']['global_line'], linestyle='--', linewidth=1)

    # Use centralized formatting function
    is_percent_fmt = plot_router._apply_yaxis_percentage_formatting(
        ax=ax, df=df, y_label=y_label, 
        value_col=value_col, 
        hypothesis_name=hypothesis_name,
        hypothesis_natural_name=hypothesis_natural_name,
        force_zero_base=True,
        **kwargs
    )

    # Adjust bars to make room for annotations
    plot_router._adjust_bars_for_annotations(ax)

    # Add value labels based on formatting result
    value_format = '{:.1f}%' if is_percent_fmt else '{:.1f}'
    for i, val in enumerate(values):
        display_val = val * 100 if is_percent_fmt else val
        ax.text(i, val, value_format.format(display_val), ha='center', va='bottom',
                color=plot_router.STYLE['colors']['text'], fontsize=8)

    # Add score components (Aligned Top-Right for compact, Top-Left for full)
    score_y_pos = 0.9 
    if score_components:
        is_root_cause = selected
        if is_root_cause:
            # Full equation - Pass the dictionary containing the score
            plot_router._add_score_components(ax, 0.02, score_y_pos - 0.08, score_components,
                                  show_score=True, show_text=False, align_right=False) 
        else:
            # Compact boxes - Pass the dictionary containing the score
            plot_router._add_score_components_compact(ax, 0, score_y_pos, score_components, align_right=True)

    # Add [selected] / [not selected] marker
    marker_y_pos = 1.02 # Position slightly above axis
    if selected:
         # Position selected marker Top-Left
         ax.text(0.015, marker_y_pos, "[selected]", ha='left', va='bottom', transform=ax.transAxes, fontsize=9, color='white',
                 bbox=dict(boxstyle='round,pad=0.3', fc=plot_router.STYLE['colors']['highlight_text'], alpha=1))
    else:
         # Position not selected marker Top-Right
         marker_x_pos = 0.99 
         ax.text(marker_x_pos, marker_y_pos, "[not selected]", ha='right', va='bottom', transform=ax.transAxes, fontsize=9, color='black',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', alpha=1))

    # Set title and labels using natural name
    plot_title = title or f"Hypothesis: {hypothesis_natural_name or hypothesis_name}"
    
    # Add delta info to title if explaining region exists and not already in title
    if explaining_region and explaining_region in df.index:
        if not (title and ("than Global" in title or "(" in title)):
            # Get region value and global value
            region_value = df.loc[explaining_region, value_col]
            global_value = df.loc['Global', value_col] if 'Global' in df.index else df[value_col].mean()
            
            # Determine if this is a percentage metric
            is_percentage = value_col.lower().endswith('_pct') or kwargs.get('is_percentage', False)
            
            # Calculate direction
            direction = "higher" if region_value > global_value else "lower"
            
            # Format delta using MetricFormatting
            delta_fmt = MetricFormatting.format_delta(
                region_value, 
                global_value, 
                is_percentage
            )
            
            # Add formatted delta to title
            plot_title += f" ({explaining_region} is {delta_fmt} {direction} than Global)"

    ax.set_title(plot_title, fontsize=11)
    ax.set_ylabel(y_label or value_col.replace('_', ' ').title(), fontsize=10)

@hypothesis_plotter
def plot_for_report(ax, hypo_result, **kwargs):
    """Plot a single dimension hypothesis for a report.
    
    This function extracts data from a hypothesis result and calls the
    hypo_bar_scored function.
    
    Args:
        ax: The matplotlib axes to draw on
        hypo_result: The hypothesis result object
        **kwargs: Additional parameters
        
    Returns:
        True if plotting was successful, False otherwise
    """
    # Extract plot data
    hypo_df = getattr(hypo_result, 'plot_data', None)
    if hypo_df is None:
        logger.warning(f"Could not get data for hypothesis plot: {hypo_result.name}")
        ax.text(0.5, 0.5, f"No plot data available for {hypo_result.name}", 
               ha='center', va='center')
        return False
    
    # Extract value column
    value_col = kwargs.get('value_col')
    if not value_col and hypo_df is not None and len(hypo_df.columns) > 0:
        # Try to determine value_col from the plot_data columns
        potential_cols = [col for col in hypo_df.columns if col != 'region']
        if potential_cols:
            value_col = potential_cols[0]
            logger.debug(f"Determined value_col '{value_col}' from plot_data columns for hypothesis: {hypo_result.name}")
        else:
            logger.error(f"Could not determine value_col for hypothesis plot: {hypo_result.name}")
            ax.text(0.5, 0.5, f"Could not determine value column for {hypo_result.name}", 
                   ha='center', va='center')
            return False
    
    # Get region
    primary_region = kwargs.get('primary_region') or kwargs.get('focus_region')
    
    # Format delta text if needed
    if kwargs.get('include_delta', False) and hasattr(hypo_result, 'value') and hasattr(hypo_result, 'global_value'):
        delta_fmt = MetricFormatting.format_delta(
            hypo_result.value, 
            hypo_result.global_value, 
            hypo_result.is_percentage if hasattr(hypo_result, 'is_percentage') else False
        )
        
        direction = "higher" if hypo_result.value > hypo_result.global_value else "lower"
        delta_text = f"({primary_region} is {delta_fmt} {direction} than Global)"
        
        title = kwargs.get('title', '')
        if title and delta_text:
            kwargs['title'] = f"{title} {delta_text}"
    
    # Prepare score components
    score_components = {
        'score': getattr(hypo_result, 'score', 0.0),
        'direction_alignment': kwargs.get('direction_alignment', 0.0),
        'consistency': kwargs.get('consistency', 0.0),
        'hypo_z_score_norm': kwargs.get('hypo_z_score_norm', 0.0),
        'explained_ratio': kwargs.get('explained_ratio', 0.0)
    }
    
    # Create kwargs for the plotter
    hypo_kwargs = {
        'region_col': 'region',
        'value_col': value_col,
        'hypothesis_name': hypo_result.name,
        'hypothesis_natural_name': getattr(hypo_result, 'natural_name', hypo_result.name),
        'explaining_region': primary_region,
        'primary_region': primary_region,
        'score_components': score_components,
        'selected': kwargs.get('selected', False),
        'title': kwargs.get('title', ''),
        'y_label': getattr(hypo_result, 'natural_name', hypo_result.name)
    }
    
    # Call the plotter
    hypo_bar_scored(ax=ax, df=hypo_df, **hypo_kwargs)
    return True

# Register the plotter
register_plotter('single_dim', plot_for_report)