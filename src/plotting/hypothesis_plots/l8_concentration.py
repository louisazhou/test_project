"""
L8 concentration hypothesis plots.

This module will contain implementations for visualizing L8 territory concentration hypotheses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, Optional, List
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.ticker import FixedLocator, PercentFormatter

from ..plot_styles import setup_style, STYLE
from ...registry import hypothesis_plotter, register_plotter

logger = logging.getLogger(__name__)

def l8_concentration_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    primary_region: str = None,
    value_col: str = None,
    region_col: str = "region",
    subregion_col: str = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    **kwargs
) -> None:
    """Generate an L8 concentration density plot.
    
    Args:
        ax: The matplotlib axes to draw on
        df: DataFrame containing the territory data with regions and values
        primary_region: The region to highlight (anomaly region)
        value_col: Column containing the metric value 
        region_col: Column containing the region
        subregion_col: Optional column for subregions or territory IDs
        title: Plot title (optional)
        subtitle: Plot subtitle (optional)
    """
    # Return early if essential data is missing
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data available for L8 concentration plot", ha='center', va='center')
        return
    
    # Ensure style consistency
    setup_style()
    
    # Extract necessary data
    if value_col is None:
        # Try to infer value column (first column that's not region or subregion)
        value_candidates = [col for col in df.columns if col != region_col and col != subregion_col]
        if value_candidates:
            value_col = value_candidates[0]
            logger.debug(f"Inferred value_col: {value_col} for L8 concentration plot")
        else:
            ax.text(0.5, 0.5, "Could not determine value column for L8 concentration plot", 
                   ha='center', va='center')
            return
    
    # Ensure all regions have an index label
    if isinstance(df, pd.DataFrame) and region_col in df.columns:
        # Group by region and calculate distribution
        grouped = df.groupby(region_col)[value_col].agg(['mean', 'count', 'std']).reset_index()
    else:
        ax.text(0.5, 0.5, f"Missing required column '{region_col}' for L8 concentration plot", 
               ha='center', va='center')
        return
    
    # Use a color palette with distinct colors for each region
    regions = grouped[region_col].unique()
    region_count = len(regions)
    
    # Use a colorblind-friendly palette from seaborn if available
    color_palette = sns.color_palette("colorblind", n_colors=max(region_count, 8))
    
    # Create a dictionary mapping regions to colors
    region_colors = {region: color_palette[i % len(color_palette)] for i, region in enumerate(regions)}
    
    # If primary region specified, emphasize it with a darker color
    if primary_region:
        # Keep the primary region color but make it more prominent in the plot
        primary_color = region_colors[primary_region]
        
    # Calculate overall data range for the linspace
    min_val = df[value_col].min() * 0.9
    max_val = df[value_col].max() * 1.1
    xs = np.linspace(min_val, max_val, 200)
    
    # Plot territory-level data distributions using gaussian_kde
    for region in grouped[region_col].unique():
        region_data = df[df[region_col] == region][value_col].dropna()
        if len(region_data) < 2:
            logger.warning(f"Skipping density for {region}, not enough data points ({len(region_data)}). Plotting markers instead.")
            if len(region_data) == 1:
                ax.plot(region_data, [0], marker='o', markersize=8, linestyle='None', 
                       label=f"{region} (single value: {region_data.iloc[0]:.3f})", 
                       color=region_colors[region])
            continue  # Skip KDE if not enough points
            
        # Plot density with KDE
        try:
            density = gaussian_kde(region_data)
            density_values = density(xs)
            
            # Adjust line properties based on whether this is the primary region
            line_alpha = 0.9 if region == primary_region else 0.7
            line_width = 2.5 if region == primary_region else 1.8
            
            ax.plot(xs, density_values, label=region, 
                   color=region_colors[region], 
                   linewidth=line_width, 
                   alpha=line_alpha)
            
            # If this is the primary region, add slightly transparent fill below curve
            if region == primary_region:
                ax.fill_between(xs, 0, density_values, 
                               color=region_colors[primary_region], 
                               alpha=0.15)
                
                # If we have subregion data for primary region, mark them on the plot
                if 'lagged_sub_regions' in kwargs and subregion_col:
                    lagged_sub_regions = kwargs.get('lagged_sub_regions', [])
                    for i, sub_name in enumerate(lagged_sub_regions):
                        sub_data = df[(df[region_col] == primary_region) & 
                                     (df[subregion_col] == sub_name)]
                        if not sub_data.empty:
                            sub_value = sub_data[value_col].iloc[0]
                            ax.axvline(sub_value, color=region_colors[primary_region], 
                                      linestyle=':', linewidth=1.2, alpha=0.8)
                            # Place text above the line, staggered to avoid overlap
                            y_pos = 0.8 - (i % 3) * 0.1
                            ax.text(sub_value, y_pos, f"{sub_name}: {sub_value:.3f}", 
                                   rotation=90, va='top', fontsize=8, 
                                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
                
        except Exception as e:
            logger.error(f"Error plotting density for region {region}: {e}")
    
    # Calculate the group statistics
    if 'Global' in grouped[region_col].values:
        global_stats = grouped[grouped[region_col] == 'Global'].iloc[0]
        global_mean = global_stats['mean']
        ax.axvline(x=global_mean, color='black', linestyle='--', 
                  linewidth=1.5, label='Global Average')
    
    # Add p10 threshold line if provided
    if 'p10_all_territory' in kwargs and kwargs['p10_all_territory'] is not None:
        p10 = kwargs['p10_all_territory']
        higher_is_better = kwargs.get('higher_is_better', True)
        threshold_label = f"Global {('10th' if higher_is_better else '90th')} Percentile"
        ax.axvline(p10, color='red', linestyle='--', linewidth=1.2,
                  label=f"{threshold_label} ({p10:.3f})")
    
    # Add primary region marker
    if primary_region and primary_region in grouped[region_col].values:
        primary_stats = grouped[grouped[region_col] == primary_region].iloc[0]
        primary_mean = primary_stats['mean']
        ax.axvline(x=primary_mean, color=region_colors[primary_region], linestyle='-', 
                  linewidth=2.0, label=f'{primary_region} Average')
    
    # Add labels and title
    if title:
        ax.set_title(title, fontsize=12)
    else:
        if primary_region:
            ax.set_title(f"L8 Concentration for {primary_region}", fontsize=12)
        else:
            ax.set_title("L8 Concentration by Region", fontsize=12)
            
    if subtitle:
        ax.text(0.5, 0.92, subtitle, ha='center', va='center', transform=ax.transAxes, 
               fontsize=9, style='italic')
    
    ax.set_xlabel(f"{value_col}", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    
    # Add legend, position it outside the plot if many regions
    if len(grouped[region_col].unique()) > 5:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    else:
        ax.legend(fontsize=9)
    
    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    # Apply percentage formatting if it's a percentage column
    if value_col.lower().endswith('_pct'):
        # Get current ticks and set both the locator and formatter
        current_ticks = ax.get_xticks()
        ax.xaxis.set_major_locator(FixedLocator(current_ticks))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

@hypothesis_plotter
def plot_for_report(ax, hypo_result, focus_region=None, primary_region=None, **kwargs):
    """Plot L8 concentration for a report.
    
    This function extracts data from a hypothesis result and calls the 
    l8_concentration_plot function.
    
    Args:
        ax: The matplotlib axes to draw on
        hypo_result: The hypothesis result object
        focus_region: The region to focus on (priority over primary_region)
        primary_region: The primary region to focus on (used if focus_region not provided)
        **kwargs: Additional parameters
        
    Returns:
        True if plotting was successful, False otherwise
    """
    if not hasattr(hypo_result, 'plots') or not hypo_result.plots:
        ax.text(0.5, 0.5, "No plots defined for L8 concentration hypothesis", 
               ha='center', va='center')
        return False
        
    # Look for a plot spec with the right key
    for plot_spec in hypo_result.plots:
        if plot_spec.plot_key == 'l8_concentration':
            # Extract context and data from the plot spec
            context = plot_spec.context.copy() if plot_spec.context else {}
            df = plot_spec.data
            
            # Prioritize parameters: focus_region > primary_region > context['primary_region']
            target_region = focus_region or primary_region or context.get('primary_region')
            
            # Create new context without primary_region to avoid conflict
            if 'primary_region' in context:
                del context['primary_region']
                
            # Remove primary_region from kwargs if present to avoid conflicts
            clean_kwargs = kwargs.copy()
            if 'primary_region' in clean_kwargs:
                del clean_kwargs['primary_region']
            
            # Call the plotting function with explicit primary_region
            l8_concentration_plot(
                ax=ax, 
                df=df, 
                primary_region=target_region, 
                **context,
                **clean_kwargs
            )
            return True
    
    # No suitable plot spec found
    ax.text(0.5, 0.5, "No L8 concentration plot found in hypothesis", 
           ha='center', va='center')
    return False

# Register the plotter
register_plotter('depth_spotter', plot_for_report) 