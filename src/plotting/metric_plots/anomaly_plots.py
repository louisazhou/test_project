"""
Anomaly-detecting metric plots.

This module implements visualizations for metrics with anomaly detection.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

# Import shared style and utility functions
from .. import plot_router

logger = logging.getLogger(__name__)


def metric_bar_anomaly(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    value_col: str,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    metric_name: Optional[str] = None,
    metric_natural_name: Optional[str] = None,
    higher_is_better: bool = True,
    ref_metric_val: Optional[float] = None,
    std: Optional[float] = None,
    z_score_threshold: float = 1.5,
    enrichment_data: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> None:
    """Plot metric values as a bar chart with anomaly highlighting.
    
    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame containing metric values
        value_col: Column containing metric values
        title: Plot title
        y_label: Y-axis label
        metric_name: Technical name of the metric
        metric_natural_name: Human-readable metric name
        higher_is_better: Whether higher values are better
        ref_metric_val: Reference value (typically global average)
        std: Standard deviation of the metric
        z_score_threshold: Z-score threshold for highlighting anomalies
        enrichment_data: Additional data about each region
    """
    logger.debug(f"metric_bar_anomaly called for {metric_name}, value_col={value_col}")
    
    plot_router.setup_style()
    
    if df.empty:
        logger.warning(f"No data provided for metric_bar_anomaly: {title}")
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        ax.set_title(title or "Metric - No Data")
        return

    # --- Log for debugging ---
    logger.debug(f"metric_bar_anomaly called for {metric_name}, value_col={value_col}")

    # --- Ensure region index --- 
    if df.index.name != 'region':
         if 'region' in df.columns:
             try:
                 df = df.set_index('region')
             except Exception as e:
                 logger.error(f"Error setting 'region' as index for metric plot {metric_name or title}: {e}")
                 ax.text(0.5, 0.5, "Error: Failed to set region index", ha='center', va='center')
                 return
         else:
             logger.error(f"Metric plot requires DataFrame indexed by region ('region') or have 'region' column. Plot: {metric_name or title}")
             ax.text(0.5, 0.5, "Error: region index/column missing", ha='center', va='center')
             return
    # --- End Ensure region Index --- 

    # --- Validate Value Column --- 
    if value_col not in df.columns:
         logger.error(f"Metric column '{value_col}' not found in DataFrame for metric plot. Available columns: {df.columns.tolist()}")
         ax.text(0.5, 0.5, f"Error: Column '{value_col}' missing", ha='center', va='center')
         return

    actual_metric_name_display = metric_natural_name or metric_name 
    df = df.sort_index()
    regions = df.index.tolist()
    x_positions = np.arange(len(regions))

    # --- Color Logic using Enrichment Data ---
    bar_colors = []
    enrichment_data = enrichment_data or {} # Ensure it's a dict

    for region in regions:
        region_enrichment = enrichment_data.get(region, {}) # Get data for this region
        is_anomaly = region_enrichment.get('is_anomaly', False)
        
        if is_anomaly:
            if region_enrichment.get('bad_anomaly', False):
                bar_colors.append(plot_router.STYLE['colors']['anomaly_negative'])
            elif region_enrichment.get('good_anomaly', False):
                bar_colors.append(plot_router.STYLE['colors']['anomaly_positive'])
            else: # Anomaly but flags missing? Default red.
                bar_colors.append(plot_router.STYLE['colors']['anomaly_negative'])
        else: # Not an anomaly or Global row (not in enrichment_data)
            bar_colors.append(plot_router.STYLE['colors']['default_bar'])
    # --- End Color Logic ---

    # Plot bars
    bars = ax.bar(x_positions, df[value_col], color=bar_colors)

    # Set x-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regions, rotation=0, ha='center')
    plt.setp(ax.get_xticklabels(), fontsize=9)

    # Add reference line and confidence band
    ax.axhline(ref_metric_val, color=plot_router.STYLE['colors']['global_line'], linestyle='--', linewidth=1)
    if std > 0:
        ax.axhspan(ref_metric_val - std * z_score_threshold,
                   ref_metric_val + std * z_score_threshold,
                   color=plot_router.STYLE['colors']['confidence_band'],
                   alpha=plot_router.STYLE['anomaly_band_alpha'],
                   label=f'±{z_score_threshold:.1f} Std Dev')
        ax.legend(loc='upper right', fontsize=8)

    # Use the centralized formatting check
    is_percent_fmt = plot_router._apply_yaxis_percentage_formatting(
        ax=ax, df=df, y_label=y_label, 
        value_col=value_col, 
        metric_name=metric_name, 
        metric_natural_name=actual_metric_name_display,
        force_zero_base=True,
        **kwargs
    )
    
    # Adjust bars lower to make room for annotations
    plot_router._adjust_bars_for_annotations(ax)

    # Now use the result to format the bar annotations
    value_format = '{:.1f}%' if is_percent_fmt else '{:.2f}'
    z_score_format = '(z={:.2f})'
    for i, region in enumerate(regions):
        val = df.loc[region, value_col]
        display_val = val * 100 if is_percent_fmt else val
        label_text = f"{value_format.format(display_val)}"
        
        # Get z-score from enrichment data if available
        region_enrichment = enrichment_data.get(region, {})
        z_score = region_enrichment.get('z_score') # Will be None if region not in enrichment (e.g., Global)
        if z_score is not None:
             label_text += f"\n{z_score_format.format(z_score)}"
        
        ax.text(i, val, label_text, ha='center', va='bottom', color=plot_router.STYLE['colors']['text'], fontsize=8)

    # Set title and labels
    ax.set_title(title or f"Metric: {actual_metric_name_display}", fontsize=11)
    ax.set_ylabel(y_label or actual_metric_name_display.replace('_',' ').title(), fontsize=10)

    # Add Anomaly Detected label if ANY anomaly exists in enrichment data
    if any(ed.get('is_anomaly', False) for ed in enrichment_data.values()):
        ax.text(0.015, 0.97, "Anomaly Detected in KPI", ha='left', va='top',
                transform=ax.transAxes, fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.3', fc=plot_router.STYLE['colors']['highlight_text'], alpha=1))

    # Apply formatting
    plot_router._apply_yaxis_percentage_formatting(
        ax=ax, df=df, y_label=y_label, 
        value_col=value_col, 
        metric_name=metric_name, 
        metric_natural_name=actual_metric_name_display,
        force_zero_base=True,
        **kwargs
    ) 