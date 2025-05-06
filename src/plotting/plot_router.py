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
        # ... (component checking logic) ...
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
    score_color = STYLE['colors'].get('score_color', '#AF7AC5')
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
    return y_min, new_max

def metric_bar_anomaly(
    ax: plt.Axes,
    df: pd.DataFrame,
    *, # Make subsequent args keyword-only
    metric_name: str, 
    ref_metric_val: float,
    std: float,
    value_col: str, # No default needed, should always be passed
    metric_natural_name: Optional[str] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    z_score_threshold: float = 1.0,
    higher_is_better: Optional[bool] = None, 
    enrichment_data: Optional[Dict[str, Dict]] = None, # Expect enrichment dict
    **kwargs # Accept extra kwargs
) -> None:
    """Plot metric values with anomaly highlighting and confidence band."""
    setup_style()
    if df.empty:
        logger.warning(f"No data provided for metric_bar_anomaly: {title}")
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        ax.set_title(title or "Metric - No Data")
        return

    # --- Log for debugging ---
    logger.info(f"metric_bar_anomaly called for {metric_name}, value_col={value_col}, columns in df: {df.columns.tolist()}")

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
                bar_colors.append(STYLE['colors']['anomaly_negative'])
            elif region_enrichment.get('good_anomaly', False):
                bar_colors.append(STYLE['colors']['anomaly_positive'])
            else: # Anomaly but flags missing? Default red.
                bar_colors.append(STYLE['colors']['anomaly_negative'])
        else: # Not an anomaly or Global row (not in enrichment_data)
            bar_colors.append(STYLE['colors']['default_bar'])
    # --- End Color Logic ---

    # Plot bars
    bars = ax.bar(x_positions, df[value_col], color=bar_colors)

    # Set x-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regions, rotation=0, ha='center')
    plt.setp(ax.get_xticklabels(), fontsize=9)

    # Add reference line and confidence band
    ax.axhline(ref_metric_val, color=STYLE['colors']['global_line'], linestyle='--', linewidth=1)
    if std > 0:
        ax.axhspan(ref_metric_val - std * z_score_threshold,
                   ref_metric_val + std * z_score_threshold,
                   color=STYLE['colors']['confidence_band'],
                   alpha=STYLE['anomaly_band_alpha'],
                   label=f'±{z_score_threshold:.1f} Std Dev')
        ax.legend(loc='upper right', fontsize=8)

    # Use the centralized formatting check
    is_percent_fmt = _apply_yaxis_percentage_formatting(
        ax=ax, df=df, y_label=y_label, 
        value_col=value_col, 
        metric_name=metric_name, 
        metric_natural_name=actual_metric_name_display,
        force_zero_base=True,
        **kwargs
    )
    
    # Adjust bars lower to make room for annotations
    _adjust_bars_for_annotations(ax)

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
        
        ax.text(i, val, label_text, ha='center', va='bottom', color=STYLE['colors']['text'], fontsize=8)

    # Set title and labels
    ax.set_title(title or f"Metric: {actual_metric_name_display}", fontsize=11)
    ax.set_ylabel(y_label or actual_metric_name_display.replace('_',' ').title(), fontsize=10)

    # Add Anomaly Detected label if ANY anomaly exists in enrichment data
    if any(ed.get('is_anomaly', False) for ed in enrichment_data.values()):
        ax.text(0.015, 0.97, "Anomaly Detected in KPI", ha='left', va='top',
                transform=ax.transAxes, fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.3', fc=STYLE['colors']['highlight_text'], alpha=1))

    # Apply formatting
    _apply_yaxis_percentage_formatting(
        ax=ax, df=df, y_label=y_label, 
        value_col=value_col, 
        metric_name=metric_name, 
        metric_natural_name=actual_metric_name_display,
        force_zero_base=True,
        **kwargs
    )

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
) -> None: # Changed return type to None
    """Plot hypothesis values, highlighting the explaining region and showing score."""
    setup_style()
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
    bar_colors = [STYLE['colors']['default_bar']] * len(regions)
    explaining_idx = regions.index(explaining_region) if explaining_region and explaining_region in regions else -1
    if explaining_idx != -1:
        bar_colors[explaining_idx] = STYLE['colors']['highlight']

    # Plot bars
    bars = ax.bar(x_positions, values, color=bar_colors)

    # Set x-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regions, rotation=0, ha='center')
    plt.setp(ax.get_xticklabels(), fontsize=9)

    # Add reference line
    if not np.isnan(ref_val):
        ax.axhline(ref_val, color=STYLE['colors']['global_line'], linestyle='--', linewidth=1)

    # Use centralized formatting function
    is_percent_fmt = _apply_yaxis_percentage_formatting(
        ax=ax, df=df, y_label=y_label, 
        value_col=value_col, 
        hypothesis_name=hypothesis_name,
        hypothesis_natural_name=hypothesis_natural_name,
        force_zero_base=True,
        **kwargs
    )

    # Adjust bars to make room for annotations
    _adjust_bars_for_annotations(ax)

    # Add value labels based on formatting result
    value_format = '{:.1f}%' if is_percent_fmt else '{:.1f}'
    for i, val in enumerate(values):
        display_val = val * 100 if is_percent_fmt else val
        ax.text(i, val, value_format.format(display_val), ha='center', va='bottom',
                color=STYLE['colors']['text'], fontsize=8)

    # Add score components (Aligned Top-Right for compact, Top-Left for full)
    score_y_pos = 0.9 
    if score_components:
        is_root_cause = selected
        if is_root_cause:
            # Full equation - Pass the dictionary containing the score
            _add_score_components(ax, 0.02, score_y_pos - 0.08, score_components,
                                  show_score=True, show_text=False, align_right=False) 
        else:
            # Compact boxes - Pass the dictionary containing the score
            _add_score_components_compact(ax, 0, score_y_pos, score_components, align_right=True)

    # Add [selected] / [not selected] marker
    marker_y_pos = 1.02 # Position slightly above axis
    if selected:
         # Position selected marker Top-Left
         ax.text(0.015, marker_y_pos, "[selected]", ha='left', va='bottom', transform=ax.transAxes, fontsize=9, color='white',
                 bbox=dict(boxstyle='round,pad=0.3', fc=STYLE['colors']['highlight_text'], alpha=1))
    else:
         # Position not selected marker Top-Right
         marker_x_pos = 0.99 
         ax.text(marker_x_pos, marker_y_pos, "[not selected]", ha='right', va='bottom', transform=ax.transAxes, fontsize=9, color='black',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', alpha=1))

    # Set title and labels using natural name
    plot_title = title or f"Hypothesis: {hypothesis_natural_name or hypothesis_name}"
    # Add delta info to title if explaining region exists
    if explaining_region and explaining_region in df.index:
        # Check for both old and new key formats (for backward compatibility)
        if score_components:
            # Only add delta information if it's not already in the title
            if not (title and ("than Global" in title or "(" in title)):
                # Try new format first
                delta_fmt = score_components.get('delta_fmt', score_components.get('hypo_delta_fmt', ''))
                direction = score_components.get('dir', score_components.get('hypo_dir', ''))
                plot_title += f" ({explaining_region} is {delta_fmt} {direction} than Global)"
    
    ax.set_title(plot_title, fontsize=11)
    ax.set_ylabel(y_label or value_col.replace('_', ' ').title(), fontsize=10)

    # Apply formatting
    _apply_yaxis_percentage_formatting(
        ax=ax, df=df, y_label=y_label, 
        value_col=value_col, 
        hypothesis_name=hypothesis_name,
        hypothesis_natural_name=hypothesis_natural_name,
        force_zero_base=True,
        **kwargs
    )

    # Adjust bars to make room for annotations
    _adjust_bars_for_annotations(ax)

def plot_summary_report(analysis_summary_item: Dict, data_registry: DataRegistry, output_dir: str, report_format: str = "detailed") -> Optional[str]:
    """Generates a detailed or succinct summary plot for a single metric.

    Args:
        analysis_summary_item: The consolidated summary dict for ONE metric.
        data_registry: The DataRegistry instance to fetch dataframes.
        output_dir: Directory to save the plot.
        report_format: "detailed" or "succinct".
       
    Returns:
        Filepath of the saved plot, or None if generation failed.
    """
    setup_style() # Ensure consistent style
    metric = analysis_summary_item['metric_name']
    logger.info(f"Generating '{report_format}' summary plot for: {metric}")

    # --- Extract Data --- 
    metric_natural_name = analysis_summary_item['metric_natural_name']
    primary_region = analysis_summary_item['primary_region']
    explanation_text = analysis_summary_item['explanation_text']
    best_hypo_result = analysis_summary_item['best_hypothesis_result']
    all_hypo_results = analysis_summary_item['all_hypotheses_for_metric']
    metric_data_key = analysis_summary_item['metric_data_key']
    metric_df = data_registry.get(metric_data_key)
    
    if metric_df is None:
         logger.error(f"Could not retrieve enriched metric data for summary plot: {metric}")
         return None

    # Separate best hypo from others
    best_hypo_name = best_hypo_result.name if best_hypo_result else None
    other_hypo_results = sorted([h for h in all_hypo_results if h.name != best_hypo_name and h.score is not None],
                              key=lambda x: x.score, reverse=True) # Sort others by score
    num_other_hypos = len(other_hypo_results)
    
    # --- Setup Gridspec Layout --- 
    # TODO: Implement layout logic for "succinct" format
    if report_format == "detailed":
         # Determine grid size based on number of hypotheses
         num_total_hypos = len(all_hypo_results)
         has_root_cause = best_hypo_result is not None

         if not has_root_cause:
             # Handle case with anomaly but no root cause - maybe just metric + top N hypos?
             # For now, let's just plot the metric if no root cause found in detailed view
             logger.info(f"No root cause found for {metric}, generating metric-only plot for detailed report.")
             fig, ax = plt.subplots(figsize=(8, 6))
             metric_kwargs = analysis_summary_item.get('metric_plot_ctx', {}) # Need ctx stored in summary
             # Need to recreate kwargs correctly here from summary data
             # Remove fallback calculations, rely solely on analysis_summary_item
             metric_bar_anomaly(ax, metric_df, 
                                 # Pass arguments explicitly by keyword
                                 metric_name=metric,
                                 ref_metric_val=analysis_summary_item['metric_ref_val'], 
                                 std=analysis_summary_item['metric_std'],
                                 metric_natural_name=metric_natural_name,
                                 title=f'Metric: {metric_natural_name} ({primary_region} anomaly)', 
                                 y_label=metric_natural_name,
                                 higher_is_better=analysis_summary_item.get('higher_is_better'),
                                 z_score_threshold=analysis_summary_item.get('z_score_threshold'),
                                 value_col=analysis_summary_item.get('metric_value_col', metric) # Use the actual column name
                                )
             gs = None # No complex grid
             axes_map = {'metric': ax}
        
         elif num_total_hypos <= 1:
             figsize = (8, 10); fig = plt.figure(figsize=figsize) # Increased height further
             gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.5) # Increased hspace more
             axes_map = {'metric': fig.add_subplot(gs[0, 0]), 'best_hypo': fig.add_subplot(gs[1, 0])}
         elif num_total_hypos <= 3: 
             figsize = (12, 7); fig = plt.figure(figsize=figsize) # Increased height further
             gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.5, wspace=0.15) # More hspace
             axes_map = {'metric': fig.add_subplot(gs[0, 0]), 'best_hypo': fig.add_subplot(gs[1, 0]),
                         'other_hypo_0': fig.add_subplot(gs[0, 1]), 'other_hypo_1': fig.add_subplot(gs[1, 1])}
         else: 
             figsize = (15, 7); fig = plt.figure(figsize=figsize) # Increased height further
             num_other_cols = 2
             num_other_rows = (num_other_hypos + num_other_cols - 1) // num_other_cols
             heights = [1] * max(2, num_other_rows) 
             gs = gridspec.GridSpec(max(2, num_other_rows), 1 + num_other_cols, 
                                   width_ratios=[2] + [1]*num_other_cols, 
                                   height_ratios=heights, hspace=0.5, wspace=0.15) # More hspace
             axes_map = {'metric': fig.add_subplot(gs[0, 0]), 'best_hypo': fig.add_subplot(gs[1, 0])}
             for i in range(num_other_hypos):
                 r = i // num_other_cols
                 c = (i % num_other_cols) + 1
                 if r < gs.nrows and c < gs.ncols: 
                     axes_map[f'other_hypo_{i}'] = fig.add_subplot(gs[r, c])
    else: # Handle succinct or unknown format
        logger.warning(f"Summary plot format '{report_format}' not implemented. Skipping plot for {metric}.")
        return None

    # --- Plot Metric --- 
    ax_metric = axes_map.get('metric')
    if ax_metric:
        metric_enrichment_data = analysis_summary_item.get('metric_enrichment_data', {})
        # Use the correct metric column - THIS IS THE KEY FIX - use the metric name directly
        metric_value_col = metric  # Use the actual metric name directly
        
        # IMPORTANT: Log the metric DataFrame columns for debugging
        original_metric_df_key = analysis_summary_item.get('metric_data_key')
        original_metric_df = data_registry.get(original_metric_df_key) if original_metric_df_key else None
        if original_metric_df is not None:
            logger.info(f"Metric DataFrame columns for {metric}: {original_metric_df.columns.tolist()}")
        
        metric_kwargs_for_plot = {
            'ref_metric_val': analysis_summary_item.get('metric_ref_val'),
            'std': analysis_summary_item.get('metric_std'),
            'metric_name': metric,
            'value_col': metric_value_col, # Use the metric name directly as column
            'metric_natural_name': metric_natural_name,
            'title': f"Metric: {metric_natural_name} ({primary_region} anomaly)", 
            'y_label': metric_natural_name,
            'higher_is_better': analysis_summary_item.get('higher_is_better'),
            'z_score_threshold': analysis_summary_item.get('z_score_threshold'),
            'enrichment_data': metric_enrichment_data 
        }
        metric_kwargs_for_plot = {k:v for k,v in metric_kwargs_for_plot.items() if v is not None}

        if original_metric_df is None:
             logger.error(f"Missing original metric data in registry for key {original_metric_df_key}")
             ax_metric.text(0.5, 0.5, "Error: Metric data missing", ha='center', va='center')
        elif 'ref_metric_val' not in metric_kwargs_for_plot or 'std' not in metric_kwargs_for_plot:
             logger.error(f"Missing ref_metric_val or std in analysis summary for metric plot: {metric}")
             ax_metric.text(0.5, 0.5, "Error: Metric context missing", ha='center', va='center')
        elif not metric_value_col: 
             logger.error(f"Could not determine metric value column for summary plot: {metric}")
             ax_metric.text(0.5, 0.5, "Error: Metric col missing", ha='center', va='center')
        else:
             # Last verification - ensure value_col exists in the DataFrame
             if metric_value_col not in original_metric_df.columns:
                 logger.error(f"Metric column '{metric_value_col}' not in DataFrame (has {original_metric_df.columns.tolist()}). Using '{metric}' instead.")
                 # Try to use the metric name itself as a fallback
                 if metric in original_metric_df.columns:
                     metric_kwargs_for_plot['value_col'] = metric
                 else:
                     logger.error(f"Neither '{metric_value_col}' nor '{metric}' found in DataFrame columns.")
                     ax_metric.text(0.5, 0.5, f"Error: Column missing\nExpected: {metric_value_col}\nAvailable: {original_metric_df.columns.tolist()}", 
                                    ha='center', va='center')
                     return None
             
             metric_bar_anomaly(ax_metric, original_metric_df, **metric_kwargs_for_plot)
             # Add annotation
             if primary_region not in ["NoAnomaly", "NoData", None]:
                  ax_metric.text(0.015, 0.97, "Anomaly Detected in KPI", ha='left', va='top', 
                           transform=ax_metric.transAxes, fontsize=9, color='white', 
                           bbox=dict(boxstyle='round,pad=0.3', fc=STYLE['colors']['highlight_text'], alpha=1))

    # --- Plot Best Hypothesis --- 
    ax_best_hypo = axes_map.get('best_hypo')
    if ax_best_hypo and best_hypo_result:
        hypo_name = best_hypo_result.name
        hypo_config = analysis_summary_item.get('hypotheses_configs', {}).get(hypo_name, {})
        # Retrieve plot_data_df directly from HypoResult object
        hypo_df = best_hypo_result.plot_data 
        
        if hypo_df is not None:
             score_components_data = best_hypo_result.key_numbers.copy() # Copy key numbers
             score_components_data['score'] = best_hypo_result.score # Add score
             
             # Get value_col correctly
             value_col = hypo_config.get('input_data', [{}])[0].get('columns', [None])[0]
             
             # Get the hypo delta values from context but ensure they're numeric
             hypo_dir_primary = analysis_summary_item.get('hypo_dir_primary', '?')
             hypo_delta_fmt = analysis_summary_item.get('hypo_delta_fmt', '')
             
             # Use the better formatted values directly from the HypoResult object
             if best_hypo_result and best_hypo_result.key_numbers:
                 delta_fmt = best_hypo_result.key_numbers.get('delta_fmt', hypo_delta_fmt)
                 direction = best_hypo_result.key_numbers.get('dir', hypo_dir_primary)
             else:
                 delta_fmt = hypo_delta_fmt
                 direction = hypo_dir_primary
             
             # Use the pre-formatted value directly - no need for special handling
             delta_text = f"({primary_region} is {delta_fmt} {direction} than Global)"
             
             hypo_kwargs = {
                 'region_col': 'region',
                 'value_col': value_col,
                 'hypothesis_name': hypo_name,
                 'hypothesis_natural_name': analysis_summary_item.get('hypo_natural_name'),
                 'explaining_region': primary_region,
                 'primary_region': primary_region,
                 'score_components': score_components_data, # Pass dict including score
                 'selected': True,
                 # Construct title with delta info for best hypo
                 'title': f"Root Cause: Hypothesis {best_hypo_result.display_rank+1}\n{analysis_summary_item.get('hypo_natural_name')} {delta_text}",
                 'y_label': analysis_summary_item.get('hypo_natural_name')
             }
             hypo_kwargs = {k:v for k,v in hypo_kwargs.items() if v is not None}
             hypo_bar_scored(ax_best_hypo, hypo_df, **hypo_kwargs)
        else:
             logger.warning(f"Could not get data for best hypothesis plot: {hypo_name}")

    # --- Plot Other Hypotheses --- 
    for i, other_hypo in enumerate(other_hypo_results):
        ax_other = axes_map.get(f'other_hypo_{i}')
        if not ax_other: continue
        
        hypo_name = other_hypo.name
        hypo_config = analysis_summary_item.get('hypotheses_configs', {}).get(hypo_name, {})
        # Retrieve plot_data_df directly from HypoResult object
        hypo_df = other_hypo.plot_data
        hypo_natural_name = hypo_config.get('natural_name', hypo_name)

        if hypo_df is not None:
            score_components_data = other_hypo.key_numbers.copy()
            score_components_data['score'] = other_hypo.score # Add score
            hypo_kwargs = {
                 'region_col': 'region',
                 'value_col': hypo_config.get('input_data', [{}])[0].get('columns', [None])[0],
                 'hypothesis_name': hypo_name,
                 'hypothesis_natural_name': hypo_natural_name,
                 'score_components': score_components_data, # Pass dict including score
                 'selected': False,
                 'title': f"Hypothesis {other_hypo.display_rank+1}\n{hypo_natural_name}",
                 'y_label': hypo_natural_name
             }
            hypo_kwargs = {k:v for k,v in hypo_kwargs.items() if v is not None}
            hypo_bar_scored(ax_other, hypo_df, **hypo_kwargs)
        else:
            logger.warning(f"Could not get data for other hypothesis plot: {hypo_name}")

    # --- Add Figure Annotations --- 
    fig_title = f"{primary_region} for {metric_natural_name} is an Anomaly" if primary_region not in ["NoAnomaly", "NoData", None] else f"{metric_natural_name} Performance"
    # Position the title even higher to make room for everything
    title_y_pos = 0.98 
    # Position explanation text much lower to avoid overlap
    explanation_y_pos = title_y_pos - 0.04 
    fig.suptitle(fig_title, fontsize=16, y=title_y_pos)
    if explanation_text:
        fig.text(0.05, explanation_y_pos, explanation_text, ha='left', va='top', fontsize=11,
                 style='italic', wrap=True, bbox=dict(boxstyle='round,pad=0.4', fc='white', lw=0.2)) 
    
    create_score_formula_on_fig(fig, y_pos=0.001) # Lowered from 0.01 to 0.005

    # --- Save --- 
    filename = f"{metric}_{primary_region}_{report_format}_summary.png"
    filepath = os.path.join(output_dir, filename)
    try:
        # Adjust figure layout - significantly more space at top
        fig.subplots_adjust(left=0.06, right=0.96, bottom=0.06, top=0.8, hspace=0.5, wspace=0.15)
        fig.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info(f"Saved summary plot: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving summary plot {filepath}: {e}")
        plt.close(fig)
        return None

def create_score_formula_on_fig(fig, y_pos = 0.005):
    """Create a colored score formula at the bottom of the figure."""
    # Components dictionary with dummy values (0.0) just for structure/names
    components_dict = {comp: 0.0 for comp in STYLE['score_component_order']}
    score_value = 0.0 
    
    # Use a figure-specific version of the scorecard helper if needed,
    # or adapt the axis one carefully. Let's try adapting _add_score_components.
    # We need to simulate the calculation for positioning correctly.
    
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
    score_label_color = STYLE['colors'].get('score_color', '#AF7AC5')

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

ROUTER = {
    "metric_bar_anomaly": metric_bar_anomaly,
    "hypo_bar_scored": hypo_bar_scored,
    "summary_report": plot_summary_report, # Add new summary plot function
} 