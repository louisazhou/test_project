"""
Detailed report plots module.

This module implements the detailed summary report visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List
import logging

from ...core.data_registry import DataRegistry
from ...plotting.metric_plots.anomaly_plots import metric_bar_anomaly
from ...plotting.hypothesis_plots.single_dim import hypo_bar_scored
from ...plotting import plot_router  # For style constants and helper functions
from ...core.types import HypoResult, MetricFormatting

logger = logging.getLogger(__name__)


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
    plot_router.setup_style() # Ensure consistent style
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
                                 z_score_threshold=analysis_summary_item.get('z_score_threshold'),
                                 value_col=analysis_summary_item.get('metric_value_col', metric), # Use the actual column name
                                 higher_is_better=analysis_summary_item.get('higher_is_better', False),
                                 enrichment_data=analysis_summary_item.get('metric_enrichment_data', {})
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
            'z_score_threshold': analysis_summary_item.get('z_score_threshold'),
            'higher_is_better': analysis_summary_item.get('higher_is_better', False),
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
                 
             # Create the metric plot
             # Pass the axis directly to the plotting function
             metric_bar_anomaly(ax=ax_metric, df=original_metric_df, **metric_kwargs_for_plot)
             # Add annotation directly to the axis
             if primary_region not in ["NoAnomaly", "NoData", None]:
                  ax_metric.text(0.015, 0.97, "Anomaly Detected in KPI", ha='left', va='top', 
                           transform=ax_metric.transAxes, fontsize=9, color='white', 
                           bbox=dict(boxstyle='round,pad=0.3', fc=plot_router.STYLE['colors']['highlight_text'], alpha=1))

    # --- Plot Best Hypothesis --- 
    ax_best_hypo = axes_map.get('best_hypo')
    if ax_best_hypo and best_hypo_result:
        hypo_name = best_hypo_result.name
        hypo_config = analysis_summary_item.get('hypotheses_configs', {}).get(hypo_name, {})
        # Retrieve plot_data_df directly from HypoResult object
        hypo_df = best_hypo_result.plot_data 
        
        if hypo_df is not None:
             # Create score components dictionary from HypoResult fields
             score_components_data = {
                 'score': best_hypo_result.score,
                 'direction_alignment': analysis_summary_item.get('score_components', {}).get(hypo_name, {}).get('direction_alignment', 0.0),
                 'consistency': analysis_summary_item.get('score_components', {}).get(hypo_name, {}).get('consistency', 0.0),
                 'hypo_z_score_norm': analysis_summary_item.get('score_components', {}).get(hypo_name, {}).get('hypo_z_score_norm', 0.0),
                 'explained_ratio': analysis_summary_item.get('score_components', {}).get(hypo_name, {}).get('explained_ratio', 0.0)
             }
             
             # Get value_col correctly
             value_col = hypo_config.get('input_data', [{}])[0].get('columns', [None])[0]
             if not value_col:
                 # Try to determine value_col from the plot_data columns
                 if hypo_df is not None and len(hypo_df.columns) > 0:
                     # Use first column that's not 'region' or index
                     potential_cols = [col for col in hypo_df.columns if col != 'region']
                     if potential_cols:
                         value_col = potential_cols[0]
                         logger.info(f"Determined value_col '{value_col}' from plot_data columns for hypothesis: {hypo_name}")
                     else:
                         logger.error(f"Could not determine value_col for hypothesis plot: {hypo_name}")
                         return None
                 else:
                     logger.error(f"Could not determine value_col for hypothesis plot: {hypo_name}")
                     return None
             else:
                 logger.info(f"Determined value_col '{value_col}' from hypotheses_configs for hypothesis: {hypo_name}")
             
             # Format the deviation description using the central utility
             delta_fmt = MetricFormatting.format_delta(
                 best_hypo_result.value, 
                 best_hypo_result.global_value, 
                 best_hypo_result.is_percentage
             )
             
             direction = "higher" if best_hypo_result.value > best_hypo_result.global_value else "lower"
             
             # Use the formatted values for the delta text
             delta_text = f"({primary_region} is {delta_fmt} {direction} than Global)"
             
             hypo_kwargs = {
                 'region_col': 'region',
                 'value_col': value_col,
                 'hypothesis_name': hypo_name,
                 'hypothesis_natural_name': best_hypo_result.natural_name,
                 'explaining_region': primary_region,
                 'primary_region': primary_region,
                 'score_components': score_components_data,
                 'selected': True,
                 # Construct title with delta info for best hypo
                 'title': f"Root Cause: Hypothesis {best_hypo_result.display_rank+1}\n{best_hypo_result.natural_name} {delta_text}",
                 'y_label': best_hypo_result.natural_name
             }
             hypo_kwargs = {k:v for k,v in hypo_kwargs.items() if v is not None}
             
             # Create the hypothesis plot - pass the axis directly
             hypo_bar_scored(ax=ax_best_hypo, df=hypo_df, **hypo_kwargs)
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
        hypo_natural_name = other_hypo.natural_name or hypo_config.get('natural_name', hypo_name)

        if hypo_df is not None:
            # Create score components dictionary from HypoResult fields
            score_components_data = {
                'score': other_hypo.score,
                'direction_alignment': analysis_summary_item.get('score_components', {}).get(hypo_name, {}).get('direction_alignment', 0.0),
                'consistency': analysis_summary_item.get('score_components', {}).get(hypo_name, {}).get('consistency', 0.0),
                'hypo_z_score_norm': analysis_summary_item.get('score_components', {}).get(hypo_name, {}).get('hypo_z_score_norm', 0.0),
                'explained_ratio': analysis_summary_item.get('score_components', {}).get(hypo_name, {}).get('explained_ratio', 0.0)
            }
            
            # Get value_col correctly
            value_col = hypo_config.get('input_data', [{}])[0].get('columns', [None])[0]
            if not value_col:
                # Try to determine value_col from the plot_data columns
                if hypo_df is not None and len(hypo_df.columns) > 0:
                    # Use first column that's not 'region' or index
                    potential_cols = [col for col in hypo_df.columns if col != 'region']
                    if potential_cols:
                        value_col = potential_cols[0]
                        logger.info(f"Determined value_col '{value_col}' from plot_data columns for hypothesis: {hypo_name}")
                    else:
                        logger.error(f"Could not determine value_col for hypothesis plot: {hypo_name}")
                        return None
                else:
                    logger.error(f"Could not determine value_col for hypothesis plot: {hypo_name}")
                    return None
            else:
                logger.info(f"Determined value_col '{value_col}' from hypotheses_configs for hypothesis: {hypo_name}")
            
            hypo_kwargs = {
                 'region_col': 'region',
                 'value_col': value_col,
                 'hypothesis_name': hypo_name,
                 'hypothesis_natural_name': hypo_natural_name,
                 'score_components': score_components_data,
                 'selected': False,
                 'title': f"Hypothesis {other_hypo.display_rank+1}\n{hypo_natural_name}",
                 'y_label': hypo_natural_name
             }
            hypo_kwargs = {k:v for k,v in hypo_kwargs.items() if v is not None}
            
            # Create the hypothesis plot - pass the axis directly
            hypo_bar_scored(ax=ax_other, df=hypo_df, **hypo_kwargs)
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
    
    plot_router.create_score_formula_on_fig(fig, y_pos=0.001) 

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