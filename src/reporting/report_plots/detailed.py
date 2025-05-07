"""
Detailed report plots module.

This module implements the detailed summary report visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List, Callable
import logging

from ...core.data_registry import DataRegistry
from ...core.types import MetricFormatting, HypoResult, PlotSpec
from ...plotting import plot_router  # For style constants and helper functions
from ...plotting.metric_plots.anomaly_plots import metric_bar_anomaly
from ...registry import get_plotter, register_plotter  # Import from central registry

logger = logging.getLogger(__name__)

def plot_summary_report_by_type(
    analysis_summary_item: Dict, 
    data_registry: DataRegistry, 
    output_dir: str, 
    summary_view_type: str = "ranked",
    report_format: str = "detailed"
) -> Optional[str]:
    """Generates a detailed summary plot for a single metric with specific view type.
    
    This function serves as a wrapper around plot_summary_report that filters 
    hypotheses based on their type and generates the appropriate view.
    
    Args:
        analysis_summary_item: The consolidated summary dict for ONE metric.
        data_registry: The DataRegistry instance to fetch dataframes.
        output_dir: Directory to save the plot.
        summary_view_type: "ranked" (for single_dim hypotheses) or "descriptive" (for other types)
        report_format: "detailed" or "succinct".
        
    Returns:
        Filepath of the saved plot, or None if generation failed.
    """
    metric = analysis_summary_item['metric_name']
    primary_region = analysis_summary_item['primary_region']
    best_hypo_result = analysis_summary_item['best_hypothesis_result']
    all_hypo_results = analysis_summary_item['all_hypotheses_for_metric']
    
    # Create a copy of the summary item to avoid modifying the original
    filtered_summary = analysis_summary_item.copy()
    
    # Deduplicate hypotheses by name to prevent duplicates
    unique_hypos = {}
    for h in all_hypo_results:
        if h.name not in unique_hypos:
            unique_hypos[h.name] = h
    
    all_hypo_results = list(unique_hypos.values())
    
    # Filter hypotheses based on view type
    if summary_view_type == "ranked":
        # For ranked view, only include single_dim hypotheses
        ranked_hypos = [h for h in all_hypo_results if getattr(h, 'type', '') == 'single_dim']
        
        # Also filter by primary_region to ensure we're only showing relevant hypotheses
        # Use a more flexible approach to check for region presence
        filtered_ranked_hypos = []
        for h in ranked_hypos:
            if not hasattr(h, 'plot_data') or h.plot_data is None:
                continue
                
            # Check various ways the region might be represented
            has_region = False
            if isinstance(h.plot_data, pd.DataFrame):
                if 'region' in h.plot_data.columns and primary_region in h.plot_data['region'].values:
                    has_region = True
                elif primary_region in getattr(h.plot_data, 'index', []):
                    has_region = True
                    
            if has_region:
                filtered_ranked_hypos.append(h)
        
        ranked_hypos = filtered_ranked_hypos
        
        if not ranked_hypos:
            logger.info(f"No single_dim hypotheses found for {metric} in {primary_region}, skipping ranked view.")
            return None
            
        # If best hypothesis is not single_dim, find the best among single_dim
        if best_hypo_result and getattr(best_hypo_result, 'type', '') != 'single_dim':
            ranked_hypos_sorted = sorted(ranked_hypos, key=lambda x: x.score if x.score is not None else -1, reverse=True)
            best_ranked_hypo = ranked_hypos_sorted[0] if ranked_hypos_sorted else None
            filtered_summary['best_hypothesis_result'] = best_ranked_hypo
        
        filtered_summary['all_hypotheses_for_metric'] = ranked_hypos
        
    elif summary_view_type == "descriptive":
        # For descriptive view, keep non-single_dim hypotheses
        descriptive_hypos = [h for h in all_hypo_results if getattr(h, 'type', '') != 'single_dim']
        
        # For descriptive hypotheses, we need special handling since they might have different structures
        filtered_desc_hypos = []
        
        for h in descriptive_hypos:
            # For specific descriptive hypotheses like closed_lost_reason, we don't require region filtering
            if getattr(h, 'type', '') == 'closed_lost_reason':
                filtered_desc_hypos.append(h)
                continue
                
            # For other descriptive hypotheses, check for region if possible
            if hasattr(h, 'plot_data') and h.plot_data is not None:
                if isinstance(h.plot_data, pd.DataFrame):
                    # Check if region is in columns or if there's any data
                    if ('region' not in h.plot_data.columns) or (h.plot_data.empty):
                        # If no region column, include it (it might be specialized data)
                        filtered_desc_hypos.append(h)
                    elif primary_region in h.plot_data['region'].values:
                        filtered_desc_hypos.append(h)
            else:
                # If no plot_data, include it anyway (can't filter)
                filtered_desc_hypos.append(h)
        
        if not filtered_desc_hypos:
            logger.info(f"No descriptive hypotheses found for {metric} in {primary_region}, skipping descriptive view.")
            return None
            
        # For descriptive hypotheses, use their descriptive_score if available
        for h in filtered_desc_hypos:
            if hasattr(h, 'descriptive_score') and h.descriptive_score is not None:
                logger.info(f"Using descriptive_score for hypothesis {h.name}: {h.descriptive_score}")
            
        filtered_summary['all_hypotheses_for_metric'] = filtered_desc_hypos
    
    # Modify the filename to include the view type
    filename_prefix = f"{metric}_{primary_region}"
    view_specific_filename = f"{filename_prefix}_{summary_view_type}_{report_format}_summary.png"
    output_path = os.path.join(output_dir, view_specific_filename)
    
    # Call the original function with filtered data and direct filename
    result = plot_summary_report(filtered_summary, data_registry, output_dir, report_format, output_filename=view_specific_filename)
    
    return result

def plot_summary_report(
    analysis_summary_item: Dict, 
    data_registry: DataRegistry, 
    output_dir: str, 
    report_format: str = "detailed",
    output_filename: Optional[str] = None
) -> Optional[str]:
    """Generates a detailed or succinct summary plot for a single metric.
    
    Args:
        analysis_summary_item: The consolidated summary dict for ONE metric.
        data_registry: The DataRegistry instance to fetch dataframes.
        output_dir: Directory to save the plot.
        report_format: "detailed" or "succinct".
        output_filename: Optional custom filename for the output plot.
        
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
    
    # Check if this is a descriptive summary type
    is_descriptive = False
    if best_hypo_result and hasattr(best_hypo_result, 'type'):
        is_descriptive = best_hypo_result.type != 'single_dim'
    
    if metric_df is None:
         logger.error(f"Could not retrieve enriched metric data for summary plot: {metric}")
         return None

    # Separate best hypo from others
    best_hypo_name = best_hypo_result.name if best_hypo_result else None
    
    # Deduplicate other hypotheses by name
    other_hypos_dict = {}
    for h in all_hypo_results:
        if h.name != best_hypo_name and h.score is not None:
            other_hypos_dict[h.name] = h
    
    other_hypo_results = sorted(list(other_hypos_dict.values()), key=lambda x: x.score, reverse=True)
    num_other_hypos = len(other_hypo_results)
    
    # --- Setup Gridspec Layout --- 
    if report_format == "detailed":
         # Determine grid size based on number of hypotheses
         num_total_hypos = len(all_hypo_results)
         has_root_cause = best_hypo_result is not None

         if not has_root_cause:
             # Handle case with anomaly but no root cause - maybe just metric + top N hypos?
             # For now, let's just plot the metric if no root cause found in detailed view
             logger.info(f"No root cause found for {metric}, generating metric-only plot for detailed report.")
             fig, ax = plt.subplots(figsize=(8, 6))
             metric_kwargs = analysis_summary_item.get('metric_plot_ctx', {})
             metric_bar_anomaly(ax, metric_df, 
                                 metric_name=metric,
                                 ref_metric_val=analysis_summary_item['metric_ref_val'], 
                                 std=analysis_summary_item['metric_std'],
                                 metric_natural_name=metric_natural_name,
                                 title=f'Metric: {metric_natural_name} ({primary_region} anomaly)', 
                                 y_label=metric_natural_name,
                                 z_score_threshold=analysis_summary_item.get('z_score_threshold'),
                                 value_col=analysis_summary_item.get('metric_value_col', metric),
                                 higher_is_better=analysis_summary_item.get('higher_is_better', False),
                                 enrichment_data=analysis_summary_item.get('metric_enrichment_data', {})
                                )
             gs = None # No complex grid
             axes_map = {'metric': ax}
        
         elif num_total_hypos <= 1:
             figsize = (8, 10); fig = plt.figure(figsize=figsize)
             gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.5)
             axes_map = {'metric': fig.add_subplot(gs[0, 0]), 'best_hypo': fig.add_subplot(gs[1, 0])}
         elif num_total_hypos <= 3: 
             figsize = (12, 7); fig = plt.figure(figsize=figsize)
             gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.5, wspace=0.15)
             axes_map = {'metric': fig.add_subplot(gs[0, 0]), 'best_hypo': fig.add_subplot(gs[1, 0]),
                         'other_hypo_0': fig.add_subplot(gs[0, 1]), 'other_hypo_1': fig.add_subplot(gs[1, 1])}
         else:
             figsize = (15, 7); fig = plt.figure(figsize=figsize)
             num_other_cols = 2
             # Limit number of other hypotheses to avoid creating too many subplots
             max_other_hypos = min(4, num_other_hypos)
             num_other_rows = (max_other_hypos + num_other_cols - 1) // num_other_cols
             heights = [1] * max(2, num_other_rows) 
             gs = gridspec.GridSpec(max(2, num_other_rows), 1 + num_other_cols, 
                                   width_ratios=[2] + [1]*num_other_cols, 
                                   height_ratios=heights, hspace=0.5, wspace=0.15)
             axes_map = {'metric': fig.add_subplot(gs[0, 0]), 'best_hypo': fig.add_subplot(gs[1, 0])}
             for i in range(max_other_hypos):
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
        metric_value_col = metric  # Use the actual metric name directly
        
        # Get metric data
        original_metric_df = data_registry.get(metric_data_key)
        
        # Log columns for debugging
        logger.debug(f"Metric DataFrame for {metric} has {len(original_metric_df.columns)} columns")
        
        # First clean the metric data: need to ensure we have 'region' and metric columns
        metric_df = original_metric_df.copy()
        
        metric_kwargs_for_plot = {
            'ref_metric_val': analysis_summary_item.get('metric_ref_val'),
            'std': analysis_summary_item.get('metric_std'),
            'metric_name': metric,
            'value_col': metric_value_col,
            'metric_natural_name': metric_natural_name,
            'title': f"Metric: {metric_natural_name} ({primary_region} anomaly)", 
            'y_label': metric_natural_name,
            'z_score_threshold': analysis_summary_item.get('z_score_threshold'),
            'higher_is_better': analysis_summary_item.get('higher_is_better', False),
            'enrichment_data': metric_enrichment_data 
        }
        metric_kwargs_for_plot = {k:v for k,v in metric_kwargs_for_plot.items() if v is not None}

        if original_metric_df is None:
             logger.error(f"Missing original metric data in registry for key {metric_data_key}")
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
        hypo_type = getattr(best_hypo_result, 'type', 'single_dim')  # Default to single_dim
        hypo_config = analysis_summary_item.get('hypotheses_configs', {}).get(hypo_name, {})
        
        # Get the appropriate plotter from the registry
        plotter = get_plotter(hypo_type)
        if plotter:
            # Get score components
            score_components = analysis_summary_item.get('score_components', {}).get(hypo_name, {})
            
            # Call the plotter with the appropriate parameters
            try:
                plotter(
                    ax=ax_best_hypo,
                    hypo_result=best_hypo_result,
                    hypo_config=hypo_config,
                    primary_region=primary_region,
                    title=f"Root Cause: {best_hypo_result.natural_name}",
                    selected=True,
                    include_delta=True,
                    direction_alignment=score_components.get('direction_alignment', 0.0),
                    consistency=score_components.get('consistency', 0.0),
                    hypo_z_score_norm=score_components.get('hypo_z_score_norm', 0.0),
                    explained_ratio=score_components.get('explained_ratio', 0.0)
                )
            except Exception as e:
                logger.error(f"Error plotting hypothesis {hypo_name}: {e}")
                ax_best_hypo.text(0.5, 0.5, f"Error plotting {hypo_name}: {str(e)}", 
                                 ha='center', va='center')
        else:
            logger.warning(f"No plotter registered for hypothesis type: {hypo_type}")
            ax_best_hypo.text(0.5, 0.5, f"No plotter available for {hypo_type}", ha='center', va='center')

    # --- Plot Other Hypotheses --- 
    for i, other_hypo in enumerate(other_hypo_results[:4]):  # Limit to max 4 other hypotheses
        ax_other = axes_map.get(f'other_hypo_{i}')
        if not ax_other: continue
        
        hypo_name = other_hypo.name
        hypo_type = getattr(other_hypo, 'type', 'single_dim')  # Default to single_dim
        hypo_config = analysis_summary_item.get('hypotheses_configs', {}).get(hypo_name, {})
        
        # Get the appropriate plotter from the registry
        plotter = get_plotter(hypo_type)
        if plotter:
            # Get score components
            score_components = analysis_summary_item.get('score_components', {}).get(hypo_name, {})
            
            # Call the plotter with the appropriate parameters
            try:
                plotter(
                    ax=ax_other,
                    hypo_result=other_hypo,
                    hypo_config=hypo_config,
                    primary_region=primary_region,
                    title=f"Hypothesis {other_hypo.display_rank+1 if other_hypo.display_rank is not None else ''}\n{other_hypo.natural_name}",
                    selected=False,
                    direction_alignment=score_components.get('direction_alignment', 0.0),
                    consistency=score_components.get('consistency', 0.0),
                    hypo_z_score_norm=score_components.get('hypo_z_score_norm', 0.0),
                    explained_ratio=score_components.get('explained_ratio', 0.0)
                )
            except Exception as e:
                logger.error(f"Error plotting hypothesis {hypo_name}: {e}")
                ax_other.text(0.5, 0.5, f"Error plotting {hypo_name}: {str(e)}", 
                             ha='center', va='center')
        else:
            logger.warning(f"No plotter registered for hypothesis type: {hypo_type}")
            ax_other.text(0.5, 0.5, f"No plotter available for {hypo_type}", ha='center', va='center')

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
    
    # Only add score formula for ranked summaries, not descriptive ones
    if not is_descriptive:
        plot_router.create_score_formula_on_fig(fig, y_pos=0.001)

    # --- Save --- 
    if output_filename:
        filename = output_filename
    else:
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

def plot_summary_report_registry(anomaly_data: Dict[str, Any], region: str, hypothesis_results: List[HypoResult]) -> pd.DataFrame:
    """Generate a summary view plot showing the metric and top hypothesis explanations.
    
    Args:
        anomaly_data: Dictionary containing anomaly data including date, value, etc.
        region: The region being analyzed
        hypothesis_results: List of hypothesis results, sorted by score
        
    Returns:
        A DataFrame containing plot metadata
    """
    logger.info(f"Plotting summary report for metric {anomaly_data.get('metric_name', 'UNKNOWN')} in region {region}")
    
    # Create the figure and define the grid layout
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    metric_name = anomaly_data.get('metric_name', 'Unknown Metric')
    
    # Plot the metric bar with anomaly
    ax_metric = fig.add_subplot(gs[0, :])
    from ...plotting.hypothesis_plots.metric_bar_anomaly import plot_metric_bar_anomaly
    
    # Create metric context for plotting
    metric_plot_context = {
        "region": region,
        "metric_name": metric_name,
        "date": anomaly_data.get('date'),
        "value": anomaly_data.get('value'),
        "anomaly_value": anomaly_data.get('anomaly_value', anomaly_data.get('value')),
        "pct_delta": anomaly_data.get('pct_delta'),
        "higher_is_better": anomaly_data.get('higher_is_better', True)
    }
    
    # Extract historic data if available
    historic_data = None
    if 'historic_data' in anomaly_data:
        historic_data = pd.DataFrame(anomaly_data['historic_data'])
    
    plot_metric_bar_anomaly(
        ax=ax_metric,
        df=historic_data,
        **metric_plot_context
    )
    
    # Create axis for each hypothesis plot
    ax_top = fig.add_subplot(gs[1, 0])
    ax_runner_up = fig.add_subplot(gs[1, 1])
    
    # Plot the top hypothesis if available
    if hypothesis_results and len(hypothesis_results) > 0:
        best_hypo = hypothesis_results[0]
        hypo_type = best_hypo.hypothesis_type
        hypo_name = best_hypo.hypothesis_name
        
        logger.info(f"Plotting top hypothesis: {hypo_name} (type: {hypo_type})")
        
        # Use the registry to get the plotter
        plotter = get_plotter(hypo_type)
        
        if plotter:
            # Use the registered plotter
            try:
                title = f"Top Hypothesis: {hypo_name}"
                if not plotter(ax=ax_top, hypo_result=best_hypo, focus_region=region, title=title):
                    ax_top.text(0.5, 0.5, f"No plot data for {hypo_name}", ha='center', va='center')
            except Exception as e:
                logger.error(f"Error plotting top hypothesis {hypo_name}: {e}")
                ax_top.text(0.5, 0.5, f"Error plotting {hypo_name}: {str(e)}", ha='center', va='center')
        else:
            ax_top.text(0.5, 0.5, f"No plotter found for hypothesis type: {hypo_type}", ha='center', va='center')
        
        ax_top.set_title(f"Top Hypothesis: {hypo_name}")
    else:
        ax_top.text(0.5, 0.5, "No hypothesis results available", ha='center', va='center')
        ax_top.set_title("Top Hypothesis")
    
    # Plot the runner-up hypothesis if available
    if hypothesis_results and len(hypothesis_results) > 1:
        runner_up = hypothesis_results[1]
        runner_up_type = runner_up.hypothesis_type
        runner_up_name = runner_up.hypothesis_name
        
        logger.info(f"Plotting runner-up hypothesis: {runner_up_name} (type: {runner_up_type})")
        
        # Use the registry to get the plotter
        plotter = get_plotter(runner_up_type)
        
        if plotter:
            # Use the registered plotter
            try:
                title = f"Runner-up: {runner_up_name}"
                if not plotter(ax=ax_runner_up, hypo_result=runner_up, focus_region=region, title=title):
                    ax_runner_up.text(0.5, 0.5, f"No plot data for {runner_up_name}", ha='center', va='center')
            except Exception as e:
                logger.error(f"Error plotting runner-up hypothesis {runner_up_name}: {e}")
                ax_runner_up.text(0.5, 0.5, f"Error plotting {runner_up_name}: {str(e)}", ha='center', va='center')
        else:
            ax_runner_up.text(0.5, 0.5, f"No plotter found for hypothesis type: {runner_up_type}", ha='center', va='center')
        
        ax_runner_up.set_title(f"Runner-up Hypothesis: {runner_up_name}")
    else:
        ax_runner_up.text(0.5, 0.5, "No runner-up hypothesis available", ha='center', va='center')
        ax_runner_up.set_title("Runner-up Hypothesis")
    
    # Format overall layout
    plt.tight_layout()
    
    # Generate the plot filename using metric and region
    metric_name_clean = metric_name.replace(' ', '_').lower()
    plot_filename = f"{metric_name_clean}_{region}_descriptive_detailed_summary.png"
    
    # Return a DataFrame with plot metadata
    return pd.DataFrame([{
        'plot_path': plot_filename,
        'metric_name': metric_name,
        'region': region,
        'plot_type': 'descriptive_detailed_summary',
        'figure': fig
    }]) 