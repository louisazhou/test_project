import os
import logging
import yaml
from typing import Dict, Any, List
import pandas as pd # Add pandas import
import numpy as np  # Add numpy import

# Import core components
from ..core.data_catalog import DataCatalog
from ..core.data_registry import DataRegistry
from ..core.metric_engine import MetricEngine
from ..core.anomaly_gate import AnomalyGate
from ..core.hypothesis_engine import HypothesisEngine
from ..core.plot_engine import PlotEngine
from ..core.narrative_engine import NarrativeEngine
# Import PlotSpec for collection
from ..core.types import MetricReport, PlotSpec, MetricFormatting
# Import presentation/upload functions
from ..core.presentation import generate_ppt, upload_to_drive
# Import utils if needed for settings conversion etc.
from ..core.utils import convert_bool 
# Import the summary plot function
from ..plotting.plot_router import plot_summary_report

# TODO: Add report builder (PPTX/MD etc.) component import
# from ..reporting.report_builder import ReportBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config if config else {}
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {}

def main():
    """Main pipeline execution function."""
    logger.info("--- Starting Refactored RCA Pipeline --- ")

    # --- Configuration --- 
    # Determine base directory relative to this script's location
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = os.path.join(base_dir, 'config')
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output') # Base output dir

    # Load configurations
    metrics_config_full = load_config(os.path.join(config_dir, 'metrics.yaml'))
    hypotheses_config_full = load_config(os.path.join(config_dir, 'hypotheses.yaml'))
    datasets_map = load_config(os.path.join(config_dir, 'datasets.yaml'))
    settings = metrics_config_full.get('settings', {})
    metrics_map = metrics_config_full.get('metrics', {})
    hypotheses_map = {h['name']: h for h in hypotheses_config_full.get('hypotheses', [])} # Convert list to map

    # Read plotting settings for individual plots
    plotting_settings = settings.get('plotting', {})
    # Read output format settings for summary report
    output_format_settings = settings.get('output_formats', {})
    summary_report_format = output_format_settings.get('summary_report_format', None) 
    if summary_report_format == 'null': summary_report_format = None
    generate_ppt_flag = convert_bool(output_format_settings.get('ppt', False))
    # Read google drive settings
    gdrive_settings = settings.get('google_drive', {})
    upload_gdrive_flag = convert_bool(gdrive_settings.get('upload_enabled', False))
    gdrive_folder_id = gdrive_settings.get('folder_id')
    if gdrive_folder_id == 'null': gdrive_folder_id = None # Handle null string

    # --- Initialization --- 
    logger.info("Initializing core components...")
    data_catalog = DataCatalog(config_dir, input_dir)
    data_registry = DataRegistry()
    # Extract specific thresholds for AnomalyGate
    anomaly_gate_settings = settings.get('anomaly_gate', {})
    z_thresh = anomaly_gate_settings.get('z_thresh', 1.0) # Default if not in settings
    delta_thresh = anomaly_gate_settings.get('delta_thresh', 0.1) # Default if not in settings
    anomaly_gate = AnomalyGate(z_thresh=z_thresh, delta_thresh=delta_thresh)
    # Pass settings to HypothesisEngine
    hypothesis_engine = HypothesisEngine(
        hypothesis_configs=hypotheses_map,
        data_catalog=data_catalog,
        data_registry=data_registry,
        settings=settings
    )
    
    # Pass the specific plotting settings to PlotEngine
    plot_engine = PlotEngine(data_registry=data_registry, settings=plotting_settings)
    # Pass hypothesis_configs to NarrativeEngine
    narrative_engine = NarrativeEngine(hypothesis_configs=hypotheses_map)
    # report_builder = ReportBuilder(settings=settings, output_dir=output_dir)

    all_metric_reports: List[MetricReport] = []
    all_plot_specs: List[PlotSpec] = [] # Collect all plot specs

    # --- Pre-load primary datasets? ---
    # Ensure the main metrics dataset is loaded before the loop
    # Need to know the dataset name from config (e.g., 'pipeline_metrics')
    main_metrics_dataset_name = None # TODO: Get this dynamically from metrics_map or assume a fixed name
    for _, metric_info in metrics_map.items():
         # Assuming the dataset comes from the first metric's config (needs better way)
         # Or perhaps it's defined globally in metrics.yaml?
         # For now, let's assume 'pipeline_metrics' based on previous logs
         main_metrics_dataset_name = "pipeline_metrics" 
         break # Only need it once
    main_metrics_data_key = None
    if main_metrics_dataset_name:
         try:
             main_metrics_data_key = data_catalog.load(main_metrics_dataset_name)
             logger.info(f"Pre-loaded dataset '{main_metrics_dataset_name}' with key: {main_metrics_data_key}")
         except Exception as e:
              logger.error(f"Failed to pre-load dataset {main_metrics_dataset_name}: {e}")
    # --- End Pre-load --- 

    if main_metrics_data_key is None:
         logger.error("Failed to load main metrics data. Aborting pipeline.")
         return # Exit if main data failed to load

    # --- Pipeline Execution --- 
    for metric_name, metric_info in metrics_map.items():
        logger.info(f"Processing metric: {metric_name}")

        # 1. Get Metric Data from pre-loaded DataFrame
        full_metrics_df = data_registry.get(main_metrics_data_key)
        if full_metrics_df is None:
            logger.error(f"Could not retrieve pre-loaded metrics data from registry using key {main_metrics_data_key}. Skipping {metric_name}.")
            continue
        
        # Select relevant columns (region and the metric itself)
        if metric_name not in full_metrics_df.columns:
            logger.warning(f"Metric column '{metric_name}' not found in dataset '{main_metrics_dataset_name}'. Skipping.")
            continue
        if 'region' not in full_metrics_df.columns:
            logger.warning(f"'region' column not found in dataset '{main_metrics_dataset_name}'. Skipping {metric_name}.")
            continue
            
        metric_df = full_metrics_df[['region', metric_name]].copy()
        # Convert value column to numeric early
        metric_df[metric_name] = pd.to_numeric(metric_df[metric_name], errors='coerce')
        metric_df = metric_df.dropna(subset=[metric_name]) # Drop rows where metric is NA
        
        # Define higher_is_better early from metric_info
        higher_is_better = metric_info.get('higher_is_better', True)
        
        # --- Set region as index --- 
        if 'region' in metric_df.columns:
             metric_df = metric_df.set_index('region')
        elif metric_df.index.name != 'region':
             # Handle case where region is neither column nor index
             logger.error(f"Metric DataFrame for {metric_name} does not contain 'region' column or index. Skipping.")
             continue
        # --- End Set Index --- 
            
        # Identify value column (first column after index)
        if len(metric_df.columns) < 1:
             logger.error(f"Metric DataFrame for {metric_name} has no value columns. Skipping.")
             continue
        value_col = metric_df.columns[0]

        # Get global value directly from the DataFrame (assuming region index is set)
        if 'Global' in metric_df.index:
             global_value = metric_df.loc['Global', value_col]
             logger.info(f"Using global value from DataFrame for {metric_name}: {global_value:.3f}")
        else:
             logger.warning(f"Could not determine global value for {metric_name} (no 'Global' row in index). Skipping anomaly detection.")
             continue
        
        metric_ref_val = global_value # Use global value as reference

        # Find anomalies, get enrichment data, and metric_std directly from the Gate
        # Pass the metric_natural_name for use in narrative generation
        metric_natural_name = metric_info.get('natural_name', metric_name)
        region_anomalies, metric_enrichment_data, metric_std = anomaly_gate.find_anomalies(
            metric_df.reset_index(), 
            metric_name, 
            global_value, 
            higher_is_better,
            metric_natural_name=metric_natural_name  # Pass natural name to the AnomalyGate
        )
        logger.info(f"Std dev for {metric_name} (from AnomalyGate): {metric_std:.6f}")
        
        # Get original metric DF key (needed for plotting)
        original_metric_df_key = main_metrics_data_key

        metric_level_plots = [] 
        # Evaluate Hypotheses for each TRUE Anomaly identified by the gate
        true_anomalies = region_anomalies # The list now ONLY contains true anomalies
        associated_hypotheses = metric_info.get('hypothesis', [])
        all_hypo_results_for_metric = []
        for anomaly in true_anomalies: 
            logger.info(f"Evaluating hypotheses for anomaly in {anomaly.region} for metric {metric_name}")
            # Pass the metric's global_value and overall_std to the engine, and the original data key
            hypo_results, hypo_plot_specs = hypothesis_engine.evaluate_hypotheses_for_anomaly(
                metric_name=metric_name,
                anomaly=anomaly, # This is RegionAnomaly
                associated_hypotheses=associated_hypotheses,
                metric_data_key=main_metrics_data_key, # Pass key to the original metric data
            )
            all_hypo_results_for_metric.extend(hypo_results) # Collect results for final summary
            all_plot_specs.extend(hypo_plot_specs) # Collect the plot specs
            
            # Generate Narratives using the ranked results
            top_hypothesis = next((r for r in hypo_results if r.display_rank == 0), None)

            # Update narrative in HypoResult objects
            for hypo_res in hypo_results:
                narrative_engine.generate_narrative(hypo_res, anomaly)  # Pass anomaly to narrative generation

            # Store anomaly and its evaluated hypotheses for potential later use (like summary report)
            anomaly.hypo_results = hypo_results
            
        # Build MetricReport
        metric_report = MetricReport(
            metric_name=metric_name,
            global_value=global_value,
            natural_name=metric_info.get('natural_name', metric_name),  # Add natural name
            is_percentage=MetricFormatting.is_percentage_metric(metric_name),  # Add is_percentage
            formatted_global_value=MetricFormatting.format_value(
                global_value, MetricFormatting.is_percentage_metric(metric_name)
            ),  # Add formatted global value
            metric_data_key=original_metric_df_key, 
            metric_enrichment_data=metric_enrichment_data, # Store enrichment data
            metric_std=metric_std,  # Use std from AnomalyGate
            anomalies=region_anomalies,
            metric_level_plots=metric_level_plots
        )
        all_metric_reports.append(metric_report)

        # Generate Metric Plot Spec using ORIGINAL data key and passing enrichment data
        metric_natural_name = metric_info.get('natural_name', metric_name)
        metric_plot_spec = PlotSpec(
            plot_key='metric_bar_anomaly',
            data_keys=[original_metric_df_key], # Use ORIGINAL key
            ctx={ # Pass context needed for display
                'value_col': value_col, 
                'title': f'Metric: {metric_natural_name}',
                'y_label': metric_natural_name,
                'metric_name': metric_name,
                'metric_natural_name': metric_natural_name,
                'higher_is_better': higher_is_better, 
                'ref_metric_val': metric_ref_val,
                'std': metric_std, # Use std from AnomalyGate
                'z_score_threshold': z_thresh
            },
            extra_data={ # Pass enrichment data directly from AnomalyGate
                 'enrichment_data': metric_enrichment_data
            }
        )
        all_plot_specs.append(metric_plot_spec)

    # --- Render Individual Plots --- 
    logger.info(f"Collected {len(all_plot_specs)} plot specifications.")
    # PlotEngine uses its internal mode ('batch' or 'inline') from settings
    saved_plot_files = plot_engine.render(all_plot_specs) 

    # --- Generate Summary Report / Visualizations (if format specified) --- 
    logger.info("Building final report(s)...")
    
    # --- Final Analysis Summary Consolidation ---
    analysis_summary = {} 
    for report in all_metric_reports:
        metric = report.metric_name
        metric_natural_name = metrics_map.get(metric, {}).get('natural_name', metric)
        # Get enrichment data directly from the report object
        metric_enrichment = report.metric_enrichment_data if report.metric_enrichment_data else {}
        
        # Retrieve original metric DF to recalculate std (if needed) or pass context
        original_metric_df_key = report.metric_data_key
        metric_df = data_registry.get(original_metric_df_key) if original_metric_df_key else None
        # Initialize these variables with defaults
        value_col_recalc = metric  # Default to metric name itself
        # Use the std directly from the report (which came from AnomalyGate)
        metric_std_for_summary = getattr(report, 'metric_std', 0.0) 
        logger.info(f"Using std for summary report for {metric}: {metric_std_for_summary:.6f}")

        primary_region = "NoAnomaly"
        best_hypothesis_result = None
        explanation_text = "No significant anomalies detected for this metric."
        metric_delta_primary = 0.0
        metric_dir_primary = "neutral"
        primary_anomaly_obj = None

        true_anomalies = [a for a in report.anomalies if a.is_anomaly]
        # Consolidate unique hypotheses across all anomalies for this metric
        all_hypo_results_dict = {}
        for anom in report.anomalies:
            for hypo_res in anom.hypo_results:
                if hypo_res.name not in all_hypo_results_dict: # Keep first encountered (or could prioritize by score)
                    all_hypo_results_dict[hypo_res.name] = hypo_res
        all_hypotheses_for_metric = list(all_hypo_results_dict.values())

        if true_anomalies:
            # Ensure explaining_hypotheses are derived from the unique list
            explaining_hypotheses = sorted([h for h in all_hypotheses_for_metric if h.score is not None and h.score > 0.5], 
                                           key=lambda x: x.score, reverse=True)
            
            if explaining_hypotheses: # Found explaining hypotheses
                best_hypothesis_result = explaining_hypotheses[0]
                # Find the anomaly object corresponding to the best hypothesis' region
                # Iterate through anomalies to find the one containing the best hypo result
                primary_anomaly_obj = None
                for anom in true_anomalies:
                    if best_hypothesis_result in anom.hypo_results: # Check if the best result is in this anomaly's list
                        primary_anomaly_obj = anom
                        break # Found the corresponding anomaly

                if primary_anomaly_obj:
                     primary_region = primary_anomaly_obj.region
                     explanation_text = best_hypothesis_result.narrative # Use the generated narrative
                     metric_delta_primary = primary_anomaly_obj.delta_pct
                     metric_dir_primary = primary_anomaly_obj.dir
                else: # Should not happen if hypo_result has region, but fallback
                     primary_region = best_hypothesis_result.region
                     explanation_text = f"Anomaly likely in {primary_region} explained by {best_hypothesis_result.name}, but anomaly details missing."

            else: # Anomalies exist, but none well explained
                # Find anomaly with highest absolute z-score
                primary_anomaly_obj = max(true_anomalies, key=lambda a: abs(a.z_score))
                primary_region = primary_anomaly_obj.region
                explanation_text = f"Anomaly detected in {primary_region} (Z={primary_anomaly_obj.z_score:.2f}), but no hypothesis strongly explains it."
                metric_delta_primary = primary_anomaly_obj.delta_pct
                metric_dir_primary = primary_anomaly_obj.dir
        else:
            # No true anomalies detected, keep defaults
             pass

        # Get hypothesis natural name if best hypo found
        hypo_natural_name = None
        hypo_delta_primary = None
        hypo_dir_primary = None
        hypo_delta_fmt = None
        if best_hypothesis_result:
             hypo_natural_name = hypotheses_map.get(best_hypothesis_result.name, {}).get('natural_name', best_hypothesis_result.name)
             hypo_delta_primary = best_hypothesis_result.key_numbers.get('hypo_delta')
             hypo_dir_primary = best_hypothesis_result.key_numbers.get('hypo_dir')
             hypo_delta_fmt = best_hypothesis_result.key_numbers.get('hypo_delta_fmt')

        # Store consolidated summary - Ensure all needed fields are present
        analysis_summary[metric] = {
            'metric_name': metric,
            'metric_natural_name': metric_natural_name,
            'metric_ref_val': report.global_value, 
            'metric_ref_val_fmt': report.formatted_global_value,  # Use the formatted value from MetricReport
            'metric_std': metric_std_for_summary if not np.isnan(metric_std_for_summary) else 0.0,
            'metric_value_col': value_col_recalc, # Store the metric column name used for std calc
            'higher_is_better': metrics_map.get(metric, {}).get('higher_is_better', True), 
            'z_score_threshold': z_thresh, 
            'primary_region': primary_region,
            'primary_anomaly_obj': primary_anomaly_obj, 
            'metric_delta_primary': metric_delta_primary,
            'metric_dir_primary': metric_dir_primary,
            'metric_value_primary': primary_anomaly_obj.value if primary_anomaly_obj else None,  # Use the value from anomaly
            'metric_value_primary_fmt': primary_anomaly_obj.formatted_value if primary_anomaly_obj else None,  # Use formatted value
            'metric_deviation_description': primary_anomaly_obj.deviation_description if primary_anomaly_obj else None,  # Use description
            'best_hypothesis_result': best_hypothesis_result, 
            'best_hypothesis_name': best_hypothesis_result.name if best_hypothesis_result else None,
            'hypo_natural_name': best_hypothesis_result.natural_name if best_hypothesis_result else hypo_natural_name,  # Use natural_name from HypoResult
            'hypo_delta_primary': hypo_delta_primary,
            'hypo_dir_primary': hypo_dir_primary,
            'hypo_delta_fmt': hypo_delta_fmt,
            'explanation_text': explanation_text,
            'all_hypotheses_for_metric': all_hypotheses_for_metric, 
            'metric_data_key': report.metric_data_key, 
            'metric_enrichment_data': metric_enrichment, # Now correctly populated
            'hypotheses_configs': hypotheses_map,
            'is_percentage_metric': report.is_percentage  # Use is_percentage from MetricReport
        }
    # --- End Final Analysis Summary Consolidation ---

    summary_plot_paths = [] # Store paths to generated summary plots if any
    if summary_report_format and summary_report_format.lower() != 'null':
         if summary_report_format in ["detailed", "succinct"]:
             logger.info(f"Generating '{summary_report_format}' summary visualization...")
             # Loop through analysis_summary and call the plot function
             for metric, summary_item in analysis_summary.items():
                 # Need to pass registry, output_dir, and format
                 summary_plot_path = plot_summary_report(
                     analysis_summary_item=summary_item, 
                     data_registry=data_registry,
                     output_dir=plot_engine.output_dir,
                     report_format=summary_report_format
                 )
                 if summary_plot_path: 
                      summary_plot_paths.append(summary_plot_path)
         else:
             logger.warning(f"Unknown summary_report_format: '{summary_report_format}'. Skipping summary visualization.")
    else:
         logger.info("No summary report format specified (or format is null), skipping summary visualization generation.")

    # --- Generate PPT (if enabled AND a summary format was specified) --- 
    ppt_path = None
    # Only generate PPT if flag is true AND a summary format was requested (implying summary plots exist or are intended)
    if generate_ppt_flag and summary_report_format and summary_report_format.lower() != 'null':
        logger.info("Generating PowerPoint presentation...")
        # Pass the analysis_summary and the specific summary_report_format needed by generate_ppt
        if analysis_summary: 
            ppt_filename = "RCA_Summary.pptx"
            ppt_path = generate_ppt(
                analysis_summary, 
                plot_engine.output_dir, # Assumes summary plots are saved here too
                ppt_filename,
                report_format=summary_report_format # Pass the summary format
            )
            if ppt_path:
                logger.info(f"PowerPoint presentation generated: {ppt_path}")
            else:
                logger.error("Failed to generate PowerPoint presentation.")
        else:
            logger.warning("Skipping PowerPoint generation: No analysis results processed for summary.")
    elif generate_ppt_flag:
         logger.info("Skipping PowerPoint generation: No summary report format specified.")
    else:
        logger.info("Skipping PowerPoint generation based on settings.")

    # --- Upload to Google Drive (if enabled and PPT exists) --- 
    if ppt_path and upload_gdrive_flag:
        logger.info("Uploading PowerPoint to Google Drive...")
        file_id = upload_to_drive(ppt_path, folder_id=gdrive_folder_id)
        if file_id:
            logger.info(f"PowerPoint uploaded to Google Drive. File ID: {file_id}")
        else:
            logger.error("Failed to upload PowerPoint to Google Drive.")
    elif ppt_path:
        logger.info("Skipping Google Drive upload based on settings.")

    # --- Final Summary Log --- 
    logger.info(f"Processed {len(all_metric_reports)} metrics.")
    # TODO: Improve final summary logging based on analysis_summary
    for report in all_metric_reports:
         num_anomalies = len(report.anomalies)
         logger.info(f"  Metric: {report.metric_name}, Anomalies: {num_anomalies}")
         # Add more details like top hypo, primary region etc.
         # Example: Log top hypo for first anomaly if exists
         if report.anomalies and report.anomalies[0].hypo_results:
              top_hypo = min(report.anomalies[0].hypo_results, key=lambda h: h.display_rank if h.display_rank is not None else float('inf'))
              logger.info(f"    Top Hypo (first anomaly): {top_hypo.name} (Score: {top_hypo.score:.3f})")

    logger.info("--- Refactored RCA Pipeline Finished --- ")

if __name__ == "__main__":
    main() 