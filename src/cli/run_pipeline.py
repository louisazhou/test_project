import os
import logging
import yaml
import argparse
import json
import pandas as pd
from typing import Dict, Any, List

# Import core components
from ..core.data_catalog import DataCatalog
from ..core.data_registry import DataRegistry
from ..core.anomaly_gate import AnomalyGate
from ..core.hypothesis_engine import HypothesisEngine
from ..core.plot_engine import PlotEngine
from ..core.narrative_engine import NarrativeEngine
from ..core.metric_engine import MetricEngine
from ..core.types import MetricReport, PlotSpec, MetricFormatting
from ..reporting.report_builder import ReportBuilder

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

def save_data_for_inspection(data, filename, tmp_dir="tmp"):
    """Save data objects to tmp directory for inspection."""
    os.makedirs(tmp_dir, exist_ok=True)
    filepath = os.path.join(tmp_dir, filename)
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath + ".csv", index=True)
    elif isinstance(data, dict):
        with open(filepath + ".json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif hasattr(data, '__dict__'):
        with open(filepath + ".json", 'w') as f:
            json.dump(data.__dict__, f, indent=2, default=str)
    else:
        with open(filepath + ".txt", 'w') as f:
            f.write(str(data))
    
    logger.info(f"Saved data for inspection to {filepath}")

def main():
    """Main pipeline execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RCA Pipeline")
    parser.add_argument("--metric", help="Only run this specific metric")
    parser.add_argument("--hypothesis", help="Only run this specific hypothesis")
    parser.add_argument("--save-data", action="store_true", help="Save data objects to tmp directory for inspection")
    args = parser.parse_args()
    
    logger.info("--- Starting Refactored RCA Pipeline --- ")

    # --- Configuration --- 
    # Determine base directory relative to this script's location
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = os.path.join(base_dir, 'config')
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output') # Base output dir
    tmp_dir = os.path.join(base_dir, 'tmp') # Directory for inspection data

    # Load and parse configuration files
    settings = load_config(os.path.join(config_dir, "settings.yaml"))
    metrics_yaml = load_config(os.path.join(config_dir, "metrics.yaml"))
    hypotheses_yaml = load_config(os.path.join(config_dir, "hypotheses.yaml"))
    datasets_config = load_config(os.path.join(config_dir, "datasets.yaml"))
    
    # Extract metrics map and hypotheses map
    metrics_map = metrics_yaml.get("metrics", {})
    hypotheses_map = {h['name']: h for h in hypotheses_yaml.get("hypotheses", [])}

    # Save configurations for inspection if requested
    if args.save_data:
        save_data_for_inspection(settings, "settings", tmp_dir)
        save_data_for_inspection(metrics_yaml, "metrics_config", tmp_dir)
        save_data_for_inspection(hypotheses_yaml, "hypotheses_config", tmp_dir)
        save_data_for_inspection(datasets_config, "datasets_map", tmp_dir)
        save_data_for_inspection(metrics_map, "metrics_map", tmp_dir)
        save_data_for_inspection(hypotheses_map, "hypotheses_map", tmp_dir)

    # Apply filters if command line args are provided
    if args.metric:
        if args.metric in metrics_map:
            logger.info(f"Filtering to run only metric: {args.metric}")
            metrics_map = {args.metric: metrics_map[args.metric]}
        else:
            logger.warning(f"Requested metric '{args.metric}' not found in configuration. Running all metrics.")
    
    if args.hypothesis:
        if args.hypothesis in hypotheses_map:
            logger.info(f"Filtering to run only hypothesis: {args.hypothesis}")
            hypotheses_map = {args.hypothesis: hypotheses_map[args.hypothesis]}
        else:
            logger.warning(f"Requested hypothesis '{args.hypothesis}' not found in configuration. Running all hypotheses.")

    # Extract specific settings
    plotting_settings = settings.get('plotting', {})
    anomaly_gate_settings = settings.get('anomaly_gate', {})
    z_thresh = anomaly_gate_settings.get('z_thresh', 1.0) # Default if not in settings
    delta_thresh = anomaly_gate_settings.get('delta_thresh', 0.1) # Default if not in settings

    # --- Initialization --- 
    logger.info("Initializing core components...")
    data_catalog = DataCatalog(config_dir, input_dir)
    data_registry = DataRegistry()
    
    # Initialize the metric engine
    metric_engine = MetricEngine(
        metrics_config=metrics_map,
        data_catalog=data_catalog,
        data_registry=data_registry
    )
    
    # Initialize anomaly detection
    anomaly_gate = AnomalyGate(z_thresh=z_thresh, delta_thresh=delta_thresh)
    
    # Initialize hypothesis engine
    hypothesis_engine = HypothesisEngine(
        hypothesis_configs=hypotheses_map,
        data_catalog=data_catalog,
        data_registry=data_registry,
        settings=settings
    )
    
    # Initialize plot engine
    plot_engine = PlotEngine(data_registry=data_registry, settings=plotting_settings)
    
    # Initialize narrative engine
    narrative_engine = NarrativeEngine(hypothesis_configs=hypotheses_map, settings=settings)
    
    # Initialize report builder
    reporting_settings = settings.get('reporting', {})
    report_builder = ReportBuilder(
        data_registry=data_registry,
        settings=reporting_settings,
        output_dir=output_dir
    )

    all_metric_reports: List[MetricReport] = []
    all_plot_specs: List[PlotSpec] = [] # Collect all plot specs

    # --- Process Metrics and Generate Reports --- 
    for metric_name, metric_info in metrics_map.items():
        logger.info(f"Processing metric: {metric_name}")

        # 1. Get Metric Data using the MetricEngine
        metric_df, global_value, _ = metric_engine.get_metric_info(metric_name)
        if metric_df is None or global_value is None:
            logger.error(f"Failed to process metric '{metric_name}'. Skipping.")
            continue
        
        # Save metric data for inspection if requested
        if args.save_data:
            save_data_for_inspection(metric_df, f"metric_df_{metric_name}", tmp_dir)
            save_data_for_inspection({"global_value": global_value}, f"global_value_{metric_name}", tmp_dir)
        
        # Define higher_is_better from metric_info
        higher_is_better = metric_info.get('higher_is_better', True)
        
        # Get metric natural name 
        metric_natural_name = metric_info.get('natural_name', metric_name)
        
        # 2. Find anomalies using AnomalyGate
        region_anomalies, metric_enrichment_data, metric_std = anomaly_gate.find_anomalies(
            metric_df.reset_index() if metric_df.index.name == 'region' else metric_df, 
            metric_name, 
            global_value, 
            higher_is_better,
            metric_natural_name=metric_natural_name
        )
        logger.info(f"Std dev for {metric_name} (from AnomalyGate): {metric_std:.6f}")
        
        # Save anomaly data for inspection if requested
        if args.save_data:
            save_data_for_inspection(region_anomalies, f"region_anomalies_{metric_name}", tmp_dir)
            save_data_for_inspection(metric_enrichment_data, f"metric_enrichment_data_{metric_name}", tmp_dir)
            save_data_for_inspection({"metric_std": metric_std}, f"metric_std_{metric_name}", tmp_dir)
        
        # Get original metric DF key (needed for plotting)
        original_metric_df_key = metric_engine._metric_cache.get(metric_name)
        
        # 3. Process each anomaly and evaluate hypotheses
        metric_level_plots = [] 
        associated_hypotheses = metric_info.get('hypothesis', [])
        
        # Filter associated hypotheses if a specific hypothesis was requested
        if args.hypothesis:
            associated_hypotheses = [h for h in associated_hypotheses if h == args.hypothesis]
            if not associated_hypotheses:
                logger.warning(f"Requested hypothesis '{args.hypothesis}' not associated with metric '{metric_name}'. Skipping hypothesis evaluation.")
        
        all_hypo_results_for_metric = []
        
        for anomaly in region_anomalies: 
            logger.info(f"Evaluating hypotheses for anomaly in {anomaly.region} for metric {metric_name}")
            
            # Evaluate hypotheses for this anomaly
            hypo_results, hypo_plot_specs = hypothesis_engine.evaluate_hypotheses_for_anomaly(
                metric_name=metric_name,
                anomaly=anomaly,
                associated_hypotheses=associated_hypotheses,
                metric_data_key=original_metric_df_key
            )
            
            # Save hypothesis results for inspection if requested
            if args.save_data:
                save_data_for_inspection(hypo_results, f"hypo_results_{metric_name}_{anomaly.region}", tmp_dir)
                save_data_for_inspection(hypo_plot_specs, f"hypo_plot_specs_{metric_name}_{anomaly.region}", tmp_dir)
            
            # Collect results for reporting
            all_hypo_results_for_metric.extend(hypo_results)
            all_plot_specs.extend(hypo_plot_specs)
            
            # Generate narratives for hypothesis results
            for hypo_res in hypo_results:
                hypo_res.narrative = narrative_engine.generate_narrative(hypo_res, anomaly)
            
            # Store hypothesis results in the anomaly object
            anomaly.hypo_results = hypo_results
            
        # Create a MetricReport object to consolidate metric data, anomalies, and plots
        metric_report = MetricReport(
            metric_name=metric_name,
            global_value=global_value,
            natural_name=metric_natural_name,
            is_percentage=MetricFormatting.is_percentage_metric(metric_name),
            metric_data_key=original_metric_df_key,
            metric_std=metric_std,
            anomalies=region_anomalies
        )
        all_metric_reports.append(metric_report)
        
        # Save metric report for inspection if requested
        if args.save_data:
            save_data_for_inspection(metric_report, f"metric_report_{metric_name}", tmp_dir)
        
        # 5. Generate metric plot specification
        metric_plot_spec = PlotSpec(
            plot_key='metric_bar_anomaly',
            data_key=original_metric_df_key,
            context={
                'value_col': metric_name,
                'title': f'Metric: {metric_natural_name}',
                'y_label': metric_natural_name,
                'metric_name': metric_name,
                'metric_natural_name': metric_natural_name,
                'higher_is_better': higher_is_better,
                'ref_metric_val': global_value,
                'std': metric_std,
                'z_score_threshold': z_thresh,
                'enrichment_data': metric_enrichment_data
            }
        )
        all_plot_specs.append(metric_plot_spec)
        
        # Save plot spec for inspection if requested
        if args.save_data:
            save_data_for_inspection(metric_plot_spec, f"metric_plot_spec_{metric_name}", tmp_dir)

    # --- Render Individual Plots --- 
    logger.info(f"Collected {len(all_plot_specs)} plot specifications.")
    saved_plot_files = plot_engine.render(all_plot_specs)
    
    # Save plot files for inspection if requested
    if args.save_data:
        save_data_for_inspection(saved_plot_files, "saved_plot_files", tmp_dir)

    # --- Generate Reports ---
    logger.info("Building final report(s)...")
    report_results = report_builder.build_report(
        all_metric_reports,
        metrics_map,
        hypotheses_map,
        z_thresh
    )
    
    # Save report results for inspection if requested
    if args.save_data:
        save_data_for_inspection(report_results, "report_results", tmp_dir)
    
    # --- Final Summary Log --- 
    logger.info(f"Processed {len(all_metric_reports)} metrics.")
    for report in all_metric_reports:
        num_anomalies = len(report.anomalies)
        logger.info(f"  Metric: {report.metric_name}, Anomalies: {num_anomalies}")
        
        if report.anomalies and report.anomalies[0].hypo_results:
            top_hypo = min(report.anomalies[0].hypo_results, key=lambda h: h.display_rank if h.display_rank is not None else float('inf'))
            logger.info(f"    Top Hypo (first anomaly): {top_hypo.name} (Score: {top_hypo.score:.3f})")

    logger.info("--- RCA Pipeline Finished --- ")

if __name__ == "__main__":
    main() 