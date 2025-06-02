import yaml
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from jinja2 import Template

# Configure logging
logger = logging.getLogger(__name__)

def load_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the parsed configuration
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.debug(f"Successfully loaded configuration from {yaml_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {yaml_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

def get_metric_info(config: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    """
    Get information for a specific metric from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        metric_name: Display name of the metric
        
    Returns:
        Dictionary containing metric information
    """
    metrics = config.get('metrics', {})
    return metrics.get(metric_name, {})

def get_hypothesis_info(config: Dict[str, Any], metric_name: str, hypo_name: str) -> Dict[str, Any]:
    """
    Get information for a specific hypothesis from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        metric_name: Display name of the metric
        hypo_name: Display name of the hypothesis
        
    Returns:
        Dictionary containing hypothesis information
    """
    metric_info = get_metric_info(config, metric_name)
    hypotheses = metric_info.get('hypotheses', {})
    return hypotheses.get(hypo_name, {})

def get_all_metrics(config: Dict[str, Any]) -> List[str]:
    """
    Get all metric names from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        
    Returns:
        List of metric display names
    """
    return list(config.get('metrics', {}).keys())

def get_relevant_hypotheses(config: Dict[str, Any], metric_name: str) -> List[str]:
    """
    Get hypotheses relevant to a specific metric.
    
    Args:
        config: Loaded configuration dictionary
        metric_name: Display name of the metric
        
    Returns:
        List of hypothesis display names relevant to the metric
    """
    metric_info = get_metric_info(config, metric_name)
    return list(metric_info.get('hypotheses', {}).keys())

def get_expected_directions(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract expected directions for all hypotheses across all metrics.
    
    Args:
        config: Loaded configuration dictionary
        
    Returns:
        Dictionary mapping hypothesis names to their expected directions
    """
    expected_directions = {}
    
    for metric_name, metric_info in config.get('metrics', {}).items():
        for hypo_name, hypo_info in metric_info.get('hypotheses', {}).items():
            # Store the expected direction for this hypothesis
            expected_directions[hypo_name] = hypo_info.get('expected_direction', 'same')
    
    return expected_directions

def get_metric_hypothesis_map(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Create a mapping from metrics to their relevant hypotheses.
    
    Args:
        config: Loaded configuration dictionary
        
    Returns:
        Dictionary mapping metric names to lists of relevant hypothesis names
    """
    metric_hypo_map = {}
    
    for metric_name, metric_info in config.get('metrics', {}).items():
        # Get the hypothesis names for this metric
        hypo_names = list(metric_info.get('hypotheses', {}).keys())
        metric_hypo_map[metric_name] = hypo_names
    
    return metric_hypo_map

def get_template(config: Dict[str, Any], metric_name: str, hypo_name: str, template_type: str = 'template') -> str:
    """
    Get a template for a specific metric-hypothesis pair.
    
    Args:
        config: Loaded configuration dictionary
        metric_name: Name of the metric (display name)
        hypo_name: Name of the hypothesis (display name)
        template_type: Type of template to retrieve ('template' or 'summary_template')
        
    Returns:
        String containing the template text
    """
    hypo_info = get_hypothesis_info(config, metric_name, hypo_name)
    return hypo_info.get(template_type, '')

def run_analysis_from_config(
    df: pd.DataFrame,
    config_path: str,
    anomalous_region: str = None,
    save_path: str = '.',
    results_path: str = None
):
    """
    Run hypothesis analysis using configuration from a YAML file.
    
    Args:
        df: DataFrame containing metric and hypothesis data
        config_path: Path to the YAML configuration file
        anomalous_region: Name of the anomalous region (if None, will be detected)
        save_path: Directory to save generated figures
        results_path: Path to save DataFrame results (if None, results are not saved)
    """
    # Import here to avoid circular import
    from rca_package.hypothesis_scorer import (
        process_metrics, save_results_to_dataframe, get_ranked_hypotheses,
        create_multi_hypothesis_plot, create_scatter_grid, add_template_text,
        add_score_formula, plot_scatter, score_all_hypotheses
    )
    from rca_package.anomaly_detector import detect_snapshot_anomaly_for_column
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Get all metrics from config
    metric_names = get_all_metrics(config)
    logger.debug(f"Found {len(metric_names)} metrics in configuration")
    
    # Get all unique hypotheses from config
    all_hypo_names = set()
    for metric_name in metric_names:
        hypo_names = get_relevant_hypotheses(config, metric_name)
        all_hypo_names.update(hypo_names)
    all_hypo_names = list(all_hypo_names)
    logger.debug(f"Found {len(all_hypo_names)} unique hypotheses in configuration")
    
    # Create metric anomaly map using the anomaly detector
    logger.info("Creating metric anomaly map")
    metric_anomaly_map = {}
    for metric_name in metric_names:
        anomaly_info = detect_snapshot_anomaly_for_column(df, 'Global', column=metric_name)
        if anomaly_info:
            metric_anomaly_map[metric_name] = anomaly_info
            if anomalous_region:  # Override detected region if specified
                metric_anomaly_map[metric_name]['anomalous_region'] = anomalous_region
    
    # Get expected directions
    expected_directions = get_expected_directions(config)
    
    # Get metric-hypothesis mapping
    metric_hypo_map = get_metric_hypothesis_map(config)
    
    # Process each metric with its relevant hypotheses
    logger.info("Processing metrics and scoring hypotheses")
    all_results = {}
    for metric_name, metric_info in config.get('metrics', {}).items():
        # Skip if metric not in DataFrame
        if metric_name not in df.columns:
            continue
            
        # Get relevant hypotheses for this metric
        hypo_names = metric_hypo_map.get(metric_name, [])
        
        # Skip if no hypotheses for this metric
        if not hypo_names:
            logger.warning(f"No hypotheses found for metric {metric_name}. Skipping.")
            continue
        
        # Process this metric with its relevant hypotheses (always use sign-based scoring)
        logger.debug(f"Scoring {len(hypo_names)} hypotheses for metric {metric_name}")
        try:
            hypo_results = score_all_hypotheses(
                df=df,
                metric_col=metric_name,
                hypo_cols=hypo_names,
                metric_anomaly_info=metric_anomaly_map[metric_name],
                expected_directions=expected_directions
            )
            
            # Store results
            all_results[metric_name] = hypo_results
        except Exception as e:
            logger.error(f"Error processing {metric_name}: {str(e)}")
            continue
    
    # Check if we got any results
    if not all_results:
        logger.error("No results were generated for any metrics.")
        return
    
    # Convert results to DataFrame
    logger.info("Converting results to DataFrame")
    df_results = save_results_to_dataframe(all_results)
    
    # Save results if path provided
    if results_path is not None:
        try:
            df_results.to_csv(results_path, index=False)
            logger.info(f"Saved results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving results to {results_path}: {str(e)}")
    
    # Create visualizations for each metric
    logger.info("Creating visualizations")
    import matplotlib.pyplot as plt
    for metric_name, hypo_results in all_results.items():
        # Skip if no results for this metric
        if not hypo_results:
            continue
            
        # Get relevant hypotheses for this metric
        hypo_names = metric_hypo_map.get(metric_name, [])
        
        # Skip if no hypotheses for this metric
        if not hypo_names:
            continue
        
        # 1. Scatter plots for each hypothesis
        for hypo_name in hypo_names:
            # Skip if hypothesis not in DataFrame
            if hypo_name not in df.columns:
                logger.warning(f"Hypothesis {hypo_name} not found in DataFrame. Skipping visualization.")
                continue
                
            # Get expected direction
            expected_direction = expected_directions.get(hypo_name, 'same')
            
            # Create and save single scatter plot
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_scatter(
                    ax=ax,
                    df=df[[metric_name, hypo_name]],
                    metric_anomaly_info=metric_anomaly_map[metric_name],
                    expected_direction=expected_direction
                )
                
                # Set more descriptive title
                ax.set_title(f"Relationship between {metric_name} and {hypo_name}")
                ax.set_xlabel(metric_name)
                ax.set_ylabel(hypo_name)
                
                plt.tight_layout()
                
                # Save with descriptive filename (always sign_based)
                filename = f"{save_path}/scatter_{metric_name}_{hypo_name}_sign_based.png"
                plt.savefig(filename, dpi=120, bbox_inches='tight')
                plt.close(fig)
                logger.debug(f"Created scatter plot: {filename}")
            except Exception as e:
                logger.error(f"Error creating scatter plot for {metric_name} and {hypo_name}: {str(e)}")
        
        # 2. Scatter grid
        try:
            scatter_grid = create_scatter_grid(
                df=df,
                metric_col=metric_name,
                hypo_cols=hypo_names,
                metric_anomaly_info=metric_anomaly_map[metric_name],
                expected_directions=expected_directions,
                figsize=(15, 10)
            )
            
            # Save with descriptive filename (always sign_based)
            filename = f"{save_path}/scatter_grid_{metric_name}_sign_based.png"
            scatter_grid.savefig(filename, dpi=120, bbox_inches='tight')
            plt.close(scatter_grid)
            logger.debug(f"Created scatter grid: {filename}")
        except Exception as e:
            logger.error(f"Error creating scatter grid for {metric_name}: {str(e)}")

def get_technical_name(config: Dict[str, Any], display_name: str, col_type: str = None) -> str:
    """
    Get the technical name for a display name from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        display_name: Display name to look up
        col_type: Type of column ('metric' or 'hypothesis'). If None, will auto-detect.
        
    Returns:
        Technical name for the display name, or the display name itself if not found
    """
    metrics = config.get('metrics', {})
    
    # If col_type is specified, search only in that category
    if col_type == 'metric':
        metric_info = metrics.get(display_name, {})
        return metric_info.get('technical_name', display_name)
    elif col_type == 'hypothesis':
        # Search through all metrics' hypotheses
        for metric_name, metric_info in metrics.items():
            hypotheses = metric_info.get('hypotheses', {})
            if display_name in hypotheses:
                return hypotheses[display_name].get('technical_name', display_name)
        return display_name
    
    # Auto-detect: first check if it's a metric
    if display_name in metrics:
        return metrics[display_name].get('technical_name', display_name)
    
    # Then check if it's a hypothesis
    for metric_name, metric_info in metrics.items():
        hypotheses = metric_info.get('hypotheses', {})
        if display_name in hypotheses:
            return hypotheses[display_name].get('technical_name', display_name)
    
    # If not found, return the display name as-is
    return display_name

def get_display_name_from_technical(config: Dict[str, Any], technical_name: str, col_type: str = None) -> str:
    """
    Get the display name for a technical name from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        technical_name: Technical name to look up
        col_type: Type of column ('metric' or 'hypothesis'). If None, will auto-detect.
        
    Returns:
        Display name for the technical name, or the technical name itself if not found
    """
    metrics = config.get('metrics', {})
    
    # If col_type is specified, search only in that category
    if col_type == 'metric':
        for display_name, metric_info in metrics.items():
            if metric_info.get('technical_name') == technical_name:
                return display_name
        return technical_name
    elif col_type == 'hypothesis':
        # Search through all metrics' hypotheses
        for metric_name, metric_info in metrics.items():
            hypotheses = metric_info.get('hypotheses', {})
            for hypo_display_name, hypo_info in hypotheses.items():
                if hypo_info.get('technical_name') == technical_name:
                    return hypo_display_name
        return technical_name
    
    # Auto-detect: first check if it's a metric
    for display_name, metric_info in metrics.items():
        if metric_info.get('technical_name') == technical_name:
            return display_name
    
    # Then check if it's a hypothesis
    for metric_name, metric_info in metrics.items():
        hypotheses = metric_info.get('hypotheses', {})
        for hypo_display_name, hypo_info in hypotheses.items():
            if hypo_info.get('technical_name') == technical_name:
                return hypo_display_name
    
    # If not found, return the technical name as-is
    return technical_name

def convert_dataframe_to_display_names(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a DataFrame with technical column names to display names.
    
    Args:
        df: DataFrame with technical column names
        config: Loaded configuration dictionary
        
    Returns:
        DataFrame with display names as column headers
    """
    # Create a copy to avoid modifying the original
    df_display = df.copy()
    
    # Create mapping from technical names to display names
    column_mapping = {}
    for col in df.columns:
        display_name = get_display_name_from_technical(config, col)
        if display_name != col:  # Only map if we found a display name
            column_mapping[col] = display_name
    
    # Rename columns
    df_display = df_display.rename(columns=column_mapping)
    
    return df_display

def get_technical_names_for_metrics(config: Dict[str, Any], metric_display_names: List[str]) -> List[str]:
    """
    Get technical names for a list of metric display names.
    
    Args:
        config: Loaded configuration dictionary
        metric_display_names: List of metric display names
        
    Returns:
        List of technical names
    """
    return [get_technical_name(config, name, 'metric') for name in metric_display_names]

def get_technical_names_for_hypotheses(config: Dict[str, Any], hypo_display_names: List[str]) -> List[str]:
    """
    Get technical names for a list of hypothesis display names.
    
    Args:
        config: Loaded configuration dictionary
        hypo_display_names: List of hypothesis display names
        
    Returns:
        List of technical names
    """
    return [get_technical_name(config, name, 'hypothesis') for name in hypo_display_names]

# Example usage
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hypothesis analysis using YAML configuration")
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to the YAML configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--region', type=str, default=None,
                        help='Name of the anomalous region (if None, will be detected)')
    parser.add_argument('--save_path', type=str, default='.', 
                        help='Directory to save generated figures')
    parser.add_argument('--results_path', type=str, default=None, 
                        help='Path to save DataFrame results (CSV)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load data
    try:
        df = pd.read_csv(args.data, index_col=0)
        logger.info(f"Loaded data from {args.data} with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading data from {args.data}: {str(e)}")
        exit(1)
    
    # Run analysis
    run_analysis_from_config(
        df=df,
        config_path=args.config,
        anomalous_region=args.region,
        save_path=args.save_path,
        results_path=args.results_path
    ) 