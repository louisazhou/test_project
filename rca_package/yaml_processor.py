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

def get_metric_info(config: Dict[str, Any], metric_col: str) -> Dict[str, Any]:
    """
    Get information for a specific metric from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        metric_col: Name of the metric column
        
    Returns:
        Dictionary containing metric information
    """
    metrics = config.get('metrics', {})
    return metrics.get(metric_col, {})

def get_hypothesis_info(config: Dict[str, Any], metric_col: str, hypo_col: str) -> Dict[str, Any]:
    """
    Get information for a specific hypothesis from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        metric_col: Name of the metric column
        hypo_col: Name of the hypothesis column
        
    Returns:
        Dictionary containing hypothesis information
    """
    metric_info = get_metric_info(config, metric_col)
    hypotheses = metric_info.get('hypotheses', {})
    return hypotheses.get(hypo_col, {})

def get_all_metrics(config: Dict[str, Any]) -> List[str]:
    """
    Get all metric column names from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        
    Returns:
        List of metric column names
    """
    return list(config.get('metrics', {}).keys())

def get_relevant_hypotheses(config: Dict[str, Any], metric_col: str) -> List[str]:
    """
    Get hypotheses relevant to a specific metric.
    
    Args:
        config: Loaded configuration dictionary
        metric_col: Name of the metric column
        
    Returns:
        List of hypothesis column names relevant to the metric
    """
    metric_info = get_metric_info(config, metric_col)
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
    
    for metric_col, metric_info in config.get('metrics', {}).items():
        for hypo_col, hypo_info in metric_info.get('hypotheses', {}).items():
            # Store the expected direction for this hypothesis
            expected_directions[hypo_col] = hypo_info.get('expected_direction', 'same')
    
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
    
    for metric_col, metric_info in config.get('metrics', {}).items():
        # Get the hypothesis columns for this metric
        hypo_cols = list(metric_info.get('hypotheses', {}).keys())
        metric_hypo_map[metric_col] = hypo_cols
    
    return metric_hypo_map

def get_template(config: Dict[str, Any], metric_col: str, hypo_col: str, template_type: str = 'template') -> str:
    """
    Get the template for a specific metric-hypothesis pair.
    
    Args:
        config: Loaded configuration dictionary
        metric_col: Name of the metric column
        hypo_col: Name of the hypothesis column
        template_type: Type of template to get ('template' or 'summary_template')
        
    Returns:
        Template string for the metric-hypothesis pair
    """
    hypo_info = get_hypothesis_info(config, metric_col, hypo_col)
    return hypo_info.get(template_type, '')

def get_display_name(config: Dict[str, Any], col_name: str, col_type: str = 'metric') -> str:
    """
    Get the display name for a metric or hypothesis.
    
    Args:
        config: Loaded configuration dictionary
        col_name: Technical column name
        col_type: 'metric' or 'hypothesis'
        
    Returns:
        Display name for the column
    """
    if col_type == 'metric':
        metric_info = get_metric_info(config, col_name)
        return metric_info.get('name', col_name)
    else:
        # For hypothesis, search across all metrics
        for metric_col, metric_info in config.get('metrics', {}).items():
            for hypo_col, hypo_info in metric_info.get('hypotheses', {}).items():
                if hypo_col == col_name:
                    return hypo_info.get('name', col_name)
        return col_name

def get_scoring_method(config: Dict[str, Any]) -> str:
    """
    Get the global scoring method from the configuration.
    
    Args:
        config: Loaded configuration dictionary
        
    Returns:
        String indicating the scoring method ('standard' or 'sign_based')
    """
    return config.get('scoring_method', 'standard')

def create_anomaly_info_from_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
    metric_col: str,
    anomalous_region: str
) -> Dict[str, Any]:
    """
    Create metric anomaly info dictionary from data.
    
    Args:
        df: DataFrame containing the metric data
        config: Loaded configuration dictionary
        metric_col: Name of the metric column
        anomalous_region: Name of the anomalous region
        
    Returns:
        Dictionary with metric anomaly information
    """
    # Get the metric value for the anomalous region
    metric_val = df.loc[anomalous_region, metric_col]
    
    # Get the global reference value
    if 'Global' in df.index:
        global_val = df.loc['Global', metric_col]
    else:
        # If no 'Global' row, use the mean of all regions
        global_val = df[metric_col].mean()
    
    # Calculate the direction and magnitude
    direction = 'higher' if metric_val > global_val else 'lower'
    magnitude = abs((metric_val - global_val) / global_val * 100) if global_val != 0 else 0
    
    # Get higher_is_better from config
    metric_info = get_metric_info(config, metric_col)
    higher_is_better = metric_info.get('higher_is_better', True)
    
    # Get the global scoring method
    scoring_method = get_scoring_method(config)
    
    # Create the anomaly info dictionary
    anomaly_info = {
        'anomalous_region': anomalous_region,
        'metric_val': metric_val,
        'global_val': global_val,
        'direction': direction,
        'magnitude': magnitude,
        'higher_is_better': higher_is_better,
        'scoring_method': scoring_method
    }
    
    return anomaly_info

def create_metric_anomaly_map(
    df: pd.DataFrame,
    config: Dict[str, Any],
    anomalous_region: str = None,
    metric_cols: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Create a dictionary mapping metrics to their anomaly information.
    
    Args:
        df: DataFrame containing the metric data
        config: Loaded configuration dictionary
        anomalous_region: Name of the anomalous region (if None, will be detected)
        metric_cols: List of metric columns to process (if None, all metrics from config)
        
    Returns:
        Dictionary mapping metric names to their anomaly information
    """
    # If no metric columns specified, use all from config
    if metric_cols is None:
        metric_cols = get_all_metrics(config)
    
    # Create empty map
    metric_anomaly_map = {}
    
    # Process each metric
    for metric_col in metric_cols:
        # Skip if metric is not in dataframe
        if metric_col not in df.columns:
            logger.warning(f"Metric {metric_col} not found in DataFrame. Skipping.")
            continue
        
        # If no anomalous region specified, detect the one with largest deviation
        if anomalous_region is None:
            # Use largest absolute relative deviation from Global
            if 'Global' in df.index:
                global_val = df.loc['Global', metric_col]
                deviations = df[metric_col].copy()
                deviations = deviations.drop('Global') if 'Global' in deviations.index else deviations
                # Calculate relative deviations
                deviations = deviations.apply(lambda x: abs((x - global_val) / global_val) if global_val != 0 else 0)
                # Find region with largest deviation
                detected_region = deviations.idxmax()
            else:
                # If no 'Global' row, use the region furthest from mean
                mean_val = df[metric_col].mean()
                deviations = df[metric_col].apply(lambda x: abs((x - mean_val) / mean_val) if mean_val != 0 else 0)
                detected_region = deviations.idxmax()
            
            current_anomalous_region = detected_region
            logger.info(f"Detected anomalous region for {metric_col}: {current_anomalous_region}")
        else:
            current_anomalous_region = anomalous_region
        
        # Create anomaly info for this metric
        anomaly_info = create_anomaly_info_from_data(
            df, config, metric_col, current_anomalous_region
        )
        
        # Store in map
        metric_anomaly_map[metric_col] = anomaly_info
    
    return metric_anomaly_map

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
        add_score_formula, plot_scatter
    )
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Get all metrics from config
    metric_cols = get_all_metrics(config)
    logger.debug(f"Found {len(metric_cols)} metrics in configuration")
    
    # Get all unique hypotheses from config
    all_hypo_cols = set()
    for metric_col in metric_cols:
        hypo_cols = get_relevant_hypotheses(config, metric_col)
        all_hypo_cols.update(hypo_cols)
    all_hypo_cols = list(all_hypo_cols)
    logger.debug(f"Found {len(all_hypo_cols)} unique hypotheses in configuration")
    
    # Create metric anomaly map
    logger.info("Creating metric anomaly map")
    metric_anomaly_map = create_metric_anomaly_map(df, config, anomalous_region, metric_cols)
    
    # Get expected directions
    expected_directions = get_expected_directions(config)
    
    # Get metric-hypothesis mapping
    metric_hypo_map = get_metric_hypothesis_map(config)
    
    # Process each metric with its relevant hypotheses
    logger.info("Processing metrics and scoring hypotheses")
    all_results = {}
    for metric_col, metric_info in config.get('metrics', {}).items():
        # Skip if metric not in DataFrame
        if metric_col not in df.columns:
            continue
            
        # Get relevant hypotheses for this metric
        hypo_cols = metric_hypo_map.get(metric_col, [])
        
        # Skip if no hypotheses for this metric
        if not hypo_cols:
            logger.warning(f"No hypotheses found for metric {metric_col}. Skipping.")
            continue
        
        # Get scoring method from config
        scoring_method = metric_info.get('scoring_method', 'standard')
        
        # Process this metric with its relevant hypotheses
        logger.debug(f"Scoring {len(hypo_cols)} hypotheses for metric {metric_col}")
        from rca_package.scoring_tools.hypothesis_scorer import score_all_hypotheses
        try:
            hypo_results = score_all_hypotheses(
                df=df,
                metric_col=metric_col,
                hypo_cols=hypo_cols,
                metric_anomaly_info=metric_anomaly_map[metric_col],
                expected_directions=expected_directions,
                scoring_method=scoring_method
            )
            
            # Store results
            all_results[metric_col] = hypo_results
        except Exception as e:
            logger.error(f"Error processing {metric_col}: {str(e)}")
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
    for metric_col, hypo_results in all_results.items():
        # Skip if no results for this metric
        if not hypo_results:
            continue
            
        # Get relevant hypotheses for this metric
        hypo_cols = metric_hypo_map.get(metric_col, [])
        
        # Skip if no hypotheses for this metric
        if not hypo_cols:
            continue
            
        # Get metric display name
        metric_display_name = get_display_name(config, metric_col, 'metric')
        
        # Get scoring method from config
        metric_info = get_metric_info(config, metric_col)
        scoring_method = metric_info.get('scoring_method', 'standard')
        
        # 1. Scatter plots for each hypothesis
        for hypo_col in hypo_cols:
            # Skip if hypothesis not in DataFrame
            if hypo_col not in df.columns:
                logger.warning(f"Hypothesis {hypo_col} not found in DataFrame. Skipping visualization.")
                continue
                
            # Get hypothesis display name
            hypo_display_name = get_display_name(config, hypo_col, 'hypothesis')
            
            # Get expected direction
            expected_direction = expected_directions.get(hypo_col, 'same')
            
            # Create and save single scatter plot
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_scatter(
                    ax=ax,
                    df=df[[metric_col, hypo_col]],
                    metric_anomaly_info=metric_anomaly_map[metric_col],
                    expected_direction=expected_direction
                )
                
                # Set more descriptive title
                ax.set_title(f"Relationship between {metric_display_name} and {hypo_display_name}")
                ax.set_xlabel(metric_display_name)
                ax.set_ylabel(hypo_display_name)
                
                plt.tight_layout()
                
                # Save with descriptive filename
                filename = f"{save_path}/scatter_{metric_col}_{hypo_col}_{scoring_method}.png"
                plt.savefig(filename, dpi=120, bbox_inches='tight')
                plt.close(fig)
                logger.debug(f"Created scatter plot: {filename}")
            except Exception as e:
                logger.error(f"Error creating scatter plot for {metric_col} and {hypo_col}: {str(e)}")
        
        # 2. Scatter grid
        try:
            scatter_grid = create_scatter_grid(
                df=df,
                metric_col=metric_col,
                hypo_cols=hypo_cols,
                metric_anomaly_info=metric_anomaly_map[metric_col],
                expected_directions=expected_directions,
                figsize=(15, 10)
            )
            
            # Save with descriptive filename
            filename = f"{save_path}/scatter_grid_{metric_col}_{scoring_method}.png"
            scatter_grid.savefig(filename, dpi=120, bbox_inches='tight')
            plt.close(scatter_grid)
            logger.debug(f"Created scatter grid: {filename}")
        except Exception as e:
            logger.error(f"Error creating scatter grid for {metric_col}: {str(e)}")
        
        # 3. Multi-hypothesis bar plot
        try:
            # Get ranked hypotheses
            ranked_hypos = get_ranked_hypotheses(hypo_results)
            
            # Create visualization
            fig = create_multi_hypothesis_plot(
                df=df,
                metric_col=metric_col,
                hypo_cols=hypo_cols,
                metric_anomaly_info=metric_anomaly_map[metric_col],
                hypo_results=hypo_results,
                ordered_hypos=ranked_hypos
            )
            
            # Get best hypothesis
            best_hypo_name, best_hypo_result = ranked_hypos[0]
            
            # Get template from config
            template = get_template(config, metric_col, best_hypo_name)
            
            # Add template text and score formula
            add_template_text(
                fig, 
                template, 
                best_hypo_name,
                best_hypo_result, 
                metric_anomaly_map[metric_col],
                metric_col
            )
            add_score_formula(fig, is_sign_based=(scoring_method == 'sign_based'))
            
            # Save with descriptive filename
            filename = f"{save_path}/multi_hypo_{metric_col}_{scoring_method}.png"
            fig.savefig(filename, dpi=120, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"Created multi-hypothesis visualization: {filename}")
        except Exception as e:
            logger.error(f"Error creating multi-hypothesis plot for {metric_col}: {str(e)}")
    
    logger.info("Analysis completed successfully")


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