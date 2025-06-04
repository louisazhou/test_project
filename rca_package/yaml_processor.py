import yaml
import os
import pandas as pd
from typing import Dict, Any, List, Optional

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
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

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

def get_technical_name(config: Dict[str, Any], display_name: str, col_type: Optional[str] = None) -> str:
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

def get_display_name_from_technical(config: Dict[str, Any], technical_name: str, col_type: Optional[str] = None) -> str:
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

def load_all_configs(config_paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Load all configuration files.
    
    Args:
        config_paths: Dictionary mapping config names to file paths
        
    Returns:
        Dictionary of loaded configurations
    """
    configs = {}
    for config_name, config_path in config_paths.items():
        try:
            configs[config_name] = load_config(config_path)
            print(f"✅ Loaded config: {config_name}")
        except Exception as e:
            print(f"❌ Failed to load config {config_name} from {config_path}: {e}")
    
    return configs 