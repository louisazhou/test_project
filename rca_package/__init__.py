"""
RCA Automation Package for Root Cause Analysis.

This package provides tools for hypothesis scoring, YAML configuration processing,
and slide generation for root cause analysis.
"""

import logging
import os
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import key functions from hypothesis_scorer
from .hypothesis_scorer import (
    score_all_hypotheses,
    process_metrics,
    get_ranked_hypotheses,
    create_multi_hypothesis_plot,
    create_scatter_grid,
    plot_scatter
)

# Import key functions from yaml_processor
from .yaml_processor import (
    load_config,
    get_metric_info,
    get_hypothesis_info,
    get_all_metrics,
    get_relevant_hypotheses,
    get_expected_directions,
    get_metric_hypothesis_map,
    get_template,
    get_display_name,
    get_scoring_method,
    create_metric_anomaly_map
)

# Import key functions from make_slides
from .make_slides import (
    create_metrics_summary_slide,
    create_metrics_presentation
)

def run_analysis(
    df,
    config_path: str,
    save_path: str = '.',
    results_path: Optional[str] = None,
    region: Optional[str] = None
):
    """
    Run a complete analysis using configuration from a YAML file.
    
    Args:
        df: DataFrame containing metric and hypothesis data
        config_path: Path to the YAML configuration file
        save_path: Directory to save generated figures
        results_path: Path to save DataFrame results (if None, results are not saved)
        region: Name of the anomalous region (if None, will be detected)
        
    Returns:
        Dictionary containing all analysis results
    """
    from .yaml_processor import run_analysis_from_config
    
    return run_analysis_from_config(
        df=df,
        config_path=config_path,
        anomalous_region=region,
        save_path=save_path,
        results_path=results_path
    )

# Define what should be available in "from rca_package import *"
__all__ = [
    # Core analysis functions
    'score_all_hypotheses',
    'process_metrics',
    'get_ranked_hypotheses',
    
    # Visualization functions
    'create_multi_hypothesis_plot',
    'create_scatter_grid',
    'plot_scatter',
    
    # YAML configuration functions
    'load_config',
    'get_metric_info',
    'get_hypothesis_info',
    'get_all_metrics',
    'get_relevant_hypotheses',
    'get_expected_directions',
    'get_metric_hypothesis_map',
    'get_template',
    'get_display_name',
    'create_metric_anomaly_map',
    'get_scoring_method',
    # High-level functions
    'run_analysis',
    
    # Presentation functions
    'create_metrics_summary_slide',
    'create_metrics_presentation'
] 