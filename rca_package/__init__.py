"""
RCA Automation Package for Root Cause Analysis.

This package provides tools for hypothesis scoring, YAML configuration processing,
and slide generation for root cause analysis.
"""

# Import key functions from hypothesis_scorer
from .hypothesis_scorer import (
    score_hypotheses_for_metrics,
    score_all_hypotheses,
    get_ranked_hypotheses,
    create_multi_hypothesis_plot,
    create_scatter_grid
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
    get_technical_name,
    get_display_name_from_technical,
    convert_dataframe_to_display_names,
    get_technical_names_for_metrics,
    get_technical_names_for_hypotheses,
    load_all_configs
)

# Import key functions from make_slides
from .make_slides import (
    SlideLayouts,
    SlideContent,
    dual_output,
    create_bar_chart,
    create_scatter_plot
)

# Import Google Drive utilities (optional)
try:
    from .google_drive_utils import (
        get_credentials_local,
        get_credentials_enterprise,
        upload_to_google_drive
    )
    _has_google_drive = True
except ImportError:
    # Google Drive utilities not available
    _has_google_drive = False
    get_credentials_local = None
    get_credentials_enterprise = None
    upload_to_google_drive = None

# Import key functions from depth_spotter
from .depth_spotter import (
    analyze_region_depth,
    plot_subregion_bars
)

# Import anomaly detector
from .anomaly_detector import detect_snapshot_anomaly_for_column

# Import slides from make_slides
from .make_slides import SlideLayouts
slides = SlideLayouts()

# Define what should be available in "from rca_package import *"
__all__ = [
    # Core analysis functions
    'score_hypotheses_for_metrics',
    'score_all_hypotheses',
    'get_ranked_hypotheses',
    'create_multi_hypothesis_plot',
    'create_scatter_grid',
    
    # Depth analysis functions
    'analyze_region_depth',
    'plot_subregion_bars',
    
    # Anomaly detection
    'detect_snapshot_anomaly_for_column',
    
    # YAML processing
    'load_config',
    'get_metric_info',
    'get_hypothesis_info',
    'get_all_metrics',
    'get_relevant_hypotheses',
    'get_expected_directions',
    'get_metric_hypothesis_map',
    'get_template',
    'get_technical_name',
    'get_display_name_from_technical',
    'convert_dataframe_to_display_names',
    'get_technical_names_for_metrics',
    'get_technical_names_for_hypotheses',
    'load_all_configs',
    
    # Slide building
    'SlideLayouts',
    'SlideContent',
    'dual_output',
    'create_bar_chart',
    'create_scatter_plot',
    'get_credentials_local',
    'get_credentials_enterprise',
    'upload_to_google_drive',
    'slides'
] 