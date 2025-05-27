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
    plot_scatter,
    build_structured_hypothesis_results,
    process_metrics_with_structured_results,
    add_score_formula,
    add_template_text,
    render_template_text
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
    get_scoring_method,
    get_technical_name,
    get_display_name_from_technical,
    convert_dataframe_to_display_names,
    get_technical_names_for_metrics,
    get_technical_names_for_hypotheses
)

# Import key functions from make_slides
from .make_slides import (
    create_metrics_summary_slide,
    add_figure_to_slide,
    add_text_to_slide,
    create_slide_with_title,
    create_figure_with_text_slide,
    create_flexible_presentation,
    add_section_slide
)

# Import key functions from depth_spotter
from .depth_spotter import (
    rate_contrib,
    additive_contrib,
    plot_subregion_bars,
    analyze_region_depth
)

# Define what should be available in "from rca_package import *"
__all__ = [
    # Core analysis functions
    'score_all_hypotheses',
    'process_metrics',
    'get_ranked_hypotheses',
    'build_structured_hypothesis_results',
    'process_metrics_with_structured_results',
    
    # Visualization functions
    'create_multi_hypothesis_plot',
    'create_scatter_grid',
    'plot_scatter',
    'add_score_formula',
    'add_template_text',
    'render_template_text',
    
    # YAML configuration functions
    'load_config',
    'get_metric_info',
    'get_hypothesis_info',
    'get_all_metrics',
    'get_relevant_hypotheses',
    'get_expected_directions',
    'get_metric_hypothesis_map',
    'get_template',
    'get_scoring_method',
    'get_technical_name',
    'get_display_name_from_technical',
    'convert_dataframe_to_display_names',
    'get_technical_names_for_metrics',
    'get_technical_names_for_hypotheses',
    
    # High-level functions
    'run_analysis',
    
    # Presentation functions
    'create_metrics_summary_slide',
    'add_figure_to_slide',
    'add_text_to_slide',
    'create_slide_with_title',
    'create_figure_with_text_slide',
    'create_flexible_presentation',
    'add_section_slide',
    
    # Depth analysis functions
    'rate_contrib',
    'additive_contrib',
    'plot_subregion_bars',
    'analyze_region_depth'
] 