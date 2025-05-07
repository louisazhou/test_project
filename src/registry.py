"""
Registry module for the RCA automation system.

This module provides central registries for various components such as:
- Hypothesis plotters
- Metric calculators
- Handler functions
"""

import logging
import functools
from typing import Dict, Any, Callable, Optional

from .core.types import MetricFormatting

logger = logging.getLogger(__name__)

# Initialize registry
HYPOTHESIS_PLOTTERS = {}

def register_plotter(hypothesis_type: str, plotter_function: Callable) -> None:
    """Register a plotting function for a hypothesis type.
    
    Args:
        hypothesis_type: The type of hypothesis this plotter handles
        plotter_function: The plotting function to register
    """
    HYPOTHESIS_PLOTTERS[hypothesis_type] = plotter_function
    logger.debug(f"Registered plotter for hypothesis type: {hypothesis_type}")
    
def get_plotter(hypothesis_type: str) -> Optional[Callable]:
    """Get a registered plotting function for a hypothesis type.
    
    Args:
        hypothesis_type: The type of hypothesis to get a plotter for
        
    Returns:
        The plotting function if registered, otherwise None
    """
    plotter = HYPOTHESIS_PLOTTERS.get(hypothesis_type)
    if plotter is None:
        logger.warning(f"No plotter registered for hypothesis type: {hypothesis_type}")
    return plotter

def hypothesis_plotter(func: Callable) -> Callable:
    """Decorator for hypothesis plotter functions.
    
    This decorator standardizes error handling and title setting across
    plotting functions to reduce code duplication.
    
    Args:
        func: The plotting function to decorate
        
    Returns:
        The wrapped function with standard error handling
    """
    @functools.wraps(func)
    def wrapper(ax, hypo_result, title=None, **kwargs):
        """Wrapper for hypothesis plotter functions.
        
        Args:
            ax: The matplotlib axes to draw on
            hypo_result: The hypothesis result object
            title: Optional title for the plot
            **kwargs: Additional parameters for the plotting function
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            if title and hasattr(ax, 'set_title'):
                ax.set_title(title)
                
            return func(ax=ax, hypo_result=hypo_result, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            if hasattr(ax, 'text'):
                ax.text(0.5, 0.5, f"Error plotting: {str(e)}", 
                      ha='center', va='center')
            return False
    
    return wrapper

def get_value_column(hypo_result, hypo_config=None) -> Optional[str]:
    """Extract the value column name from hypothesis results or configuration.
    
    Args:
        hypo_result: The hypothesis result object
        hypo_config: Optional hypothesis configuration
        
    Returns:
        The name of the value column if found, None otherwise
    """
    # Try to get from explicit plot_data.value_col attribute
    if hasattr(hypo_result, 'plot_data') and hasattr(hypo_result.plot_data, 'value_col'):
        return hypo_result.plot_data.value_col
    
    # Try to get from plot_data DataFrame columns
    if hasattr(hypo_result, 'plot_data') and hasattr(hypo_result.plot_data, 'columns'):
        # Look for common value column patterns
        value_candidates = [col for col in hypo_result.plot_data.columns 
                           if any(pattern in col.lower() for pattern in 
                                 ['value', 'metric', 'score', 'pct', 'rate'])]
        if value_candidates:
            return value_candidates[0]
    
    # Try to get from config
    if hypo_config and 'value_col' in hypo_config:
        return hypo_config['value_col']
    
    return None

def format_plot_components(delta, higher_is_better=True, use_pct=False):
    """Format common components for hypothesis plots.
    
    Args:
        delta: The delta value to format
        higher_is_better: Whether higher values are better
        use_pct: Whether to format as percentage
        
    Returns:
        Dictionary with formatted text and colors
    """
    formatter = MetricFormatting()
    delta_text = formatter.format_delta_text(delta, higher_is_better, use_pct)
    
    return {
        'delta_text': delta_text,
        'delta_color': 'green' if (delta > 0) == higher_is_better else 'red',
        'icon': '▲' if (delta > 0) == higher_is_better else '▼'
    } 