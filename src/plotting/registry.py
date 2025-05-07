"""
Plotting registry module.

This module provides centralized registration of plotting functions
for different hypothesis types, ensuring a standardized interface
between hypothesis handlers and visualization.
"""

import logging
import functools
from typing import Callable, Dict, Any, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Global registry of hypothesis plotters
HYPOTHESIS_PLOTTERS = {}

def register_plotter(hypothesis_type: str, plotter_func: Callable):
    """Register a plotting function for a specific hypothesis type.
    
    Args:
        hypothesis_type: The type of hypothesis this plotter handles
        plotter_func: The function that will plot this hypothesis type
    """
    HYPOTHESIS_PLOTTERS[hypothesis_type] = plotter_func
    logger.debug(f"Registered plotter for hypothesis type: {hypothesis_type}")
    
def get_plotter(hypothesis_type: str) -> Optional[Callable]:
    """Get the registered plotting function for a hypothesis type.
    
    Args:
        hypothesis_type: The type of hypothesis to retrieve a plotter for
        
    Returns:
        The registered plotting function or None if no plotter exists
    """
    plotter = HYPOTHESIS_PLOTTERS.get(hypothesis_type)
    if plotter is None:
        logger.warning(f"No plotter registered for hypothesis type: {hypothesis_type}")
    return plotter

def hypothesis_plotter(func):
    """Decorator for hypothesis plotting functions.
    
    This decorator handles common error handling, title setting, and
    data extraction patterns to reduce code duplication across plotters.
    
    Args:
        func: The plotting function to decorate
        
    Returns:
        Wrapped function with standardized error handling
    """
    @functools.wraps(func)
    def wrapper(ax, hypo_result, **kwargs):
        if ax is None:
            logger.error(f"Cannot plot {func.__name__}: No axes provided")
            return False
            
        try:
            # Call the actual plotting function
            result = func(ax, hypo_result, **kwargs)
            
            # Set title if provided and not already set by the function
            if kwargs.get('title') and not ax.get_title():
                ax.set_title(kwargs.get('title'), fontsize=12)
                
            return result if result is not None else True
        except Exception as e:
            logger.error(f"Error plotting {func.__name__}: {e}")
            ax.text(0.5, 0.5, f"Error: Could not plot {hypo_result.name if hasattr(hypo_result, 'name') else 'hypothesis'}", 
                   ha='center', va='center')
            return False
    
    return wrapper

def get_value_column(hypo_result, hypo_config=None, **kwargs):
    """Extract the value column name from hypothesis result or config.
    
    Args:
        hypo_result: The hypothesis result object
        hypo_config: Optional config dictionary for the hypothesis
        
    Returns:
        The column name to use for values, or None if it cannot be determined
    """
    # First check if it's directly specified in kwargs
    value_col = kwargs.get('value_col')
    if value_col:
        return value_col
        
    # Next try to get from hypothesis config
    if hypo_config:
        value_col = hypo_config.get('input_data', [{}])[0].get('columns', [None])[0]
        if value_col:
            return value_col
    
    # Finally try to extract from the plot data
    if hasattr(hypo_result, 'plot_data') and hypo_result.plot_data is not None:
        df = hypo_result.plot_data
        if hasattr(df, 'columns'):
            # Use first column that's not 'region'
            potential_cols = [col for col in df.columns if col != 'region']
            if potential_cols:
                logger.info(f"Determined value_col '{potential_cols[0]}' from plot_data columns")
                return potential_cols[0]
    
    # Couldn't determine value column
    logger.error(f"Could not determine value column for hypothesis plot")
    return None
    
def format_plot_components(hypo_result, primary_region=None, **kwargs):
    """Format common components for hypothesis plots.
    
    Args:
        hypo_result: The hypothesis result object
        primary_region: The primary region being analyzed
        
    Returns:
        Dictionary of formatted plot components
    """
    from ..core.types import MetricFormatting
    
    components = {}
    
    # Format delta text if needed
    if kwargs.get('include_delta', False) and hasattr(hypo_result, 'value') and hasattr(hypo_result, 'global_value'):
        # Only proceed if we have all the necessary values
        if hypo_result.value is not None and hypo_result.global_value is not None and primary_region:
            is_percentage = hasattr(hypo_result, 'is_percentage') and hypo_result.is_percentage
            
            delta_fmt = MetricFormatting.format_delta(
                hypo_result.value,
                hypo_result.global_value,
                is_percentage
            )
            
            direction = "higher" if hypo_result.value > hypo_result.global_value else "lower"
            components['delta_text'] = f"({primary_region} is {delta_fmt} {direction} than Global)"
            
            # Update title with delta info if requested
            if 'title' in kwargs and components.get('delta_text'):
                components['title'] = f"{kwargs['title']} {components['delta_text']}"
            else:
                components['title'] = kwargs.get('title')
    else:
        components['title'] = kwargs.get('title')
        
    return components 