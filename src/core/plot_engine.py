from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import os
import importlib
import logging
import pathlib
from .data_registry import DataRegistry
from .types import PlotSpec
from ..plotting.plot_router import ROUTER

logger = logging.getLogger(__name__)

class PlotEngine:
    """Renders PlotSpec objects into plots (inline or saved files) using plotting functions."""

    def __init__(self, data_registry: DataRegistry, settings: Dict[str, Any]):
        self.data_registry = data_registry
        # Example settings structure: { 'plotting': {'enabled': True, 'mode': 'batch', 'output_dir': 'output/figs'} }
        self.settings = settings.get('plotting', {})
        self.plot_enabled = self.settings.get('enabled', True)
        self.plot_mode = self.settings.get('mode', 'batch')  # 'batch' or 'inline'
        self.output_dir = self.settings.get('output_dir', 'output/figs')

        if self.plot_enabled and self.plot_mode == 'batch':
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"PlotEngine initialized. Mode: '{self.plot_mode}', Output directory: '{self.output_dir}'")
        elif self.plot_enabled:
            logger.info(f"PlotEngine initialized. Mode: '{self.plot_mode}'")
        else:
            logger.info("PlotEngine initialized. Plotting disabled.")

    def render(self, plot_specs: Union[PlotSpec, List[PlotSpec]]) -> Optional[List[Union[pathlib.Path, plt.Figure]]]:
        """Renders one or more PlotSpec objects.

        Args:
            plot_specs: A single PlotSpec or list of PlotSpec objects to render.

        Returns:
            A list of file paths where plots were saved (in batch mode), or None if plotting is disabled or mode is inline.
        """
        if not self.plot_enabled:
            logger.debug("Plotting is disabled, skipping render.")
            return None

        # Convert single PlotSpec to list
        if isinstance(plot_specs, PlotSpec):
            plot_specs = [plot_specs]

        # Track unique plot specs to avoid duplicates
        seen_plots = set()
        saved_files = []

        for spec in plot_specs:
            try:
                # Create a unique identifier for this plot spec
                plot_id = (
                    spec.plot_key,
                    str(spec.data_keys),  # Convert list to string
                    str(sorted(spec.extra_data.items())) if spec.extra_data else None,  # Convert dict to string
                    str(sorted(spec.ctx.items())) if spec.ctx else None  # Convert dict to string
                )

                # Skip if we've already rendered this exact plot
                if plot_id in seen_plots:
                    logger.debug(f"Skipping duplicate plot spec: {spec.plot_key}")
                    continue
                seen_plots.add(plot_id)

                # Get plot function from router
                plot_function = ROUTER.get(spec.plot_key)
                if not plot_function:
                    logger.error(f"Unknown plot key: {spec.plot_key}")
                    continue

                # Prepare data for the plotting function
                df = None
                if spec.data_keys: # If keys provided, load from registry (e.g., for metric plot)
                    primary_key = spec.data_keys[0]
                    df = self.data_registry.get(primary_key)
                    if df is None:
                         logger.warning(f"DataFrame for key '{primary_key}' not found in registry for plot '{spec.plot_key}'.")
                         # Try getting df from extra_data as fallback
                         df = spec.extra_data.get('df') if spec.extra_data else None
                elif spec.extra_data and 'df' in spec.extra_data: # If no keys, look for df in extra_data (e.g., for hypo plot)
                     df = spec.extra_data.get('df')
                
                if df is None or df.empty: # Check if we failed to get data
                    logger.error(f"No data available (registry or extra_data) for plot '{spec.plot_key}'")
                    continue

                # --- Create Figure and Axes ---
                fig, ax = plt.subplots(figsize=(10, 6))
                if fig is None or ax is None:
                     logger.error(f"Failed to create figure and axes for plot spec '{spec.plot_key}'")
                     continue
                # --- End Create Figure and Axes ---

                # Prepare keyword arguments for the plotting function
                keyword_args = {} 
                if spec.ctx:
                    keyword_args.update(spec.ctx)
                if spec.extra_data:
                    keyword_args.update(spec.extra_data)
                
                # Remove df/dfs/ax if they exist in kwargs, as they are passed positionally
                keyword_args.pop('ax', None)
                keyword_args.pop('df', None) 
                keyword_args.pop('dfs', None)

                # Call the plot function positionally for ax, df, then keywords
                try:
                    plot_function(ax, df, **keyword_args) # Pass ax, df positionally
                except Exception as plot_err:
                     logger.error(f"Error calling plot function '{spec.plot_key}': {plot_err}", exc_info=True)

                # Handle output based on mode
                if self.plot_mode == 'inline':
                    plt.show()
                elif self.plot_mode == 'batch':
                    try:
                        # Generate filename based on plot type
                        metric_name = keyword_args.get('metric_name') or keyword_args.get('metric', 'unknown_metric') # Use metric_name if passed
                        hypothesis_name = keyword_args.get('hypothesis_name', None) # Get hypo name
                        
                        if spec.plot_key == 'metric_bar_anomaly':
                            filename = f"metric_{metric_name}.png"
                        elif spec.plot_key == 'hypo_bar_scored' and hypothesis_name:
                             filename = f"hypo_{hypothesis_name}_for_{metric_name}.png"
                        else:
                            # Fallback filename (includes region/hypo if available)
                            region = keyword_args.get('region', 'all_regions')
                            hypo_fallback = hypothesis_name or keyword_args.get('hypothesis', 'no_hypo') # Use hypo name if available
                            filename = f"{metric_name}_{region}_{hypo_fallback}_{spec.plot_key}.png"

                        filepath = os.path.join(self.output_dir, filename)

                        fig.savefig(filepath, bbox_inches='tight', dpi=150)
                        saved_files.append(filepath)
                        logger.info(f"Saved plot to {filepath}")
                    except Exception as e:
                        logger.error(f"Error saving plot: {e}", exc_info=True)
                    finally:
                        plt.close(fig)  # Close figure after saving
                else:
                    logger.warning(f"Unknown plot mode: {self.plot_mode}")
                    plt.close(fig)  # Close figure if mode is unknown

            except Exception as e:
                logger.error(f"Error rendering plot spec '{spec.plot_key}': {e}", exc_info=True)
                if 'fig' in locals() and plt.fignum_exists(fig.number):
                    plt.close(fig)

        return saved_files if self.plot_mode == 'batch' else None

    def close_all(self):
        """Close all open figures."""
        plt.close('all')

def render(spec: PlotSpec, mode: str = "batch") -> Union[pathlib.Path, plt.Figure]:
    """Render a plot based on the provided specification.
    
    Args:
        spec: PlotSpec containing plot configuration
        mode: Either "batch" (save to file) or "inline" (return figure)
        
    Returns:
        Either a Path to the saved file or a matplotlib Figure
    """
    # Get data from registry
    dfs = [DataRegistry.get(k) for k in spec.data_keys]
    
    # Generate plot using router
    fig = ROUTER[spec.plot_key](dfs, **spec.extra_data)
    
    if mode == "batch":
        # Create output directory if it doesn't exist
        output_dir = pathlib.Path("tmp/figs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        path = output_dir / f"{hash(spec)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
        
    elif mode == "inline":
        return fig
        
    else:
        raise ValueError(f"Invalid mode: {mode}") 
