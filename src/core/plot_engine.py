from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import os
import logging
import pathlib
from .data_registry import DataRegistry
from .types import PlotSpec
from ..plotting.plot_router import route_plot

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
                    spec.data_key,
                    str(sorted(str(spec.context.items()))) if spec.context else None
                )

                # Skip if we've already rendered this exact plot
                if plot_id in seen_plots:
                    logger.debug(f"Skipping duplicate plot spec: {spec.plot_key}")
                    continue
                seen_plots.add(plot_id)

                # Get the data for plotting
                df = None
                
                # First check if data is already provided in the spec
                if spec.data is not None:
                    df = spec.data
                # Then try to get data using data_key
                elif spec.data_key:
                    df = self.data_registry.get(spec.data_key)
                
                if df is None or df.empty: # Check if we failed to get data
                    logger.error(f"No data available for plot '{spec.plot_key}'")
                    continue

                # Create a new PlotSpec with the optimized structure
                ready_spec = PlotSpec(
                    plot_key=spec.plot_key,
                    data=df,
                    context=spec.context.copy()
                )
                
                # Route the plot through the plot_router
                filepath = route_plot(ready_spec, plot_engine=self)
                if filepath:
                    saved_files.append(filepath)

            except Exception as e:
                logger.error(f"Error rendering plot spec '{spec.plot_key}': {e}", exc_info=True)

        return saved_files if self.plot_mode == 'batch' else None

    def save_plot(self, fig: plt.Figure, filename: str) -> Optional[str]:
        """Save a plot to a file in the output directory.
        
        Args:
            fig: The matplotlib figure to save
            filename: The filename to save as
            
        Returns:
            The path to the saved file, or None if saving failed
        """
        if not self.plot_enabled or self.plot_mode != 'batch':
            plt.close(fig)
            return None
            
        try:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, bbox_inches='tight', dpi=150)
            logger.debug(f"Saved plot to {filepath}")
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"Error saving plot {filename}: {e}")
            plt.close(fig)
            return None

    def close_all(self):
        """Close all open figures."""
        plt.close('all') 
