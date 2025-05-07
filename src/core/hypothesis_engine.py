from typing import Dict, Any, List, Optional, Tuple
import importlib
import logging

from .data_catalog import DataCatalog
from .data_registry import DataRegistry
from .types import RegionAnomaly, HypoResult, PlotSpec

logger = logging.getLogger(__name__)

class HypothesisEngine:
    """Routes anomaly context to the appropriate hypothesis handler and collects results."""

    def __init__(self, hypothesis_configs: Dict[str, Any], data_catalog: DataCatalog, data_registry: DataRegistry, settings: Dict[str, Any]):
        self.hypothesis_configs = hypothesis_configs
        self.data_catalog = data_catalog
        self.data_registry = data_registry
        self.settings = settings
        logger.info(f"HypothesisEngine initialized with {len(hypothesis_configs)} hypothesis configurations.")

    def evaluate_hypotheses_for_anomaly(
        self,
        metric_name: str,
        anomaly: RegionAnomaly, # Specific anomaly instance
        associated_hypotheses: List[str],
        metric_data_key: str,       # Key to the original metric data in registry
    ) -> Tuple[List[HypoResult], List[PlotSpec]]:
        """Evaluate all relevant hypotheses for a given anomaly on a metric."""
        results: List[HypoResult] = []
        plot_specs: List[PlotSpec] = []

        logger.info(f"Evaluating {len(associated_hypotheses)} hypotheses for {metric_name} anomaly in {anomaly.region}")

        for hypo_name in associated_hypotheses:
            handler_config = self.hypothesis_configs.get(hypo_name)
            if not handler_config:
                logger.warning(f"No configuration found for hypothesis: {hypo_name}")
                continue

            handler_module_path = handler_config.get('hypothesis_type')
            if not handler_module_path:
                logger.warning(f"No handler module specified for hypothesis: {hypo_name}")
                continue
            
            try:
                # Dynamically import the handler module
                # Check if handler_module_path contains a class specification
                if '.' in handler_module_path:
                    module_name, class_name = handler_module_path.rsplit('.', 1)
                else:
                    # If no class name specified, assume it's "Handler"
                    module_name = handler_module_path
                    class_name = "Handler"
                
                # Ensure module path is relative to src directory if not already dot-prefixed
                if not module_name.startswith('src.') and not module_name.startswith('.'):
                    full_module_path = f"src.handlers.{module_name}" 
                else:
                    full_module_path = module_name
                
                logger.debug(f"Loading handler: module={full_module_path}, class={class_name}")
                module = importlib.import_module(full_module_path)
                HandlerClass = getattr(module, class_name)
                
                # Instantiate handler - pass necessary context
                # Handler needs data_catalog for loading its own data
                # and potentially settings for its specific behavior.
                handler = HandlerClass(
                    hypothesis_config=handler_config,
                    data_catalog=self.data_catalog, 
                    data_registry=self.data_registry, # For storing/retrieving intermediate data if needed by handler
                    settings=self.settings # Pass global settings
                )
                
                # Call the handler's run method
                # Pass the specific anomaly, and now metric_data_key
                hypo_result, hypo_plot_spec_from_handler = handler.run(
                    metric_name=metric_name, 
                    anomaly=anomaly,
                    metric_data_key=metric_data_key, 
                )

                if hypo_result:
                    results.append(hypo_result)
                if hypo_plot_spec_from_handler: # Changed variable name
                    # Add selected flag to the plot spec context if it's the top hypothesis after ranking (done later)
                    # For now, just collect the spec as returned by the handler.
                    plot_specs.append(hypo_plot_spec_from_handler) 

            except ImportError as e:
                logger.error(f"Could not import handler module for type '{handler_module_path}': {e}")
                continue
            except Exception as e:
                # Log the full traceback for better debugging
                logger.exception(f"Error evaluating hypothesis '{hypo_name}': {e}") 
                continue

        # Sort results by score for display ranking
        # Filter out None scores before sorting if necessary
        results_with_scores = [r for r in results if r.score is not None]
        results_without_scores = [r for r in results if r.score is None]
        results_with_scores.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks only to those with scores
        for i, result in enumerate(results_with_scores):
            result.display_rank = i
            
        # Combine lists back
        final_results = results_with_scores + results_without_scores

        if results_with_scores: # Log only if ranks were assigned
             logger.info(f"Assigned display ranks to {len(results_with_scores)} hypothesis results.")
             
        # The final_results list is now ranked. This is what we use for plot spec selection.
        ranked_results = final_results 

        # --- Create PlotSpecs AFTER ranking (or refine existing ones) --- 
        # The plot_specs list already contains specs from handlers. We need to update their 'selected' flag.
        final_plot_specs = []
        for spec in plot_specs: # Iterate over specs collected from handlers
            is_selected_plot = False
            # Check if this plot corresponds to the top-ranked hypothesis
            if ranked_results and spec.context.get('hypothesis_name') == ranked_results[0].name:
                is_selected_plot = True
            
            # Update the existing spec's context
            spec.context['selected'] = is_selected_plot 
            final_plot_specs.append(spec)

        return ranked_results, final_plot_specs

    def _assign_display_ranks(self, results: List[HypoResult]) -> List[HypoResult]:
        # Sort results by score for display ranking
        # Filter out None scores before sorting if necessary
        results_with_scores = [r for r in results if r.score is not None]
        results_without_scores = [r for r in results if r.score is None]
        results_with_scores.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks only to those with scores
        for i, result in enumerate(results_with_scores):
            result.display_rank = i
            
        # Combine lists back
        final_results = results_with_scores + results_without_scores

        if results_with_scores: # Log only if ranks were assigned
             logger.info(f"Assigned display ranks to {len(results_with_scores)} hypothesis results.")
             
        return final_results 