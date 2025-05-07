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
    ) -> List[HypoResult]:
        """Evaluate all relevant hypotheses for a given anomaly on a metric."""
        results: List[HypoResult] = []

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
                    # Map hypothesis types to module paths if not directly specified
                    if handler_module_path == 'single_dim':
                        module_name = 'src.handlers.single_dim'
                    elif handler_module_path == 'depth_spotter':
                        module_name = 'src.handlers.depth_spotter'
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
                
                # Log handler call
                logger.debug(f"Calling handler '{handler.name}' for hypothesis '{hypo_name}' on metric '{metric_name}' for anomaly in '{anomaly.region}'")
                
                hypo_result_obj, hypo_plot_spec_obj = handler.run(
                    metric_name=metric_name, 
                    anomaly=anomaly, 
                    metric_data_key=metric_data_key
                )
                logger.debug(f"Handler '{handler.name}' returned HypoResult: {hypo_result_obj.name if hypo_result_obj else 'None'}, PlotSpec: {'Exists' if hypo_plot_spec_obj else 'None'}")

                if hypo_result_obj:
                    if hypo_plot_spec_obj:
                        if hypo_result_obj.plots is None: 
                            hypo_result_obj.plots = []
                            logger.debug(f"Initialized .plots for HypoResult {hypo_result_obj.name}")
                        
                        hypo_result_obj.plots.append(hypo_plot_spec_obj)
                        logger.debug(f"Appended PlotSpec to HypoResult {hypo_result_obj.name}. .plots now has {len(hypo_result_obj.plots)} items. First item context: {hypo_result_obj.plots[0].context if hypo_result_obj.plots else 'N/A'}")
                    
                    results.append(hypo_result_obj)
            except ImportError as e:
                logger.error(f"Could not import handler module for type '{handler_module_path}': {e}")
                continue
            except Exception as e:
                # Log the full traceback for better debugging
                logger.exception(f"Error evaluating hypothesis '{hypo_name}': {e}") 
                continue

        # Separate single_dim hypotheses (for ranking) from descriptive hypotheses
        single_dim_results = [r for r in results if r.type == 'single_dim' and r.score is not None]
        descriptive_results = [r for r in results if r.type != 'single_dim']
        
        # Only rank single_dim hypotheses by score
        single_dim_results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign display ranks only to single_dim hypotheses
        for i, result in enumerate(single_dim_results):
            result.display_rank = i
            
        # Combine lists, with ranked single_dim hypotheses first
        final_results = single_dim_results + descriptive_results

        if single_dim_results:
             logger.info(f"Assigned display ranks to {len(single_dim_results)} single_dim hypothesis results.")
        if descriptive_results:
             logger.info(f"Found {len(descriptive_results)} descriptive hypothesis results (not ranked).")
             
        # Update 'selected' flag in PlotSpecs WITHIN HypoResults
        if single_dim_results:  # Only mark selection if we have ranked results
            top_ranked_hypo_name = single_dim_results[0].name if single_dim_results else None
            for res in final_results:
                if res.plots:
                    for spec_in_result in res.plots:
                        is_selected_plot = False
                        # Only select the top ranked single_dim hypothesis
                        if res.type == 'single_dim' and top_ranked_hypo_name and spec_in_result.context.get('hypothesis_name') == top_ranked_hypo_name:
                            is_selected_plot = True
                        spec_in_result.context['selected'] = is_selected_plot
        
        return final_results

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