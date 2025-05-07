from typing import Dict, Any, Optional, List
import logging
import yaml
import os
import re
import pandas as pd
import numpy as np

# Assuming HypoResult is defined
from .types import HypoResult, RegionAnomaly, MetricFormatting

logger = logging.getLogger(__name__)

class NarrativeEngine:
    """Generates human-readable narratives based on analysis results."""

    def __init__(self, hypothesis_configs: Dict[str, Any], data_registry=None, settings=None):
        self.hypothesis_configs = hypothesis_configs
        # data_registry is no longer needed here for key_numbers if passed via narrative_context
        self.data_registry = data_registry # Keep for now if other parts use it, but can be removed if only for key_numbers
        self.settings = settings or {}
        logger.info("NarrativeEngine initialized.")

    def generate_narrative(self, hypo_obj: HypoResult, anomaly: Optional[RegionAnomaly] = None) -> str:
        """Generate a narrative for a hypothesis result, optionally with anomaly context.
        
        Args:
            hypo_obj: Hypothesis result object, expected to have a `narrative_context` dict.
            anomaly: Optional anomaly context.
            
        Returns:
            A string narrative.
        """
        if not hypo_obj or not hypo_obj.name:
            logger.warning("NarrativeEngine: HypoResult or hypo_obj.name is missing, cannot generate narrative.")
            return ""

        # 1. Start with a base context from general hypo_obj and anomaly fields.
        ctx = {
            'hypo_name': hypo_obj.natural_name or hypo_obj.name,
            'hypothesis_name': hypo_obj.natural_name or hypo_obj.name, # For template consistency
            'score': f"{hypo_obj.score:.3f}" if hypo_obj.score is not None else "N/A",
            'hypo_value_fmt': hypo_obj.get_formatted_value(),
            'global_hypo_value_fmt': hypo_obj.get_formatted_global_value(),
            'hypo_deviation_description': hypo_obj.get_deviation_description(),
            'hypo_dir': MetricFormatting.get_direction(hypo_obj.value, hypo_obj.global_value),
            'hypo_delta': MetricFormatting.format_delta(hypo_obj.value, hypo_obj.global_value, hypo_obj.is_percentage),
            'ref_hypo_val': hypo_obj.get_formatted_global_value(),
        }

        if anomaly:
            ctx['region'] = anomaly.region
            ctx['metric_name'] = getattr(anomaly, 'metric_name', "the metric") # Get metric_name safely
            ctx['metric_dir'] = anomaly.dir
            ctx['metric_value_fmt'] = anomaly.get_formatted_value()
            ctx['metric_deviation'] = anomaly.get_deviation_description()
        else:
            # Provide fallbacks if anomaly is not present but template might expect these
            ctx.setdefault('region', "[Overall]")
            ctx.setdefault('metric_name', "the metric")
            ctx.setdefault('metric_dir', "N/A")
            ctx.setdefault('metric_value_fmt', "N/A")
            ctx.setdefault('metric_deviation', "N/A")

        # 2. Merge the handler-provided narrative_context.
        # This will overwrite any common keys from step 1 if the handler provides them,
        # which is intended as the handler's context is more specific.
        if hypo_obj.narrative_context:
            logger.debug(f"NarrativeEngine: Merging narrative_context from handler for {hypo_obj.name}: {list(hypo_obj.narrative_context.keys())}")
            ctx.update(hypo_obj.narrative_context)
        else:
            logger.warning(f"NarrativeEngine: No narrative_context provided by handler for {hypo_obj.name}.")

        # 3. Get template from the hypothesis config
        hypo_config = self.hypothesis_configs.get(hypo_obj.name, {})
        template = hypo_config.get('insight_template', '')
            
        logger.info(f"==================== NARRATIVE GENERATION ({hypo_obj.name}) ====================")
        # logger.debug(f"Narrative Context for {hypo_obj.name}: {ctx}") # Potentially very verbose

        if not template:
            logger.warning(f"No insight_template for '{hypo_obj.name}'. Using minimal template.")
            template = "{hypothesis_name} was evaluated for {metric_name} in {region}. Score: {score}. Summary: {hypo_deviation_description}."
        
        # 4. Ensure all keys required by the template are present, providing fallbacks.
        # Since handlers are now responsible for pre-formatting or providing "N/A",
        # this loop primarily catches major omissions or typos in keys.
        missing_keys = []
        current_template_keys = self._get_template_keys(template)
        for key_in_template in current_template_keys:
            if key_in_template not in ctx:
                missing_keys.append(key_in_template)
                logger.warning(f"NarrativeEngine: Key '{key_in_template}' in template for '{hypo_obj.name}' not found in final context. Using fallback string.")
                ctx[key_in_template] = f"[Data for {key_in_template} missing]"
        
        if missing_keys:
            logger.warning(f"NarrativeEngine: Final missing keys for '{hypo_obj.name}' before formatting: {missing_keys}")

        # 5. Format the template.
        # No need to dynamically alter template string anymore, as handlers pre-format.
        try:
            formatted_narrative = template.format(**ctx)
            return formatted_narrative.strip('\n')
        except KeyError as e:
            logger.error(f"NarrativeEngine: KeyError formatting narrative for {hypo_obj.name}. Missing key: {e}. Context had: {list(ctx.keys())}")
            return f"Error generating narrative for {hypo_obj.name}: key {e} not found."
        except Exception as e:
            logger.error(f"NarrativeEngine: Unexpected error formatting narrative for {hypo_obj.name}: {e}", exc_info=True)
            return f"Error generating narrative for {hypo_obj.name}."

    def _get_template_keys(self, template_string: str) -> List[str]:
        """Extract unique keys (identifiers) from a format string."""
        return [match.split(':')[0].split('!')[0] for match in re.findall(r'{(.*?)}', template_string) if match.isidentifier() or (match and match.split(':')[0].split('!')[0].isidentifier())] 