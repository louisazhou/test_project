from typing import Dict, Any, Optional
import logging

# Assuming HypoResult is defined
from .types import HypoResult, RegionAnomaly

logger = logging.getLogger(__name__)

class NarrativeEngine:
    """Generates human-readable narratives based on analysis results."""

    def __init__(self, hypothesis_configs: Dict[str, Any]):
        self.hypothesis_configs = hypothesis_configs
        logger.info("NarrativeEngine initialized.")

    def generate_narrative(self, hypo_obj: HypoResult, anomaly: Optional[RegionAnomaly] = None) -> str:
        """Generates a narrative for the hypothesis result.
        
        Args:
            hypo_obj: HypoResult object containing hypothesis evaluation data
            anomaly: Optional RegionAnomaly object linked to this hypothesis
        
        Returns:
            Generated narrative string
        """
        if hypo_obj is None:
            return ""
        
        # Create a new context for this specific narrative generation
        ctx = {} # Start with a fresh context

        # Use natural names for all name fields to match template expectations
        if hypo_obj.natural_name:
            ctx['hypo_name'] = hypo_obj.natural_name  # The natural name from HypoResult
        
        # 2. Add all pre-calculated and formatted key_numbers
        if hasattr(hypo_obj, 'key_numbers') and hypo_obj.key_numbers:
            # Add all the key numbers directly
            ctx.update(hypo_obj.key_numbers)
        
        # 3. Add anomaly data if provided
        if anomaly:
            # Get region from anomaly
            ctx['region'] = anomaly.region
            
            # Get direction (higher/lower)
            ctx['metric_dir'] = anomaly.dir
            
            # Add formatted values if available
            if hasattr(anomaly, 'formatted_value') and anomaly.formatted_value:
                ctx['metric_value_fmt'] = anomaly.formatted_value
                
            # Use the deviation description as metric_deviation 
            if hasattr(anomaly, 'deviation_description') and anomaly.deviation_description:
                ctx['metric_deviation'] = anomaly.deviation_description
                
            # Get metric natural name from anomaly if available
            if hasattr(anomaly, 'metric_natural_name') and anomaly.metric_natural_name:
                ctx['metric_name'] = anomaly.metric_natural_name
        
        # 4. Get template from the hypothesis config
        hypo_config = self.hypothesis_configs.get(hypo_obj.name, {})
        template = hypo_config.get('insight_template', '')
            
        # Log key context variables for debugging - more comprehensive
        logger.info(f"==================== NARRATIVE GENERATION ====================")
        logger.info(f"Generating narrative for hypothesis: {hypo_obj.name}")
        
        # 5. Use a minimal generic template if no specific template is found
        if not template:
            logger.warning(f"No insight_template found in config for hypothesis '{hypo_obj.name}'. Using minimal template.")
            # Minimal template now uses keys expected from key_numbers
            template = "The {region} region shows an anomaly in {metric_name}. This is potentially related to {hypo_name} ({hypo_deviation_description})."
        
        # 6. Ensure all keys required by the template are present, providing fallbacks
        missing_keys = []
        for template_key in self._get_template_keys(template):
            if template_key not in ctx:
                missing_keys.append(template_key)
                # Special handling for certain template keys with our restructured data
                if template_key == "metric_deviation" and 'deviation_description' in ctx:
                    # Use deviation_description as a fallback for metric_deviation
                    ctx["metric_deviation"] = ctx["deviation_description"]
                elif template_key == "ref_hypo_val" and "global_value_fmt" in ctx:
                    ctx["ref_hypo_val"] = ctx["global_value_fmt"]
                else:
                    logger.warning(f"Missing required key '{template_key}' for template of hypothesis '{hypo_obj.name}'. Using fallback.")
                    # Provide a noticeable fallback placeholder
                    ctx[template_key] = f"[missing_{template_key}]"
        
        if missing_keys:
            logger.warning(f"Missing template keys: {missing_keys}")
        
        # 7. Format the narrative
        try:
            narrative = template.format(**ctx)
            narrative = narrative.rstrip('\n') # Remove trailing empty lines
            
            # Log the generated narrative
            logger.info(f"FULL GENERATED NARRATIVE:\n{narrative}")
            logger.info(f"==================== END NARRATIVE ====================")
            
            hypo_obj.narrative = narrative
            return narrative
        except KeyError as e:
            logger.error(f"Missing key '{e}' in context when formatting template for hypothesis '{hypo_obj.name}'")
            hypo_obj.narrative = f"Could not generate narrative: missing {e} in context (available: {sorted(ctx.keys())})"
            return hypo_obj.narrative
        except Exception as ex:
            logger.error(f"Error formatting narrative for '{hypo_obj.name}': {ex}", exc_info=True)
            hypo_obj.narrative = f"Error generating narrative for {hypo_obj.name}."
            return hypo_obj.narrative

    def _get_template_keys(self, template):
        """Extract all keys used in a template string.
        
        This handles cases where curly braces are used within the template
        for purposes other than variable substitution.
        """
        import re
        # Find all keys within curly braces 
        # Use a more robust regex pattern that handles edge cases
        pattern = r'\{([^{}]+)\}'
        matches = re.findall(pattern, template)
        
        # Filter out any non-identifier strings (caused by formatting specs)
        keys = []
        for match in matches:
            # Remove format specifiers if present
            if ':' in match:
                match = match.split(':', 1)[0]
            # Strip whitespace
            match = match.strip()
            # Only add if it's a valid variable name
            if match and not match.startswith(('.', '!', '=', '+', '-', '*', '/')):
                keys.append(match)
        
        return keys 