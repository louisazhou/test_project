from typing import Dict, Any, Optional
import logging
import yaml
import os
import re

# Assuming HypoResult is defined
from .types import HypoResult, RegionAnomaly, MetricFormatting

logger = logging.getLogger(__name__)

class NarrativeEngine:
    """Generates human-readable narratives based on analysis results."""

    def __init__(self, hypothesis_configs: Dict[str, Any], settings):
        self.hypothesis_configs = hypothesis_configs
        self.settings = settings
        logger.info("NarrativeEngine initialized.")

    def generate_narrative(self, hypo_obj: HypoResult, anomaly: Optional[RegionAnomaly] = None) -> str:
        """Generates a narrative for the hypothesis result.
        
        Args:
            hypo_obj: The hypothesis result object
            anomaly: Optional anomaly object providing context for the narrative
            
        Returns:
            String containing the generated narrative
        """
        if not hypo_obj or not hypo_obj.name:
            return ""
        
        # 1. Start with common template variables
        ctx = {
            'hypo_name': hypo_obj.natural_name or hypo_obj.name,
            'metric_name': "the metric",  # Default placeholder, will be updated from anomaly
            'score': f"{hypo_obj.score:.2f}" if hypo_obj.score is not None else "N/A"
        }
        
        # 2. Add hypo specific data
        ctx['value'] = hypo_obj.value
        ctx['global_value'] = hypo_obj.global_value
        ctx['hypo_value_fmt'] = hypo_obj.get_formatted_value()
        ctx['global_value_fmt'] = hypo_obj.get_formatted_global_value()
        ctx['hypo_deviation_description'] = hypo_obj.get_deviation_description()
        
        # Calculate additional fields using the centralized utility
        if hypo_obj.value is not None and hypo_obj.global_value is not None:
            # Get direction using utility
            direction = MetricFormatting.get_direction(hypo_obj.value, hypo_obj.global_value)
            ctx['dir'] = direction
            ctx['hypo_dir'] = direction  # Required for templates
            
            # Format delta using utility
            ctx['hypo_delta'] = MetricFormatting.format_delta(
                hypo_obj.value, hypo_obj.global_value, hypo_obj.is_percentage
            )
            
            # Add formatted global value as ref_hypo_val
            ctx['ref_hypo_val'] = hypo_obj.get_formatted_global_value()
        
        # 3. Add anomaly data if provided
        if anomaly:
            # Get region from anomaly
            ctx['region'] = anomaly.region
            
            # Get direction (higher/lower)
            ctx['metric_dir'] = anomaly.dir
            
            # Add formatted values for the metric
            ctx['metric_value_fmt'] = anomaly.get_formatted_value()
            
            # Use the deviation description as metric_deviation 
            ctx['metric_deviation'] = anomaly.get_deviation_description()
        
        # 4. Get template from the hypothesis config
        hypo_config = self.hypothesis_configs.get(hypo_obj.name, {})
        template = hypo_config.get('insight_template', '')
            
        # Log key context variables for debugging - more comprehensive
        logger.info(f"==================== NARRATIVE GENERATION ====================")
        logger.info(f"Generating narrative for hypothesis: {hypo_obj.name}")
        
        # 5. Use a minimal generic template if no specific template is found
        if not template:
            logger.warning(f"No insight_template found in config for hypothesis '{hypo_obj.name}'. Using minimal template.")
            # Minimal template uses keys from direct fields
            template = "The {region} region shows an anomaly in {metric_name}. This is potentially related to {hypo_name} ({hypo_deviation_description})."
        
        # 6. Ensure all keys required by the template are present, providing fallbacks
        missing_keys = []
        for template_key in self._get_template_keys(template):
            if template_key not in ctx:
                missing_keys.append(template_key)
                # Provide a default placeholder for any missing keys
                logger.warning(f"Missing required key '{template_key}' for template of hypothesis '{hypo_obj.name}'. Using fallback.")
                ctx[template_key] = f"[missing_{template_key}]"
        
        if missing_keys:
            logger.warning(f"Missing template keys: {missing_keys}")
        
        # 7. Format the template with the context variables
        try:
            formatted_narrative = template.format(**ctx)
            formatted_narrative = formatted_narrative.strip('\n')
            return formatted_narrative
        except KeyError as e:
            logger.error(f"Error formatting narrative template for {hypo_obj.name}: {e}")
            return f"Error generating narrative for {hypo_obj.name}: missing key {e}"
        except Exception as e:
            logger.error(f"Unexpected error formatting narrative for {hypo_obj.name}: {e}")
            return f"Error generating narrative for {hypo_obj.name}: {e}"

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