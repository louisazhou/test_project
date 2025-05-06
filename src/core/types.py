import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PlotSpec:
    """Specification for a plot to be rendered."""
    plot_key: str  # Identifier for the type of plot (e.g., 'metric_bar', 'hypothesis_heatmap')
    data_keys: List[str]  # Keys to retrieve raw DataFrames from DataRegistry
    extra_data: Dict[str, Any] = field(default_factory=dict)  # Additional computed data (deltas, p10, lift, etc.)
    ctx: Dict[str, Any] = field(default_factory=dict)  # Context: metric, region, hypothesis, title etc.

@dataclass
class HypoResult:
    """Result object containing evaluation details for a single hypothesis."""
    name: str
    type: str
    narrative: str
    key_numbers: Dict[str, Any]
    plots: List[PlotSpec]
    natural_name: Optional[str] = None  # Added to store hypothesis natural name directly
    plot_data: Optional[pd.DataFrame] = None  # Consider removing if not needed downstream
    score: Optional[float] = None
    display_rank: Optional[int] = None

@dataclass
class RegionAnomaly:
    """Summary of an anomaly identified in a specific region for a metric."""
    region: str
    dir: str  # "higher" or "lower"
    delta_pct: float
    z_score: float
    is_anomaly: bool = True  # Default to True since this is an anomaly object
    value: Optional[float] = None  # Added to store actual metric value
    formatted_value: Optional[str] = None  # Added to store formatted metric value
    deviation_description: Optional[str] = None  # Added for narrative generation
    hypo_results: List['HypoResult'] = field(default_factory=list)

@dataclass
class MetricReport:
    """Consolidated report for a single metric across all regions."""
    metric_name: str
    global_value: float
    natural_name: Optional[str] = None  # Added to store metric natural name
    is_percentage: bool = False  # Added to store metric format type
    formatted_global_value: Optional[str] = None  # Added formatted global value
    metric_data_key: Optional[str] = None
    metric_enrichment_data: Optional[Dict[str, Dict[str, Any]]] = None
    metric_std: Optional[float] = None  # Store the std dev for consistency between individual and summary plots
    anomalies: List[RegionAnomaly] = field(default_factory=list)  # Empty list if no anomalies
    metric_level_plots: List['PlotSpec'] = field(default_factory=list)  # Plots relevant to the overall metric (e.g., global bar chart)


class MetricFormatting:
    """Centralized utility for formatting values consistently across the pipeline.
    
    This class provides static methods to handle formatting of metrics and
    hypothesis values according to consistent rules. Use this instead of
    duplicating format logic across the codebase.
    """
    
    @staticmethod
    def is_percentage_metric(metric_name: str) -> bool:
        """Determine if a metric is a percentage based on name conventions."""
        return "_pct" in metric_name.lower()
    
    @staticmethod
    def format_delta(
        value: float, 
        reference: float, 
        is_percentage: bool = False, 
        include_sign: bool = False
    ) -> str:
        """Format a delta value consistently based on whether it's a percentage.
        
        Args:
            value: The current value
            reference: The reference value (e.g., global mean)
            is_percentage: Whether the value is a percentage (affects format)
            include_sign: Whether to include +/- sign
            
        Returns:
            Formatted delta string (e.g., "4.5pp" or "28.0%")
        """
        delta = value - reference
        sign = "+" if delta > 0 else "-" if delta < 0 else ""
        if not include_sign:
            sign = ""
            
        if is_percentage:
            # For percentage metrics, use absolute difference in percentage points (pp)
            # Note: values are already in decimal form (0.XX), so multiply by 100 to get percentage points
            return f"{sign}{abs(delta) * 100:.1f}pp"
        else:
            # For non-percentage metrics, use relative difference as percentage
            rel_diff = (delta / reference) if reference != 0 else 0
            return f"{sign}{abs(rel_diff) * 100:.1f}%"
    
    @staticmethod
    def format_value(value: float, is_percentage: bool = False) -> str:
        """Format a raw value based on its type.
        
        Args:
            value: The value to format
            is_percentage: Whether the value is a percentage
            
        Returns:
            Formatted value string (e.g., "45.2%" or "123.4")
        """
        if is_percentage:
            return f"{value * 100:.1f}%"
        else:
            return f"{value:.2f}"
    
    @staticmethod
    def get_direction(value: float, reference: float) -> str:
        """Get the direction of a comparison.
        
        Args:
            value: The current value
            reference: The reference value (e.g., global mean)
            
        Returns:
            Direction as string: "higher", "lower", or "similar"
        """
        if value > reference:
            return "higher"
        elif value < reference:
            return "lower"
        else:
            return "similar"
    
    @staticmethod
    def create_deviation_description(
        delta_fmt: str, 
        direction: str,
        reference_label: str = "the global average"
    ) -> str:
        """Create a standardized deviation description.
        
        Args:
            delta_fmt: Pre-formatted delta string (e.g., "4.5pp")
            direction: Direction string ("higher", "lower", "similar")
            reference_label: What to compare against (default: "the global average")
            
        Returns:
            Complete deviation description
        """
        if direction == "higher":
            return f"{delta_fmt} higher than {reference_label}"
        elif direction == "lower":
            return f"{delta_fmt} lower than {reference_label}"
        else:
            return f"at about the same level as {reference_label}" 