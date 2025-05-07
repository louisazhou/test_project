import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PlotSpec:
    """Specification for a plot to be rendered.
    
    PlotSpec provides a standardized way to define what should be plotted.
    
    Attributes:
        plot_key: Identifier for the type of plot (e.g., 'metric_bar', 'hypothesis_heatmap')
        data: The actual DataFrame to plot
        context: Combined context information: metric, region, hypothesis, formatting, and any additional data 
                needed by the plotting function
        data_key: Optional registry key to retrieve the DataFrame if not directly provided
    """
    plot_key: str
    data: Optional[pd.DataFrame] = None
    context: Dict[str, Any] = field(default_factory=dict)
    data_key: Optional[str] = None

@dataclass
class HypoResult:
    """Result object containing evaluation details for a single hypothesis.
    
    Attributes:
        name: Identifier for the hypothesis
        type: Type of hypothesis (e.g., 'single_dim', 'multi_dim')
        narrative: Generated text explaining the hypothesis evaluation
        score: Score value indicating how well the hypothesis explains the anomaly
        display_rank: Ranking position for display (lower is better)
        natural_name: Human-readable name of the hypothesis
        value: The value of the hypothesis for the affected region
        global_value: The global reference value for the hypothesis
        is_percentage: Whether the hypothesis value should be formatted as a percentage
        z_score: Z-score of the hypothesis value relative to other regions
        plot_data: The DataFrame with all regions' data for this hypothesis
    """
    name: str
    type: str
    narrative: str
    score: Optional[float] = None
    display_rank: Optional[int] = None
    natural_name: Optional[str] = None
    value: Optional[float] = None
    global_value: Optional[float] = None
    is_percentage: bool = False
    z_score: Optional[float] = None
    plot_data: Optional[pd.DataFrame] = None
    
    def get_formatted_value(self) -> str:
        """Get the formatted value of the hypothesis."""
        return MetricFormatting.format_value(self.value, self.is_percentage)
            
    def get_formatted_global_value(self) -> str:
        """Get the formatted global value of the hypothesis."""
        return MetricFormatting.format_value(self.global_value, self.is_percentage)
            
    def get_deviation_description(self) -> str:
        """Generate a description of how the value deviates from the global value."""
        return MetricFormatting.create_deviation_description(
            self.value, 
            self.global_value, 
            self.is_percentage
        )

@dataclass
class RegionAnomaly:
    """Summary of an anomaly identified in a specific region for a metric.
    
    Attributes:
        region: The region name where the anomaly was detected
        dir: Direction of the anomaly ("higher" or "lower")
        delta_pct: Percentage difference from the global average
        z_score: Statistical significance of the anomaly
        is_anomaly: Whether this is a true anomaly (always True for RegionAnomaly objects)
        value: Actual metric value for this region
        is_percentage: Whether the value should be formatted as a percentage
        global_value: The global reference value for comparison
        hypo_results: List of hypothesis evaluation results for this anomaly
        good_anomaly: Whether this is a good anomaly (default to False and will be populated by the anomaly gate)
        bad_anomaly: Whether this is a bad anomaly (default to False and will be populated by the anomaly gate)
    """
    region: str
    dir: str  # "higher" or "lower"
    delta_pct: float
    z_score: float
    is_anomaly: bool = True  # Default to True since this is an anomaly object
    value: Optional[float] = None
    is_percentage: bool = False
    global_value: Optional[float] = None
    hypo_results: List['HypoResult'] = field(default_factory=list)
    good_anomaly: bool = False #default to False
    bad_anomaly: bool = False #default to False
    
    def get_formatted_value(self) -> str:
        """Get the formatted value of the anomaly."""
        return MetricFormatting.format_value(self.value, self.is_percentage)
            
    def get_deviation_description(self) -> str:
        """Generate a description of how the value deviates from the global value."""
        if self.value is None or self.global_value is None:
            return f"{'above' if self.dir == 'higher' else 'below'} the global average"
            
        return MetricFormatting.create_deviation_description(
            self.value, 
            self.global_value, 
            self.is_percentage
        )

@dataclass
class MetricReport:
    """Consolidated report for a single metric across all regions.
    
    Attributes:
        metric_name: Identifier for the metric
        global_value: Global average value of the metric
        natural_name: Human-readable name of the metric
        is_percentage: Whether the metric should be formatted as a percentage
        metric_data_key: Registry key to retrieve the metric data
        metric_std: Standard deviation of the metric across regions
        anomalies: List of detected anomalies for this metric
    """
    metric_name: str
    global_value: float
    natural_name: Optional[str] = None
    is_percentage: bool = False
    metric_data_key: Optional[str] = None
    metric_std: Optional[float] = None
    anomalies: List[RegionAnomaly] = field(default_factory=list)
    
    def get_formatted_global_value(self) -> str:
        """Get the formatted global value of the metric."""
        return MetricFormatting.format_value(self.global_value, self.is_percentage)
            
    def get_anomaly_for_region(self, region: str) -> Optional[RegionAnomaly]:
        """Get the anomaly for a specific region if it exists."""
        for anomaly in self.anomalies:
            if anomaly.region == region:
                return anomaly
        return None


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
        if value is None or reference is None:
            return "N/A"
            
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
            if reference == 0:
                return "N/A"
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
        if value is None:
            return "N/A"
            
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
        if value is None or reference is None:
            return "unknown"
            
        if value > reference:
            return "higher"
        elif value < reference:
            return "lower"
        else:
            return "similar"
    
    @staticmethod
    def create_deviation_description(
        value: float,
        reference: float, 
        is_percentage: bool = False,
        reference_label: str = "the global average"
    ) -> str:
        """Create a standardized deviation description.
        
        Args:
            value: The current value
            reference: The reference value (e.g., global mean)
            is_percentage: Whether the value is a percentage
            reference_label: What to compare against (default: "the global average")
            
        Returns:
            Complete deviation description
        """
        if value is None or reference is None:
            return f"cannot be compared to {reference_label}"
            
        direction = MetricFormatting.get_direction(value, reference)
        delta_fmt = MetricFormatting.format_delta(value, reference, is_percentage)
        
        if direction == "higher":
            return f"{delta_fmt} higher than {reference_label}"
        elif direction == "lower":
            return f"{delta_fmt} lower than {reference_label}"
        else:
            return f"at about the same level as {reference_label}" 