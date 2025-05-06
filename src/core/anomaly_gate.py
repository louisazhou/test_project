import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from .types import RegionAnomaly, MetricFormatting

logger = logging.getLogger(__name__)

def is_anomalous(metric_val: float, global_val: float, std: float, threshold: float = 1.0) -> Tuple[bool, float]:
    """Check if a metric value is anomalous based on z-score threshold.
    
    Args:
        metric_val: Current metric value
        global_val: Global reference value
        std: Standard deviation
        threshold: Z-score threshold (default: 1.0)
        
    Returns:
        Tuple of (is_anomalous, z_score)
    """
    # Calculate z-score
    z_score = (metric_val - global_val) / std if std > 0 else 0
    
    # Check if absolute z-score exceeds threshold
    is_anomaly = abs(z_score) >= threshold
    
    return is_anomaly, z_score

class AnomalyGate:
    """Identifies significant anomalies based on multiple statistical criteria."""

    def __init__(self, z_thresh: float = 1.0, delta_thresh: float = 0.1):
        self.z_thresh = z_thresh
        self.delta_thresh = delta_thresh
        logger.info(f"AnomalyGate initialized with z_thresh={self.z_thresh}, delta_thresh={self.delta_thresh}")

    def _detect_anomaly_votes(self, 
                              all_values: pd.Series, 
                              current_value: float, 
                              ref_value: float
                             ) -> Tuple[float, float, float, int]:
        """Detect if a value is anomalous using multiple statistical methods and return votes."""
        votes = 0
        # Ensure all_values contains only numeric types for calculations
        numeric_values = pd.to_numeric(all_values, errors='coerce').dropna()
        if numeric_values.empty:
            logger.warning("No valid numeric values found for anomaly vote calculation.")
            return 0.0, 0.0, 0.0, 0 # Return defaults if no valid data

        # Calculate basic statistics
        mean = numeric_values.mean()
        std = numeric_values.std()
        z_score = (current_value - mean) / std if std > 0 else 0.0
        delta = (current_value - ref_value) / ref_value if ref_value != 0 else 0.0

        # 1. Z-score method
        if abs(z_score) >= self.z_thresh:
            votes += 1
            
        # 2. Delta method
        if abs(delta) >= self.delta_thresh:
            votes += 1
            
        # 3. IQR method (only if enough data)
        if len(numeric_values) >= 4:
            q1 = numeric_values.quantile(0.25)
            q3 = numeric_values.quantile(0.75)
            iqr = q3 - q1
            if iqr > 1e-9: # Avoid division by zero or issues with constant data
                 lower_bound = q1 - 1.5 * iqr
                 upper_bound = q3 + 1.5 * iqr
                 if current_value < lower_bound or current_value > upper_bound:
                     votes += 1
        
        # 4. 95% CI method (using std dev)
        if std > 0:
            lower_ci = mean - 1.96 * std
            upper_ci = mean + 1.96 * std
            if current_value < lower_ci or current_value > upper_ci:
                votes += 1
        
        # 5. 10/90 Percentile method (only if enough data)
        if len(numeric_values) >= 10: # Need enough points for percentiles
            p10 = numeric_values.quantile(0.1)
            p90 = numeric_values.quantile(0.9)
            if current_value < p10 or current_value > p90:
                votes += 1
        
        return z_score, std, delta, votes

    def find_anomalies(self, df: pd.DataFrame, metric_name: str, global_value: float, higher_is_better: bool, metric_natural_name: Optional[str] = None) -> Tuple[List[RegionAnomaly], Dict[str, Dict[str, Any]], float]:
        """Finds anomalies using voting and returns anomaly objects, enrichment details, and the overall metric standard deviation.
        
        Args:
            df: DataFrame containing the metric data
            metric_name: Technical name of the metric column
            global_value: Global reference value for the metric
            higher_is_better: Whether higher values are considered better for this metric
            metric_natural_name: Human-readable name of the metric (optional)
            
        Returns:
            Tuple of (anomalies, enrichment_data_map, overall_metric_std)
        """
        anomalies = []
        results = [] # Store intermediate results with votes
        enrichment_data_map = {} # Store enrichment details for all regions
        overall_metric_std = 0.0  # Initialize overall std for the metric

        if 'region' not in df.columns:
            logger.error("AnomalyGate requires 'region' column in the DataFrame.")
            return [], {}, 0.0
        if metric_name not in df.columns:
             logger.error(f"AnomalyGate requires metric column '{metric_name}' in the DataFrame.")
             return [], {}, 0.0

        all_values = df.loc[df['region'] != 'Global', metric_name]
        
        # Use natural name if provided, otherwise use technical name
        display_name = metric_natural_name if metric_natural_name else metric_name
        
        # Determine if this is a percentage metric for formatting
        is_percentage = MetricFormatting.is_percentage_metric(metric_name)
        
        first_call_to_detect = True
        for index, row in df.iterrows():
            current_region = row['region']
            if current_region == 'Global':
                continue

            current_value = row[metric_name]
            # std_from_detect is the overall std of the metric (excluding Global)
            z_score, std_from_detect, delta, votes = self._detect_anomaly_votes(all_values, current_value, global_value)
            
            if first_call_to_detect and std_from_detect > 0:
                overall_metric_std = std_from_detect
                first_call_to_detect = False

            # Use centralized direction determination
            anomaly_dir = MetricFormatting.get_direction(current_value, global_value)

            results.append({
                'region': current_region,
                'value': current_value,
                'z_score': z_score,
                'delta_pct': delta,
                'dir': anomaly_dir,
                'votes': votes,
                'is_anomaly': False # Determined later based on max votes
            })

        if not results:
            return [], {}, 0.0
            
        # Determine max votes
        max_votes = max(r['votes'] for r in results) if results else 0
        
        # Determine final anomaly status and create enrichment map
        for r in results:
             r['is_anomaly'] = (r['votes'] == max_votes and max_votes > 0)
             # Calculate good/bad based on final status and directionality
             is_bad = r['is_anomaly'] and ( (r['dir'] == 'higher' and not higher_is_better) or \
                                           (r['dir'] == 'lower' and higher_is_better) )
             is_good = r['is_anomaly'] and not is_bad
             
             # Store enrichment data for this region
             enrichment_data_map[r['region']] = {
                 'is_anomaly': r['is_anomaly'],
                 'z_score': r['z_score'],
                 'delta_pct': r['delta_pct'],
                 'anomaly_dir': r['dir'],
                 'good_anomaly': is_good,
                 'bad_anomaly': is_bad,
                 'votes': r['votes']
             }

             # Create RegionAnomaly object ONLY if it IS an anomaly
             if r['is_anomaly']:
                 # Create formatted values for this anomaly
                 formatted_value = MetricFormatting.format_value(r['value'], is_percentage)
                 delta_fmt = MetricFormatting.format_delta(r['value'], global_value, is_percentage)
                 deviation_description = MetricFormatting.create_deviation_description(
                     delta_fmt=delta_fmt,
                     direction=r['dir'],
                     reference_label="the global average"
                 )
                 
                 # Create the anomaly with all the formatted values
                 anomaly = RegionAnomaly(
                     region=r['region'],
                     dir=r['dir'],
                     delta_pct=r['delta_pct'],
                     z_score=r['z_score'],
                     is_anomaly=True # Explicitly True
                 )
                 
                 # Add additional fields needed for narrative generation
                 anomaly.value = r['value']
                 anomaly.formatted_value = formatted_value
                 anomaly.deviation_description = deviation_description
                 anomaly.metric_name = metric_name
                 anomaly.metric_natural_name = display_name
                 anomaly.delta_fmt = delta_fmt
                 anomalies.append(anomaly)
             
        logger.info(f"Anomaly detection complete for {metric_name}. Max votes: {max_votes}. Found {len(anomalies)} anomalies. Overall std: {overall_metric_std:.4f}")
        return anomalies, enrichment_data_map, overall_metric_std 