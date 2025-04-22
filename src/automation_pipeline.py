import pandas as pd
import os
from data_processor import DataProcessor
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging

from visualization import RCAVisualizer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from presentation import generate_ppt, upload_to_drive

class EvaluateHypothesis:
    """Evaluates hypotheses to explain metric anomalies across regions.
    
    This class implements a statistical framework to:
    1. Detect anomalies in metrics using multiple methods (z-score, delta, IQR, CI)
    2. Score how well each hypothesis explains the anomaly
    3. Generate human-readable explanations using configurable templates
    
    The analysis supports both single-dimensional hypotheses (where the hypothesis metric's dimension 
    is the same as the monitored metric's dimension) and multi-dimensional hypotheses (where the 
    hypothesis metric has one more layer of dimension than the monitored metric).
    
    Key components:
    - Anomaly Detection: Centralized storage of anomaly detection results per metric-region pair
    - Score Calculation: Separate storage for hypothesis scoring results
    - Explanation Generation: Templates for human-readable explanations
    
    Attributes:
        z_thresh (float): Z-score threshold for anomaly detection
        delta_thresh (float): Threshold for significant deviation from global
        corr_weight (float): Weight for correlation in final score
        delta_weight (float): Weight for delta in final score
        max_expected_z (float): Maximum expected z-score for normalization
        hypothesis_configs (Dict[str, Any]): Hypothesis configuration dictionary
        metrics_config (Dict[str, Any]): Metrics configuration dictionary
        anomaly_detection_results (Dict[Tuple[str, str], Dict[str, Any]]): 
            Cached anomaly detection results by (metric, region)
        score_calculation_results (List[Dict[str, Any]]): 
            List of hypothesis scoring results
        final_analysis_summary (Dict[str, Dict[str, Any]]): Storage for final analysis summary per metric
    """
    
    def __init__(
        self,
        z_thresh: float = 1.0,
        delta_thresh: float = 0.10,
        hypothesis_configs: Optional[Dict[str, Any]] = None,
        metrics_config: Optional[Dict[str, Any]] = None,
        region_column: str = "L4"
    ) -> None:
        """Initialize the hypothesis evaluator with detection thresholds and scoring weights.
        
        Args:
            z_thresh: Z-score threshold for anomaly detection (default: 1.0)
            delta_thresh: Threshold for significant deviation from global (default: 0.10)
            hypothesis_configs: Dictionary mapping hypothesis names to their configurations
            region_column: Column name for regions (default: "L4")
        """
        self.z_thresh = z_thresh
        self.delta_thresh = delta_thresh
        self.max_expected_z = 3.0  # For z-score normalization
        self.hypothesis_configs = hypothesis_configs or {}
        self.metrics_config = metrics_config or {}
        self.region_column = region_column  
        # Centralized storage for anomaly detection results
        # Key: (metric, region), Value: {z_score, std, delta, votes, is_candidate_anomaly}
        self.anomaly_detection_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Storage for score calculation results
        self.score_calculation_results: List[Dict[str, Any]] = []
        self.final_analysis_summary: Dict[str, Dict[str, Any]] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def detect_anomaly(
        self, 
        values: pd.Series, 
        current_value: float, 
        ref_value: float
    ) -> Tuple[float, float, float, int]:
        """Detect if a value is anomalous using multiple statistical methods.
        
        Uses 5 methods to detect anomalies:
        1. Z-score: Deviation from mean in standard deviations
        2. Delta: Percentage difference from reference value
        3. IQR: Outlier detection using interquartile range
        4. 95% CI: Outside confidence interval
        5. 10/90 Percentile: Outside expected range
            
        Returns:
            Tuple of (z_score, std, delta, votes) where votes is the number of
            methods that detected an anomaly (0-5)
        """
        votes = 0
        
        # Calculate basic statistics
        mean = values.mean()
        std = values.std()
        z_score = (current_value - mean) / std if std > 0 else 0 # Handle zero std dev
        delta = (current_value - ref_value) / ref_value if ref_value > 0 else 0 # Handle zero ref value
        
        # 1. Z-score method
        if abs(z_score) > self.z_thresh:
            votes += 1
            
        # 2. Delta method
        if abs(delta) > self.delta_thresh:
            votes += 1
            
        # 3. IQR method
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        if current_value < lower_bound or current_value > upper_bound:
            votes += 1
        
        # 4. 95% CI method
        lower_ci = mean - 1.96 * std if std > 0 else mean
        upper_ci = mean + 1.96 * std if std > 0 else mean
        if current_value < lower_ci or current_value > upper_ci:
            votes += 1
        
        # 5. 10/90 Percentile method
        p10 = values.quantile(0.1)
        p90 = values.quantile(0.9)
        if current_value < p10 or current_value > p90:
            votes += 1
        
        return z_score, std, delta, votes

    def calculate_score(
        self,
        metric_series: pd.Series,
        hypo_series: pd.Series,
        metric_val: float,
        hypo_val: float,
        ref_metric_val: float,
        ref_hypo_val: float,
        hypothesis_name: str
    ) -> Tuple[float, float, float, float, float, bool]:
        """Calculate how well a hypothesis explains a metric anomaly based on new logic.
        
        The score combines:
        1. Direction Alignment (30%): Checks if metric and hypothesis deviations align with config.
        2. Consistency (30%): Absolute correlation between metric and hypothesis series.
        3. Hypothesis Z-score (20%): Normalized z-score of the hypothesis value.
        4. Explained Ratio (20%): How much of the metric deviation is explained by hypothesis deviation.
            
        Returns:
            Tuple of (direction_alignment, consistency, hypo_z_score_norm, explained_ratio, final_score, explains)
        """
        # Calculate deltas
        metric_delta = (metric_val - ref_metric_val) / ref_metric_val if ref_metric_val > 0 else 0
        hypo_delta = (hypo_val - ref_hypo_val) / ref_hypo_val if ref_hypo_val > 0 else 0
        
        # --- Calculate Raw Components ---
        
        # Raw Consistency (Correlation)
        common_index = metric_series.index.intersection(hypo_series.index)
        metric_aligned = metric_series.loc[common_index].dropna()
        hypo_aligned = hypo_series.loc[common_index].dropna()
        common_aligned_index = metric_aligned.index.intersection(hypo_aligned.index)
        raw_consistency = 0.0
        if len(common_aligned_index) > 1:
            try:
                # Calculate raw correlation (can be negative)
                raw_consistency = metric_aligned.loc[common_aligned_index].corr(hypo_aligned.loc[common_aligned_index])
            except Exception:
                raw_consistency = 0.0 # Handle cases where correlation fails
        raw_consistency = raw_consistency if not np.isnan(raw_consistency) else 0.0 # Ensure consistency is not NaN
        
        # Hypothesis Z-score
        hypo_std = hypo_series.std()
        hypo_z_score = (hypo_val - ref_hypo_val) / hypo_std if hypo_std > 0 else 0
        
        # --- Normalize/Score Components ---
        
        # 1. Direction Alignment (30%)
        direction_alignment = 0.0
        hypothesis_config = self.hypothesis_configs.get(hypothesis_name, {})
        expected_direction = hypothesis_config.get('evaluation', {}).get('direction', 'same') # Default to 'same'
        
        # Sign of consistency
        consistency_sign = np.sign(raw_consistency) if raw_consistency != 0 else 0
        
        if expected_direction == 'opposite':
            if consistency_sign < 0:
                direction_alignment = 1.0
        elif expected_direction == 'same':
            if consistency_sign > 0:
                direction_alignment = 1.0
        # Else, direction_alignment remains 0.0
        
        # 2. Consistency (30%)
        consistency = abs(raw_consistency) # Use absolute correlation for score
        
        # 3. Hypothesis Z-score Norm (20%)
        abs_hypo_z = abs(hypo_z_score)
        if abs_hypo_z > 3:
            hypo_z_score_norm = 1.0
        elif abs_hypo_z > 2:
            hypo_z_score_norm = 0.7
        elif abs_hypo_z > 1:
            hypo_z_score_norm = 0.6
        else:
            hypo_z_score_norm = 0.3
            
        # 4. Explained Ratio (20%)
        explained_ratio = min(abs(hypo_delta) / abs(metric_delta), 1.0) if abs(metric_delta) > 1e-6 else 0
        
        # --- Calculate Final Score ---
        final_score = (
            0.3 * direction_alignment +
            0.3 * consistency +
            0.2 * hypo_z_score_norm +
            0.2 * explained_ratio
        )
        
        # Determine if hypothesis explains the anomaly
        explains = final_score > 0.5 # Threshold remains 0.5
        
        return direction_alignment, consistency, hypo_z_score_norm, explained_ratio, final_score, explains

    def get_or_calculate_anomaly(
        self, 
        metric_series: pd.Series,
        region: str,
        metric: str,
        region_metric: float,
        ref_metric_val: float
    ) -> Dict[str, Any]:
        """Get existing anomaly detection result or calculate if not available.
        
        This method implements a caching mechanism for anomaly detection results.
        It first checks if results exist for the given metric-region pair, and if not,
        calculates and stores them.
        
        Args:
            metric_series: Series containing all values for the metric
            region: Region name
            metric: Metric name
            region_metric: Current region's metric value
            ref_metric_val: Global reference value for the metric
            
        Returns:
            Dictionary containing anomaly detection results including:
            - z_score: Number of standard deviations from mean
            - std: Standard deviation
            - delta: Percentage difference from global
            - votes: Number of detection methods that flagged anomaly
            - is_candidate_anomaly: Whether this might be an anomaly
            - is_anomaly: Whether this is confirmed as an anomaly
        """
        key = (metric, region)
        
        if key not in self.anomaly_detection_results:
            # Calculate anomaly detection if not already done
            z_score, std, delta, votes = self.detect_anomaly(
                metric_series, region_metric, ref_metric_val
            )
            is_candidate_anomaly = votes >= 2
            
            # Store the result
            self.anomaly_detection_results[key] = {
                self.region_column: region,
                "metric": metric,
                "metric_val": region_metric,
                "ref_metric_val": ref_metric_val,
                "z_score": z_score if not np.isnan(z_score) else 0,
                "std": std if not np.isnan(std) else 0,
                "delta": delta if not np.isnan(delta) else 0,
                "votes": votes,
                "is_candidate_anomaly": is_candidate_anomaly,
                "is_anomaly": False  # Will be updated in post-processing
            }
        
        return self.anomaly_detection_results[key]

    def analyze_single(
        self,
        region_df: pd.DataFrame,
        metric: str,
        hypothesis_name: str
    ) -> Optional[pd.DataFrame]:
        """Analyze a single-dimensional hypothesis against a metric.
        
        Evaluates how well a single hypothesis explains metric anomalies in different regions.
        The analysis includes:
        1. Calculating z-scores and deltas from global values
        2. Computing correlations between metric and hypothesis values
        3. Determining if the hypothesis direction matches the metric direction
        4. Generating a final score and explanation based on the analysis
        
        Args:
            region_df: DataFrame indexed by region containing metric and hypothesis values
            metric: Name of the metric column to analyze
            hypothesis_name: Name of the hypothesis being tested
            
        Returns:
            DataFrame containing analysis results with columns for z-scores, correlations,
            direction matches, and explanations. Returns None if analysis cannot be performed.
        """
        hypothesis_config = self.hypothesis_configs[hypothesis_name]
        value_column = hypothesis_config['input_data']['value_column']
        
        score_results = []
        metric_series = region_df[metric]
        hypo_series = region_df[value_column]
        ref_metric_val = region_df.loc["Global", metric]
        ref_hypo_val = region_df.loc["Global", value_column]

        for region in region_df.index:
            if region == "Global":
                continue

            region_metric = region_df.loc[region, metric]
            region_hypo = region_df.loc[region, value_column]

            # Get or calculate anomaly detection results
            anomaly_result = self.get_or_calculate_anomaly(
                metric_series, region, metric, region_metric, ref_metric_val
            )
            
            # Calculate score and get detailed components
            direction_alignment, consistency, hypo_z_score_norm, explained_ratio, score, is_candidate_RC = self.calculate_score(
                metric_series, hypo_series,
                region_metric, region_hypo,
                ref_metric_val, ref_hypo_val,
                hypothesis_name
            )
            
            # Calculate hypothesis delta for explanation
            hypo_delta = (region_hypo - ref_hypo_val) / ref_hypo_val if ref_hypo_val > 0 else 0
            
            # Initialize with empty reason and False explains
            reason = ""
            explains = False
            
            score_results.append({
                self.region_column: region,
                "metric": metric,
                "hypothesis": hypothesis_name,
                "metric_val": region_metric,
                "hypothesis_val": region_hypo,
                "ref_hypo_val": ref_hypo_val,
                "hypothesis_delta": hypo_delta,
                "direction_alignment": direction_alignment,
                "consistency": consistency,
                "hypo_z_score_norm": hypo_z_score_norm,
                "explained_ratio": explained_ratio,
                "score": score,
                "is_candidate_RC": is_candidate_RC,
                "explains": explains,
                "reason": reason
            })

        # Process results for this hypothesis
        if score_results:
            processed_results = self.process_results_and_format_explanations(score_results)
            if processed_results is not None:
                return processed_results

        return None

    def analyze_multi(
        self,
        multi_df: pd.DataFrame,
        metric: str,
        region_column: str
    ) -> Optional[pd.DataFrame]:
        """Analyze a multi-dimensional hypothesis against a metric.
        
        Evaluates hypotheses that involve multiple dimensions (e.g., product categories)
        by analyzing weighted averages and distributions. The analysis includes:
        1. Computing weighted averages for each region
        2. Analyzing distribution shifts between regions
        3. Identifying significant deviations from global patterns
        4. Generating explanations based on dimensional analysis
        
        Args:
            multi_df: DataFrame containing multi-dimensional data with regions and categories
            metric: Name of the metric to analyze
            region_column: Name of the column containing region information
            
        Returns:
            DataFrame containing analysis results with weighted averages, distribution
            insights, and explanations. Returns None if analysis cannot be performed.
        """
        all_results = []
        
        # Process all multi-dimensional hypotheses
        for hypothesis_name, hypothesis_config in self.hypothesis_configs.items():
            if hypothesis_config.get('type') != 'multi_dim_groupby':
                continue
            
            # Get the dimensional analysis config
            dim_config = hypothesis_config.get('dimensional_analysis', {})
            if not dim_config:
                self.logger.warning(f"Warning: No dimensional analysis config found for hypothesis {hypothesis_name}")
                continue
            
            # Get weight and value columns
            weight_col = None
            value_col = None
            # Use the column names defined in the config
            input_data_config = hypothesis_config.get('input_data', {})
            value_col = input_data_config.get('value_column')
            # Get weight column from dimensional analysis config if it exists
            dim_config = hypothesis_config.get('dimensional_analysis', {})
            if dim_config and 'metrics' in dim_config:
                for metric_config in dim_config.get('metrics', []):
                    if metric_config.get('is_weight', False):
                        weight_col = metric_config.get('column')
                        break
            
            if not value_col: # Weight col is optional
                self.logger.warning(f"Warning: Missing value column in hypothesis {hypothesis_name}")
                continue
            
            try:
                # --- Refactored Weighted Average Calculation --- 
                def weighted_average(group):
                    try:
                        return np.average(group[value_col], weights=group[weight_col] if weight_col else None)
                    except ZeroDivisionError: # Handle cases where weights sum to zero
                        return np.nan

                # Group by region and calculate weighted average
                weighted_avgs_series = multi_df.groupby(region_column).apply(weighted_average)
                weighted_avg_df = weighted_avgs_series.reset_index(name=f'{value_col}_weighted')
                
                # Calculate global weighted average separately (considering all data)
                ref_hypo_weighted_avg = weighted_average(multi_df)
                # --- End Refactor --- 
                
                # Get metric values for each region
                # Use first() as metric value should be consistent per region in this context
                metric_values = multi_df.groupby(region_column)[metric].first().reset_index()
                # Merge weighted averages with metric values
                analysis_data = pd.merge(weighted_avg_df, metric_values, on=region_column)
                
                score_results = []
                # Iterate through the combined analysis data
                for _, row in analysis_data.iterrows():
                    if row[region_column] == "Global": # Should not happen with groupby but good check
                        continue
                    
                    # Get or calculate anomaly detection results
                    # Need the original metric series for anomaly detection context
                    # Use the previously merged metric_values for simplicity here
                    metric_series_for_anomaly = analysis_data.set_index(region_column)[metric]
                    # Find the global metric value for reference
                    global_metric_row = analysis_data[analysis_data[region_column] == "Global"]
                    ref_metric_val = global_metric_row[metric].iloc[0] if not global_metric_row.empty else metric_series_for_anomaly.mean()
                    
                    anomaly_result = self.get_or_calculate_anomaly(
                        metric_series_for_anomaly, row[region_column], metric, row[metric], ref_metric_val
                    )
                    
                    # Calculate score using weighted values
                    direction_alignment, consistency, hypo_z_score_norm, explained_ratio, score, is_candidate_RC = self.calculate_score(
                        analysis_data.set_index(region_column)[metric], # Metric series
                        analysis_data.set_index(region_column)[f'{value_col}_weighted'], # Weighted hypo series
                        row[metric], row[f'{value_col}_weighted'], # Current region values
                        ref_metric_val, # Reference metric value
                        ref_hypo_weighted_avg, # Reference weighted hypothesis value
                        hypothesis_name # Pass hypothesis name
                    )
                    
                    # Calculate hypothesis delta for explanation
                    hypo_delta = (row[f'{value_col}_weighted'] - ref_hypo_weighted_avg) / ref_hypo_weighted_avg if ref_hypo_weighted_avg != 0 else 0
                    
                    # Initialize with empty reason and False explains
                    reason = ""
                    explains = False

                    score_results.append({
                        self.region_column: row[region_column],
                        "metric": metric,
                        "hypothesis": hypothesis_name,
                        "metric_val": row[metric],
                        "hypothesis_val": row[f'{value_col}_weighted'],
                        "ref_hypo_val": ref_hypo_weighted_avg,
                        "hypothesis_delta": hypo_delta,
                        "direction_alignment": direction_alignment,
                        "consistency": consistency,
                        "hypo_z_score_norm": hypo_z_score_norm,
                        "explained_ratio": explained_ratio,
                        "score": score,
                        "is_candidate_RC": is_candidate_RC,
                        "explains": explains,
                        "reason": reason
                    })

                # Process results for this hypothesis
                if score_results:
                    processed_results = self.process_results_and_format_explanations(score_results)
                    if processed_results is not None:
                        all_results.append(processed_results)
            
            except Exception as e:
                self.logger.error(f"Error in multi-dimensional analysis for hypothesis {hypothesis_name}: {str(e)}")
                continue
                
        # Combine all results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return None

    def process_results_and_format_explanations(
        self,
        results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Process analysis results and format human-readable explanations.
        
        This method:
        1. Consolidates results into a DataFrame
        2. Updates anomaly status based on vote ranking
        3. Generates explanations for significant anomalies
        4. Formats explanations using hypothesis templates
        
        The method combines information from both anomaly detection and
        hypothesis scoring to create comprehensive explanations.
        
        Args:
            results: List of dictionaries containing analysis results
        Returns:
            DataFrame with processed results including:
            - Updated anomaly and explanation flags
            - Formatted explanation text
            - Combined statistical measures
        """
        df = pd.DataFrame(results)
        if df.empty:
            return df
        
        # --- Refactored Explanation Generation --- 
        # 1. Update anomaly status first (this modifies self.anomaly_detection_results)
        self.update_anomaly_status_by_metric()
        
        # 2. Convert anomaly results dictionary to DataFrame for merging
        anomaly_list = []
        for (metric, region), anomaly_data in self.anomaly_detection_results.items():
            anomaly_list.append({
                'metric': metric,
                self.region_column: region,
                'is_anomaly': anomaly_data.get('is_anomaly', False),
                'delta': anomaly_data.get('delta', 0) # Need delta for explanation
            })
        anomaly_df = pd.DataFrame(anomaly_list)
        
        # 3. Merge score results with anomaly status
        merged_df = pd.merge(df, anomaly_df, on=['metric', self.region_column], how='left')
        # Fill NA for is_anomaly (if a score exists but no anomaly calculation happened?)
        merged_df['is_anomaly'].fillna(False, inplace=True)
        merged_df['delta'].fillna(0, inplace=True)

        # 4. Define a function to generate explanation text
        def generate_explanation(row):
            if row['is_anomaly'] and row['is_candidate_RC']:
                hypothesis_config = self.hypothesis_configs.get(row['hypothesis'], {})
                template = hypothesis_config.get('insight_template', '')
                metric_natural_name = self.metrics_config.get('metrics', {}).get(row['metric'], {}).get('natural_name', row['metric'])
                if ('ref_hypo_val' in row and '_pct' not in row['hypothesis']):
                    ref_hypo_val_formatted = f"{int(row.get('ref_hypo_val', 0))}"
                elif ('ref_hypo_val' in row and '_pct' in row['hypothesis']):
                    ref_hypo_val_formatted = f"{int(row.get('ref_hypo_val', 0)*100):}%"
                else:
                    ref_hypo_val_formatted = 'N/A'
                
                template_context = {
                    'region': row[self.region_column],
                    'metric_name': metric_natural_name,
                    'metric_deviation': int(abs(row['delta'] * 100)),
                    'metric_dir': 'lower' if row['delta'] < 0 else 'higher',
                    'hypo_dir': 'lower' if row.get('hypothesis_delta', 0) < 0 else 'higher',
                    'hypothesis_deviation': int(abs(row.get('hypothesis_delta', 0) * 100)),
                    'explained_ratio': int(row.get('explained_ratio', 0) * 100),
                    'final_score': int(row.get('score', 0) * 100),
                    'hypo_delta': f"{abs(row.get('hypothesis_delta', 0) * 100):.1f}%",
                    'ref_hypo_val': ref_hypo_val_formatted
                }
                
                if template:
                    try:
                        return template.format(**template_context)
                    except KeyError as e:
                        self.logger.warning(f"Missing key '{e}' in template context for hypothesis {row['hypothesis']}")
                        return "Explanation unavailable due to missing data."
                else:
                    # Fallback explanation
                    return (
                        f"In {row[self.region_column]}, the value is {row['hypothesis_val']:.1f}, "
                        f"{template_context['hypo_dir']} than Global. "
                        f"This may explain the {template_context['metric_dir']} trend in {row['metric']}."
                    )
            else:
                return "" # No explanation needed

        # 5. Apply the function to create 'reason' and update 'explains'
        merged_df['reason'] = merged_df.apply(generate_explanation, axis=1)
        merged_df['explains'] = merged_df['reason'] != ""
        # --- End Refactor --- 
        
        # Drop temporary columns added during merge if necessary (is_anomaly, delta from anomaly_df)
        return merged_df.drop(columns=['is_anomaly', 'delta'], errors='ignore')
    
    def update_anomaly_status_by_metric(self) -> None:
        """Update is_anomaly flag based on vote ranking for each metric.
        
        This method processes all anomaly detection results and marks true anomalies
        based on voting results. For each metric:
        1. Groups results by metric
        2. Finds the maximum vote count
        3. Marks all regions with max votes as anomalies
        """
        # Group anomaly detection results by metric
        metric_groups: Dict[str, List[Dict[str, Any]]] = {}
        for key, result in self.anomaly_detection_results.items():
            metric = result['metric']
            if metric not in metric_groups:
                metric_groups[metric] = []
            metric_groups[metric].append(result)
        
        # For each metric, mark regions with max votes as anomalies
        for metric, results in metric_groups.items():
            # Get maximum votes for this metric
            max_votes = max(result['votes'] for result in results)
            # Update is_anomaly for regions with max votes
            for result in results:
                if result['votes'] == max_votes:
                    result['is_anomaly'] = True
    
    def get_anomaly_detection_table(self) -> pd.DataFrame:
        """Return the anomaly detection results as a DataFrame.
            
        Returns:
            DataFrame containing all anomaly detection results with columns:
            region, metric, metric_val, ref_metric_val, z_score, std, delta,
            votes, is_candidate_anomaly, is_anomaly
        """
        return pd.DataFrame(list(self.anomaly_detection_results.values()))

    def finalize_all_analyses(self, all_intermediate_results: List[pd.DataFrame], metrics_config: Dict[str, Any]) -> None:
        """Consolidate all intermediate results, flag anomalies, and determine the final summary for each metric.

        Args:
            all_intermediate_results: List of DataFrames, one per hypothesis analysis run.
            metrics_config: The overall metrics configuration dictionary.
        """
        self.final_analysis_summary = {} # Reset summary

        if not all_intermediate_results:
            self.logger.warning("No intermediate results provided to finalize analysis.")
            return

        # --- Consolidate and get anomaly data ---
        final_df = pd.concat(all_intermediate_results, ignore_index=True)
        final_df = final_df.sort_values(['metric', self.region_column])
        
        # Update anomaly status based on votes (important before getting table)
        self.update_anomaly_status_by_metric()
        anomaly_df = self.get_anomaly_detection_table()

        if anomaly_df.empty:
            self.logger.warning("No anomaly detection results available during finalization.")
            # We might still want summaries even without anomalies, so don't return yet.

        # --- Add good/bad anomaly flags ---
        if not anomaly_df.empty and 'metric' in anomaly_df.columns:
            try:
                # Ensure metrics_config and the 'metrics' key exist
                if metrics_config and 'metrics' in metrics_config:
                    anomaly_df['higher_is_better'] = anomaly_df['metric'].apply(
                        lambda x: metrics_config['metrics'].get(x, {}).get('higher_is_better', False) # Default to False if metric missing
                    )
                else:
                     self.logger.warning("Metrics configuration is missing or malformed. Cannot determine 'higher_is_better'.")
                     anomaly_df['higher_is_better'] = False # Default

                is_anomaly = anomaly_df['is_anomaly']
                delta_positive = anomaly_df['delta'] > 0
                wants_higher = anomaly_df['higher_is_better']
                anomaly_df['good_anomaly'] = is_anomaly & ((wants_higher & delta_positive) | (~wants_higher & ~delta_positive))
                anomaly_df['bad_anomaly'] = is_anomaly & ((wants_higher & ~delta_positive) | (~wants_higher & delta_positive))
            except KeyError as e:
                 self.logger.error(f"Metric key error during anomaly flagging: {e}. Check metrics config structure.")
                 # Assign default values if error occurs
                 anomaly_df['good_anomaly'] = False
                 anomaly_df['bad_anomaly'] = False
            except Exception as e:
                 self.logger.error(f"Unexpected error during anomaly flagging: {e}")
                 anomaly_df['good_anomaly'] = False
                 anomaly_df['bad_anomaly'] = False
        else:
            # Ensure columns exist even if flagging fails or anomaly_df is empty initially
            if 'good_anomaly' not in anomaly_df.columns: anomaly_df['good_anomaly'] = False
            if 'bad_anomaly' not in anomaly_df.columns: anomaly_df['bad_anomaly'] = False


        # --- Determine summary for each metric ---
        for metric_name in final_df['metric'].unique():
            # Filter data for the current metric
            metric_anomaly_data = anomaly_df[anomaly_df['metric'] == metric_name].copy()
            metric_hypotheses_data = final_df[final_df['metric'] == metric_name].copy()

            # Initialize summary fields
            primary_region, best_hypothesis_name, explanation_text = None, None, ""
            metric_delta, metric_dir = 0.0, "neutral"
            hypo_delta, hypo_dir = 0.0, "neutral"
            metric_natural_name = metrics_config.get('metrics', {}).get(metric_name, {}).get('natural_name', metric_name)
            hypo_natural_name = ""


            if metric_anomaly_data.empty:
                self.logger.warning(f"No anomaly data found for metric '{metric_name}' during result processing.")
                # Still add an entry to the summary, but with limited info
                self.final_analysis_summary[metric_name] = {
                    'metric_anomaly_data': metric_anomaly_data, # Will be empty
                    'metric_hypotheses_data': metric_hypotheses_data,
                    'primary_region': "NoData",
                    'best_hypothesis_name': None,
                    'explanation_text': "No anomaly data available.",
                    'metric_natural_name': metric_natural_name,
                    'hypo_natural_name': None,
                    'metric_delta': None,
                    'metric_dir': None,
                    'hypo_delta': None,
                    'hypo_dir': None
                }
                continue # Skip to next metric

            # Find primary region and best hypothesis
            anomalous_regions_df = metric_anomaly_data[metric_anomaly_data['is_anomaly']]
            explaining_hypotheses = metric_hypotheses_data[metric_hypotheses_data['explains']]

            primary_region_row = None
            if not explaining_hypotheses.empty:
                best_explaining_row = explaining_hypotheses.loc[explaining_hypotheses['score'].idxmax()]
                primary_region = best_explaining_row[self.region_column]
                best_hypothesis_name = best_explaining_row['hypothesis']
                explanation_text = best_explaining_row.get('reason', '')
                # Get deltas/dirs for the best hypothesis in the primary region
                hypo_delta = best_explaining_row.get('hypothesis_delta', 0.0)
                hypo_natural_name = self.hypothesis_configs.get(best_hypothesis_name, {}).get('input_data', {}).get('natural_name', best_hypothesis_name)

                # Find the anomaly data row for the primary region
                primary_region_row = metric_anomaly_data[metric_anomaly_data[self.region_column] == primary_region]

            elif not anomalous_regions_df.empty:
                # Fallback to highest absolute z-score anomaly if no explaining hypothesis
                primary_region_row = anomalous_regions_df.loc[[anomalous_regions_df['z_score'].abs().idxmax()]] # Use double brackets to keep DataFrame structure
                primary_region = primary_region_row[self.region_column].iloc[0]
                explanation_text = f"Anomaly detected in {primary_region}, but no hypothesis strongly explains it."
            else:
                # No confirmed anomalies, but maybe still plot performance?
                primary_region = "NoAnomaly"
                explanation_text = "No significant anomalies detected for this metric."
                # Find the row for 'Global' or the first region if 'Global' isn't present for context
                primary_region_row = metric_anomaly_data[metric_anomaly_data[self.region_column] == "Global"]
                if primary_region_row.empty and not metric_anomaly_data.empty:
                     primary_region_row = metric_anomaly_data.iloc[[0]]


            # Extract metric delta and direction for the primary region (or fallback)
            if primary_region_row is not None and not primary_region_row.empty:
                 metric_delta = primary_region_row['delta'].iloc[0]
            else:
                 # Fallback if no primary region row found (should be rare)
                 metric_delta = 0.0
                 self.logger.warning(f"Could not determine metric delta for primary region of {metric_name}")


            # Determine directions
            metric_dir = 'lower' if metric_delta < 0 else ('higher' if metric_delta > 0 else 'neutral')
            hypo_dir = 'lower' if hypo_delta < 0 else ('higher' if hypo_delta > 0 else 'neutral')

            # Store the final summary for this metric
            self.final_analysis_summary[metric_name] = {
                'metric_anomaly_data': metric_anomaly_data,
                'metric_hypotheses_data': metric_hypotheses_data,
                'primary_region': primary_region,
                'best_hypothesis_name': best_hypothesis_name,
                'explanation_text': explanation_text,
                'metric_natural_name': metric_natural_name,
                'hypo_natural_name': hypo_natural_name if best_hypothesis_name else None,
                'metric_delta': metric_delta,
                'metric_dir': metric_dir,
                'hypo_delta': hypo_delta if best_hypothesis_name else None,
                'hypo_dir': hypo_dir if best_hypothesis_name else None
            }

        self.logger.info(f"Final analysis summary generated for {len(self.final_analysis_summary)} metrics.")

class RootCauseAnalysisPipeline:
    """Main pipeline for running root cause analysis on metrics data.
    
    This class orchestrates the end-to-end process of:
    1. Processing metrics and hypothesis data
    2. Running anomaly detection on metrics
    3. Evaluating hypotheses against anomalies
    4. Generating explanations and insights
    
    The pipeline handles both single-dimensional and multi-dimensional analyses:
    - Single-dimensional: Direct comparison of metrics and hypotheses
    - Multi-dimensional: Analysis of metrics against weighted averages of dimensional data
    
    Attributes:
        metrics_config (Dict[str, Any]): Configuration for metrics analysis
        hypothesis_configs (Dict[str, Any]): Configuration for hypothesis evaluation
        single_dim_df (pd.DataFrame): Processed single dimension data
        multi_dim_df (pd.DataFrame): Processed multi-dimensional data
        region_column (str): Name of the column containing region information
        intermediate_results (List[pd.DataFrame]): Storage for intermediate analysis results
        final_df (pd.DataFrame): Processed results with explanations
        anomaly_df (pd.DataFrame): Anomaly detection results
        evaluator (EvaluateHypothesis): Instance of hypothesis evaluator
    """
    
    def __init__(
        self,
        metrics_config: Dict[str, Any],
        hypothesis_configs: Dict[str, Any],
        single_dim_df: pd.DataFrame,
        multi_dim_df: pd.DataFrame,
        region_column: str = "L4"
    ) -> None:
        """Initialize the RCA pipeline with configurations and processed data.
        
        Args:
            metrics_config: Dictionary containing metrics configuration
            hypothesis_configs: Dictionary containing hypothesis configurations
            single_dim_df: Processed single dimension DataFrame
            multi_dim_df: Processed multi dimension DataFrame
            region_column: Name of the region column in the DataFrames
        """
        self.metrics_config = metrics_config
        self.hypothesis_configs = hypothesis_configs
        self.single_dim_df = single_dim_df
        self.multi_dim_df = multi_dim_df
        self.region_column = region_column
        
        # Initialize containers for results
        self.intermediate_results: List[pd.DataFrame] = []
        self.final_df: pd.DataFrame = pd.DataFrame()
        self.anomaly_df: pd.DataFrame = pd.DataFrame()
        
        # Initialize hypothesis evaluator with configs
        self.evaluator = EvaluateHypothesis(hypothesis_configs=hypothesis_configs, metrics_config=metrics_config, region_column=region_column)

        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def run_all_metrics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run analysis for all metrics, collect intermediate results, and finalize.

        This method:
        1. Iterates through all metrics and their hypotheses.
        2. Calls the appropriate analysis methods in EvaluateHypothesis.
        3. Collects intermediate results.
        4. Calls EvaluateHypothesis.finalize_all_analyses to process results.
        5. Stores final DataFrames (optional, could rely solely on evaluator state).
        
        Returns:
            Tuple containing:
            - final_df (pd.DataFrame): Concatenated intermediate results (before explanation formatting).
            - anomaly_df (pd.DataFrame): Anomaly detection results.
        """
        self.intermediate_results = [] # Reset intermediate results

        for metric_name, metric_config in self.metrics_config['metrics'].items():
            self.logger.info(f"\nAnalyzing metric: {metric_name}")
            
            # Get relevant hypotheses for this metric
            for hypothesis_name in metric_config['hypothesis']:
                hypothesis_config = self.hypothesis_configs[hypothesis_name]
                
                # Handle single dimension hypotheses
                if hypothesis_config['type'] == 'single_dim':
                    # Get the value column from hypothesis config
                    value_column = hypothesis_config.get('input_data', {}).get('value_column')
                    if not value_column:
                        self.logger.warning(f"Warning: No value column found in hypothesis config for {hypothesis_name}")
                        continue
                    
                    # Debug print
                    self.logger.info(f"\nDebug: Processing single dimension hypothesis {hypothesis_name}")
                    self.logger.info(f"Looking for columns: {metric_name}, {value_column}")
                    self.logger.info(f"Available columns in single_dim_df: {self.single_dim_df.columns.tolist()}")
                    
                    # Verify columns exist in the DataFrame
                    required_columns = [self.region_column, metric_name, value_column]
                    missing_columns = [col for col in required_columns if col not in self.single_dim_df.columns]
                    if missing_columns:
                        self.logger.warning(f"Warning: Missing columns {missing_columns} for hypothesis {hypothesis_name}")
                        continue
                    
                    # Select only needed columns and set index
                    analysis_df = self.single_dim_df[required_columns].copy()
                    analysis_df.set_index(self.region_column, inplace=True)
                    
                    # Debug print
                    self.logger.info(f"\nAnalysis DataFrame for {hypothesis_name}:")
                    self.logger.info(analysis_df)
                    
                    # Run single dimension analysis
                    results = self.evaluator.analyze_single(
                        region_df=analysis_df,
                        metric=metric_name,
                        hypothesis_name=hypothesis_name
                    )
                    if results is not None:
                        self.intermediate_results.append(results)
                
                # Handle multi-dimensional hypotheses
                elif hypothesis_config['type'] == 'multi_dim_groupby':
                    # Get configuration values
                    dim_config = hypothesis_config['dimensional_analysis']
                    dimension_column = dim_config['dimension_column']
                    
                    # Find weight and value columns from metrics configuration
                    weight_column = None
                    value_column = None
                    for metric in dim_config['metrics']:
                        if metric['is_weight']:
                            weight_column = metric['column']
                        else:
                            value_column = metric['column']
                    
                    if not weight_column or not value_column:
                        self.logger.warning(f"Warning: Missing weight or value column in {hypothesis_name} config")
                        continue
                    
                    # Debug print
                    self.logger.info(f"\nDebug: Processing multi dimension hypothesis {hypothesis_name}")
                    self.logger.info(f"Looking for columns: {metric_name}, {value_column}, {weight_column}, {dimension_column}")
                    self.logger.info(f"Available columns in multi_dim_df: {self.multi_dim_df.columns.tolist()}")
                    
                    # Verify columns exist in the DataFrame
                    required_columns = [
                        self.region_column,
                        dimension_column,
                        metric_name,
                        value_column,
                        weight_column
                    ]
                    missing_columns = [col for col in required_columns if col not in self.multi_dim_df.columns]
                    if missing_columns:
                        self.logger.warning(f"Warning: Missing columns {missing_columns} for hypothesis {hypothesis_name}")
                        continue
                    
                    # Select only needed columns
                    analysis_df = self.multi_dim_df[required_columns].copy()
                    
                    # Run multi-dimensional analysis
                    results = self.evaluator.analyze_multi(
                        multi_df=analysis_df,
                        metric=metric_name,
                    region_column=self.region_column
                )
                    if results is not None:
                        self.intermediate_results.append(results)
        
        # --- Consolidate intermediate results (still useful) --- 
        if self.intermediate_results:
            self.final_df = pd.concat(self.intermediate_results, ignore_index=True)
            self.final_df = self.final_df.sort_values(['metric', self.region_column])
        else:
            self.logger.warning("No intermediate results generated. Final DF is empty.")
            self.final_df = pd.DataFrame()

        # --- Finalize Analysis using Evaluator --- 
        self.evaluator.finalize_all_analyses(self.intermediate_results, self.metrics_config)
        
        # --- Get final anomaly results (still useful) --- 
        self.anomaly_df = self.evaluator.get_anomaly_detection_table()
        if self.anomaly_df.empty:
            self.logger.warning("No anomaly detection results available after finalization.")
        
        # Return the basic consolidated DataFrames (or modify return type if not needed)
        return self.final_df, self.anomaly_df # Or return None if these aren't used elsewhere

    def generate_visualizations(self, output_dir: str = "output") -> None:
        """Generate DETAILED visualizations based on analysis results.

        Creates a combined plot for each metric, showing the metric performance
        and related hypothesis plots (including non-root-cause ones).

        Args:
            output_dir: Directory to save visualization images.
        """
        # Access the final summary from the evaluator
        final_summary = self.evaluator.final_analysis_summary
        if not final_summary:
            self.logger.warning("No analysis summary found. Skipping detailed visualizations.")
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
            viz = RCAVisualizer()
            metrics = list(final_summary.keys())
            self.logger.info(f"Generating detailed visualizations for {len(metrics)} metrics")

            for metric_idx, (metric, summary_data) in enumerate(final_summary.items()):
                self.logger.info(f"--- Processing detailed: {metric} ---")

                # --- Extract Common Data --- 
                metric_anomaly_data = summary_data['metric_anomaly_data']
                metric_hypotheses_data = summary_data['metric_hypotheses_data']
                primary_region = summary_data['primary_region']
                best_hypothesis_name = summary_data['best_hypothesis_name']
                # Use description and prepend text
                top_explanation = summary_data.get('explanation_text', '')
                if best_hypothesis_name:
                    hypo_config = self.evaluator.hypothesis_configs.get(best_hypothesis_name, {})
                    desc = hypo_config.get('description', top_explanation)
                    top_explanation = f"Root cause of the anomaly: {desc}"
                metric_natural_name = summary_data.get('metric_natural_name', metric)
                hypo_natural_name = summary_data.get('hypo_natural_name', best_hypothesis_name)
                metric_delta = summary_data.get('metric_delta', 0.0)
                metric_dir = summary_data.get('metric_dir', 'neutral')
                hypo_delta = summary_data.get('hypo_delta', 0.0)
                hypo_dir = summary_data.get('hypo_dir', 'neutral')

                has_confirmed_anomalies = primary_region not in ["NoAnomaly", "NoData", None]
                has_root_cause = has_confirmed_anomalies and best_hypothesis_name is not None

                # --- Handle Performance Only Plot --- 
                if not has_confirmed_anomalies:
                    self.logger.info(f"No confirmed anomalies for {metric}. Plotting performance only.")
                    fig_perf, ax_perf = plt.subplots(figsize=(7, 4))
                    viz.plot_metric(ax_perf, metric_anomaly_data, metric_name=metric, region_column=self.region_column,
                                  z_score_threshold=self.evaluator.z_thresh, title=f"Performance: {metric_natural_name}", y_label=metric_natural_name)
                    plt.tight_layout()
                    fig_perf.savefig(os.path.join(output_dir, f"{metric}_performance.png"), bbox_inches='tight', dpi=150)
                    plt.close(fig_perf)
                    continue

                # --- Setup Detailed Layout --- 
                # (This is the full layout logic for the detailed visualization)
                fig = None; gs = None; ax_metric = None; ax_best_hypo = None; other_axes = []
                hypotheses_for_metric = metric_hypotheses_data['hypothesis'].unique()
                num_hypotheses = len(hypotheses_for_metric)
                other_hypotheses = []
                if best_hypothesis_name: 
                    other_hypotheses = [h for h in hypotheses_for_metric if h != best_hypothesis_name]
                else: 
                    other_hypotheses = list(hypotheses_for_metric)
                num_other_hypotheses = len(other_hypotheses)

                if best_hypothesis_name:
                    if num_hypotheses <= 1:
                        figsize = (8, 7); fig = plt.figure(figsize=figsize)
                        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
                        ax_metric = fig.add_subplot(gs[0]); ax_best_hypo = fig.add_subplot(gs[1])
                    elif num_hypotheses <= 3:
                        figsize = (12, 7); fig = plt.figure(figsize=figsize)
                        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.35, wspace=0.3)
                        ax_metric = fig.add_subplot(gs[0, 0]); ax_best_hypo = fig.add_subplot(gs[1, 0])
                        other_axes = [fig.add_subplot(gs[i, 1]) for i in range(min(2, num_other_hypotheses))]
                    else:
                        figsize = (15, 7); fig = plt.figure(figsize=figsize)
                        num_other_cols = 2
                        num_other_rows = (num_other_hypotheses + num_other_cols - 1) // num_other_cols
                        heights = [1] * max(2, num_other_rows)
                        gs = gridspec.GridSpec(max(2, num_other_rows), 1 + num_other_cols, 
                                              width_ratios=[2] + [1]*num_other_cols, 
                                              height_ratios=heights, hspace=0.4, wspace=0.35)
                        ax_metric = fig.add_subplot(gs[0, 0]); ax_best_hypo = fig.add_subplot(gs[1, 0])
                        for i in range(num_other_hypotheses):
                            r = i // num_other_cols; c = (i % num_other_cols) + 1
                            if r < gs.nrows and c < gs.ncols: 
                                other_axes.append(fig.add_subplot(gs[r, c]))
                            else: 
                                self.logger.warning(f"GridSpec index out of bounds for hypothesis {i} in {metric}")
                else: # No root cause found
                    figsize = (12, 4 * ((num_hypotheses + 1) // 2)); fig = plt.figure(figsize=figsize)
                    num_cols = 2
                    num_rows = (num_hypotheses + num_cols - 1) // num_cols
                    gs = gridspec.GridSpec(num_rows, 1 + num_cols, width_ratios=[2] + [1]*num_cols, hspace=0.4, wspace=0.3)
                    ax_metric = fig.add_subplot(gs[:, 0]); ax_best_hypo = None
                    for i in range(num_hypotheses):
                        r = i // num_cols; c = (i % num_cols) + 1
                        if r < gs.nrows and c < gs.ncols: 
                            other_axes.append(fig.add_subplot(gs[r, c]))
                        else: 
                            self.logger.warning(f"GridSpec index out of bounds for hypothesis {i} in {metric} (no root cause)")
                
                hypo_plot_idx = 1 # Start hypothesis numbering at 1
                
                if fig is None or ax_metric is None: # Check layout setup
                    self.logger.error(f"Detailed layout setup failed for {metric}.")
                    if fig: plt.close(fig)
                    continue

                # --- Plot Metric Data ---
                metric_axis_title = f"Metric: {metric_natural_name} ({primary_region} is {metric_delta:.1%} {metric_dir} than Global)"
                viz.plot_metric(ax_metric, metric_anomaly_data, metric_name=metric, region_column=self.region_column,
                                z_score_threshold=self.evaluator.z_thresh, title=metric_axis_title, y_label=metric_natural_name)
                # Add Anomaly Indicator if needed
                if has_confirmed_anomalies:
                    ax_metric.text(0.015, 0.97, "Anomaly Detected in KPI", ha='left', va='top', 
                                   transform=ax_metric.transAxes, fontsize=10, color='white', 
                                   bbox=dict(boxstyle='round,pad=0.3', fc='dodgerblue', alpha=1))

                # --- Plot Best Hypothesis (Root Cause) ---
                if has_root_cause and ax_best_hypo:
                    best_hypo_data = metric_hypotheses_data[metric_hypotheses_data['hypothesis'] == best_hypothesis_name]
                    # Construct two-line title for root cause with index
                    rc_title_line1 = f"Hypothesis {hypo_plot_idx}"
                    rc_title_line2 = f"{hypo_natural_name} ({primary_region} is {hypo_delta:.1%} {hypo_dir} than Global)"
                    rc_axis_title = f"{rc_title_line1}\n{rc_title_line2}" 
                    ax_best_hypo = viz.plot_hypothesis(ax_best_hypo, best_hypo_data, best_hypothesis_name,
                                      self.region_column, explaining_region=primary_region,
                                      primary_anomaly_region=primary_region, title=rc_axis_title, 
                                      y_label=hypo_natural_name, show_score_components=True)
                    hypo_plot_idx += 1 # Increment index after plotting
                    ax_best_hypo.text(0.015, 1.1, "[selected]", ha='left', va='top', 
                                   transform=ax_best_hypo.transAxes, fontsize=10, color='white', 
                                   bbox=dict(boxstyle='round,pad=0.3', fc='dodgerblue', alpha=1))

                # --- Plot Other Hypotheses ---
                for i, (ax_other, hypo_name) in enumerate(zip(other_axes, other_hypotheses)):
                    if ax_other is None: continue
                    hypo_config = self.evaluator.hypothesis_configs.get(hypo_name, {})
                    other_hypo_natural_name = hypo_config.get('input_data', {}).get('natural_name', hypo_name)
                    other_hypo_data = metric_hypotheses_data[metric_hypotheses_data['hypothesis'] == hypo_name]
                    # Construct two-line title for other hypotheses with index
                    other_title_line1 = f"Hypothesis {hypo_plot_idx}"
                    other_title_line2 = f"{other_hypo_natural_name}"
                    other_axis_title = f"{other_title_line1}\n{other_title_line2}"
                    viz.plot_hypothesis(ax_other, other_hypo_data, hypo_name, self.region_column,
                                      explaining_region=None, primary_anomaly_region=primary_region,
                                      title=other_axis_title, 
                                      y_label=other_hypo_natural_name, show_score_components=True)
                    hypo_plot_idx += 1 # Increment index after plotting
                    ax_other.text(0.8, 1.1, "[not selected]", ha='left', va='top', 
                                   transform=ax_other.transAxes, fontsize=10, color='black', 
                                   bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', alpha=1))

                # --- Add Figure-Level Annotations ---
                figure_title = f"{primary_region} for {metric_natural_name} is an Anomaly"
                fig.suptitle(figure_title, fontsize=16, y=0.98)

                # Use the explanation text directly from the summary
                top_explanation_from_summary = summary_data.get('explanation_text', '') # Get the raw text
                if top_explanation_from_summary:
                    # Display with larger font and adjusted position/bbox
                    fig.text(0.06, 0.93, top_explanation_from_summary, ha='left', va='top', fontsize=12, # Increased font size
                             style='italic', wrap=True, bbox=dict(boxstyle='round,pad=0.5', fc='white', lw=0.2)) # Increased padding

                viz.create_score_formula(fig)

                # --- Adjust Layout & Save ---
                fig.subplots_adjust(left=0.07, right=0.95, bottom=0.1, top=0.77, hspace=0.4, wspace=0.2)

                filename = os.path.join(output_dir, f"{metric}_{primary_region}_detailed_summary.png")
                fig.savefig(filename, bbox_inches='tight', dpi=150)
                plt.close(fig)
                self.logger.info(f"Saved detailed visualization: {filename}")

            self.logger.info(f"Detailed visualization generation complete.")
        except Exception as e:
            self.logger.error(f"Error generating detailed visualizations: {str(e)}", exc_info=True)

    def generate_visualizations_succinct(self, output_dir: str = "output") -> None:
        """Generate succinct visualizations using gridspec for layout.

        Metric/Root Cause plots on the left, Legends/Explanations on the right.
        Includes legends for bar colors/lines and explanations for score components.

        Args:
            output_dir: Directory to save visualization images.
        """
        final_summary = self.evaluator.final_analysis_summary
        if not final_summary:
            self.logger.warning("No analysis summary found. Skipping succinct visualizations.")
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
            viz = RCAVisualizer()
            metrics = list(final_summary.keys())
            self.logger.info(f"Generating succinct visualizations for {len(metrics)} metrics")

            for metric, summary_data in final_summary.items():
                self.logger.info(f"--- Processing succinct: {metric} ---")

                # --- Extract Common Data --- 
                metric_anomaly_data = summary_data['metric_anomaly_data']
                metric_hypotheses_data = summary_data['metric_hypotheses_data']
                primary_region = summary_data['primary_region']
                best_hypothesis_name = summary_data['best_hypothesis_name']
                # Get the raw explanation text
                top_explanation_from_summary = summary_data.get('explanation_text', '') 
                # Natural names and deltas
                metric_natural_name = summary_data.get('metric_natural_name', metric)
                hypo_natural_name = summary_data.get('hypo_natural_name', best_hypothesis_name)
                metric_delta = summary_data.get('metric_delta', 0.0)
                metric_dir = summary_data.get('metric_dir', 'neutral')
                hypo_delta = summary_data.get('hypo_delta', 0.0)
                hypo_dir = summary_data.get('hypo_dir', 'neutral')
                has_confirmed_anomalies = primary_region not in ["NoAnomaly", "NoData", None]
                has_root_cause = has_confirmed_anomalies and best_hypothesis_name is not None

                # --- Handle Performance Only Plot --- 
                if not has_confirmed_anomalies:
                    self.logger.info(f"No confirmed anomalies for {metric}. Plotting performance only (succinct).")
                    fig_perf, ax_perf = plt.subplots(figsize=(8, 7)) # Single axis is fine here
                    viz.plot_metric(ax_perf, metric_anomaly_data, metric_name=metric, region_column=self.region_column,
                                  z_score_threshold=self.evaluator.z_thresh, title=f"Performance: {metric_natural_name}", y_label=metric_natural_name)
                    plt.tight_layout()
                    fig_perf.savefig(os.path.join(output_dir, f"{metric}_performance.png"), bbox_inches='tight', dpi=150)
                    plt.close(fig_perf)
                    continue

                # --- Setup Advanced GridSpec Layout with Nested Grids --- 
                figsize = (12, 7) # Wider figure to accommodate side panels
                fig = plt.figure(figsize=figsize)
                
                # First create the main column division (2 columns)
                main_gs = gridspec.GridSpec(1, 2, width_ratios=[2.3, 1], wspace=0.05)
                
                # Create separate GridSpecs for each column
                # Left column: even 50/50 split for metric and root cause
                left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 0], 
                                                         height_ratios=[1, 1], hspace=0.35)
                
                # Right column: uneven split with more space for explanations
                right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 1], 
                                                          height_ratios=[1.5, 3.5], hspace=0.2)
                
                # Create the axes
                if has_root_cause:
                    ax_metric = fig.add_subplot(left_gs[0, 0])  # Top left
                    ax_best_hypo = fig.add_subplot(left_gs[1, 0])  # Bottom left
                else: 
                    # For metric-only view, span both rows in left column
                    ax_metric = fig.add_subplot(left_gs[:, 0])
                    ax_best_hypo = None
                
                # Right column axes remain the same conceptually
                ax_legend = fig.add_subplot(right_gs[0, 0])  # Top right
                ax_score_expl = fig.add_subplot(right_gs[1, 0])  # Bottom right
                
                if not has_root_cause:
                    # Hide the bottom-right axis if no root cause plot
                    ax_score_expl.set_visible(False)

                # --- Plot Metric Data --- 
                metric_axis_title = f"Metric: {metric_natural_name} ({primary_region} is {abs(metric_delta):.1%} {metric_dir} than Global)"
                viz.plot_metric(ax_metric, metric_anomaly_data, metric_name=metric, region_column=self.region_column,
                                z_score_threshold=self.evaluator.z_thresh, title=metric_axis_title, y_label=metric_natural_name)
                # Add Anomaly Indicator if needed
                if has_confirmed_anomalies:
                    ax_metric.text(0.02, 0.95, "Anomaly Detected for KPI", ha='left', va='top', 
                                   transform=ax_metric.transAxes, fontsize=9, color='red', 
                                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

                # --- Plot Best Hypothesis (Root Cause) --- 
                if has_root_cause and ax_best_hypo:
                    best_hypo_data = metric_hypotheses_data[metric_hypotheses_data['hypothesis'] == best_hypothesis_name]
                    # Construct two-line title for root cause
                    rc_title_line = f"{hypo_natural_name} ({primary_region} is {abs(hypo_delta):.1%} {hypo_dir} than Global)"
                    rc_axis_title = f"{rc_title_line}" # Use newline character
                    viz.plot_hypothesis(ax_best_hypo, best_hypo_data, best_hypothesis_name,
                                      self.region_column, explaining_region=primary_region,
                                      primary_anomaly_region=primary_region, title=rc_axis_title, # Use new title
                                      y_label=hypo_natural_name, show_score_components=True)
                                      
                # --- Plot Legends and Explanations in Dedicated Axes --- 
                viz.plot_color_legend(ax_legend)
                if has_root_cause: # Only show score explanation if there's a root cause plot
                     viz.plot_score_component_explanations(ax_score_expl)
                
                # --- Add Figure-Level Annotations --- 
                figure_title = f"{primary_region} for {metric_natural_name} is an Anomaly"
                fig.suptitle(figure_title, fontsize=16, y=0.98)

                # Use the explanation text directly from the summary
                if top_explanation_from_summary:
                    # Display with larger font and adjusted position/bbox
                    fig.text(0.10, 0.93, top_explanation_from_summary, ha='left', va='top', fontsize=12, # Increased font size
                             style='italic', wrap=True, bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', lw=0.5)) # Increased padding

                viz.create_score_formula(fig)

                # --- Adjust Layout & Save --- 
                fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.88, hspace=0.35, wspace=0.15)

                filename_suffix = "succinct_summary"
                # Correct determination of filename based on root cause and anomaly status
                if not has_root_cause:
                    if has_confirmed_anomalies:
                         # Anomaly exists, but no hypothesis explains it
                         primary_region = "AnomalyOnly" 
                         filename_suffix = "anomaly_only"
                    else:
                         # No anomaly confirmed (e.g., primary_region is 'NoAnomaly' or 'NoData')
                         primary_region = "NoAnomaly"
                         filename_suffix = "performance"
                
                # Construct filename using updated primary_region and suffix
                filename = os.path.join(output_dir, f"{metric}_{primary_region}_{filename_suffix}.png")
                fig.savefig(filename, bbox_inches='tight', dpi=150)
                plt.close(fig)
                self.logger.info(f"Saved succinct visualization: {filename}")

            self.logger.info(f"Succinct visualization generation complete.")
        except Exception as e:
            self.logger.error(f"Error generating succinct visualizations: {str(e)}", exc_info=True)

# Update main script to save multi-dimensional data as well
if __name__ == "__main__":
    """Main script for running the RCA pipeline.
    
    This script:
    1. Sets up necessary directories
    2. Initializes the data processor (loading configs and settings)
    3. Prepares input data
    4. Runs the RCA pipeline
    5. Saves results to output files
    6. Generates visualizations (detailed or succinct based on settings)
    7. Optionally generates a PowerPoint presentation
    8. Optionally uploads the PowerPoint to Google Drive
    """
    # Initialize paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, 'config')
    input_dir = os.path.join(base_dir, 'input')
    tmp_dir = os.path.join(base_dir, 'tmp')
    output_dir = os.path.join(base_dir, 'output') # Define output_dir earlier

    # --- Setup Logging --- 
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__) # Logger for the main script
    logger.info("--- Starting RCA Pipeline --- ")

    # Create tmp and output directories if they don't exist
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DataProcessor
    logger.info("Initializing Data Processor...")
    processor = DataProcessor(config_dir, input_dir, region_column="L4")
    settings = processor.get_settings() # Get the loaded settings
    
    # Load and process data
    logger.info("Preparing data...")
    single_dim_df, multi_dim_df = processor.prepare_data()
    
    # Initialize and run the pipeline
    logger.info("Initializing RCA Pipeline...")
    pipeline = RootCauseAnalysisPipeline(
        processor.get_metrics_config(),
        processor.get_hypothesis_configs(),
        single_dim_df,
        multi_dim_df,
        region_column=processor.region_column
    )

    logger.info("Running analysis for all metrics...")
    pipeline.run_all_metrics()

    # Save output tables
    logger.info("Saving analysis results...")
    try:
        # Use final_df and anomaly_df from the evaluator if needed, but they are also stored in pipeline
        # Example: Saving the final summary might be more useful than raw intermediate
        # For now, keep saving the pipeline's final_df and anomaly_df
        if not pipeline.final_df.empty:
            debug_output_path = os.path.join(tmp_dir, 'rca_intermediate_results.csv')
            pipeline.final_df.to_csv(debug_output_path, index=False)
            logger.info(f"Intermediate results saved to: {debug_output_path}")
        else: logger.warning("Pipeline final_df is empty, skipping save.")
        
        if not pipeline.anomaly_df.empty:
            anomaly_output_path = os.path.join(tmp_dir, 'rca_anomaly_detection.csv')
            pipeline.anomaly_df.to_csv(anomaly_output_path, index=False)
            logger.info(f"Anomaly detection results saved to: {anomaly_output_path}")
        else: logger.warning("Pipeline anomaly_df is empty, skipping save.")

        # Maybe save the final summary from the evaluator?
        if pipeline.evaluator.final_analysis_summary:
             summary_list = []
             for metric, data in pipeline.evaluator.final_analysis_summary.items():
                 summary_list.append({
                     'metric': metric,
                     'primary_region': data['primary_region'],
                     'best_hypothesis': data['best_hypothesis_name'],
                     'explanation': data['explanation_text'],
                     'metric_natural_name': data['metric_natural_name'],
                     'hypo_natural_name': data['hypo_natural_name'],
                     'metric_delta': data['metric_delta'],
                     'metric_dir': data['metric_dir'],
                     'hypo_delta': data['hypo_delta'],
                     'hypo_dir': data['hypo_dir'],
                 })
             summary_df = pd.DataFrame(summary_list)
             summary_output_path = os.path.join(output_dir, 'rca_final_summary.csv')
             summary_df.to_csv(summary_output_path, index=False)
             logger.info(f"Final analysis summary saved to: {summary_output_path}")
             
    except Exception as e:
        logger.error(f"Error saving output files: {e}")


    # --- Generate Visualizations (Based on Settings) --- 
    logger.info(f"Generating visualizations (type: {settings['visualization_type']})...")
    if settings['visualization_type'] == 'succinct':
        pipeline.generate_visualizations_succinct(output_dir)
    elif settings['visualization_type'] == 'detailed':
        pipeline.generate_visualizations(output_dir)
    else:
        logger.warning(f"Unknown visualization type '{settings['visualization_type']}'. Defaulting to detailed.")
        pipeline.generate_visualizations(output_dir) # Default to detailed

    # --- Generate and Upload PPT (Based on Settings) --- 
    ppt_path = None
    if settings.get('generate_ppt', False): # Check the setting
        logger.info("Generating PowerPoint presentation...")
        if pipeline.evaluator.final_analysis_summary: # Check if there are results
            ppt_filename = "RCA_Summary.pptx"
            # Pass the final summary from the evaluator and the visualization type
            ppt_path = generate_ppt(
                pipeline.evaluator.final_analysis_summary, 
                output_dir, 
                ppt_filename,
                visualization_type=settings['visualization_type'] # Pass the visualization type
            )
            if ppt_path:
                logger.info(f"PowerPoint presentation generated: {ppt_path}")
            else:
                logger.error("Failed to generate PowerPoint path.")
        else:
            logger.warning("Skipping PowerPoint generation: No analysis results found.")
    else:
        logger.info("Skipping PowerPoint generation based on settings.")

    # Upload if PPT was generated and setting is true
    if ppt_path and settings.get('upload_to_drive', False):
        logger.info("Uploading PowerPoint to Google Drive...")
        drive_folder_id = settings.get('drive_folder_id') # Get folder ID from settings
        if drive_folder_id == 'null': # Handle null string from YAML
            drive_folder_id = None 
            
        file_id = upload_to_drive(ppt_path, folder_id=drive_folder_id)
        if file_id:
            logger.info(f"PowerPoint uploaded to Google Drive. File ID: {file_id}")
        else:
            logger.error("Failed to upload PowerPoint to Google Drive.")
    elif ppt_path: # If PPT generated but upload setting is false
        logger.info("Skipping Google Drive upload based on settings.")

    logger.info("--- RCA Pipeline Finished ---")