import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from ..core.data_catalog import DataCatalog
from ..core.data_registry import DataRegistry
from ..core.types import RegionAnomaly, HypoResult, PlotSpec, MetricFormatting

logger = logging.getLogger(__name__)

class Handler:
    """Handler for single-dimension hypotheses."""
    
    def __init__(self, hypothesis_config: Dict[str, Any], 
                 data_catalog: DataCatalog, 
                 data_registry: DataRegistry, 
                 settings: Dict[str, Any]):
        self.hypothesis_config = hypothesis_config
        self.data_catalog = data_catalog
        self.data_registry = data_registry
        self.settings = settings
        self.name = hypothesis_config.get('name', 'UnknownHypothesis')
        self.display_rank = 0 # Default, will be updated by engine
    
    def _load_hypothesis_data(self, input_data_configs: List[Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """Loads and prepares hypothesis data based on the first input_data config."""
        if not input_data_configs:
            logger.warning(f"No input_data configured for hypothesis {self.name}. Skipping data loading.")
            return None, None, None
        
        config = input_data_configs[0]
        source_name = config.get('dataset')
        value_col = config.get('columns', [None])[0]
        region_col = config.get('region_col', 'region')

        if not source_name or not value_col:
            logger.error(f"Hypothesis {self.name} input_data missing 'dataset' or 'columns'.")
            return None, None, None

        try:
            data_key = self.data_catalog.load(source_name)
            if not data_key:
                logger.error(f"Failed to load data for source '{source_name}' in {self.name}.")
                return None, None, None
            
            df_full = self.data_registry.get(data_key)
            if df_full is None or df_full.empty:
                logger.error(f"No data in registry for key '{data_key}' (source: '{source_name}') for {self.name}.")
                return None, None, None

            if value_col not in df_full.columns or region_col not in df_full.columns:
                logger.error(f"Hypo value or region column not in dataframe from source '{source_name}'.")
                return None, None, None

            df = df_full[[region_col, value_col]].copy()
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df = df.dropna(subset=[value_col])
            df = df.set_index(region_col)
            return df, value_col, region_col
        except Exception as e:
            logger.error(f"Error loading/processing data for hypothesis {self.name}: {e}", exc_info=True)
            return None, None, None

    def _calculate_hypothesis_values(self, df_hypo: pd.DataFrame, anomaly_region: str, hypo_value_col: str) -> Tuple[Optional[float], Optional[float]]:
        """Extracts hypothesis value for the anomaly region and the global/reference value."""
        if anomaly_region not in df_hypo.index:
            logger.warning(f"Anomaly region '{anomaly_region}' not in hypothesis '{self.name}' data.")
            return None, None
        hypo_val_region = df_hypo.loc[anomaly_region, hypo_value_col]

        ref_val_hypo = df_hypo.loc['Global', hypo_value_col] if 'Global' in df_hypo.index else df_hypo[hypo_value_col].mean()
        if 'Global' not in df_hypo.index:
            logger.warning(f"'Global' ref not in hypo '{self.name}' data. Using mean.")
        return hypo_val_region, ref_val_hypo

    def _calculate_score_components(self, metric_name: str, metric_data_key: str, 
                                   anomaly: RegionAnomaly, df_hypo: pd.DataFrame, hypo_val_region: float, ref_val_hypo: float, 
                                   hypo_value_col: str) -> Dict[str, Any]:
        """Calculates all components needed for the hypothesis score."""
        # Initialize raw_consistency to default value
        raw_consistency = 0.0
        
        # Deltas and Z-scores for hypothesis
        hypo_delta = (hypo_val_region - ref_val_hypo) / ref_val_hypo if ref_val_hypo != 0 else 0
        hypo_dir_calculated = MetricFormatting.get_direction(hypo_val_region, ref_val_hypo)
        hypo_std_dev = df_hypo.loc[df_hypo.index != 'Global', hypo_value_col].std()
        hypo_z_score = (hypo_val_region - ref_val_hypo) / hypo_std_dev if hypo_std_dev > 0 else 0

        # Load full metric_df for correlation-based consistency
        metric_df_full = self.data_registry.get(metric_data_key)
        if metric_df_full is None or metric_df_full.empty:
            logger.error(f"Could not load metric_df for '{self.name}'. Cannot calculate correlation.")
            
        else:
            hypo_region_col = df_hypo.index.name # Get region col name from df_hypo index
            if metric_df_full.index.name != hypo_region_col:
                if hypo_region_col in metric_df_full.columns:
                    metric_df_full = metric_df_full.set_index(hypo_region_col)
                else:
                    logger.error(f"Metric DF for {metric_data_key} missing region index/col '{hypo_region_col}'.")
                    
            if metric_name not in metric_df_full.columns:
                logger.error(f"Metric col '{metric_name}' not in metric_df for key {metric_data_key}.")   
            
            # Only try to calculate if we haven't set raw_consistency to 0.0 due to errors
            if raw_consistency == 0.0 and metric_df_full is not None and not metric_df_full.empty and hypo_region_col and metric_name in metric_df_full.columns:
                metric_series = metric_df_full[metric_name]
                hypo_series = df_hypo[hypo_value_col]
                common_index = metric_series.index.intersection(hypo_series.index).drop('Global', errors='ignore')
                metric_aligned = metric_series.loc[common_index].dropna()
                hypo_aligned = hypo_series.loc[common_index].dropna()
                common_aligned_index = metric_aligned.index.intersection(hypo_aligned.index)
                if len(common_aligned_index) > 1:
                    try:
                        raw_consistency = metric_aligned.loc[common_aligned_index].corr(hypo_aligned.loc[common_aligned_index])
                    except Exception:
                        raw_consistency = 0.0
                # else raw_consistency remains 0.0
        
        raw_consistency = raw_consistency if not np.isnan(raw_consistency) else 0.0
        consistency = abs(raw_consistency)

        abs_hypo_z = abs(hypo_z_score)
        hypo_z_score_norm = 1.0 if abs_hypo_z > 3 else (0.7 if abs_hypo_z > 2 else (0.6 if abs_hypo_z > 1 else 0.3))

        expected_direction = self.hypothesis_config.get('evaluation', {}).get('direction', 'same')
        consistency_sign = np.sign(raw_consistency) if raw_consistency != 0 else 0
        direction_alignment = 0.0
        if (expected_direction == 'opposite' and consistency_sign < 0) or \
           (expected_direction == 'same' and consistency_sign > 0):
            direction_alignment = 1.0

        metric_delta = anomaly.delta_pct
        explained_ratio = min(abs(hypo_delta) / abs(metric_delta), 1.0) if abs(metric_delta) > 1e-9 else 0.0
        
        return {
            "direction_alignment": direction_alignment,
            "consistency": consistency,
            "hypo_z_score_norm": hypo_z_score_norm,
            "explained_ratio": explained_ratio,
            "raw_consistency_correlation": raw_consistency,
            "hypo_delta": hypo_delta,
            "hypo_dir_calculated": hypo_dir_calculated,
            "hypo_z_score": hypo_z_score
        }

    def _calculate_final_score(self, score_components: Dict[str, Any]) -> float:
        weights = {
            'direction_alignment': self.hypothesis_config.get('score_weights', {}).get('direction_alignment', 0.3),
            'consistency': self.hypothesis_config.get('score_weights', {}).get('consistency', 0.3),
            'hypo_z_score_norm': self.hypothesis_config.get('score_weights', {}).get('hypo_z_score_norm', 0.2),
            'explained_ratio': self.hypothesis_config.get('score_weights', {}).get('explained_ratio', 0.2)
        }
        score = (
            score_components["direction_alignment"] * weights['direction_alignment'] +
            score_components["consistency"] * weights['consistency'] +
            score_components["hypo_z_score_norm"] * weights['hypo_z_score_norm'] +
            score_components["explained_ratio"] * weights['explained_ratio']
        )
        return max(0, min(score, 1.0))

    def _create_plot_spec(self, df_hypo: pd.DataFrame, anomaly_region: str, 
                          hypo_region_col: str, hypo_value_col: str, 
                          score_components: Dict[str, Any], final_score: float, metric_name: str) -> PlotSpec:
        hypo_natural_name = self.hypothesis_config.get('natural_name', self.name)
        return PlotSpec(
            plot_key='hypo_bar_scored',
            data=df_hypo,
            context={
                'hypothesis_name': self.name,
                'hypothesis_natural_name': hypo_natural_name,
                'metric_name': metric_name,
                'title': hypo_natural_name,  # Use natural name in title
                'y_label': hypo_natural_name,  # Use natural name for y-axis
                'region_col': hypo_region_col,
                'value_col': hypo_value_col, 
                'explaining_region': anomaly_region,
                'primary_region': anomaly_region,
                'score_components': {
                    'score': final_score,
                    'direction_alignment': score_components.get('direction_alignment', 0.0),
                    'consistency': score_components.get('consistency', 0.0),
                    'hypo_z_score_norm': score_components.get('hypo_z_score_norm', 0.0),
                    'explained_ratio': score_components.get('explained_ratio', 0.0)
                },
                'selected': False 
            }
        )
    
    def run(self, metric_name: str, 
            anomaly: RegionAnomaly, 
            metric_data_key: str,
            ) -> Tuple[Optional[HypoResult], Optional[PlotSpec]]:
        logger.info(f"Running single_dim handler for {self.name} on {metric_name} in {anomaly.region}")

        df_hypo, hypo_value_col, hypo_region_col = self._load_hypothesis_data(self.hypothesis_config.get('input_data', []))
        if df_hypo is None or hypo_value_col is None or hypo_region_col is None:
            return None, None

        hypo_val_region, ref_val_hypo = self._calculate_hypothesis_values(df_hypo, anomaly.region, hypo_value_col)
        if hypo_val_region is None or ref_val_hypo is None:
            return None, None

        # Confirm the RegionAnomaly object already contains metric information (from AnomalyGate)
        if not hasattr(anomaly, 'value') or not hasattr(anomaly, 'global_value'):
            logger.error(f"RegionAnomaly object is missing required attributes for {metric_name} in {anomaly.region}")
            return None, None
        
        score_components = self._calculate_score_components(
            metric_name, metric_data_key, anomaly, 
            df_hypo, hypo_val_region, ref_val_hypo, hypo_value_col
        )
        if not score_components: # If something went wrong in score calculation
             logger.error(f"Could not calculate score components for {self.name}")
             return None, None

        final_score = self._calculate_final_score(score_components)
        
        # Formatting for hypothesis values
        is_percent_hypo = MetricFormatting.is_percentage_metric(hypo_value_col)
        
        hypo_natural_name = self.hypothesis_config.get('natural_name', self.name)
        
        # Extract values directly from key_numbers
        value = hypo_val_region
        global_value = ref_val_hypo
        
        # Create the hypothesis result
        result = HypoResult(
            name=self.name,
            type="single_dim",
            narrative=None,  # Narrative will be generated separately
            score=final_score,
            display_rank=self.display_rank,
            natural_name=hypo_natural_name,
            value=value,
            global_value=global_value,
            is_percentage=is_percent_hypo,
            plot_data=df_hypo
        )
        
        hypo_plot_spec = self._create_plot_spec(df_hypo, anomaly.region, hypo_region_col, hypo_value_col, score_components, final_score, metric_name)
        
        logger.info(f"Hypothesis {self.name} for metric {metric_name} in {anomaly.region} scored: {final_score:.3f}")
        return result, hypo_plot_spec
