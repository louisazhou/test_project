import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Dict, Any, Optional, Tuple, List
import logging
import os
import matplotlib.pyplot as plt

from ..core.data_catalog import DataCatalog
from ..core.data_registry import DataRegistry
from ..core.types import RegionAnomaly, HypoResult, PlotSpec
from ..plotting.hypothesis_plots.l8_concentration import l8_concentration_plot
from ..plotting import plot_router

logger = logging.getLogger(__name__)

class Handler:
    """Handler for depth_spotter hypothesis - identify L8 concentration patterns."""
    
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
    
    def run(self, metric_name: str, 
            anomaly: RegionAnomaly, 
            metric_data_key: str,
            ) -> Tuple[Optional[HypoResult], Optional[PlotSpec]]:
        """Execute depth_spotter analysis on L8 territory data."""
        logger.info(f"Running depth_spotter handler for {self.name} on {metric_name} in {anomaly.region}")

        # 1. Load data
        input_data_configs = self.hypothesis_config.get('input_data', [])
        if not input_data_configs:
            logger.warning(f"No input_data configured for hypothesis {self.name}. Skipping data loading.")
            return None, None
            
        config = input_data_configs[0]
        dataset_name = config.get('dataset')
        columns = config.get('columns', [])
        
        if not dataset_name or len(columns) < 3:
            logger.error(f"Hypothesis {self.name} input_data missing 'dataset' or requires at least 3 columns.")
            return None, None
            
        try:
            # Extract column names
            metric_col = columns[0]  # First column is the metric value
            region_col = columns[1]  # Second column is the region (L4)
            subregion_col = columns[2]  # Third column is the subregion (L8)
            
            # Load dataset using the DataCatalog.load method
            data_key = self.data_catalog.load(dataset_name, columns) # Pass all required columns
            if data_key is None:
                logger.error(f"Failed to load data for source '{dataset_name}' in {self.name}.")
                return None, None
                
            # Get the actual dataframe from the registry
            full_df = self.data_registry.get(data_key) # Renamed to full_df for clarity
            if full_df is None or full_df.empty:
                logger.error(f"Empty dataframe or failed to retrieve from registry for {self.name}.")
                return None, None
            
            # Ensure all required columns are present after potential renaming by DataCatalog
            required_final_columns = [metric_col, region_col, subregion_col]
            if not all(col in full_df.columns for col in required_final_columns):
                logger.error(f"Required columns {required_final_columns} missing in dataframe for {self.name} after loading. Available: {list(full_df.columns)}.")
                return None, None
                
            primary_region_name = anomaly.region # The anomalous region
            region_df_for_stats = full_df[full_df[region_col] == primary_region_name]
            global_values = full_df[full_df[region_col] != 'Global'][metric_col].dropna().values # Use full_df here
            
            # 2. Apply the depth_spotter algorithm
            values_for_stats = region_df_for_stats[metric_col].dropna().values
            
            if len(values_for_stats) < 2:
                logger.warning(f"Not enough L8 data points for primary region {primary_region_name} for stats. At least 2 required.")
                # We might still want to plot if other regions have data, so don't return yet unless full_df is too small overall
                # return None, None 
                
            # Calculate statistics for the primary anomalous region
            l4_mean = np.mean(values_for_stats) if len(values_for_stats) > 0 else np.nan
            std_l4 = np.std(values_for_stats) if len(values_for_stats) > 0 else np.nan
            skew_l4 = skew(values_for_stats) if len(values_for_stats) > 1 else np.nan # skew needs at least 2 points
            global_std = np.std(global_values) if len(global_values) > 0 else np.nan
            skew_trend = "left-skewed" if skew_l4 < 0 else ("right-skewed" if skew_l4 > 0 else "symmetric")
            
            higher_is_better = True  # Default assumption
            metrics_config = self.data_registry._instance_store.get('metrics_config', {})
            if metric_name in metrics_config:
                higher_is_better = metrics_config.get(metric_name, {}).get('higher_is_better', True)
            
            # Calculate concentration statistics for the primary region
            lagged_sub_regions = []
            low_l8_ratio = np.nan
            p10_all_territory = np.nan

            if global_values.size > 0:
                if higher_is_better:
                    p10_all_territory = np.percentile(global_values, 10)
                    if len(values_for_stats) > 0:
                        low_l8_ratio = np.mean(values_for_stats < p10_all_territory)
                        lagged_sub_regions = region_df_for_stats[region_df_for_stats[metric_col] < p10_all_territory] \
                            .sort_values(by=metric_col) \
                            .iloc[:min(3, len(region_df_for_stats))][subregion_col].tolist()
                else:
                    p10_all_territory = np.percentile(global_values, 90)
                    if len(values_for_stats) > 0:
                        low_l8_ratio = np.mean(values_for_stats > p10_all_territory)
                        lagged_sub_regions = region_df_for_stats[region_df_for_stats[metric_col] > p10_all_territory] \
                            .sort_values(by=metric_col, ascending=False) \
                            .iloc[:min(3, len(region_df_for_stats))][subregion_col].tolist()
            
            bottom_l8_names = ', '.join(lagged_sub_regions) if lagged_sub_regions else "None identified"
            
            # Calculate score based on stats for the primary region
            score_weight_std = 0.3
            score_weight_skew = 0.3
            score_weight_ratio = 0.4
            
            std_score = min(1.0, std_l4 / (global_std * 2)) if global_std > 0 and not np.isnan(std_l4) and not np.isnan(global_std) else 0.5
            skew_score = abs(min(1.0, skew_l4 / 2)) if not np.isnan(skew_l4) and skew_l4 != 0 else 0
            ratio_score = min(1.0, low_l8_ratio * 3) if not np.isnan(low_l8_ratio) else 0 
            
            total_score = (
                std_score * score_weight_std +
                skew_score * score_weight_skew +
                ratio_score * score_weight_ratio
            )
            
            # 3. Prepare narrative_context (specific to primary region)
            # Values should be pre-formatted as strings for direct template use.
            narrative_ctx = {
                'l4_mean': f"{l4_mean:.3f}" if not np.isnan(l4_mean) else "N/A",
                'std_l4': f"{std_l4:.3f}" if not np.isnan(std_l4) else "N/A",
                'skew_l4': f"{skew_l4:.2f}" if not np.isnan(skew_l4) else "N/A",
                'global_std': f"{global_std:.3f}" if not np.isnan(global_std) else "N/A",
                'skew_trend': skew_trend if not pd.isna(skew_trend) and skew_trend else "N/A",
                'low_l8_ratio': f"{low_l8_ratio:.1%}" if not np.isnan(low_l8_ratio) else "N/A",
                'bottom_l8_names': bottom_l8_names if bottom_l8_names else "None identified",
                'p10_all_territory': f"{p10_all_territory:.3f}" if not np.isnan(p10_all_territory) else "N/A"
            }
            
            # Prepare context for the plotting function
            l8_plot_context = {
                'primary_region': primary_region_name,
                'value_col': metric_col,
                'region_col': region_col,
                'subregion_col': subregion_col,
                'p10_all_territory': p10_all_territory,
                'lagged_sub_regions': lagged_sub_regions,
                'higher_is_better': higher_is_better,
                'title': f"L8 Concentration - {metric_name} ({primary_region_name})",
                'metric_name': metric_name,  # Add metric name to the context
                'hypothesis_name': self.name,  # Add hypothesis name for consistent naming
                'focus_region': primary_region_name,  # Used for detailed view
                'territory_data': full_df  # Store the full dataframe for reporting
            }

            # CHANGED: Don't create a separate plot file, instead create a PlotSpec
            # Create a plot specification that will be used by the report builder
            plot_spec = PlotSpec(
                plot_key='l8_concentration',
                data=full_df,
                context=l8_plot_context,
                data_key=None
            )

            # 5. Create HypoResult
            result = HypoResult(
                name=self.name,
                type='depth_spotter',
                score=total_score, 
                descriptive_score=total_score,  # Add descriptive_score for non-single_dim hypotheses
                display_rank=self.display_rank,
                natural_name=self.hypothesis_config.get('natural_name', self.name),
                value=l4_mean, 
                global_value=np.mean(global_values) if global_values.size > 0 else np.nan, 
                is_percentage=True, 
                plot_data=full_df, # Store data for the detailed view to use
                plots=[plot_spec],  # Add the PlotSpec to the plots list
                plot_path=None,  # No pre-rendered plot path anymore
                narrative_context=narrative_ctx
            )
            
            return result, plot_spec
            
        except Exception as e:
            logger.exception(f"Error in depth_spotter handler for {self.name}: {e}")
            return None, None 