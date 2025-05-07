import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

# Assuming DataCatalog is in the same directory or accessible via PYTHONPATH
from .data_catalog import DataCatalog
# Assuming metrics.yaml structure is loaded elsewhere and passed
from .data_registry import DataRegistry

logger = logging.getLogger(__name__)

class MetricEngine:
    """Computes metrics defined in metrics.yaml using data from DataCatalog."""

    def __init__(self, metrics_config: Dict[str, Any], data_catalog: DataCatalog, data_registry: DataRegistry):
        """Initialize the metric engine.
        
        Args:
            metrics_config: The metrics configuration dictionary from metrics.yaml
            data_catalog: DataCatalog instance for loading data
            data_registry: DataRegistry instance for storing and retrieving data
        """
        self.metrics_config = metrics_config
        self.data_catalog = data_catalog
        self.data_registry = data_registry
        self._metric_cache: Dict[str, str] = {}  # Stores registry keys of cached metric DataFrames

    def _calculate_metric(self, metric_name: str, metric_info: Dict[str, Any]) -> Optional[str]:
        """Calculate a single metric based on its definition.
        
        Args:
            metric_name: Name of the metric to calculate
            metric_info: Configuration for this metric
            
        Returns:
            Registry key for the DataFrame with region and metric value columns, or None if calculation fails
        """
        # Get input data configuration
        input_data = metric_info.get('input_data', {})
        if not input_data:
            logger.error(f"No input_data configuration found for metric '{metric_name}'")
            return None
            
        dataset_name = input_data.get('dataset')
        required_columns = input_data.get('columns', [])
        
        if not dataset_name or not required_columns:
            logger.error(f"Missing dataset or columns configuration for metric '{metric_name}'")
            return None
        
        # Load the dataset
        registry_key = self.data_catalog.load(dataset_name, cols=required_columns)
        if registry_key is None:
            logger.error(f"Failed to load data for metric '{metric_name}'")
            return None
            
        df = self.data_registry.get(registry_key)
        if df is None:
            logger.error(f"Failed to retrieve data for metric '{metric_name}' from registry")
            return None
            
        # Get the region column name (should be 'region' after renaming in DataCatalog)
        region_col = 'region'  # This is now handled by datasets.yaml rename configuration
            
        # Ensure we have the necessary columns
        if region_col not in df.columns:
            logger.error(f"Region column '{region_col}' not found in dataset for metric '{metric_name}'")
            return None
            
        if metric_name not in df.columns:
            logger.error(f"Metric column '{metric_name}' not found in dataset")
            return None
            
        # Convert metric values to numeric, handling any errors
        df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')
        df = df.dropna(subset=[metric_name])
            
        # Add global row if not present
        if 'Global' not in df[region_col].values:
            # Calculate global average
            global_value = df[metric_name].mean()
            global_row = pd.DataFrame({region_col: ['Global'], metric_name: [global_value]})
            df = pd.concat([df, global_row], ignore_index=True)
            logger.info(f"Added 'Global' row with value {global_value:.4f} for metric '{metric_name}'")
            
        # Store the updated DataFrame in the registry
        updated_key = f"metric_engine_{metric_name}_{pd.util.hash_pandas_object(df).sum()}"
        self.data_registry.set(updated_key, df)
        logger.debug(f"Stored processed metric DataFrame with key: {updated_key}")
            
        return updated_key

    def get_metric_df(self, metric_name: str) -> Optional[pd.DataFrame]:
        """Get a DataFrame containing the region and calculated value for the specified metric.
        
        Args:
            metric_name: The name of the metric as defined in metrics.yaml
            
        Returns:
            DataFrame with columns [region, metric_name] or None if calculation fails
        """
        if metric_name in self._metric_cache:
            logger.debug(f"Returning cached DataFrame for metric '{metric_name}'")
            return self.data_registry.get(self._metric_cache[metric_name])

        if metric_name not in self.metrics_config:
            logger.error(f"Metric '{metric_name}' not found in metrics configuration.")
            return None

        metric_info = self.metrics_config[metric_name]
        logger.info(f"Calculating metric: {metric_name}")

        registry_key = self._calculate_metric(metric_name, metric_info)
        if registry_key is not None:
            self._metric_cache[metric_name] = registry_key
            logger.info(f"Successfully calculated and cached metric '{metric_name}'")
            return self.data_registry.get(registry_key)
        else:
            logger.error(f"Failed to calculate metric '{metric_name}'")
            return None

    def get_global_value(self, metric_name: str) -> Optional[float]:
        """Get the global value for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Global value or None if not available
        """
        df = self.get_metric_df(metric_name)
        if df is None:
            return None
            
        global_row = df[df['region'] == 'Global']
        if global_row.empty:
            logger.warning(f"No 'Global' row found for metric '{metric_name}'")
            return None
            
        return float(global_row[metric_name].iloc[0])
    
    def get_metric_info(self, 
                       metric_name: str, 
                       reset_cache: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[Dict[str, Any]]]:
        """Get comprehensive information about a metric including DataFrame, global value, and config.
        
        Args:
            metric_name: Name of the metric
            reset_cache: Whether to ignore cached values and recalculate
            
        Returns:
            Tuple of (metric_df, global_value, metric_config) or (None, None, None) if not available
        """
        # Clear cache for this metric if requested
        if reset_cache and metric_name in self._metric_cache:
            del self._metric_cache[metric_name]
            
        # Get the metric DataFrame
        metric_df = self.get_metric_df(metric_name)
        if metric_df is None:
            return None, None, None
            
        # Get the global value
        global_value = self.get_global_value(metric_name)
        if global_value is None:
            return metric_df, None, None
            
        # Get the metric configuration
        metric_config = self.metrics_config.get(metric_name, {})
            
        return metric_df, global_value, metric_config
