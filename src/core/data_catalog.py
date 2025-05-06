import pandas as pd
from typing import List, Optional, Dict, Any, Tuple, Union
import os
import yaml
import logging
from .data_registry import DataRegistry

logger = logging.getLogger(__name__)

class DataCatalog:
    """Loads datasets defined in datasets.yaml, applying renaming and selecting columns."""

    def __init__(self, config_dir: str, input_dir: str):
        self.config_path = os.path.join(config_dir, 'datasets.yaml')
        self.input_dir = input_dir
        self._config = self._load_config()
        self._cache: Dict[Tuple[str, Optional[Tuple[str, ...]]], str] = {}  # Maps (name, cols) to registry key

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded dataset configuration from {self.config_path}")
            return config.get('datasets', {})
        except FileNotFoundError:
            logger.error(f"Dataset configuration file not found: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading dataset configuration: {e}")
            return {}

    def load(self, name: str, cols: Optional[List[str]] = None) -> Optional[str]:
        """Loads a specific dataset, potentially only specified columns, applying renaming.
        Returns a registry key instead of the DataFrame directly.

        Args:
            name: The key of the dataset in datasets.yaml.
            cols: Optional list of final column names required after renaming.
                  If None, all columns are loaded.

        Returns:
            A registry key for the loaded DataFrame or None if loading fails.
        """
        if name not in self._config:
            logger.error(f"Dataset '{name}' not found in configuration.")
            return None

        # Build cache key
        cache_key = (name, tuple(sorted(cols)) if cols else None)
        
        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Returning cached registry key for {cache_key}")
            return self._cache[cache_key]

        dataset_config = self._config[name]
        file_path = os.path.join(self.input_dir, dataset_config['path'])
        rename_config = dataset_config.get('rename', {})
        dtypes = dataset_config.get('dtypes', {})

        try:
            logger.info(f"Loading dataset '{name}' from {file_path}")
            
            # Create reverse mapping for column selection
            reverse_map = {v: k for k, v in rename_config.items()}
            
            # Map requested columns back to CSV column names
            csv_cols = None
            if cols:
                csv_cols = []
                for col in cols:
                    if col in reverse_map:
                        csv_cols.append(reverse_map[col])
                    else:
                        csv_cols.append(col)
            
            # Load the data
            df = pd.read_csv(file_path, usecols=csv_cols, na_values=None, keep_default_na=False)
            
            # Apply dtypes if specified
            for col, dtype in dtypes.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Failed to convert column {col} to dtype {dtype}: {e}")

            # Rename columns according to the mapping
            df.rename(columns=rename_config, inplace=True)

            # Store in registry and cache the registry key
            registry_key = DataRegistry.put(df)
            self._cache[cache_key] = registry_key
            logger.info(f"Successfully loaded dataset '{name}' and stored with registry key: {registry_key}")
            return registry_key

        except FileNotFoundError:
            logger.error(f"Input file not found for dataset '{name}': {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset '{name}' from {file_path}: {e}")
            return None

    def list_datasets(self) -> List[str]:
        """Returns a list of available dataset names."""
        return list(self._config.keys()) 