import pandas as pd
from typing import Dict, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)

class DataRegistry:
    """In-memory cache for DataFrames, addressable by a content-based hash key."""
    _store: Dict[str, pd.DataFrame] = {}

    def __init__(self):
        """Initialize a new instance-based data registry."""
        # Instance-level storage
        self._instance_store: Dict[str, pd.DataFrame] = {}

    @classmethod
    def put(cls, df: pd.DataFrame) -> str:
        """Stores a DataFrame and returns a hash-based key (class method)."""
        key = f'k{hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()}'
        cls._store[key] = df.copy()  # Store a copy
        logger.debug(f"Stored DataFrame with key: {key}, Shape: {df.shape}")
        return key

    @classmethod
    def get(cls, key: str) -> Optional[pd.DataFrame]:
        """Retrieves a DataFrame by its key (class method)."""
        df = cls._store.get(key)
        if df is not None:
            logger.debug(f"Retrieved DataFrame with key: {key}, Shape: {df.shape}")
            return df.copy()  # Return a copy
        else:
            logger.warning(f"Key not found in registry: {key}")
            return None

    @classmethod
    def exists(cls, key: str) -> bool:
        """Checks if a key exists in the registry (class method)."""
        return key in cls._store

    @classmethod
    def clear(cls):
        """Clears the entire registry (class method)."""
        count = len(cls._store)
        cls._store.clear()
        logger.info(f"Cleared DataRegistry (removed {count} items).")
    
    # Instance methods for instance-based registry
    def set(self, key: str, df: pd.DataFrame) -> str:
        """Stores a DataFrame with the given key (instance method)."""
        # First try to find in class store
        if key in self._store:
            logger.warning(f"Key {key} already exists in class registry. Using instance registry.")
        
        self._instance_store[key] = df.copy()  # Store a copy in instance storage
        logger.debug(f"Stored DataFrame in instance registry with key: {key}, Shape: {df.shape}")
        return key
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieves a DataFrame by its key (instance method).
        Checks instance store first, then class store."""
        # First try to find in instance store
        df = self._instance_store.get(key)
        if df is not None:
            logger.debug(f"Retrieved DataFrame from instance registry with key: {key}, Shape: {df.shape}")
            return df.copy()  # Return a copy
        
        # Fall back to class store
        df = self._store.get(key)
        if df is not None:
            logger.debug(f"Retrieved DataFrame from class registry with key: {key}, Shape: {df.shape}")
            return df.copy()  # Return a copy
        
        logger.warning(f"Key not found in instance or class registry: {key}")
        return None 