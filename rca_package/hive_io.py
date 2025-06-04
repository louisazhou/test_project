import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

# Convert numpy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_): 
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='index')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def load_from_hive(query: str, conn_str: str) -> pd.DataFrame:
    """
    Load data from Hive using a query.
    
    Args:
        query: SQL query to execute
        conn_str: Connection string for Hive
        
    Returns:
        DataFrame containing the query results
    """
    # TODO: Implement using pyhive or similar
    pass

def save_to_hive(df: pd.DataFrame, table: str, conn_str: str, 
                partition_cols: Optional[list] = None) -> None:
    """
    Save DataFrame to a Hive table.
    
    Args:
        df: DataFrame to save
        table: Target table name
        conn_str: Connection string for Hive
        partition_cols: Optional list of partition columns
    """
    # TODO: Implement using pyhive or similar
    pass 

# =====================================================================
# SIMPLE ANALYTICAL RECORD SCHEMA
# =====================================================================

def create_analytical_record(
    metric: str,
    technical_name: str, 
    region: str,
    hypothesis_name: str,
    hypothesis_technical_name: str,
    hypothesis_type: str,
    score: Optional[float],
    selected: bool,
    summary_text: str,
    text: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a standardized analytical record with minimal common fields 
    and everything else in payload.
    
    Args:
        metric: Display name of metric (e.g., "CLI Closed %")
        technical_name: Technical column name (e.g., "cli_closed_pct")
        region: Anomalous region (e.g., "AM-NA")
        hypothesis_name: Subject name (e.g., "SLI/AM") 
        hypothesis_technical_name: Subject technical name (e.g., "SLI_per_AM")
        hypothesis_type: Analysis type ("scorer", "depth", etc.)
        summary_text: Brief summary
        text: Full text description
        payload: All raw analytical data, dataframes, components, etc.
    
    Returns:
        Dictionary with standardized structure
    """
    record = {
        "metric": metric,
        "technical_name": technical_name,
        "region": region, 
        "hypothesis_name": hypothesis_name,
        "hypothesis_technical_name": hypothesis_technical_name,
        "hypothesis_type": hypothesis_type,
        "summary_text": summary_text,
        "text": text,
        "payload": convert_numpy_types(payload)
    }
    
    return record

def save_analytical_records_json(records: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save analytical records to JSON file.
    
    Args:
        records: List of analytical record dictionaries
        filepath: Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(records, f, indent=2, default=str)

def load_analytical_records_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load analytical records from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of analytical record dictionaries
    """
    with open(filepath, 'r') as f:
        return json.load(f) 