import pandas as pd
from typing import Dict, Any, Optional
import logging

from ..core.data_catalog import DataCatalog
from ..core.data_registry import DataRegistry
from ..core.types import RegionAnomaly, HypoResult, PlotSpec

logger = logging.getLogger(__name__)

def handle(
    hypo_name: str,
    hypo_config: Dict[str, Any],
    metric_name: str, 
    anomaly: RegionAnomaly,
    data_catalog: DataCatalog,
    data_registry: DataRegistry
) -> Optional[HypoResult]:
    """Handler for fit_by_vertical hypothesis type."""
    logger.info(f"Running fit_by_vertical handler for hypothesis: {hypo_name}")

    # Placeholder implementation
    # 1. Load vertical data (value_column, region_column, vertical_column)
    # 2. Calculate lift and z-score for the anomalous region vs global/peers
    # 3. Prepare key numbers (lift, z-score, top/bottom verticals)
    # 4. Create PlotSpec (e.g., dual-axis bar or scatter)
    # 5. Build HypoResult (include score based on lift/z-score)

    key_numbers = {'status': 'Not Implemented'}
    plot_specs = []
    score = None # Needs calculation
    narrative = f"Handler for {hypo_name} (fit_by_vertical) is not implemented yet."

    result = HypoResult(
        name=hypo_name,
        type='fit_by_vertical',
        narrative=narrative,
        key_numbers=key_numbers,
        plots=plot_specs,
        score=score
    )
    return result 