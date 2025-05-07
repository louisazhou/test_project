import pandas as pd
import yaml
import logging
from typing import Dict, Any, Optional, Tuple, List

from ..core.data_catalog import DataCatalog
from ..core.data_registry import DataRegistry
from ..core.types import RegionAnomaly, HypoResult, PlotSpec

logger = logging.getLogger(__name__)

class Handler:
    """Handler for Closed-Lost Reason Over-indexing hypothesis."""

    def __init__(self, hypothesis_config: Dict[str, Any],
                 data_catalog: DataCatalog,
                 data_registry: DataRegistry,
                 settings: Dict[str, Any]):
        self.hypothesis_config = hypothesis_config
        self.data_catalog = data_catalog
        self.data_registry = data_registry
        self.settings = settings # General settings, could include output_dir if handler saves plots
        self.name = hypothesis_config.get('name', 'UnknownClosedLostReasonHypothesis')
        self.display_rank = 0  # Default, will be updated by engine

        # Hard-code configuration values
        self.reason_tags_yaml_path = "config/closed_lost_reason_tags.yaml"
        self.threshold_pct_diff = 0.05
        self.min_excess_loss_dollar = 50000
        self.plot_top_n = 5

        # Load reason tags YAML using the hardcoded path
        if not self.reason_tags_yaml_path: # Should not happen with hardcoding
            logger.error(f"[{self.name}] 'reason_tags_yaml_path' is unexpectedly not set.")
            self.reason_yaml_data = {}
        else:
            try:
                with open(self.reason_tags_yaml_path) as f:
                    self.reason_yaml_data = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"[{self.name}] Error loading YAML from {self.reason_tags_yaml_path}: {e}")
                self.reason_yaml_data = {}
        
        # Configurable thresholds from hypothesis_config (REMOVED - NOW HARDCODED)
        # self.threshold_pct_diff = hypothesis_config.get('config', {}).get('threshold_pct_diff', 0.05)
        # self.min_excess_loss_dollar = hypothesis_config.get('config', {}).get('min_excess_loss_dollar', 100000)
        # self.plot_top_n = hypothesis_config.get('config', {}).get('plot_top_n', 5)

    def _load_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """Loads the primary dataset for closed-lost reasons."""
        input_data_configs = self.hypothesis_config.get('input_data', [])
        if not input_data_configs:
            logger.warning(f"[{self.name}] No input_data configured. Skipping data loading.")
            return None

        # Assuming the first input_data config is for the reasons data
        config = input_data_configs[0]
        dataset_name = config.get('dataset')
        # Potentially other columns like region_col, reason_col, value_col can be fetched from config
        # For now, assume structure of mock_closed_lost_reasons.csv: region, reason, opportunity_lost, pct_of_total

        if not dataset_name:
            logger.error(f"[{self.name}] Hypothesis input_data missing 'dataset'.")
            return None

        try:
            data_key = self.data_catalog.load(dataset_name)
            if not data_key:
                logger.error(f"[{self.name}] Failed to load data for dataset '{dataset_name}'.")
                return None
            
            df = self.data_registry.get(data_key)
            if df is None or df.empty:
                logger.error(f"[{self.name}] No data in registry for key '{data_key}' (dataset: '{dataset_name}').")
                return None
            
            # Basic validation - ensure required columns exist
            required_cols = ['region', 'reason', 'opportunity_lost', 'pct_of_total']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"[{self.name}] Dataset '{dataset_name}' is missing one or more required columns: {required_cols}. Found: {df.columns.tolist()}")
                return None
            
            # Convert opportunity_lost to numeric, coercing errors
            df['opportunity_lost'] = pd.to_numeric(df['opportunity_lost'], errors='coerce')
            df['pct_of_total'] = pd.to_numeric(df['pct_of_total'], errors='coerce')
            df.dropna(subset=['opportunity_lost', 'pct_of_total'], inplace=True)

            return df
        except Exception as e:
            logger.error(f"[{self.name}] Error loading/processing data for dataset '{dataset_name}': {e}", exc_info=True)
            return None

    def _compute_reason_analytics(self, df: pd.DataFrame, focus_region: str, baseline_region: str = "Global") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Compute overindex ratio and other analytics.
        Adapts compute_overindex_ratio from new_test.py.
        Returns merged_df (for lookups) and analytics_df (for insights and plotting, focused on the region).
        """
        if df is None or df.empty:
            return None, None
        
        # Ensure focus_region and baseline_region are in the df
        available_regions = df['region'].unique()
        if focus_region not in available_regions:
            logger.warning(f"[{self.name}] Focus region '{focus_region}' not found in dataset. Available: {available_regions}")
            return None, None
        if baseline_region not in available_regions:
            logger.warning(f"[{self.name}] Baseline region '{baseline_region}' not found in dataset. Using overall mean if needed, but over-indexing might be compromised. Available: {available_regions}")
            # For this specific logic, Global is crucial. If not present, this analysis is less meaningful.
            return None, None


        total_opportunity_by_region = df.groupby('region')["opportunity_lost"].sum().to_dict()
        
        # Pivot for percentage and value
        pct_pivot = df.pivot_table(index="reason", columns="region", values="pct_of_total", fill_value=0).reset_index()
        val_pivot = df.pivot_table(index="reason", columns="region", values="opportunity_lost", fill_value=0).reset_index()
        
        merged_df = pd.merge(pct_pivot, val_pivot, on="reason", suffixes=("_pct", "_val"))

        if f"{focus_region}_pct" not in merged_df.columns or f"{baseline_region}_pct" not in merged_df.columns:
            logger.error(f"[{self.name}] Pivoted data missing columns for focus_region '{focus_region}' or baseline_region '{baseline_region}'. Check input data.")
            return None, None

        delta_pct = merged_df[f"{focus_region}_pct"] - merged_df[f"{baseline_region}_pct"]
        
        # Avoid division by zero or with zero in denominator; result in NaN or inf, then fill
        overindex_ratio = (merged_df[f"{focus_region}_pct"] / merged_df[f"{baseline_region}_pct"]).replace([float('inf'), -float('inf')], pd.NA).fillna(0)

        expected_loss_at_global_pct = total_opportunity_by_region[focus_region] * merged_df[f"{baseline_region}_pct"]
        excess_loss_dollar = merged_df[f"{focus_region}_val"] - expected_loss_at_global_pct
        
        analytics_df = pd.DataFrame({
            "reason": merged_df["reason"],
            "delta_pct": delta_pct,
            "overindex_ratio": overindex_ratio,
            "expected_loss_at_global_pct": expected_loss_at_global_pct,
            "excess_loss_$": excess_loss_dollar,
            "category": merged_df["reason"].map(lambda r: self.reason_yaml_data.get(r, {}).get("category", "unknown")),
            "potential_action": merged_df["reason"].map(lambda r: self.reason_yaml_data.get(r, {}).get("potential_action", "review manually"))
        })
        
        # For plotting, we typically want rows where excess_loss_dollar is positive (over-indexed)
        # analytics_df = analytics_df[analytics_df["excess_loss_$"] > 0]

        return merged_df, analytics_df


    def _generate_closed_lost_narrative(self, merged_df: pd.DataFrame, analytics_df: pd.DataFrame, focus_region: str) -> str:
        """Fill the insight template for closed-lost reasons overindexing.
        Adapts fill_insight_template from new_test.py.
        """
        if analytics_df is None or analytics_df.empty or merged_df is None or merged_df.empty:
            return "No significant closed-lost reason insights found."

        # Filter for significant reasons first
        significant_reasons_df = analytics_df[
            (analytics_df["delta_pct"] > self.threshold_pct_diff) & 
            (analytics_df["excess_loss_$"] > self.min_excess_loss_dollar)
        ].copy()

        if significant_reasons_df.empty:
            return "No closed-lost reasons significantly over-indexed based on current thresholds."

        # Sort by excess_loss_$ to prioritize most impactful reasons
        # and take top N for the narrative (can be same as plot_top_n or different)
        # Using self.plot_top_n for consistency in what's highlighted
        top_n_for_narrative = self.plot_top_n 
        sorted_significant_reasons = significant_reasons_df.sort_values("excess_loss_$", ascending=False).head(top_n_for_narrative)

        narrative_parts = []
        intro = (
            f"In {focus_region}, several closed-lost reasons are significantly over-indexed compared to Global, "
            f"contributing to excess opportunity loss. Key areas for attention ({len(sorted_significant_reasons)} highlighted):"
        )
        narrative_parts.append(intro)

        for _, row in sorted_significant_reasons.iterrows():
            reason = row["reason"]
            region_pct_series = merged_df.loc[merged_df["reason"] == reason, f"{focus_region}_pct"]
            global_pct_series = merged_df.loc[merged_df["reason"] == reason, "Global_pct"]

            if region_pct_series.empty or global_pct_series.empty:
                logger.warning(f"[{self.name}] Missing data in merged_df for reason '{reason}' in narrative generation.")
                continue
            
            region_pct_val = region_pct_series.values[0] * 100
            global_pct_val = global_pct_series.values[0] * 100
            # delta_pct_val = row["delta_pct"] * 100 # This is already in the intro context implicitly
            excess_dollar_val = row["excess_loss_$"]
            mapped_category = row["category"]
            mapped_action = row["potential_action"]
            
            reason_detail = (
                f"\n  • \"{reason}\" (Category: {mapped_category}):\n"
                f"    - Contributes ${excess_dollar_val:,.0f} more in lost opportunity than expected "
                f"(accounts for {region_pct_val:.1f}% of loss in {focus_region} vs. {global_pct_val:.1f}% globally).\n"
                f"    - Recommended Action: {mapped_action}"
            )
            narrative_parts.append(reason_detail)
        
        if len(narrative_parts) == 1: # Only intro was added, meaning loop didn't run (should be caught by significant_reasons_df.empty but as a safeguard)
             return "No closed-lost reasons significantly over-indexed based on current thresholds after filtering for top N."

        return "\n".join(narrative_parts)


    def run(self, metric_name: str,
            anomaly: RegionAnomaly,
            metric_data_key: str, # Key for the main metric's data, not used by this handler directly
           ) -> Tuple[Optional[HypoResult], Optional[PlotSpec]]:
        logger.info(f"[{self.name}] Running for metric '{metric_name}' in region '{anomaly.region}'")

        raw_reasons_df = self._load_and_prepare_data()
        if raw_reasons_df is None:
            logger.warning(f"[{self.name}] Failed to load or prepare data. Aborting.")
            return None, None

        merged_df, analytics_df = self._compute_reason_analytics(raw_reasons_df, anomaly.region)
        if analytics_df is None or merged_df is None:
            logger.warning(f"[{self.name}] Failed to compute reason analytics for region '{anomaly.region}'. Aborting.")
            return None, None
        
        narrative = self._generate_closed_lost_narrative(merged_df, analytics_df, anomaly.region)

        # Revised score logic:
        if "No closed-lost reasons significantly over-indexed" in narrative or \
           "No significant closed-lost reason insights found." in narrative:
            descriptive_score = 0.1
        else:
            descriptive_score = 0.75 # Default score if there are insights
        
        # Override to low score if max impact is too small, regardless of narrative text
        if analytics_df is not None and not analytics_df.empty and analytics_df["excess_loss_$"].max() < self.min_excess_loss_dollar:
            descriptive_score = 0.1 
            # Update narrative if score is lowered due to this specific condition, to avoid confusion
            if not ("No closed-lost reasons significantly over-indexed" in narrative or "No significant closed-lost reason insights found." in narrative):
                narrative += "\n(Note: Overall impact below minimum threshold for high score.)"
        elif analytics_df is None or analytics_df.empty: # Should be caught by narrative checks, but as a safeguard
            descriptive_score = 0.05 # Very low score if no analytics_df at all
        
        # For the plot, we need the analytics_df which contains 'reason' and 'excess_loss_$'
        # The plot function itself will pick top_n and sort.
        plot_data_df = analytics_df[['reason', 'excess_loss_$']].copy()
        # Ensure 'reason' is a column for the plot function, not an index if it became one.
        if plot_data_df.index.name == 'reason':
            plot_data_df = plot_data_df.reset_index()


        # Default value and global_value for HypoResult (can be a summary stat like total excess loss)
        hypo_value = analytics_df['excess_loss_$'].sum() # Example: total excess loss for the region
        hypo_global_value = 0 # This hypothesis type doesn't have a direct "global value" in the same sense as single_dim

        result = HypoResult(
            name=self.name,
            type="closed_lost_reason", # New type
            narrative=narrative,
            score=None,  # Set to None to ensure it's not used for ranking
            descriptive_score=descriptive_score,  # Store the score here for descriptive use
            display_rank=self.display_rank, # Will be set by HypothesisEngine
            natural_name=self.hypothesis_config.get('natural_name', "Closed-Lost Reason Analysis"),
            value=hypo_value, 
            global_value=hypo_global_value,
            is_percentage=False, # The primary value (excess loss) is a dollar amount
            plot_data=plot_data_df, # Data specifically for the plot
            # plot_path will be None as we are using PlotSpec
            # narrative_context can be added if specific pre-formatted values are needed for a template
        )

        # Create PlotSpec
        plot_spec = PlotSpec(
            plot_key='plot_closed_lost_overindex', # Matches the function in new plotting file
            data=plot_data_df.copy(), # Pass the data needed by the plot
            context={
                'focus_region': anomaly.region,
                'top_n': self.plot_top_n,
                'value_col': 'excess_loss_$',
                'reason_col': 'reason',
                'title': f"Top Over-Indexed Lost Reasons in {anomaly.region}",
                # Add any other parameters your plot_closed_lost_overindex function might need from context
            }
        )
        
        logger.info(f"[{self.name}] Completed for region '{anomaly.region}'. Score: {descriptive_score:.2f}")
        return result, plot_spec 