# pyre-ignore-all-errors
"""
Clean Oaxaca-Blinder Decomposition Analysis

A well-structured implementation of Oaxaca-Blinder decomposition with:
- Clean separation of mathematical core, business logic, and edge case detection
- Comprehensive Simpson's paradox detection
- Multiple baseline methods
- Business narrative generation
- Mathematical validation

Usage:
    from rca_package.oaxaca_blinder_clean import run_oaxaca_analysis
    
    # Clean interface
    result = run_oaxaca_analysis(
        df=df,
        region_column="territory",
        numerator_column="wins", 
        denominator_column="opportunities",
        category_columns=["product"],
        subcategory_columns=["vertical"]
    )
    
    # Use methods on the result
    narrative = result.get_narrative_for_region("US")
    summary = result.get_summary_stats()
    executive_report = result.generate_executive_report("US")
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Iterable
from dataclasses import dataclass
from enum import Enum
import ast 
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# CORE CONSTANTS AND TYPES
# =============================================================================

class GapSignificance(Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

class BusinessConclusion(Enum):
    PERFORMANCE_DRIVEN = "performance_driven"      # Pure execution differences
    COMPOSITION_DRIVEN = "composition_driven"      # Pure allocation differences  
    SIMPSON_PARADOX = "simpson_paradox"            # Segment performance contradicts aggregate
    MATHEMATICAL_CANCELLATION = "cancellation"     # Effects cancel out to zero
    NO_SIGNIFICANT_GAP = "no_gap"                  # Performs in line with baseline

class PerformanceDirection(Enum):
    OUTPERFORMS = "outperforms"
    UNDERPERFORMS = "underperforms" 
    IN_LINE = "performs in line with"
    
class ParadoxType(Enum):
    """Types of Simpson's Paradox detected."""
    NONE = "none"                    # No paradox detected
    MATHEMATICAL = "mathematical"    # Performance direction reversal

@dataclass
class AnalysisThresholds:
    """Centralized thresholds for consistent business logic."""
    # DECIMAL thresholds (e.g., 0.02 == 2pp)
    significant_gap: float = 0.01        # 1pp - business significant
    minor_gap: float = 0.005             # 0.5pp - minor but detectable  
    high_impact_gap: float = 0.02        # 2pp - high business impact
    near_zero_gap: float = 0.005         # 0.5pp - essentially no gap
    
    # Execution difference thresholds
    significant_rate_diff: float = 0.01  # 1pp rate difference
    minor_rate_diff: float = 0.005       # 0.5pp rate difference
    
    # Narrative thresholds
    visual_indicator_threshold: float = 0.01  # 1pp - for visual indicators (🟢/🔴)
    cumulative_contribution_threshold: float = 0.6  # 60% - for ranking top contributors
    
    # PERCENT-POINT thresholds (explicitly in pp)
    significant_allocation_diff_pp: float = 10.0  # 10pp - significant allocation difference
    meaningful_allocation_diff_pp: float = 5.0   # 5pp - meaningful allocation difference
    
    # Mathematical validation
    mathematical_tolerance: float = 1e-12  # Decomposition validation tolerance - make this tighter
    
    # Simpson gating (all decimals, not pp)
    paradox_material_gap: float = 0.015     # ≥1.5pp magnitude on either side
    paradox_disagree_share: float = 0.40    # ≥40% exposure disagrees with pooled direction
    paradox_impact_swing: float = 0.010     # ≥1.0pp swing between pooled and common-mix
    
    # Simpson direction policy
    # "under_only"  → flag only regions that UNDERperform overall but would OVERperform under common-mix
    # "over_only"   → flag only regions that OVERperform overall but would UNDERperform under common-mix
    # "both"        → flag either direction (previous behavior)
    paradox_direction: str = "under_only"

@dataclass 
class NarrativeDecision:
    """Complete narrative decision with supporting evidence."""
    # Core classification
    root_cause_type: BusinessConclusion
    performance_direction: PerformanceDirection
    gap_significance: GapSignificance
    
    # Generated narrative text
    narrative_text: str
    
    # Supporting evidence (ready for display)
    supporting_table_df: pd.DataFrame
    key_metrics: Dict[str, float]  # gap_magnitude, primary_contribution, etc.
    
    # Metadata
    baseline_type: str

@dataclass
class ParadoxReport:
    """Simpson's paradox detection result - detection only, no narratives."""
    paradox_detected: bool
    affected_regions: List[str]
    paradox_type: ParadoxType  # Enum-driven paradox classification
    business_impact: GapSignificance  # Impact level of the paradox

@dataclass
class CancellationReport:
    """Report on mathematical cancellation scenarios - detection only, no narratives."""
    cancellation_detected: bool
    cancellation_type: str  # "identical_rates" vs "identical_mix" vs "both"
    affected_regions: List[str]

@dataclass
class AnalysisResult:
    """Complete analysis result with all components."""
    # Core analysis data (keep existing)
    decomposition_df: pd.DataFrame
    regional_gaps: pd.DataFrame
    mathematical_validation: Dict[str, float]
    
    # New structured narrative system
    narrative_decisions: Dict[str, NarrativeDecision]  # region -> decision
    
    # Edge cases
    paradox_report: ParadoxReport
    cancellation_report: CancellationReport
    
    # Internal state (baseline_type passed from orchestrator)
    baseline_type: str
      
    def get_summary_stats(self) -> Dict:
        """
        Get organized original DataFrames for portfolio analysis.
        
        Returns the actual analysis DataFrames ranked and organized for insight,
        avoiding over-processed statistics that dilute truth.
        
        Returns:
            Dict containing:
            - regional_performance_summary: Regional gaps ranked by performance
            - top_category_contributors: Biggest drivers across all regions
            - narrative_decisions_summary: Root cause classifications by region
            - paradox_regions: Regions with Simpson's paradox detected
            - cancellation_regions: Regions with mathematical cancellation
        """
        logger.info("🎯 GENERATING ORGANIZED DATAFRAMES for portfolio analysis")
        
        if len(self.regional_gaps) == 0:
            return {"error": "No regional data available for analysis"}
        
        # 1. Regional Performance Summary - ranked by net gap with root cause info merged
        regional_summary = self.regional_gaps[['region', 'total_net_gap', 'total_performance_gap', 'total_construct_gap']].copy()
        
        # Convert to percentage points for readability
        regional_summary['total_net_gap_pp'] = (regional_summary['total_net_gap'] * 100).round(1).astype(str) + 'pp'
        regional_summary['total_performance_gap_pp'] = (regional_summary['total_performance_gap'] * 100).round(1).astype(str) + 'pp'
        regional_summary['total_construct_gap_pp'] = (regional_summary['total_construct_gap'] * 100).round(1).astype(str) + 'pp'
        
        # Add root cause information
        for region in regional_summary['region']:
            if region in self.narrative_decisions:
                decision = self.narrative_decisions[region]
                regional_summary.loc[regional_summary['region'] == region, 'root_cause'] = decision.root_cause_type.value
                regional_summary.loc[regional_summary['region'] == region, 'performance_direction'] = decision.performance_direction.value
        
        # Sort by magnitude (use original numeric values for sorting)
        regional_summary = regional_summary.sort_values('total_net_gap', ascending=False)
        
        # Keep only formatted columns for display
        regional_summary = regional_summary[['region', 'total_net_gap_pp', 'total_performance_gap_pp', 'total_construct_gap_pp', 'root_cause', 'performance_direction']]
        
        # 2. Top Category Contributors - biggest absolute drivers across portfolio
        category_contributors = self.decomposition_df[['region', 'category', 'net_gap', 'construct_gap_contribution', 'performance_gap_contribution']].copy()
        
        # Convert to percentage points for readability
        category_contributors['net_gap_pp'] = (category_contributors['net_gap'] * 100).round(1).astype(str) + 'pp'
        category_contributors['construct_gap_pp'] = (category_contributors['construct_gap_contribution'] * 100).round(1).astype(str) + 'pp'
        category_contributors['performance_gap_pp'] = (category_contributors['performance_gap_contribution'] * 100).round(1).astype(str) + 'pp'
        
        category_contributors['abs_gap'] = category_contributors['net_gap'].abs()
        top_contributors = category_contributors.nlargest(15, 'abs_gap')
        
        # Keep only formatted columns for display
        top_contributors = top_contributors[['region', 'category', 'net_gap_pp', 'construct_gap_pp', 'performance_gap_pp']]
        
        # 4. Edge Case Regions
        paradox_regions = self.paradox_report.affected_regions if self.paradox_report.paradox_detected else []
        cancellation_regions = self.cancellation_report.affected_regions if self.cancellation_report.cancellation_detected else []
        
        return {
            "regional_performance_summary": regional_summary,
            "top_category_contributors": top_contributors, 
            "paradox_regions": paradox_regions,
            "cancellation_regions": cancellation_regions
        }
    

    

    
    def get_narrative_for_region(self, region: str) -> NarrativeDecision:
        """
        Get complete NarrativeDecision for a region.
        
        Args:
            region: Region identifier to get narrative for
            
        Returns:
            NarrativeDecision containing:
            - root_cause_type: BusinessConclusion enum (composition_driven, performance_driven, etc.)
            - performance_direction: PerformanceDirection enum (outperforms, underperforms)
            - narrative_text: Complete business narrative text
            - supporting_table_df: Evidence table with gaps and indicators
            
        Raises:
            ValueError: If region has no narrative decision available
        """
        logger.info(f"🎯 RETRIEVING NARRATIVE DECISION for {region}")
        
        # Access pre-computed narrative decisions
        if region not in self.narrative_decisions:
            raise ValueError(f"No narrative decision available for region: {region}")
        
        decision = self.narrative_decisions[region]
        logger.info(f"🎯 FOUND: {decision.root_cause_type.value} analysis for {region}")
        return decision
    
    def generate_executive_report(self, region: str) -> Dict:
        """
        Generate executive report data for a specific region.
        
        Returns structured data instead of formatted text, following the same
        pattern as get_summary_stats().
        
        Args:
            region: Region identifier for report generation
            
        Returns:
            Dict containing:
            - narrative_text: Root cause narrative
            - supporting_evidence: Evidence table DataFrame  
            - analysis_summary: Key metrics and classifications
        """
        logger.info(f"🎯 GENERATING EXECUTIVE REPORT DATA for {region}")
        
        # Get pre-computed narrative decision (contains everything we need)
        if region not in self.narrative_decisions:
            return {"error": f"No narrative decision available for region {region}"}
        
        narrative_decision = self.narrative_decisions[region]
        
        # Return structured data for downstream formatting
        return {
            "narrative_text": narrative_decision.narrative_text,
            "supporting_evidence": narrative_decision.supporting_table_df,
            "analysis_summary": {
                "region": region,
                "gap_magnitude_pp": f"{narrative_decision.key_metrics['gap_magnitude']*100:.1f}pp",
                "root_cause": narrative_decision.root_cause_type.value,
                "performance_direction": narrative_decision.performance_direction.value,
                "gap_significance": narrative_decision.gap_significance.value
            }
        }

# =============================================================================
# CORE MATHEMATICAL ENGINE
# =============================================================================

class OaxacaCore:
    """Core mathematical functions for Oaxaca-Blinder decomposition."""
    
    @staticmethod
    def calculate_rates(
        data: pd.DataFrame, 
        category_columns: List[str],
        numerator_column: str,
        denominator_column: str
    ) -> pd.Series:
        """Calculate rates for each category combination."""
        if not category_columns:
            # Overall rate calculation
            total_num = data[numerator_column].sum()
            total_denom = data[denominator_column].sum()
            overall_rate = total_num / total_denom if total_denom > 0 else 0
            return pd.Series([overall_rate], index=[0])
        
        # Group by categories and calculate rates
        grouped = data.groupby(category_columns).agg({
            numerator_column: "sum", 
            denominator_column: "sum"
        })
        
        rates = grouped[numerator_column] / grouped[denominator_column]
        return rates.fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def calculate_mix(
        data: pd.DataFrame,
        category_columns: List[str], 
        denominator_column: str
    ) -> pd.Series:
        """Calculate mix percentages for each category combination."""
        if not category_columns:
            return pd.Series([1.0], index=[0])
        
        mix = data.groupby(category_columns)[denominator_column].sum()
        return mix / mix.sum()
    
    @staticmethod
    def decompose_gap(
        region_mix: float,
        region_rate: float,
        baseline_mix: float, 
        baseline_rate: float,
        method: str = "two_part"
    ) -> Tuple[float, float, float]:
        """
        Core gap decomposition logic.
        
        Returns:
            (construct_gap, performance_gap, net_gap)
        """
        if method == "two_part":
            # B-baseline (rest-of-world baseline)
            construct_gap = (region_mix - baseline_mix) * baseline_rate
            performance_gap = region_mix * (region_rate - baseline_rate)
            
        elif method == "reverse":
            # A-baseline (region baseline)
            construct_gap = (region_mix - baseline_mix) * region_rate
            performance_gap = baseline_mix * (region_rate - baseline_rate)
            
        elif method == "symmetric":
            # Symmetric (path-independent)
            avg_rate = (region_rate + baseline_rate) / 2
            avg_mix = (region_mix + baseline_mix) / 2
            construct_gap = (region_mix - baseline_mix) * avg_rate
            performance_gap = avg_mix * (region_rate - baseline_rate)
            
        else:
            raise ValueError(f"Invalid method: {method}. Use 'two_part', 'reverse', or 'symmetric'")
        
        net_gap = region_mix * region_rate - baseline_mix * baseline_rate
        return construct_gap, performance_gap, net_gap
    
    @staticmethod
    def validate_decomposition(
        actual_gaps: Dict[str, float],
        decomposed_gaps: Dict[str, float], 
        tolerance: float
    ) -> Dict[str, float]:
        """Validate that decomposition sums correctly to actual gaps."""
        errors = {r: abs(actual_gaps[r] - decomposed_gaps.get(r, np.nan))
                  for r in actual_gaps}
        max_error = np.nanmax(list(errors.values())) if errors else 0.0
        
        # ASSERT MATHEMATICAL VALIDATION (fail fast on errors)
        assert max_error < tolerance, (
            f"Decomposition validation failed: max_error={max_error:.8e} ≥ tol={tolerance:.1e}"
        )
        
        logger.debug(f"[MATH] DECOMPOSITION_VALIDATION: max_error={max_error:.8f}, valid={max_error < tolerance}")
        
        return {"max_error": float(max_error), **{f"err_{k}": v for k, v in errors.items()}}


# =============================================================================
# BUSINESS LOGIC LAYER
# =============================================================================

class NarrativeTemplateEngine:
    """Clean template-driven narrative generation for ALL cases."""
    
    def __init__(self, thresholds: AnalysisThresholds = None):
        self.thresholds = thresholds or AnalysisThresholds()
    
    def create_narrative_decision(
        self, 
        region: str,
        decomposition_df: pd.DataFrame,
        regional_gaps: pd.DataFrame,
        baseline_type: str,
        business_conclusion: BusinessConclusion,
        performance_direction: PerformanceDirection,
        gap_significance: GapSignificance,
        cancellation_report=None
    ) -> NarrativeDecision:
        """Single method to create NarrativeDecision for ANY case."""
        
        # Get region data from enriched DataFrames
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        gap_magnitude = region_totals["gap_magnitude"]
        total_gap = region_totals["total_net_gap"]
        total_construct_gap = region_totals["total_construct_gap"]
        total_performance_gap = region_totals["total_performance_gap"]
        
        # Step 1: Generate narrative text using templates
        narrative_text = self._generate_template_text(
            region, decomposition_df, regional_gaps, baseline_type, business_conclusion, performance_direction
        )
        
        # Add cancellation warning for NO_SIGNIFICANT_GAP cases
        if (business_conclusion == BusinessConclusion.NO_SIGNIFICANT_GAP and 
            cancellation_report and cancellation_report.cancellation_detected and 
            region in cancellation_report.affected_regions):
            narrative_text += ", but note offsetting composition and performance effects at the category level (cancellation), which makes the topline fragile."
        
        # Step 4: Create supporting table
        supporting_table = self._create_template_table(region, decomposition_df, business_conclusion)
        
        return NarrativeDecision(
            root_cause_type=business_conclusion,
            performance_direction=performance_direction,
            gap_significance=gap_significance,
            narrative_text=narrative_text,
            supporting_table_df=supporting_table,
            key_metrics={
                "gap_magnitude": gap_magnitude,  # Already in decimal form (e.g., 0.05 for 5pp)
                "total_gap": total_gap,          # Already in decimal form  
                "construct_gap": total_construct_gap,  # Already in decimal form
                "performance_gap": total_performance_gap,  # Already in decimal form
            },
            baseline_type=baseline_type
        )
    
    def _generate_template_text(
        self, 
        region: str,
        decomposition_df: pd.DataFrame,
        regional_gaps: pd.DataFrame,
        baseline_type: str,
        conclusion: BusinessConclusion, 
        direction: PerformanceDirection
    ) -> str:
        """Template-driven narrative text generation for ALL cases."""
        
        # Get region data
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        gap_magnitude = region_totals["gap_magnitude"] * 100  # Convert to pp for display
        
        baseline_name = self._get_baseline_name(baseline_type)
        direction_text = direction.value
        
        # Template selection based on BusinessConclusion
        if conclusion == BusinessConclusion.MATHEMATICAL_CANCELLATION:
            return self._cancellation_template(region, decomposition_df, regional_gaps, baseline_name)
        elif conclusion == BusinessConclusion.SIMPSON_PARADOX:
            return self._paradox_template(region, decomposition_df, regional_gaps, baseline_name, direction_text, gap_magnitude)
        elif conclusion == BusinessConclusion.COMPOSITION_DRIVEN:
            return self._composition_template(region, decomposition_df, regional_gaps, baseline_name, direction_text, gap_magnitude)
        elif conclusion == BusinessConclusion.PERFORMANCE_DRIVEN:
            return self._performance_template(region, decomposition_df, regional_gaps, baseline_name, direction_text, gap_magnitude)
        else:  # NO_SIGNIFICANT_GAP
            return self._no_gap_template(region, regional_gaps, baseline_name)
    
    def _create_template_table(self, region: str, decomposition_df: pd.DataFrame, conclusion: BusinessConclusion) -> pd.DataFrame:
        """Template-driven table creation for ALL cases."""
        
        # Get region data from enriched DataFrame
        region_data = decomposition_df[decomposition_df["region"] == region]
        
        if conclusion == BusinessConclusion.MATHEMATICAL_CANCELLATION:
            return self._create_cancellation_table(region_data)
        elif conclusion == BusinessConclusion.SIMPSON_PARADOX:
            return self._create_paradox_table(region_data)
        else:
            return self._create_standard_table(region_data)
    
    # Template methods for different narrative types
    def _get_baseline_name(self, baseline_type: str) -> str:
        """Convert baseline type to display name."""
        baseline_names = {
            'rest_of_world': 'rest of world',
            'global_average': 'global average', 
            'top_performer': 'top performer',
            'previous_period': 'previous period'
        }
        return baseline_names.get(baseline_type, baseline_type)
    
    def _cancellation_template(self, region: str, decomposition_df: pd.DataFrame, regional_gaps: pd.DataFrame, baseline_name: str) -> str:
        """Template for mathematical cancellation narrative."""
        return (f"{region} performs in line with {baseline_name} baseline "
                f"due to mathematical cancellation where individual category effects "
                f"offset each other at the aggregate level.")
    
    def _paradox_template(self, region: str, decomposition_df: pd.DataFrame, regional_gaps: pd.DataFrame, baseline_name: str, direction_text: str, gap_magnitude: float) -> str:
        """Template for Simpson's paradox narrative following gold standard pattern."""
        # Use ACTUAL overall rates from the decomposition
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        actual_region_rate = region_totals["region_overall_rate"]
        actual_baseline_rate = region_totals["baseline_overall_rate"]
        
        # Identify segments where region outperforms vs underperforms
        outperform_segments = []
        allocation_issues = []
        
        # Work directly with numeric data from category breakdown
        region_data = decomposition_df[decomposition_df["region"] == region]
        for _, row in region_data.iterrows():
            category_name = row.get(
                'category_clean',
                self._clean_category_display_name(row.get('category'))
            )
            # Use pre-enriched columns instead of manual calculation
            region_rate_pct = row['region_rate_pct']
            baseline_rate_pct = row['rest_rate_pct']
            region_mix_display = row['region_mix_display']
            baseline_mix_display = row['rest_mix_display']
            
            # Track segment performance for "despite" clause (use raw values for comparison)
            if row['region_rate'] > row['rest_rate']:
                outperform_segments.append(f"{category_name} ({region_rate_pct} vs {baseline_rate_pct})")
            
            # Track allocation issues for "due to" clause (use raw values for calculation)
            mix_diff_pp = (row['region_mix_pct'] - row['rest_mix_pct']) * 100  # Convert to percentage points
            if abs(mix_diff_pp) >= self.thresholds.significant_allocation_diff_pp:
                if mix_diff_pp > 0:
                    allocation_issues.append(f"over-concentration in {category_name} ({region_mix_display} vs {baseline_mix_display} baseline)")
                else:
                    allocation_issues.append(f"under-concentration in {category_name} ({region_mix_display} vs {baseline_mix_display} baseline)")
        
        # Build gold standard paradox pattern
        if outperform_segments and direction_text == PerformanceDirection.UNDERPERFORMS.value:
            # Classic Simpson's Paradox: outperforms in segments but underperforms overall
            segment_text = " and ".join(outperform_segments[:3])  # Top segments
            
            # Create allocation explanation
            if allocation_issues:
                allocation_text = " and ".join(allocation_issues[:2])
            else:
                allocation_text = "allocation issues that offset segment-level strengths"
            
            return (f"{region} {direction_text} {baseline_name} by {gap_magnitude:.1f}pp "
                   f"({actual_region_rate:.0%} vs {actual_baseline_rate:.0%}) despite "
                   f"{segment_text} due to {allocation_text}.")
        else:
            # Fallback with actual data
            contributing_segments = []
            region_data = decomposition_df[decomposition_df["region"] == region]
            for _, row in region_data.iterrows():
                category_name = row.get(
                'category_clean',
                self._clean_category_display_name(row.get('category'))
            )
                # Use pre-enriched net_gap_pp column
                net_gap_pp = row['net_gap_pp']
                # Use minor_gap threshold for meaningful contributors (convert numeric pp to decimal for comparison)
                if abs(row['net_gap_numeric'] / 100) >= self.thresholds.minor_gap:
                    contributing_segments.append(f"{category_name} ({net_gap_pp})")
            
            segments_text = ", ".join(contributing_segments[:3]) if contributing_segments else "complex dynamics"
            
            return (f"{region} {direction_text} {baseline_name} by {gap_magnitude:.1f}pp "
                   f"({actual_region_rate:.0%} vs {actual_baseline_rate:.0%}) with {segments_text}.")
    
    def _composition_template(self, region: str, decomposition_df: pd.DataFrame, regional_gaps: pd.DataFrame, baseline_name: str, direction_text: str, gap_magnitude: float) -> str:
        """Template for composition-driven narrative following gold standard pattern."""
        # Use overall rates from enriched DataFrames
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        overall_region_rate = region_totals["region_overall_rate"]
        overall_baseline_rate = region_totals["baseline_overall_rate"]
        
        # Get performance in each category for "despite" clause
        performance_segments = []
        allocation_issues = []
        
        region_data = decomposition_df[decomposition_df["region"] == region]
        for _, row in region_data.iterrows():
            category_name = row.get(
                'category_clean',
                self._clean_category_display_name(row.get('category'))
            )
            # Use pre-enriched columns
            region_rate_pct = row['region_rate_pct']
            baseline_rate_pct = row['rest_rate_pct']
            region_mix_display = row['region_mix_display']
            baseline_mix_display = row['rest_mix_display']
            
            # Track performance in each segment for "despite" clause (use raw values for comparison)
            rate_diff = row['region_rate'] - row['rest_rate']
            # Use minor_rate_diff threshold for meaningful rate differences
            if abs(rate_diff) >= self.thresholds.minor_rate_diff:
                if rate_diff > 0:
                    performance_segments.append(f"outperforming in {category_name} ({region_rate_pct} vs {baseline_rate_pct})")
                else:
                    performance_segments.append(f"underperforming in {category_name} ({region_rate_pct} vs {baseline_rate_pct})")
            else:
                performance_segments.append(f"same performance in {category_name} ({region_rate_pct} vs {baseline_rate_pct})")
            
            # Track allocation issues for "due to" clause (use raw values for calculation)
            mix_diff_pp = (row['region_mix_pct'] - row['rest_mix_pct']) * 100  # Convert to percentage points
            if abs(mix_diff_pp) >= self.thresholds.meaningful_allocation_diff_pp:
                if mix_diff_pp > 0:
                    allocation_issues.append(f"over-concentration in {category_name} ({region_mix_display} vs {baseline_mix_display} baseline)")
                else:
                    allocation_issues.append(f"under-allocation to {category_name} ({region_mix_display} vs {baseline_mix_display} baseline)")
        
        # Build gold standard pattern: "despite...due to"
        despite_clause = " and ".join(performance_segments[:2]) if performance_segments else "mixed segment performance"
        
        # Use composition-focused language for allocation issues
        due_to_clause = ", ".join(allocation_issues[:2]) if allocation_issues else "allocation issues"
        
        return f"{region} {direction_text} {baseline_name} by {gap_magnitude:.1f}pp ({overall_region_rate:.0%} vs {overall_baseline_rate:.0%}) despite {despite_clause} due to {due_to_clause}."
    
    def _performance_template(self, region: str, decomposition_df: pd.DataFrame, regional_gaps: pd.DataFrame, baseline_name: str, direction_text: str, gap_magnitude: float) -> str:
        """Template for performance-driven narrative - using backup logic."""
        
        # Use overall rates from enriched DataFrames
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        overall_region_rate = region_totals["region_overall_rate"]
        overall_baseline_rate = region_totals["baseline_overall_rate"]
        
        # Get top contributors from enriched DataFrame
        region_data = decomposition_df[decomposition_df["region"] == region]
        
        top_contributors = region_data[region_data["is_top_contributor"] == True].nlargest(3, 'abs_gap_numeric')
        
        # Use pre-computed contributor text
        category_text = ", ".join(top_contributors['contributor_text'].tolist()) if len(top_contributors) > 0 else "various factors"
        
        # Use performance-focused language for driving factors
        return f"{region} {direction_text} the {baseline_name} by {gap_magnitude:.1f}pp ({overall_region_rate:.0%} vs. {overall_baseline_rate:.0%}), driven by {category_text}"
    
    def _no_gap_template(self, region: str, regional_gaps: pd.DataFrame, baseline_name: str) -> str:
        """Template for no significant gap narrative."""
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        total_gap = region_totals["total_net_gap"]
        base_narrative = (f"{region} performs in line with {baseline_name} baseline "
                         f"with no significant performance difference "
                         f"({total_gap*100:+.1f}pp gap)")
        
        # Add cancellation warning if this region has offsetting effects
        # Note: We need access to cancellation_report here, but it's not passed in.
        # For now, return base narrative. The warning logic should be added at calling site.
        return base_narrative
    
    def _create_cancellation_table(self, category_breakdown: pd.DataFrame) -> pd.DataFrame:
        """Template table for cancellation cases."""
        return self._create_standard_table(category_breakdown)  # Same format for now
    
    def _create_paradox_table(self, category_breakdown: pd.DataFrame) -> pd.DataFrame:
        """Template table for paradox cases."""
        return self._create_standard_table(category_breakdown)  # Same format for now
    
    def _create_standard_table(self, category_breakdown: pd.DataFrame) -> pd.DataFrame:
        """Template table for standard cases with proper ranking by gap importance."""
        # Use all columns from the enriched DataFrame (don't filter out enriched columns!)
        table = category_breakdown.copy()
        
        # Sort by absolute gap size so most important contributors appear first
        # (abs_gap_numeric already exists from enrichment, no need to recreate)
        table = table.sort_values('abs_gap_numeric', ascending=False)
        
        # Use pre-enriched display columns (ALL DataFrames should be enriched by this point)
        table['Region%'] = table['region_rate_pct']
        table['Baseline%'] = table['rest_rate_pct'] 
        table['Gap_pp'] = table['net_gap_pp']
        table['Mix%'] = table['region_mix_display']
        
        # Add status indicators and business insights with contribution ranking
        def get_status_with_rank(gap, rank):
            # Use visual_indicator_threshold instead of hardcoded value
            threshold = self.thresholds.visual_indicator_threshold
            if gap > threshold:
                return f"🟢 Strength #{rank}"
            elif gap < -threshold:
                return f"🔴 Gap #{rank}"
            else:
                return f"⚪ Neutral"
        
        # Add ranking to show which gaps matter most
        table['rank'] = range(1, len(table) + 1)
        table['Status'] = table.apply(lambda row: get_status_with_rank(row['net_gap'], row['rank']), axis=1)
        
        # Clean category names using unified method
        table['Category'] = table['category'].apply(lambda name: self._clean_category_display_name(name))
        
        return table[['Status', 'Category', 'Region%', 'Baseline%', 'Gap_pp', 'Mix%']]
    
    def _clean_category_display_name(self, category_name) -> str:
        """Clean category names for business display."""
        # Handle tuple directly
        if isinstance(category_name, tuple) and len(category_name) == 2:
            return f"{category_name[0]} - {category_name[1]}"
        
        # Handle string representation of tuple
        elif isinstance(category_name, str) and category_name.startswith("('") and category_name.endswith("')"):
            try:
                parsed = ast.literal_eval(category_name)
                if isinstance(parsed, tuple) and len(parsed) == 2:
                    
                    return f"{parsed[0]} - {parsed[1]}"
            except:
                pass
        
        return str(category_name)

class BusinessAnalyzer:
    """Pure business logic for interpreting prepared analysis data."""
    
    def __init__(self, thresholds: AnalysisThresholds = None):
        self.thresholds = thresholds or AnalysisThresholds()
        self.template_engine = NarrativeTemplateEngine(thresholds)
    
    def assess_gap_significance(self, gap_magnitude: float) -> GapSignificance:
        """Assess business significance of gap magnitude."""
        g = abs(gap_magnitude)  # decimal; no *100 here
        
        if g > self.thresholds.high_impact_gap:
            return GapSignificance.HIGH
        elif g > self.thresholds.significant_gap:
            return GapSignificance.MEDIUM
        else:
            return GapSignificance.LOW
    
    def generate_regional_narrative(
        self,
        region: str,
        decomposition_df: pd.DataFrame,
        regional_gaps: pd.DataFrame,
        baseline_type: str,
        paradox_report: Optional[ParadoxReport] = None,
        cancellation_report: Optional[CancellationReport] = None
    ) -> NarrativeDecision:
        """Single flow narrative generation using template engine for all cases."""
        logger.info(f"Entering: BusinessAnalyzer.generate_regional_narrative()")
        logger.info(f"[DATA] INPUT: region={region}, baseline_type={baseline_type}")
        
        # Get region data from enriched DataFrames
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        total_gap = region_totals["total_net_gap"]
        total_construct_gap = region_totals["total_construct_gap"]
        total_performance_gap = region_totals["total_performance_gap"]
        gap_magnitude = region_totals["gap_magnitude"]
        
        logger.info(f"[DATA] METRICS: gap={total_gap:.3f}, construct={total_construct_gap:.3f}, performance={total_performance_gap:.3f}")
        
        # Step 1: Determine BusinessConclusion (handles ALL cases including edge cases)
        business_conclusion = self._classify_business_conclusion(
            region, total_gap, total_construct_gap, total_performance_gap, paradox_report, cancellation_report
        )
        
        # Step 2: Determine PerformanceDirection (universal)
        performance_direction = self._determine_performance_direction(total_gap)
        
        # Step 3: Assess gap significance for narrative context
        gap_significance = self.assess_gap_significance(gap_magnitude)
        
        # Step 4: Use template engine to create NarrativeDecision (single path for everything)
        narrative_decision = self.template_engine.create_narrative_decision(
            region, decomposition_df, regional_gaps, baseline_type, business_conclusion, performance_direction, gap_significance, cancellation_report
        )
        
        logger.info(f"[CLEAN] FINAL CLASSIFICATION: {business_conclusion.value} + {performance_direction.value}")
        return narrative_decision
    
    def _classify_business_conclusion(
        self, 
        region: str,
        total_gap: float,
        total_construct_gap: float,
        total_performance_gap: float,
        paradox_report: Optional[ParadoxReport] = None,
        cancellation_report: Optional[CancellationReport] = None
    ) -> BusinessConclusion:
        """Data-driven classification using edge case detection results."""
        
        # Trust the detection reports - they already contain the specialized narratives
        # If the detectors found edge cases AND created narratives, use them directly
        if (cancellation_report and 
            cancellation_report.cancellation_detected and 
            region in cancellation_report.affected_regions):
            # Don't escalate to a separate conclusion; treat as "no_gap with risk"
            if abs(total_gap) < self.thresholds.near_zero_gap:
                return BusinessConclusion.NO_SIGNIFICANT_GAP
            
        if (paradox_report and 
            paradox_report.paradox_detected and 
            paradox_report.business_impact in {GapSignificance.HIGH, GapSignificance.MEDIUM} and
            region in paradox_report.affected_regions):
            logger.info(f"[CLASSIFICATION] Using SIMPSON_PARADOX for {region} - from ParadoxReport")
            return BusinessConclusion.SIMPSON_PARADOX
        
        # Use consistent gap significance logic
        gap_magnitude = abs(total_gap)
        gap_significance = self.assess_gap_significance(gap_magnitude)
        if gap_significance == GapSignificance.LOW:
            return BusinessConclusion.NO_SIGNIFICANT_GAP
        
        # Use direct region-level gap comparison instead of cross-region averages
        # This is more accurate for individual region classification
        region_construct_gap = abs(total_construct_gap)
        region_performance_gap = abs(total_performance_gap)
        
        # Use centralized threshold for consistent business logic
        dominance_threshold = self.thresholds.minor_gap  # 0.5pp threshold
        
        if region_construct_gap > region_performance_gap + dominance_threshold:
            return BusinessConclusion.COMPOSITION_DRIVEN
        elif region_performance_gap > region_construct_gap + dominance_threshold:
            return BusinessConclusion.PERFORMANCE_DRIVEN
        else:
            # In balanced cases, default to performance_driven as it's more actionable
            return BusinessConclusion.PERFORMANCE_DRIVEN

    def _determine_performance_direction(self, total_gap: float) -> PerformanceDirection:
        """Pure data-driven performance direction logic."""
        # Use near_zero_gap threshold instead of hardcoded value
        if abs(total_gap) < self.thresholds.near_zero_gap:
            return PerformanceDirection.IN_LINE
        elif total_gap > 0:
            return PerformanceDirection.OUTPERFORMS
        else:
            return PerformanceDirection.UNDERPERFORMS

# =============================================================================
# EDGE CASE DETECTION
# =============================================================================

class EdgeCaseDetector:
    """
    Detection-only class for mathematical edge cases.
    
    This class only detects edge cases and returns detection reports.
    Narrative generation is handled by BusinessAnalyzer → NarrativeTemplateEngine flow.
    
    Responsibilities:
    - detect_simpson_paradox() → ParadoxReport  
    - detect_mathematical_cancellation() → CancellationReport
    
    Does not create narratives or tables (that's BusinessAnalyzer's job).
    """
    
    def __init__(self, thresholds: AnalysisThresholds = None):
        self.thresholds = thresholds or AnalysisThresholds()
    
    def _paradox_direction_ok(self, pooled: float, common: float) -> bool:
        mode = getattr(self.thresholds, "paradox_direction", "both")
        if mode == "under_only":
            return pooled < 0 and common > 0
        if mode == "over_only":
            return pooled > 0 and common < 0
        # "both"
        return (pooled * common) < 0
    
    def _pair_gap_vs_common_mix(self, rows: pd.DataFrame) -> Tuple[float, float]:
        """
        New helper for Simpson detection using common-mix comparator.
        
        Args:
            rows: per product; must include cols: region_rate, rest_rate, region_mix_pct, rest_mix_pct
        
        Returns:
            Tuple of (pooled_gap, common_mix_gap) in decimal form
        """
        # pooled headline diff (decimal)
        pooled = (rows['region_mix_pct']*rows['region_rate']).sum() - (rows['rest_mix_pct']*rows['rest_rate']).sum()
        
        # common-mix weights (exposure-based). If you don't have trials, use (mA+mB)
        if 'region_trials' in rows and 'rest_trials' in rows:
            w = rows['region_trials'].fillna(0) + rows['rest_trials'].fillna(0)
        else:
            w = rows['region_mix_pct'] + rows['rest_mix_pct']
        w = w / w.sum() if w.sum() else w
        common = (w * (rows['region_rate'] - rows['rest_rate'])).sum()
        
        return float(pooled), float(common)
    
    def detect_mathematical_cancellation(
        self,
        decomposition_df: pd.DataFrame,
        regional_gaps: pd.DataFrame,
    ) -> CancellationReport:
        """
        Flag regions where category-level effects materially oppose each other,
        leaving the topline ~zero. Avoids conflating pure composition gaps with cancellation.
        """
        affected_regions = []

        for _, row in regional_gaps.iterrows():
            region = row["region"]
            total_net = float(row["total_net_gap"])
            total_c = float(row["total_construct_gap"])
            total_p = float(row["total_performance_gap"])

            # 1) topline is essentially zero
            near_zero_topline = abs(total_net) < self.thresholds.near_zero_gap

            # 2) material opposing components
            material_components = (
                (abs(total_c) >= self.thresholds.significant_gap) or
                (abs(total_p) >= self.thresholds.significant_gap)
            )
            opposing = (total_c * total_p) < 0

            # 3) internal cancellation evidence: category gaps are large vs tiny topline
            reg_rows = decomposition_df[decomposition_df["region"] == region]
            sum_abs_cat = float(reg_rows["net_gap"].abs().sum())
            internal_cancel = sum_abs_cat >= (2.0 * abs(total_net) + self.thresholds.minor_gap)

            if near_zero_topline and material_components and opposing and internal_cancel:
                affected_regions.append(region)
        
        return CancellationReport(
            cancellation_detected=len(affected_regions) > 0,
            cancellation_type="aggregate_cancel",
            affected_regions=affected_regions
        )

    def detect_simpson_paradox(
        self,
        aggregate_results: pd.DataFrame,
        detailed_results: pd.DataFrame,
        top_n_focus: int = 3,
        parent_index: int = 0  # 0 = child[0] is the parent; 1 = child[1]; etc.
    ) -> ParadoxReport:
        """
        Detect Simpson's paradox between a parent rollup and its child breakdown.
        parent_index controls which position in the child tuple is considered the 'parent'.
        """
        logger.info("[ROUTING] ENTERING: EdgeCaseDetector.detect_simpson_paradox()")
        logger.info(f"[DATA] INPUT: {len(aggregate_results)} aggregate rows, {len(detailed_results)} detailed rows")
        
        paradox_cases = []
        affected_regions = set()

        def _to_tuple(x):
            if isinstance(x, tuple):
                return x
            s = str(x)
            if s.startswith("("):
                try:
                    t = ast.literal_eval(s)
                    if isinstance(t, (list, tuple)):
                        return tuple(t)
                except Exception:
                    pass
            return (s,)

        def _match_parent(parent, child, idx):
            t = _to_tuple(child)
            idx = min(idx, len(t) - 1)  # safe clamp
            return str(t[idx]) == str(parent)

        for region in aggregate_results["region"].unique():
            agg_region = aggregate_results[aggregate_results["region"] == region]
            for _, agg_row in agg_region.iterrows():
                parent_label = agg_row["category"]
                detailed_subset = detailed_results[
                    (detailed_results["region"] == region) & 
                    detailed_results["category"].apply(lambda x: _match_parent(parent_label, x, parent_index))
                ]
                if len(detailed_subset) <= 1:
                    continue

                # Aggregate-level RATE gap (direction at parent)
                pooled = float(agg_row["region_rate"] - agg_row["rest_rate"])

                # Child-level COMMON-MIX rate gap
                _, common = self._pair_gap_vs_common_mix(detailed_subset)

                # Materiality gating under unified thresholds
                if {'region_trials', 'rest_trials'}.issubset(detailed_subset.columns):
                    w = (detailed_subset['region_trials'].fillna(0) + detailed_subset['rest_trials'].fillna(0))
                else:
                    w = (detailed_subset['region_mix_pct'] + detailed_subset['rest_mix_pct'])
                w = w / w.sum() if w.sum() else w

                rate_diff = (detailed_subset['region_rate'] - detailed_subset['rest_rate'])
                disagree_share = float(w[np.sign(rate_diff) == (-1 if pooled > 0 else 1)].sum()) if pooled != 0 else float(w[rate_diff != 0].sum())
                impact_change = abs(abs(common) - abs(pooled))

                flip = self._paradox_direction_ok(pooled, common)
                material = (
                    max(abs(pooled), abs(common)) >= self.thresholds.paradox_material_gap and
                    disagree_share >= self.thresholds.paradox_disagree_share and
                    impact_change >= self.thresholds.paradox_impact_swing
                )

                logger.debug(f"[PARADOX] region={region} parent={parent_label} pooled={pooled:+.4f} "
                             f"common={common:+.4f} disagree_share={disagree_share:.2f} "
                             f"impactΔ={impact_change:+.4f}")

                if flip and material:
                    paradox_cases.append({
                        "region": region,
                        "category": parent_label,
                        "paradox_type": ParadoxType.MATHEMATICAL,
                        "aggregate_rate_gap": pooled,
                        "common_mix_rate_gap": common,
                        "disagree_share": disagree_share,
                        "impact_change": impact_change
                    })
                    affected_regions.add(region)

        if not paradox_cases:
            return ParadoxReport(False, [], ParadoxType.NONE, GapSignificance.LOW)

        max_gap = max(max(abs(c["aggregate_rate_gap"]), abs(c["common_mix_rate_gap"])) for c in paradox_cases)
        business_impact = BusinessAnalyzer(self.thresholds).assess_gap_significance(max_gap)
        return ParadoxReport(True, list(affected_regions), ParadoxType.MATHEMATICAL, business_impact)

    def detect_portfolio_simpson(self, product_rows: pd.DataFrame) -> ParadoxReport:
        """
        Portfolio-level Simpson detection across PRODUCTS for each region:
        Compare overall pooled gap vs common-mix rate gap (per region over its products).
        Expects columns: ['region','region_rate','rest_rate','region_mix_pct','rest_mix_pct']
        Optional: 'region_trials','rest_trials' for cleaner weights.
        """
        cases, affected = [], set()
        for region in product_rows['region'].unique():
            rows = product_rows[product_rows['region'] == region]
            if len(rows) <= 1:
                continue

            # pooled headline diff (decimal)
            pooled = float(
                (rows['region_mix_pct'] * rows['region_rate']).sum()
                - (rows['rest_mix_pct'] * rows['rest_rate']).sum()
            )

            # common-mix weights across PRODUCTS
            if {'region_trials', 'rest_trials'}.issubset(rows.columns):
                w = rows['region_trials'].fillna(0) + rows['rest_trials'].fillna(0)
            else:
                w = rows['region_mix_pct'] + rows['rest_mix_pct']
            w = w / w.sum() if w.sum() else w

            # common-mix rate gap
            diff = rows['region_rate'] - rows['rest_rate']
            common = float((w * diff).sum())

            # how much exposure disagrees with the pooled direction
            if pooled == 0:
                disagree_share = float(w[diff != 0].sum())
            else:
                disagree_share = float(w[np.sign(diff) == -np.sign(pooled)].sum())

            impact_change = abs(abs(common) - abs(pooled))
            flip = self._paradox_direction_ok(pooled, common)
            material = (
                max(abs(pooled), abs(common)) >= self.thresholds.paradox_material_gap and
                disagree_share >= self.thresholds.paradox_disagree_share and
                impact_change >= self.thresholds.paradox_impact_swing
            )

            if flip and material:
                cases.append({
                    "region": region,
                    "paradox_type": ParadoxType.MATHEMATICAL,
                    "aggregate_rate_gap": pooled,
                    "common_mix_rate_gap": common,
                    "disagree_share": disagree_share,
                    "impact_change": impact_change
                })
                affected.add(region)

        if not cases:
            return ParadoxReport(False, [], ParadoxType.NONE, GapSignificance.LOW)

        max_mag = max(max(abs(c["aggregate_rate_gap"]), abs(c["common_mix_rate_gap"])) for c in cases)
        impact = (GapSignificance.HIGH if max_mag >= self.thresholds.high_impact_gap
                  else GapSignificance.MEDIUM if max_mag >= self.thresholds.significant_gap
                  else GapSignificance.LOW)

        return ParadoxReport(True, list(affected), ParadoxType.MATHEMATICAL, impact)

# =============================================================================
# HELPER FUNCTIONS FOR MULTI-AXIS PARADOX DETECTION
# =============================================================================

def _rollup_by_axis(df_rows: pd.DataFrame, parent_index: int) -> pd.DataFrame:
    """
    Build parent-level aggregate rows from fine-grained rows by selecting element
    at parent_index from the category tuple. No category names are hard-coded.
    Expects df_rows with: ['region','category','region_rate','rest_rate','region_mix_pct','rest_mix_pct'].
    """
    def _to_tuple(x):
        if isinstance(x, tuple):
            return x
        s = str(x)
        if s.startswith("("):
            try:
                t = ast.literal_eval(s)
                if isinstance(t, (list, tuple)):
                    return tuple(t)
            except Exception:
                pass
        return (s,)

    tmp = df_rows.copy()
    tmp["_cat_tuple"] = tmp["category"].apply(_to_tuple)
    tmp["_parent"] = tmp["_cat_tuple"].apply(lambda t: t[parent_index] if len(t) > parent_index else t[-1])

    def _agg(g):
        r_mix = g["region_mix_pct"].sum()
        b_mix = g["rest_mix_pct"].sum()
        r_rate = float((g["region_mix_pct"] * g["region_rate"]).sum() / r_mix) if r_mix > 0 else 0.0
        b_rate = float((g["rest_mix_pct"]   * g["rest_rate"]).sum()   / b_mix) if b_mix > 0 else 0.0
        return pd.Series({
            "region_rate": r_rate,
            "rest_rate": b_rate,
            "region_mix_pct": float(r_mix),
            "rest_mix_pct": float(b_mix),
            "category": g.name[1],    # the parent label
            "region": g.name[0]
        })

    rolled = tmp.groupby(["region", "_parent"], as_index=False).apply(_agg)
    rolled = rolled.reset_index(drop=True)
    return rolled[["region","category","region_rate","rest_rate","region_mix_pct","rest_mix_pct"]]


def _merge_paradox(p1: ParadoxReport, p2: ParadoxReport) -> ParadoxReport:
    order = {GapSignificance.LOW: 0, GapSignificance.MEDIUM: 1, GapSignificance.HIGH: 2}
    if not p1.paradox_detected and not p2.paradox_detected:
        return ParadoxReport(False, [], ParadoxType.NONE, GapSignificance.LOW)
    regions = list(set(p1.affected_regions) | set(p2.affected_regions))
    impact = p1.business_impact if order[p1.business_impact] >= order[p2.business_impact] else p2.business_impact
    return ParadoxReport(True, regions, ParadoxType.MATHEMATICAL, impact)

# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_oaxaca_analysis(
    df: pd.DataFrame,
    region_column: str = "region",
    numerator_column: str = "numerator", 
    denominator_column: str = "denominator",
    category_columns: Union[str, List[str]] = "product",
    subcategory_columns: Optional[List[str]] = None,
    baseline_type: str = "rest_of_world",
    decomposition_method: str = "two_part",
    detect_edge_cases: bool = True,
    detect_cancellation: bool = True,
    thresholds: AnalysisThresholds = None
) -> AnalysisResult:
    """
    Enhanced Oaxaca-Blinder analysis with comprehensive edge case detection.
    
    Args:
        df: Input DataFrame with regional performance data
        region_column: Column containing region identifiers
        numerator_column: Column containing numerator for rate calculation
        denominator_column: Column containing denominator for rate calculation
        category_columns: Categories for decomposition (str or List[str])
        subcategory_columns: Optional subcategories for paradox detection
        baseline_type: Baseline method ("rest_of_world", "global_average", or region name)
        decomposition_method: Method for decomposition ("two_part", "reverse", "symmetric")
        detect_edge_cases: Whether to run edge case detection
        thresholds: Custom analysis thresholds
    
    Returns:
        AnalysisResult with decomposition, insights, and edge case detection
    """
    
    # Initialize components
    thresholds = thresholds or AnalysisThresholds()
    
    # Ensure category_columns is a list
    if isinstance(category_columns, str):
        category_columns = [category_columns]
    
    # Validate required columns
    required_columns = [region_column, numerator_column, denominator_column] + category_columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Step 1: Core decomposition analysis
    decomposition_results = []
    regions = df[region_column].unique()
    actual_gaps = {}  # For validation
    regional_overall_rates = {}  # Store calculated overall rates
    
    for region in regions:
        # Get region and baseline data
        region_data = filter_region_data(df, region_column, region)
        
        try:
            baseline_mix, baseline_rates = get_baseline_data(
                df, region_column, baseline_type, region,
                numerator_column, denominator_column, category_columns
            )
        except ValueError as e:
            logger.warning(f"Skipping region {region}: {e}")
            continue
        
        # Calculate region metrics
        region_mix = OaxacaCore.calculate_mix(region_data, category_columns, denominator_column)
        region_rates = OaxacaCore.calculate_rates(region_data, category_columns, numerator_column, denominator_column)
        
        # Calculate actual gap for validation - cache baseline overall rate
        region_overall_rate = OaxacaCore.calculate_rates(region_data, [], numerator_column, denominator_column).iloc[0]
        b_all_mix, b_all_rates = get_baseline_data(
            df, region_column, baseline_type, region, numerator_column, denominator_column, []
        )
        baseline_overall_rate = b_all_rates.iloc[0]
        actual_gaps[region] = region_overall_rate - baseline_overall_rate
        
        # IDENTITY CHECKS: Explicit topline validation
        # Region topline check
        lhs = (region_mix * region_rates).sum()
        rhs = region_overall_rate
        assert abs(lhs - rhs) < 1e-12, f"Region topline mismatch for {region}: {lhs} vs {rhs}"
        
        # Baseline topline check (using cached b_all_rates)
        assert abs((baseline_mix * baseline_rates).sum() - baseline_overall_rate) < 1e-12, \
               f"Baseline topline mismatch for {region}"
        
        # Store overall rates for narrative generation
        regional_overall_rates[region] = {
            'region_overall_rate': region_overall_rate,
            'baseline_overall_rate': baseline_overall_rate
        }
        
        # Align indices for consistent calculation
        baseline_rates = baseline_rates.reindex(region_mix.index, fill_value=0.0)
        baseline_mix = baseline_mix.reindex(region_mix.index, fill_value=0.0)
        
        # Process each category
        for category in region_mix.index:
            # Get values (with defaults for missing categories)
            region_mix_val = region_mix.get(category, 0)
            baseline_mix_val = baseline_mix.get(category, 0) 
            region_rate_val = region_rates.get(category, 0)
            baseline_rate_val = baseline_rates.get(category, 0)
            
            # Decompose gap
            construct_gap, performance_gap, net_gap = OaxacaCore.decompose_gap(
                region_mix_val, region_rate_val,
                baseline_mix_val, baseline_rate_val,
                decomposition_method
            )
            
            # Calculate additional metrics
            performance_index = region_rate_val / baseline_rate_val if baseline_rate_val != 0 else np.nan
            
            # Create result record
            decomposition_results.append({
                "region": region,
                "category": str(category),
                "region_mix_pct": region_mix_val,
                "rest_mix_pct": baseline_mix_val,
                "region_rate": region_rate_val,
                "rest_rate": baseline_rate_val,
                "construct_gap_contribution": construct_gap,
                "performance_gap_contribution": performance_gap,
                "net_gap": net_gap,
                "performance_index": performance_index
            })
    
    # Convert to DataFrame
    decomposition_df = pd.DataFrame(decomposition_results)
    
    # Step 2: Calculate regional aggregates
    regional_gaps = _calculate_regional_aggregates(decomposition_df, regional_overall_rates)
    
    # Step 2.5: EARLY ROUTING - Check for single-category cases and route appropriately
    # This prevents late routing issues and provides clear decision path
    single_category_regions = set()
    if subcategory_columns:
        for region in regions:
            region_data = decomposition_df[decomposition_df["region"] == region]
            unique_categories = region_data["category"].nunique()
            if unique_categories == 1:
                single_category_regions.add(region)
                logger.info(f"🎯 EARLY ROUTING: {region} has single category '{region_data['category'].iloc[0]}' - will use subcategory analysis")
    
    # Step 2.6: Create business analyzer
    business_analyzer = BusinessAnalyzer(thresholds)
    
    # Step 3: Mathematical validation
    decomposed_gaps = {
        row["region"]: row["total_net_gap"] 
        for _, row in regional_gaps.iterrows()
    }
    mathematical_validation = OaxacaCore.validate_decomposition(actual_gaps, decomposed_gaps, thresholds.mathematical_tolerance)
    
    # Log decomposition breakdown for each region (detailed debug info)
    for region in regions:
        region_decomp = decomposition_df[decomposition_df["region"] == region]
        for _, row in region_decomp.iterrows():
            logger.debug(f"[MATH] DECOMPOSITION: {region}-{row['category']}: construct={row['construct_gap_contribution']:+.3f}, performance={row['performance_gap_contribution']:+.3f}, net={row['net_gap']:+.3f}")
    
    # Step 4: Edge case detection (BEFORE business insights)
    paradox_report = ParadoxReport(False, [], ParadoxType.NONE, GapSignificance.LOW)
    subcategory_results = None
    edge_detector = EdgeCaseDetector(thresholds)

    # (A) Parent→child when you pass one axis + subcategories (e.g., roll up Product, children=(Product,Vertical))
    if detect_edge_cases and subcategory_columns:
        subcategory_results = run_oaxaca_analysis(
            df=df,
            region_column=region_column,
            numerator_column=numerator_column,
            denominator_column=denominator_column,
            category_columns=category_columns + subcategory_columns,
            baseline_type=baseline_type,
            decomposition_method=decomposition_method,
            detect_edge_cases=False,  # avoid recursion
            thresholds=thresholds
        )
        p_parent_child = edge_detector.detect_simpson_paradox(
            aggregate_results=decomposition_df,
            detailed_results=subcategory_results.decomposition_df,
            top_n_focus=3,
            parent_index=0  # children are tuples (category_columns..., subcategory_columns...), parent is first position
        )
        if p_parent_child.paradox_detected:
            paradox_report = _merge_paradox(paradox_report, p_parent_child)

    # (B) NEW: Portfolio-level (overall ↔ current portfolio granularity)
    p_portfolio = edge_detector.detect_portfolio_simpson(product_rows=decomposition_df)
    if p_portfolio.paradox_detected:
        paradox_report = _merge_paradox(paradox_report, p_portfolio)

    # (C) NEW: If you passed MULTIPLE axes directly (e.g., category_columns=['A','B'] and no subcategories),
    # roll up separately by each axis and scan both directions. We only use tuple positions (no names).
    if detect_edge_cases and len(category_columns) > 1 and (not subcategory_columns):
        # parent = axis 0
        agg_axis0 = _rollup_by_axis(decomposition_df, parent_index=0)
        p_axis0 = edge_detector.detect_simpson_paradox(
            aggregate_results=agg_axis0,
            detailed_results=decomposition_df,
            top_n_focus=3,
            parent_index=0
        )
        if p_axis0.paradox_detected:
            paradox_report = _merge_paradox(paradox_report, p_axis0)

        # parent = axis 1
        agg_axis1 = _rollup_by_axis(decomposition_df, parent_index=1)
        p_axis1 = edge_detector.detect_simpson_paradox(
            aggregate_results=agg_axis1,
            detailed_results=decomposition_df,
            top_n_focus=3,
            parent_index=1
        )
        if p_axis1.paradox_detected:
            paradox_report = _merge_paradox(paradox_report, p_axis1)
    
    # Enrich DataFrames with all computed columns needed downstream
    enriched_decomposition_df = enrich_decomposition_df(decomposition_df, thresholds)
    enriched_regional_gaps = enrich_regional_gaps_df(regional_gaps)
    
    # Step 6: Mathematical cancellation detection
    cancellation_report = CancellationReport(
        cancellation_detected=False,
        cancellation_type="none",
        affected_regions=[]
    )
    
    if detect_cancellation:
        edge_detector = EdgeCaseDetector(thresholds)
        cancellation_report = edge_detector.detect_mathematical_cancellation(
            decomposition_df, regional_gaps
        )
        
        # sanity-check: every flagged region truly looks like cancellation
        for r in cancellation_report.affected_regions:
            row = enriched_regional_gaps.loc[enriched_regional_gaps["region"] == r].iloc[0]
            assert abs(row["total_net_gap"]) < thresholds.near_zero_gap, f"{r}: cancellation flagged but net gap not near zero"
            assert row["total_construct_gap"] * row["total_performance_gap"] < 0, f"{r}: cancellation flagged but components not opposing"
    
    # Step 5: Generate business insights using clean architecture
    narrative_decisions = {}
    
    for region in regions:
        if region in regional_gaps["region"].values:
            # For Simpson's Paradox, allow all detection paths (subcategory, portfolio, or multi-axis rollups)
            is_paradox_case = (
                detect_edge_cases and
                getattr(paradox_report, 'paradox_detected', False) and
                region in getattr(paradox_report, 'affected_regions', [])
            )

            # Use early routing decision for single-category cases
            is_single_category = (region in single_category_regions and subcategory_results is not None)

            if is_paradox_case or is_single_category:
                reason = "Simpson's Paradox" if is_paradox_case else "single category analysis (early routing detected)"
                logger.info(f"🎯 COORDINATOR: Using HYBRID data for {reason} in {region}")

                if subcategory_results:
                    analysis_decomposition_df = enrich_decomposition_df(subcategory_results.decomposition_df, thresholds)
                else:
                    # paradox found without subcategory rerun (portfolio or multi-axis rollups)
                    analysis_decomposition_df = enriched_decomposition_df

                analysis_regional_gaps = enriched_regional_gaps
            else:
                analysis_decomposition_df = enriched_decomposition_df
                analysis_regional_gaps = enriched_regional_gaps
            
            # CLEAN ARCHITECTURE: Generate narrative using enriched DataFrames
            logger.info(f"🎯 COORDINATOR: Generating narrative for {region}")
            narrative_decision = business_analyzer.generate_regional_narrative(
                region=region,
                decomposition_df=analysis_decomposition_df,
                regional_gaps=analysis_regional_gaps,
                baseline_type=baseline_type,
                paradox_report=paradox_report if region in getattr(paradox_report, 'affected_regions', []) else None,
                cancellation_report=cancellation_report if region in getattr(cancellation_report, 'affected_regions', []) else None
            )
            
            # Store NarrativeDecision in narrative_decisions
            narrative_decisions[region] = narrative_decision
            
            # ASSERT BUSINESS LOGIC CORRECTNESS (fail fast on wrong classifications)
            region_totals = analysis_regional_gaps[analysis_regional_gaps["region"] == region].iloc[0]
            total_gap = region_totals["total_net_gap"]
            if abs(total_gap) > thresholds.near_zero_gap:  # Use proper threshold
                expected_direction = PerformanceDirection.OUTPERFORMS if total_gap > 0 else PerformanceDirection.UNDERPERFORMS
                actual_direction = narrative_decision.performance_direction
                assert actual_direction == expected_direction, f"Wrong direction for {region}: gap={total_gap:.3f}, got {actual_direction.value}, expected {expected_direction.value}"
            
            logger.info(f"[ROUTING] COORDINATOR: Stored {narrative_decision.root_cause_type.value} analysis for {region}")
            logger.info(f"[DECISION] FINAL_CLASSIFICATION: {region} → {narrative_decision.root_cause_type.value} + {narrative_decision.performance_direction.value}")
    
    # Create analysis result with narrative_decisions
    logger.info("[ROUTING] COORDINATOR: Creating AnalysisResult with narrative decisions")
    
    # LOG DATA SUMMARY: What the analysis found
    logger.info(f"[DATA] ANALYSIS_SUMMARY: {len(decomposition_df)} decomposition rows across {len(regions)} regions")
    logger.info(f"[DECISION] EDGE_CASES: Paradox={paradox_report.paradox_detected}, Cancellation={cancellation_report.cancellation_detected}")
    logger.info(f"[DECISION] NARRATIVE_DECISIONS: {len(narrative_decisions)} regions with enum-driven analysis")
    for region, decision in narrative_decisions.items():
        logger.info(f"[DECISION] ENUM_CLASSIFICATION: {region} → {decision.root_cause_type.value} + {decision.performance_direction.value}")
    
    analysis_result = AnalysisResult(
        decomposition_df=enriched_decomposition_df,
        regional_gaps=enriched_regional_gaps,
        mathematical_validation=mathematical_validation,
        narrative_decisions=narrative_decisions,  # Clean enum-driven narrative system
        paradox_report=paradox_report,
        cancellation_report=cancellation_report,
        baseline_type=baseline_type  # Pass baseline_type from orchestrator
    )
    
    return analysis_result

# =============================================================================
# DATAFRAME ENRICHMENT UTILITIES
# =============================================================================

def enrich_decomposition_df(decomposition_df: pd.DataFrame, thresholds: AnalysisThresholds = None) -> pd.DataFrame:
    """Enrich decomposition DataFrame with all computed columns needed downstream."""
    thresholds = thresholds or AnalysisThresholds()
    enriched = decomposition_df.copy()
    
    # Add formatting columns for display
    enriched['region_rate_pct'] = (enriched['region_rate'] * 100).round(1).astype(str) + '%'
    enriched['rest_rate_pct'] = (enriched['rest_rate'] * 100).round(1).astype(str) + '%'
    enriched['net_gap_pp'] = (enriched['net_gap'] * 100).round(1).astype(str) + 'pp'
    enriched['region_mix_display'] = (enriched['region_mix_pct'] * 100).round(1).astype(str) + '%'
    enriched['rest_mix_display'] = (enriched['rest_mix_pct'] * 100).round(1).astype(str) + '%'
    
    # Add absolute gap for ranking and cumulative analysis
    enriched['abs_gap'] = enriched['net_gap'].abs()
    
    # CRITICAL: Keep numeric values for calculations
    enriched['net_gap_numeric'] = (enriched['net_gap'] * 100).round(1)  # Numeric pp for calculations
    enriched['abs_gap_numeric'] = enriched['net_gap_numeric'].abs()
    
    # Add clean category names for display
    enriched['category_clean'] = enriched['category'].apply(_clean_category_display_name)
    
    # Add contributor text (formatted for narrative)
    enriched['contributor_text'] = enriched.apply(lambda row: 
        f"{row['category_clean']} ({row['net_gap_pp']})", axis=1)
    
    # Add positive/negative contributor text separately - use numeric values for comparison
    enriched['positive_contributor_text'] = enriched.apply(lambda row:
        f"{row['category_clean']} (+{row['net_gap_numeric']:.0f}pp)" if row['net_gap_numeric'] > 0 else None, axis=1)
    enriched['negative_contributor_text'] = enriched.apply(lambda row:
        f"{row['category_clean']} ({row['net_gap_numeric']:.0f}pp)" if row['net_gap_numeric'] < 0 else None, axis=1)
    
    # Add ranking within each region - use numeric values for consistent ranking
    enriched['gap_rank'] = enriched.groupby('region')['abs_gap_numeric'].rank(method='dense', ascending=False)
    
    # Add cumulative contribution analysis per region - use numeric values consistently
    for region in enriched['region'].unique():
        region_mask = enriched['region'] == region
        region_data = enriched[region_mask].sort_values('abs_gap_numeric', ascending=False)
        
        # If only one category, it's always a top contributor
        if len(region_data) == 1:
            enriched.loc[region_mask, 'cumsum_impact'] = region_data['abs_gap_numeric'].iloc[0]
            enriched.loc[region_mask, 'is_top_contributor'] = True
        else:
            total_impact = region_data['abs_gap_numeric'].sum()
            cumsum_threshold = total_impact * thresholds.cumulative_contribution_threshold
            
            # Calculate cumulative sum in order of importance
            region_data_sorted = region_data.sort_values('abs_gap_numeric', ascending=False)
            cumsum_impact = region_data_sorted['abs_gap_numeric'].cumsum()
            
            # Map back to original positions in enriched DataFrame
            for idx, (orig_idx, row) in enumerate(region_data_sorted.iterrows()):
                enriched.loc[orig_idx, 'cumsum_impact'] = cumsum_impact.iloc[idx]
                enriched.loc[orig_idx, 'is_top_contributor'] = cumsum_impact.iloc[idx] <= cumsum_threshold
            
            # IMPORTANT: Ensure at least the top contributor is marked if none qualify
            # This handles cases where threshold is too restrictive for small datasets
            region_top_contributors = enriched[region_mask]['is_top_contributor'].sum()
            if region_top_contributors == 0 and len(region_data) > 0:
                # Mark the largest contributor as a top contributor - use numeric values
                largest_contrib_idx = region_data['abs_gap_numeric'].idxmax()
                enriched.loc[largest_contrib_idx, 'is_top_contributor'] = True
    
    # Add status indicators using enum values
    enriched['performance_status'] = enriched['net_gap'].apply(
        lambda x: PerformanceDirection.OUTPERFORMS.value if x > 0 
                 else PerformanceDirection.UNDERPERFORMS.value if x < 0 
                 else PerformanceDirection.IN_LINE.value
    )
    
    return enriched

def enrich_regional_gaps_df(regional_gaps: pd.DataFrame) -> pd.DataFrame:
    """Enrich regional gaps DataFrame with computed columns."""
    enriched = regional_gaps.copy()
    
    # Add formatted columns
    enriched['total_gap_pp'] = (enriched['total_net_gap'] * 100).round(1).astype(str) + 'pp'
    enriched['construct_gap_pp'] = (enriched['total_construct_gap'] * 100).round(1).astype(str) + 'pp'
    enriched['performance_gap_pp'] = (enriched['total_performance_gap'] * 100).round(1).astype(str) + 'pp'
    
    enriched['region_overall_rate_pct'] = (enriched['region_overall_rate'] * 100).round(1).astype(str) + '%'
    enriched['baseline_overall_rate_pct'] = (enriched['baseline_overall_rate'] * 100).round(1).astype(str) + '%'
    
    # Add magnitude and direction
    enriched['gap_magnitude'] = enriched['total_net_gap'].abs()
    enriched['gap_direction'] = enriched['total_net_gap'].apply(
        lambda x: PerformanceDirection.OUTPERFORMS.value if x > 0 else PerformanceDirection.UNDERPERFORMS.value
    )
    
    return enriched

def _clean_category_display_name(category) -> str:
    """Clean category name for display."""
    if isinstance(category, tuple):
        return ' → '.join(str(c) for c in category)
    return str(category)

def _calculate_regional_aggregates(
    decomposition_df: pd.DataFrame, 
    regional_overall_rates: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Calculate regional-level gap aggregates and include overall rates."""
    
    regional_gaps = decomposition_df.groupby("region").agg({
        "construct_gap_contribution": "sum",
        "performance_gap_contribution": "sum", 
        "net_gap": "sum",
        "region_mix_pct": "sum",  # Should sum to 1.0
        "rest_mix_pct": "sum"     # Should sum to 1.0
    }).reset_index()
    
    regional_gaps = regional_gaps.rename(columns={
        "construct_gap_contribution": "total_construct_gap",
        "performance_gap_contribution": "total_performance_gap",
        "net_gap": "total_net_gap"
    })
    
    # Include overall rates in regional aggregates
    regional_gaps['region_overall_rate'] = regional_gaps['region'].map(
        lambda r: regional_overall_rates[r]['region_overall_rate']
    )
    regional_gaps['baseline_overall_rate'] = regional_gaps['region'].map(
        lambda r: regional_overall_rates[r]['baseline_overall_rate']
    )
    
    return regional_gaps

def filter_region_data(
    df: pd.DataFrame,
    region_column: str,
    target_region: str,
    exclude: bool = False
) -> pd.DataFrame:
    """Filter DataFrame for specific region data."""
    if exclude:
        return df[df[region_column] != target_region]
    else:
        return df[df[region_column] == target_region]

def get_baseline_data(
    df: pd.DataFrame,
    region_column: str,
    baseline_type: str,
    current_region: str,
    numerator_column: str,
    denominator_column: str,
    category_columns: List[str]
) -> Tuple[pd.Series, pd.Series]:
    """Get baseline mix and rates based on baseline type."""
    
    if baseline_type == "rest_of_world":
        # Default behavior - exclude current region
        baseline_data = filter_region_data(df, region_column, current_region, exclude=True)
        
    elif baseline_type == "global_average":
        # Use all data as baseline (including current region)
        baseline_data = df
        
    elif baseline_type == "top_performer":
        # Create synthetic "top performer" baseline using best rate for each category
        # This avoids the issue of one region being best overall but not best in each category
        
        # Exclude current region from consideration
        other_regions_data = df[df[region_column] != current_region]
        
        if len(other_regions_data) == 0:
            raise ValueError(
                f"No other regions available for top_performer baseline for region {current_region}"
            )
        
        # For each category, find the best performing region and use that rate
        # Calculate rates by category for all other regions
        category_rates_by_region = {}
        for region in other_regions_data[region_column].unique():
            region_data = other_regions_data[other_regions_data[region_column] == region]
            region_rates = OaxacaCore.calculate_rates(
                region_data, category_columns, numerator_column, denominator_column
            )
            category_rates_by_region[region] = region_rates
        
        # Find best rate for each category across all regions
        all_categories = set()
        for rates in category_rates_by_region.values():
            all_categories.update(rates.index)
        
        best_rates = {}
        for category in all_categories:
            category_rates = []
            for region, rates in category_rates_by_region.items():
                if category in rates and not pd.isna(rates[category]):
                    category_rates.append(rates[category])
            
            if category_rates:
                best_rates[category] = max(category_rates)
        
        # Calculate mix using all other regions (not just the best performer)
        baseline_mix = OaxacaCore.calculate_mix(
            other_regions_data, category_columns, denominator_column
        )
        baseline_rates = pd.Series(best_rates)
        
        logger.info(
            f"Top performer baseline created using best rates across categories for {current_region}"
        )
        return baseline_mix, baseline_rates
        
    elif baseline_type == "previous_period":
        # This would require time-series data with a period column
        # For now, raise an error with guidance
        raise NotImplementedError(
            "previous_period baseline requires time-series data. "
            "Use oaxaca_blinder_time_series function instead."
        )
        
    elif baseline_type in df[region_column].unique():
        # Specific region baseline
        baseline_data = df[df[region_column] == baseline_type]
        
    else:
        raise ValueError(
            f"Invalid baseline_type: {baseline_type}. "
            f"Must be 'rest_of_world', 'global_average', 'top_performer', "
            f"'previous_period', or a valid region name from: {df[region_column].unique()}"
        )
    
    if len(baseline_data) == 0:
        raise ValueError(f"No data available for baseline_type: {baseline_type}")
    
    baseline_mix = OaxacaCore.calculate_mix(baseline_data, category_columns, denominator_column)
    baseline_rates = OaxacaCore.calculate_rates(baseline_data, category_columns, numerator_column, denominator_column)
    
    return baseline_mix, baseline_rates

# =============================================================================
# MINIMAL UNIT TEST FUNCTIONS
# =============================================================================

def _assert_oaxaca_identity_rows(df):
    """Minimal unit test you can drop in - validate Oaxaca identity per region."""
    # df is your per-product decomposition for ONE region
    lhs = (df['region_mix_pct']*df['region_rate']).sum() - (df['rest_mix_pct']*df['rest_rate']).sum()
    rhs = df['net_gap'].sum()
    assert abs(lhs - rhs) < 1e-12, f"Row identity failed: {lhs} vs {rhs}"

def _assert_oaxaca_identity_totals(regional_gaps):
    """Validate that totals split adds up correctly."""
    assert np.allclose(
        regional_gaps['total_net_gap'],
        regional_gaps['total_construct_gap'] + regional_gaps['total_performance_gap'],
        atol=1e-12
    ), "Totals split does not add up"


# --- AUTO SLICE SELECTION ---

@dataclass
class BestSliceDecision:
    focus_region: str
    best_categories: Tuple[str, ...]
    score: float
    score_breakdown: Dict[str, float]
    top_segments: pd.DataFrame
    analysis_result: AnalysisResult  # full result for the winning cut


def _normalize_0_1(x, lo, hi):
    if hi <= lo: return 0.0
    x = max(lo, min(hi, x))
    return (x - lo) / (hi - lo)


def _topk_evidence(decomp_region_df: pd.DataFrame, thresholds: AnalysisThresholds, k: int = 5):
    """Return top-k rows by |gap| plus concentration/coverage metrics."""
    d = decomp_region_df.sort_values("abs_gap_numeric", ascending=False).copy()
    if len(d) == 0:
        return d.head(0), 0.0, 0.0, 0.0

    topk = d.head(k)
    concentration = float(topk["abs_gap_numeric"].sum() / (d["abs_gap_numeric"].sum() or 1.0))
    exposure_share = float(topk["region_mix_pct"].sum())  # 0..1
    # Penalize tiny segments among top-k (default tiny < 3% exposure)
    tiny_cutoff = 0.03
    tiny_rate = float((topk["region_mix_pct"] < tiny_cutoff).mean())
    return topk, concentration, exposure_share, tiny_rate


def _score_cut(
    df: pd.DataFrame,
    region: str,
    category_cols: Iterable[str],
    thresholds: AnalysisThresholds,
    baseline_type: str,
    numerator_column: str,
    denominator_column: str,
    decomposition_method: str
) -> Tuple[float, Dict[str, float], AnalysisResult, pd.DataFrame]:
    """
    Run Oaxaca for this cut, compute a single score, and return diagnostics.
    """
    # 1) Run the analysis for this cut
    ar = run_oaxaca_analysis(
        df=df,
        region_column="region",
        numerator_column=numerator_column,
        denominator_column=denominator_column,
        category_columns=list(category_cols),
        subcategory_columns=None,           # parent→child not needed to rank cuts
        baseline_type=baseline_type,
        decomposition_method=decomposition_method,
        detect_edge_cases=True,             # keep paradox/cancellation signals
        thresholds=thresholds
    )

    # 2) Pull the region's rows
    rg = ar.regional_gaps[ar.regional_gaps["region"] == region]
    if len(rg) == 0:
        return 0.0, {"impact_norm": 0, "concentration": 0, "coverage": 0, "paradox_bonus": 0, "small_penalty": 0}, ar, ar.decomposition_df.head(0)

    total_gap = float(abs(rg["total_net_gap"].iloc[0]))            # decimal
    # normalize impact to [0,1] using a cap (e.g., 10pp)
    impact_norm = _normalize_0_1(total_gap, 0.0, 0.10)

    # 3) Region-level decomposition rows
    dreg = ar.decomposition_df[ar.decomposition_df["region"] == region]

    # 4) Top-k concentration / coverage / small-seg penalty
    topk, conc, exposure_share, tiny_rate = _topk_evidence(dreg, thresholds, k=5)

    # 5) Paradox bonus (if this cut surfaced a meaningful paradox for this region)
    paradox_bonus = 0.0
    if getattr(ar.paradox_report, "paradox_detected", False) and region in getattr(ar.paradox_report, "affected_regions", []):
        if ar.paradox_report.business_impact in {GapSignificance.MEDIUM, GapSignificance.HIGH}:
            paradox_bonus = 0.25  # fixed bump; you can tune

    # 6) Compose the score (all 0..1 terms except bonus/penalty)
    # weights: actionability first (conc), then impact, then coverage; penalize tiny segments
    coverage = exposure_share  # already 0..1
    small_penalty = 0.10 * tiny_rate

    score = (
        0.45 * conc +
        0.25 * impact_norm +
        0.20 * coverage +
        paradox_bonus -
        small_penalty
    )

    breakdown = {
        "impact_norm": impact_norm,
        "concentration": conc,
        "coverage": coverage,
        "paradox_bonus": paradox_bonus,
        "small_penalty": small_penalty
    }
    return float(score), breakdown, ar, topk


def auto_find_best_slice(
    df: pd.DataFrame,
    *,
    candidate_cuts: Iterable[Iterable[str]],
    focus_region: Optional[str] = None,
    baseline_type: str = "rest_of_world",
    numerator_column: str = "wins",
    denominator_column: str = "total",
    thresholds: AnalysisThresholds = None,
    decomposition_method: str = "two_part",
    portfolio_mode: bool = False
) -> BestSliceDecision:
    """
    Try multiple 'cuts' (different category_columns) and pick the one that most cleanly
    explains the problem for a region.

    Modes:
      - focus_region provided → optimize for that region.
      - portfolio_mode=True   → pick the region with the largest |negative| net gap first,
                                then find the best cut for that region.

    candidate_cuts: e.g., [["product"], ["vertical"], ["product","vertical"], ["channel"], ...]
    """
    thresholds = thresholds or AnalysisThresholds()

    # Choose region if portfolio_mode
    target_region = focus_region
    if portfolio_mode and focus_region is None:
        # Roll up with a simple, low-cost cut to identify the worst region (use the widest cut if provided)
        wide_cut = max(candidate_cuts, key=lambda cols: len(list(cols)))
        ar_probe = run_oaxaca_analysis(
            df=df,
            region_column="region",
            numerator_column=numerator_column,
            denominator_column=denominator_column,
            category_columns=list(wide_cut),
            baseline_type=baseline_type,
            decomposition_method=decomposition_method,
            detect_edge_cases=False,
            thresholds=thresholds
        )
        # pick most negative gap (largest underperformance)
        g = ar_probe.regional_gaps.sort_values("total_net_gap").reset_index(drop=True)
        if len(g) == 0:
            raise ValueError("No regions found to evaluate.")
        target_region = str(g.iloc[0]["region"])  # worst first

    if target_region is None:
        # fallback: pick the region with the largest |gap|
        ar_probe = run_oaxaca_analysis(
            df=df,
            region_column="region",
            numerator_column=numerator_column,
            denominator_column=denominator_column,
            category_columns=list(next(iter(candidate_cuts))),
            baseline_type=baseline_type,
            decomposition_method=decomposition_method,
            detect_edge_cases=False,
            thresholds=thresholds
        )
        gg = ar_probe.regional_gaps.reindex(
            ar_probe.regional_gaps["total_net_gap"].abs().sort_values(ascending=False).index
        )
        target_region = str(gg.iloc[0]["region"])

    # Evaluate all cuts
    best: Optional[BestSliceDecision] = None
    for cols in candidate_cuts:
        cols = tuple(cols)
        try:
            score, breakdown, ar, topk = _score_cut(
                df=df,
                region=target_region,
                category_cols=cols,
                thresholds=thresholds,
                baseline_type=baseline_type,
                numerator_column=numerator_column,
                denominator_column=denominator_column,
                decomposition_method=decomposition_method
            )
        except Exception as e:
            logger.debug(f"[AUTO-SLICE] Skipping {cols}: {e}")
            continue

        if (best is None) or (score > best.score):
            best = BestSliceDecision(
                focus_region=target_region,
                best_categories=cols,
                score=score,
                score_breakdown=breakdown,
                top_segments=topk[["category","region_rate_pct","rest_rate_pct","net_gap_pp","region_mix_display"]],
                analysis_result=ar
            )

    if best is None:
        raise RuntimeError("Auto slice failed to evaluate any candidate cuts.")

    return best
