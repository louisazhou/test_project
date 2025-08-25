# pyre-ignore-all-errors
"""
Oaxaca-Blinder Decomposition Analysis Engine

Four-layer architecture with single source of truth for all analytics.

Core Architecture:
    1. Math (OaxacaCore): Pure Oaxaca-Blinder decomposition with validation  
    2. Detection (EdgeCaseDetector): Simpson's Paradox and mathematical cancellation
    3. Narrative (NarrativeTemplateEngine): Text composition and evidence tables (computed once)
    4. Presentation (present_executive_pack_for_slides): Assembly only - no analytics

Key Driver Functions:
    run_oaxaca_analysis() - Core analysis engine (numeric results)
    auto_find_best_slice() - Smart dimensional cut selection
    
    AnalysisResult.present_executive_pack_for_slides() - Single slide factory returning (slide_spec, payload)
    build_oaxaca_exec_pack() - Unified driver for AUTO/FIXED modes with alternatives

Usage Examples:

    # Standard analysis (numeric by default)
    from rca_package.oaxaca_blinder import run_oaxaca_analysis
    result = run_oaxaca_analysis(
        df=data, region_column="region", category_columns=["product"],
        numerator_column="wins", denominator_column="total"
    )
    summary = result.get_summary_stats()  # numeric
    summary_display = result.get_summary_stats(format_for_display=True)
    
    # Generate slide-ready bundle from existing analysis (returns tuple)
    slide_spec, payload = result.present_executive_pack_for_slides(
        region="WEST", max_charts=2, charts=["rates_and_mix", "top_drivers"]
    )
    
    # Unified driver for AUTO mode (finds best cut + alternatives)
    from rca_package.oaxaca_blinder import build_oaxaca_exec_pack
    pack = build_oaxaca_exec_pack(
        df=data, 
        candidate_cuts=[["product"], ["channel"]],
        focus_region="WEST",  # or None for portfolio mode
        alternatives=1        # include 1 alternative cut
    )
    
    # FIXED mode for specific cut
    pack = build_oaxaca_exec_pack(
        df=data,
        fixed_cut=["product"],
        fixed_region="WEST"
    )
    
    # Extract slide components from unified API
    for slide in pack["slides"]:
        summary_text = slide["summary"]["summary_text"]
        slide_info = slide["slide_info"]
        for spec in slide_info["figure_generators"]:
            fig, ax = spec["function"](**spec["params"])
            caption = spec["caption"]

Architecture:
    - Core produces numeric DataFrames only (no display strings/emojis)
    - Display formatting via slide drivers or format_for_display=True
    - Visualization registry in separate viz_oaxaca.py module
    - Clean separation: math → business logic → presentation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Iterable, Any, Callable
from dataclasses import dataclass
from enum import Enum
import ast 
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---- UTILITY FUNCTIONS ----

def as_tuple(category) -> tuple:
    """Robustly coerce category (tuple, stringified tuple, scalar) to a tuple."""
    if isinstance(category, tuple):
        return category
    s = str(category)
    if s.startswith("("):
        try:
            t = ast.literal_eval(s)
            if isinstance(t, (list, tuple)):
                return tuple(t)
        except Exception:
            pass
    return (s,)

def pretty_category(category) -> str:
    """Consistent display name for any category."""
    t = as_tuple(category)
    return " → ".join(str(x) for x in t)

def gap_direction_from(x: float, thresholds: 'AnalysisThresholds') -> 'PerformanceDirection':
    """Determine performance direction from gap value using thresholds."""
    if abs(x) < thresholds.near_zero_gap:
        return PerformanceDirection.IN_LINE
    return PerformanceDirection.OUTPERFORMS if x > 0 else PerformanceDirection.UNDERPERFORMS

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

    minor_rate_diff: float = 0.005       # 0.5pp rate difference
    
    # Narrative thresholds
    visual_indicator_threshold: float = 0.01  # 1pp - for visual indicators (🟢/🔴)
    cumulative_contribution_threshold: float = 0.6  # 60% - for ranking top contributors
    
    # PERCENT-POINT thresholds (explicitly in pp)
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
    
    # Evidence bundle (single source of truth)
    evidence_tables: Dict[str, pd.DataFrame]   # {"rates_mix": df, "top_drivers": df, "contradiction": df}
    contradiction_segments: List[str]          # ranked labels used by "despite …"

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
      
    def get_summary_stats(self, format_for_display: bool = False) -> Dict:
        """
        Get organized original DataFrames for portfolio analysis.
        
        Returns the actual analysis DataFrames ranked and organized for insight,
        avoiding over-processed statistics that dilute truth.
        
        Returns:
            Dict containing:
            - regional_performance_summary: Regional gaps ranked by performance
            - top_category_contributors: Biggest drivers across all regions
            - paradox_regions: Regions with Simpson's paradox detected
            - cancellation_regions: Regions with mathematical cancellation
        """
        logger.info("🎯 GENERATING ORGANIZED DATAFRAMES for portfolio analysis")
        
        if len(self.regional_gaps) == 0:
            return {"error": "No regional data available for analysis"}
        
        regional_summary = self.regional_gaps[['region','total_net_gap','total_performance_gap','total_construct_gap']].copy()

        # merge root causes (no strings yet)
        for region in regional_summary['region']:
            if region in self.narrative_decisions:
                dec = self.narrative_decisions[region]
                regional_summary.loc[regional_summary['region'] == region, 'root_cause'] = dec.root_cause_type.value
                regional_summary.loc[regional_summary['region'] == region, 'performance_direction'] = dec.performance_direction.value
        regional_summary = regional_summary.sort_values('total_net_gap', ascending=False)
        
        # top contributors (numeric)
        top_contributors = self.decomposition_df[['region','category','net_gap','construct_gap_contribution','performance_gap_contribution','abs_gap']].copy()
        top_contributors = top_contributors.nlargest(15, 'abs_gap')

        if format_for_display:
            base_cols = ['region','region_overall_rate','baseline_overall_rate']
            rs = regional_summary.merge(self.regional_gaps[base_cols], on='region', how='left')
            rs = format_regional_gaps_for_display(rs)
            tc = top_contributors.copy()
            tc['net_gap_pp'] = (tc['net_gap']*100).round(1).astype(str)+'pp'
            tc['construct_gap_pp'] = (tc['construct_gap_contribution']*100).round(1).astype(str)+'pp'
            tc['performance_gap_pp'] = (tc['performance_gap_contribution']*100).round(1).astype(str)+'pp'
            tc = tc[['region','category','net_gap_pp','construct_gap_pp','performance_gap_pp']]
            return {
                "regional_performance_summary": rs[['region','total_net_gap_pp','total_performance_gap_pp','total_construct_gap_pp','root_cause','performance_direction']],
                "top_category_contributors": tc,
                "paradox_regions": self.paradox_report.affected_regions if self.paradox_report.paradox_detected else [],
                "cancellation_regions": self.cancellation_report.affected_regions if self.cancellation_report.cancellation_detected else []
            }

        # raw numeric by default
        return {
            "regional_performance_summary": regional_summary.reset_index(drop=True),
            "top_category_contributors": top_contributors[['region','category','net_gap','construct_gap_contribution','performance_gap_contribution']].reset_index(drop=True),
            "paradox_regions": self.paradox_report.affected_regions if self.paradox_report.paradox_detected else [],
            "cancellation_regions": self.cancellation_report.affected_regions if self.cancellation_report.cancellation_detected else []
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
    
    def present_executive_pack_for_slides(
        self,
        region: str,
        *,
        max_charts: int = 2,
        charts: Optional[List[str]] = None,   # explicit chart keys from viz.VIZ_REGISTRY
        table_rows: int = 10,
        custom_title: Optional[str] = None  # Allow custom title override
    ) -> Tuple[Dict, Dict]:
        """
        Slide-friendly bundle for a single region.
        - No files saved; we return function+params specs (callables) and DFs.
        - 'charts' lets the caller force which exhibits to render (order respected).
          Valid names: rates_and_mix, driver_matrix, top_drivers
        - If 'charts' is None, we auto-pick based on root cause.
        """
        # 1) One-liner summary for the summary page
        decision = self.narrative_decisions[region]
        summary_text = decision.narrative_text

        # 2) Get shares from key_metrics (computed once in narrative engine)
        gap_pp   = float(decision.key_metrics["total_gap"]) * 100
        reg_pp   = float(self.regional_gaps[self.regional_gaps["region"] == region].iloc[0]["region_overall_rate"]) * 100
        base_pp  = float(self.regional_gaps[self.regional_gaps["region"] == region].iloc[0]["baseline_overall_rate"]) * 100
        
        # Use pre-computed shares from key_metrics or calculate simple ones
        mix_abs = abs(float(decision.key_metrics["construct_gap"]))
        exe_abs = abs(float(decision.key_metrics["performance_gap"]))
        total_abs = mix_abs + exe_abs
        shares = {"mix_share": mix_abs/total_abs if total_abs > 0 else 0.5, 
                  "exe_share": exe_abs/total_abs if total_abs > 0 else 0.5}

        # 3) Supporting evidence table (top rows)
        root = getattr(decision.root_cause_type, "value", str(decision.root_cause_type))
        evidence_df = decision.supporting_table_df.head(table_rows).copy()
        tables = decision.evidence_tables

        # 3.1) Build richer template text that includes table markers
        top_drivers_text = ", ".join(
            f"{category} ({r['Net (pp)']:+.1f}pp)"
            for category, r in tables["top_drivers_table"].head(3).iterrows()
        )
        despite_text = ("Despite " + " and ".join(decision.contradiction_segments[:2]) + "."
                       if decision.contradiction_segments else "")
        
        template_text = f"""{decision.narrative_text}

Top drivers: {top_drivers_text}. {despite_text}

## Detailed Analysis

{{{{ rates_mix_table }}}}

## Impact Breakdown  

{{{{ top_drivers_table }}}}""".strip()

        # 4) Decide which charts to show (driver logic lives here)
        chart_names: List[str]
        if charts is not None:
            chart_names = charts
        else:
            # All cases now default to the single useful chart
            chart_names = ["rates_and_mix"]

        # 5) Convert chart names → figure_generators spec
        chart_registry = {
            "rates_and_mix": plot_rates_and_mix_panel,
        }
        figure_generators = []
        for name in chart_names[:max_charts]:
            if name not in chart_registry: 
                continue
            fn = chart_registry[name]
            cap = {
                # Decision-grade visualizations only
                "rates_and_mix":    f"Left: Success rates comparison. Right: Exposure mix comparison. Leaders can verify narrative claims instantly.",
            }.get(name, "")
            figure_generators.append({
                "function": fn,
                "params": {"result": self, "region": region},
                "caption": cap
            })

        # 6) Assemble the single slide spec
        slide_spec = {
            "summary": {"summary_text": summary_text},
            "slide_info": {
                "title": custom_title or f"{region}",
                "template_text": template_text,
                "template_params": {
                    "region": region,
                    "gap_pp": f"{gap_pp:+.1f}pp",
                    "region_rate_pp": f"{reg_pp:.1f}pp",
                    "baseline_rate_pp": f"{base_pp:.1f}pp",
                    "mix_share": shares["mix_share"],
                    "exe_share": shares["exe_share"],
                },
                "figure_generators": figure_generators,
                "dfs": {"supporting_evidence": evidence_df, **tables},
                "layout_type": "text_tables_then_figure",
                "total_hypotheses": 1,  # Each Oaxaca slide represents one hypothesis
            },
        }

        # Minimal, non-duplicated payload (one source of truth)
        payload = {
            "analysis_summary": {
                "region": region,
                "root_cause": root,
                "gap_pp": gap_pp,
            },
            "paradox_regions": self.paradox_report.affected_regions
                if getattr(self.paradox_report, "paradox_detected", False) else [],
            "cancellation_regions": self.cancellation_report.affected_regions
                if getattr(self.cancellation_report, "cancellation_detected", False) else [],
        }

        return slide_spec, payload

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
        
        # Build evidence tables centrally (single source of truth)
        rows = decomposition_df[decomposition_df["region"] == region]
        direction = gap_direction_from(region_totals["total_net_gap"], self.thresholds)
        tables, segs = self._build_evidence_tables(rows, direction)
        
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
            baseline_type=baseline_type,
            evidence_tables=tables,
            contradiction_segments=segs
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
        if conclusion == BusinessConclusion.SIMPSON_PARADOX:
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
        
        # All cases use the same standard table format
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

    
    def _paradox_template(self, region: str, decomposition_df: pd.DataFrame, regional_gaps: pd.DataFrame, baseline_name: str, direction_text: str, gap_magnitude: float) -> str:
        """Template for Simpson's paradox narrative following gold standard pattern."""
        # Use ACTUAL overall rates from the decomposition
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        actual_region_rate = region_totals["region_overall_rate"]
        actual_baseline_rate = region_totals["baseline_overall_rate"]
        
        # Use centralized contradiction logic
        region_rows = decomposition_df[decomposition_df["region"] == region]
        direction = gap_direction_from(region_totals["total_net_gap"], self.thresholds)
        contradicts, _, segs = self._contradiction_evidence(direction, region_rows)
        
        # Use centralized allocation issues
        allocation_issues = self._build_allocation_issues(region_rows)
        allocation_text = ", ".join(allocation_issues[:2]) if allocation_issues else "allocation issues that offset segment-level strengths"
        
        # Use the same contradiction block everywhere
        if contradicts and segs and direction_text == PerformanceDirection.UNDERPERFORMS.value:
            seg_text = " and ".join(segs[:3])
            return (f"{region} {direction_text} {baseline_name} by {gap_magnitude:.1f}pp "
                   f"({actual_region_rate:.0%} vs {actual_baseline_rate:.0%}) despite "
                   f"{seg_text} due to {allocation_text}.")
        else:
            # Fallback with actual data
            contributing_segments = []
            region_data = decomposition_df[decomposition_df["region"] == region]
            for _, row in region_data.iterrows():
                category_name = row.get(
                'category_clean',
                pretty_category(row.get('category'))
            )
                # Calculate net gap in percentage points for display
                net_gap_pp = f"{row['net_gap']*100:+.1f}pp"
                # Use minor_gap threshold for meaningful contributors
                if abs(row['net_gap']) >= self.thresholds.minor_gap:
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
        
        # Use centralized contradiction logic
        region_rows = decomposition_df[decomposition_df["region"] == region]
        direction = gap_direction_from(region_totals["total_net_gap"], self.thresholds)
        contradicts, _, segs = self._contradiction_evidence(direction, region_rows)
        
        # Use centralized allocation issues
        allocation_issues = self._build_allocation_issues(region_rows)
        due_to_clause = ", ".join(allocation_issues[:2]) if allocation_issues else "allocation issues"
        
        # Use the same contradiction block everywhere
        if contradicts and segs:
            seg_text = " and ".join(segs[:3])
            return f"{region} {direction_text} {baseline_name} by {gap_magnitude:.1f}pp ({overall_region_rate:.0%} vs {overall_baseline_rate:.0%}) despite {seg_text} due to {due_to_clause}."
        else:
            # straight driver
            return f"{region} {direction_text} {baseline_name} by {gap_magnitude:.1f}pp ({overall_region_rate:.0%} vs {overall_baseline_rate:.0%}), driven by {due_to_clause}."
    
    def _performance_template(self, region: str, decomposition_df: pd.DataFrame, regional_gaps: pd.DataFrame, baseline_name: str, direction_text: str, gap_magnitude: float) -> str:
        """Template for performance-driven narrative - using backup logic."""
        
        # Use overall rates from enriched DataFrames
        region_totals = regional_gaps[regional_gaps["region"] == region].iloc[0]
        overall_region_rate = region_totals["region_overall_rate"]
        overall_baseline_rate = region_totals["baseline_overall_rate"]
        
        region_data = decomposition_df[decomposition_df["region"] == region]
        top = (region_data.assign(_abs=region_data["performance_gap_contribution"].abs())
                         .sort_values("_abs", ascending=False)
                         .head(3)[["category_clean","region_rate","rest_rate","performance_gap_contribution"]])

        drivers = ", ".join(
            f"{r.category_clean} ({r.region_rate*100:.1f}% vs {r.rest_rate*100:.1f}%, "
            f"{r.performance_gap_contribution*100:+.1f}pp impact)"
            for r in top.itertuples()
        ) if len(top) else "various factors"

        return (f"{region} {direction_text} the {baseline_name} by {gap_magnitude:.1f}pp "
                f"({overall_region_rate:.0%} vs {overall_baseline_rate:.0%}), "
                f"driven by {drivers}.")
    
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
    
    def _contradiction_evidence(
        self,
        direction: PerformanceDirection,
        rows: pd.DataFrame,
        *,
        min_rate_pp: Optional[float] = None,
        min_exposure: float = 0.02  # ignore slivers <2% of combined exposure
    ) -> Tuple[bool, float, list]:
        """
        Returns: (contradicts, disagree_share, segments_list)
          - contradicts: True iff weighted share of segments that go against the overall direction is material
          - disagree_share: 0..1 of exposure pulling opposite to headline (same definition as detector)
          - segments_list: ["Cat A (+1.2pp)", ...] top few contradicting segments
        """
        
        # Early-out for in-line cases (no meaningful contradiction possible)
        if direction == PerformanceDirection.IN_LINE:
            return False, 0.0, []
            
        # thresholds
        min_rate_pp = (min_rate_pp if min_rate_pp is not None
                       else self.thresholds.minor_rate_diff * 100.0)  # convert to pp

        # Use shared weight computation (same as detector)
        w = _weights_common(rows)

        # compute rate deltas (pp)
        rate_dx_pp = (rows["region_rate"] - rows["rest_rate"]) * 100.0
        # exposure filter
        exposure_ok = w >= min_exposure
        # material rate filter
        rate_ok = rate_dx_pp.abs() >= min_rate_pp

        # opposite sign vs overall direction
        overall_sign = 1 if direction == PerformanceDirection.OUTPERFORMS else -1
        opposite = (rate_dx_pp * overall_sign) < 0

        mask = exposure_ok & rate_ok & opposite
        disagree_share = float(w[mask].sum())
        contradicts = disagree_share >= getattr(self.thresholds, "paradox_disagree_share", 0.40)

        # Build a ranked list of contradicting segments (by weight * |rate dx|)
        tmp = rows.loc[mask].copy()
        if len(tmp):
            tmp["_label"] = tmp.apply(
                lambda r: r.get("category_clean", r.get("category", "")), axis=1
            )
            tmp["_impact_rank"] = (w.loc[tmp.index] * rate_dx_pp.loc[tmp.index].abs())
            tmp = tmp.sort_values("_impact_rank", ascending=False)
            segments = [
                f"{lbl} ({dx:+.1f}pp)" for lbl, dx in zip(tmp["_label"], rate_dx_pp.loc[tmp.index])
            ]
        else:
            segments = []

        return contradicts, disagree_share, segments

    def _build_allocation_issues(self, rows: pd.DataFrame) -> List[str]:
        """Allocation phrases with unlabeled % and pp impact."""
        out = []
        for _, r in rows.iterrows():
            dx_pp = (r["region_mix_pct"] - r["rest_mix_pct"]) * 100
            if abs(dx_pp) >= self.thresholds.meaningful_allocation_diff_pp:
                name = r.get("category_clean", pretty_category(r.get("category")))
                r_pct = r["region_mix_pct"] * 100
                b_pct = r["rest_mix_pct"] * 100
                impact_pp = r.get("construct_gap_contribution", 0.0) * 100
                out.append(
                    f"{'under' if dx_pp<0 else 'over'}-allocation in {name} "
                    f"({r_pct:.1f}% vs {b_pct:.1f}%, {impact_pp:+.1f}pp impact)"
                )
        return out

    def _build_evidence_tables(self, rows: pd.DataFrame, direction: PerformanceDirection) -> Tuple[Dict[str,pd.DataFrame], List[str]]:
        """Build all evidence tables in one place - single source of truth."""
        # contradiction (single source of truth)
        contradicts, _, segs = self._contradiction_evidence(direction, rows)

        # Get the region name from the first row for column names
        region_name = rows.iloc[0]["region"] if not rows.empty else "Region"

        # Build rates_mix table with Category as index and region name in columns
        # Let make_slides.py handle percentage formatting based on column names
        category_index = rows.apply(lambda r: r.get("category_clean", r["category"]), axis=1)
        rates_mix = pd.DataFrame({
            f"{region_name} rate": rows["region_rate"].values,  # Will be auto-detected as rate for formatting
            "Peers rate": rows["rest_rate"].values,
            "Rate Δ (pp)": np.round((rows["region_rate"]-rows["rest_rate"]).values*100, 1),  # pp = percentage points
            f"{region_name} mix pct": rows["region_mix_pct"].values,  # Will be auto-detected for % formatting  
            "Peers mix pct": rows["rest_mix_pct"].values,
            "Mix Δ (pp)": np.round((rows["region_mix_pct"]-rows["rest_mix_pct"]).values*100, 1),   
        }, index=category_index)
        
        if "net_gap" in rows:
            rates_mix["Net impact (pp)"] = np.round((rows["net_gap"]*100).values, 1)  # Use signed net gap, not abs
            rates_mix = rates_mix.sort_values("Net impact (pp)", key=abs, ascending=False)  # Sort by absolute value

        # Build top_drivers table with Category as index
        top = (rows.assign(_abs=(rows["net_gap"].abs()*100))
                    .sort_values("_abs", ascending=False)
                    .head(10))
        top_category_index = top.apply(lambda r: r.get("category_clean", r["category"]), axis=1)
        top_drivers = pd.DataFrame({
            "Net (pp)": np.round((top["net_gap"]*100).values, 1),
            "Mix (pp)": np.round((top["construct_gap_contribution"]*100).values, 1), 
            "Execution (pp)": np.round((top["performance_gap_contribution"]*100).values, 1)
        }, index=top_category_index)

        if contradicts:
            # Build contradiction table to match the segments list
            # Category is now the index, so filter by index
            category_names = [s.split(" (")[0] for s in segs]
            contradiction = rates_mix.loc[
                rates_mix.index.isin(category_names)
            ][["Rate Δ (pp)", "Mix Δ (pp)"]]
        else:
            contradiction = pd.DataFrame(columns=["Rate Δ (pp)", "Mix Δ (pp)"])

        return ({"rates_mix_table": rates_mix,
                 "top_drivers_table": top_drivers,
                 "contradiction_evidence": contradiction},
                segs)
    
    def _create_standard_table(self, category_breakdown: pd.DataFrame) -> pd.DataFrame:
        """Template table for standard cases with proper ranking by gap importance."""
        # Use all columns from the enriched DataFrame (don't filter out enriched columns!)
        table = category_breakdown.copy()
        
        # Sort by absolute gap size so most important contributors appear first
        # (abs_gap_numeric already exists from enrichment, no need to recreate)
        table = table.sort_values('abs_gap_numeric', ascending=False)
        
        # Format columns inline from numeric values
        table['Region%'] = (table['region_rate'] * 100).round(1).astype(str) + '%'
        table['Baseline%'] = (table['rest_rate'] * 100).round(1).astype(str) + '%'
        table['Gap_pp'] = (table['net_gap'] * 100).round(1).astype(str) + 'pp'
        table['Mix%'] = (table['region_mix_pct'] * 100).round(1).astype(str) + '%'
        
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
        table['Category'] = table['category'].apply(pretty_category)
        
        return table[['Status', 'Category', 'Region%', 'Baseline%', 'Gap_pp', 'Mix%']]
    


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
        performance_direction = gap_direction_from(total_gap, self.thresholds)
        
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
        (Fixed) pooled and common-mix gaps, decimals.
        
        Args:
            rows: per product; must include cols: region_rate, rest_rate, region_mix_pct, rest_mix_pct
        
        Returns:
            Tuple of (pooled_gap, common_mix_gap) in decimal form
        """
        # pooled headline diff (decimal)
        pooled = (rows['region_mix_pct']*rows['region_rate']).sum() - (rows['rest_mix_pct']*rows['rest_rate']).sum()
        
        # common-mix weights (fixed bug: use helper method)
        w = _weights_common(rows)
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
        
        def _match_parent(parent, child, idx):
            t = as_tuple(child)
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
                # Use consolidated helper methods
                disagree_share = _disagree_share_common(detailed_subset, pooled)
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

            # Use consolidated helper methods
            pooled, common = self._pair_gap_vs_common_mix(rows)
            disagree_share = _disagree_share_common(rows, pooled)
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
    tmp = df_rows.copy()
    tmp["_cat_tuple"] = tmp["category"].apply(as_tuple)
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
        
        # CRITICAL FIX: Align on the union of ALL cells (not just region cells)
        # This prevents math validation failures from dropped baseline-only cells
        
        # Pre-reindex validation
        assert abs(region_mix.sum() - 1.0) < 1e-12, f"Region mix doesn't sum to 1.0: {region_mix.sum()}"
        assert abs(baseline_mix.sum() - 1.0) < 1e-12, f"Baseline mix doesn't sum to 1.0: {baseline_mix.sum()}"
        
        # Use full union of cells from all sources
        all_idx = (region_mix.index
                  .union(baseline_mix.index)
                  .union(region_rates.index)
                  .union(baseline_rates.index))

        region_mix   = region_mix.reindex(all_idx,   fill_value=0.0)
        baseline_mix = baseline_mix.reindex(all_idx, fill_value=0.0)

        # Let rates be NaN where undefined; they won't matter when mix=0
        region_rates   = region_rates.reindex(all_idx)
        baseline_rates = baseline_rates.reindex(all_idx)

        # If a rate is NaN on one side but that side has exposure, borrow the other side's rate;
        # if both NaN, set 0 (will be multiplied by 0 exposure on both sides anyway).
        region_rates   = region_rates.fillna(baseline_rates).fillna(0.0)
        baseline_rates = baseline_rates.fillna(region_rates).fillna(0.0)

        idx_to_iterate = all_idx
        
        # Process each category
        for category in idx_to_iterate:
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

            
            # Create result record - store both tuple and string for consistency
            decomposition_results.append({
                "region": region,
                "category": str(category),  # display string
                "region_mix_pct": region_mix_val,
                "rest_mix_pct": baseline_mix_val,
                "region_rate": region_rate_val,
                "rest_rate": baseline_rate_val,
                "construct_gap_contribution": construct_gap,
                "performance_gap_contribution": performance_gap,
                "net_gap": net_gap,

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
            parent_index=0
        )
        if p_axis0.paradox_detected:
            paradox_report = _merge_paradox(paradox_report, p_axis0)

        # parent = axis 1
        agg_axis1 = _rollup_by_axis(decomposition_df, parent_index=1)
        p_axis1 = edge_detector.detect_simpson_paradox(
            aggregate_results=agg_axis1,
            detailed_results=decomposition_df,
            parent_index=1
        )
        if p_axis1.paradox_detected:
            paradox_report = _merge_paradox(paradox_report, p_axis1)
    
    # Enrich DataFrames with all computed columns needed downstream
    enriched_decomposition_df = compute_derived_metrics_df(decomposition_df, thresholds)
    enriched_regional_gaps = compute_derived_regional_gaps(regional_gaps)
    
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
                    analysis_decomposition_df = compute_derived_metrics_df(subcategory_results.decomposition_df, thresholds)
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

def compute_derived_metrics_df(decomposition_df: pd.DataFrame, thresholds: AnalysisThresholds = None) -> pd.DataFrame:
    """Compute numeric derived metrics without display formatting."""
    thresholds = thresholds or AnalysisThresholds()
    enriched = decomposition_df.copy()

    # NUMERIC helpers only
    enriched['abs_gap'] = enriched['net_gap'].abs()
    enriched['abs_gap_numeric'] = (enriched['net_gap'] * 100).abs()       # still numeric
    enriched['category_clean'] = enriched['category'].apply(pretty_category)

    # cumsum_impact / is_top_contributor (numeric, no strings)
    for region in enriched['region'].unique():
        m = enriched['region'] == region
        r = enriched[m].sort_values('abs_gap_numeric', ascending=False)
        if len(r) == 1:
            enriched.loc[m, 'cumsum_impact'] = r['abs_gap_numeric'].iloc[0]
            enriched.loc[m, 'is_top_contributor'] = True
        else:
            tot = r['abs_gap_numeric'].sum()
            thr = tot * thresholds.cumulative_contribution_threshold
            cs = r['abs_gap_numeric'].cumsum()
            for idx, v in zip(r.index, cs):
                enriched.loc[idx, 'cumsum_impact'] = v
                enriched.loc[idx, 'is_top_contributor'] = v <= thr
            if enriched[m]['is_top_contributor'].sum() == 0 and len(r) > 0:
                enriched.loc[r['abs_gap_numeric'].idxmax(), 'is_top_contributor'] = True

    return enriched

def compute_derived_regional_gaps(regional_gaps: pd.DataFrame) -> pd.DataFrame:
    """Compute numeric derived metrics for regional gaps."""
    enriched = regional_gaps.copy()
    enriched['gap_magnitude'] = enriched['total_net_gap'].abs()
    enriched['gap_direction'] = enriched['total_net_gap'].apply(
        lambda x: PerformanceDirection.OUTPERFORMS.value if x > 0 else PerformanceDirection.UNDERPERFORMS.value
    )
    return enriched

# ================================
# SHARED SIMPSON PARADOX HELPERS
# ================================

def _weights_common(rows: pd.DataFrame) -> pd.Series:
    """Compute exposure weights consistently across EdgeCaseDetector and NarrativeTemplateEngine."""
    if {'region_trials','rest_trials'}.issubset(rows.columns):
        w = rows['region_trials'].fillna(0) + rows['rest_trials'].fillna(0)
    else:
        w = rows['region_mix_pct'].fillna(0) + rows['rest_mix_pct'].fillna(0)
    return w / w.sum() if w.sum() else w

def _disagree_share_common(rows: pd.DataFrame, pooled: float) -> float:
    """Compute disagree share consistently across EdgeCaseDetector and NarrativeTemplateEngine."""
    w = _weights_common(rows)
    diff = rows['region_rate'] - rows['rest_rate']
    return float(w[diff != 0].sum()) if pooled == 0 else float(w[np.sign(diff) == -np.sign(pooled)].sum())

# ================================
# DISPLAY FORMATTING FUNCTIONS  
# ================================

def format_regional_gaps_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Add display formatting to regional gaps DataFrame."""
    d = df.copy()
    for col in ['total_net_gap','total_construct_gap','total_performance_gap']:
        d[col + '_pp'] = (d[col] * 100).round(1).astype(str) + 'pp'
    d['region_overall_rate_pct'] = (d['region_overall_rate'] * 100).round(1).astype(str) + '%'
    d['baseline_overall_rate_pct'] = (d['baseline_overall_rate'] * 100).round(1).astype(str) + '%'
    return d

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
    decomposition_method: str,
    region_column: str = "region" 
) -> Tuple[float, Dict[str, float], AnalysisResult, pd.DataFrame]:
    """
    Run Oaxaca for this cut, compute a single score, and return diagnostics.
    """
    # 1) Run the analysis for this cut
    ar = run_oaxaca_analysis(
        df=df,
        region_column=region_column,                # pass through
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
    portfolio_mode: bool = False,
    region_column: str = "region"   
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
            region_column=region_column,             # pass through
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
            region_column=region_column,             # pass through
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
                decomposition_method=decomposition_method,
                region_column=region_column                # pass through
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
                top_segments=topk[["category","region_rate","rest_rate","net_gap","region_mix_pct"]],
                analysis_result=ar
            )

    if best is None:
        raise RuntimeError("Auto slice failed to evaluate any candidate cuts.")

    return best


# --- Titles only (no internal slugs) -----------------------------------------
def _format_cut_name(cols: Iterable[str]) -> str:
    return " × ".join(cols).title()

def _title_for_cut(cols: Iterable[str], metric_name: Optional[str]) -> str:
    name = _format_cut_name(cols)
    return f"{metric_name} - {name}" if metric_name else name


def build_oaxaca_exec_pack(
    df: pd.DataFrame,
    *,
    # choose ONE path:
    candidate_cuts: Iterable[Iterable[str]] = (),   # AUTO mode (rank cuts)
    fixed_cut: Optional[Iterable[str]] = None,      # FIXED mode (render this cut)
    fixed_region: Optional[str] = None,             # optional with fixed_cut
    focus_region: Optional[str] = None,             # used by AUTO mode

    # analysis plumbing:
    region_column: str = "region",
    numerator_column: str = "wins",
    denominator_column: str = "total",
    baseline_type: str = "rest_of_world",
    decomposition_method: str = "two_part",
    thresholds: AnalysisThresholds = None,

    # presentation knobs:
    max_charts: int = 2,
    charts: Optional[List[str]] = None,
    metric_name: Optional[str] = None,
    alternatives: int = 1  # AUTO mode: render this many alternative cuts (0 = none)
) -> Dict:
    """
    Returns:
      {
        "slides": {
          "Product": { "summary": {...}, "slide_info": {...} },
          "Vertical": { "summary": {...}, "slide_info": {...} },
          "Product × Vertical": { "summary": {...}, "slide_info": {...} },
          ...
        },
        "payload": <primary payload with best_slice info>
      }
    """
    thresholds = thresholds or AnalysisThresholds()

    # Unpack the (slide_spec, payload) from present_executive_pack_for_slides
    def _exec_from_result(ar: AnalysisResult, region: str, title: str, cut_cols: Iterable[str], role: str):
        slide, payload = ar.present_executive_pack_for_slides(
            region, max_charts=max_charts, charts=charts, custom_title=title
        )
        slide = {**slide, "cut": list(cut_cols), "role": role}
        return slide, payload

    slides: List[Dict] = []
    payload: Optional[Dict] = None

    # ---------------------- FIXED MODE ---------------------------------------
    if fixed_cut is not None:
        fixed_cut = tuple(fixed_cut)
        region = fixed_region or focus_region
        if region is None:
            probe = run_oaxaca_analysis(
                df=df,
                region_column=region_column,
                numerator_column=numerator_column,
                denominator_column=denominator_column,
                category_columns=list(fixed_cut),
                baseline_type=baseline_type,
                decomposition_method=decomposition_method,
                thresholds=thresholds,
                detect_edge_cases=False,
            )
            pick = probe.regional_gaps.reindex(
                probe.regional_gaps["total_net_gap"].abs().sort_values(ascending=False).index
            )
            region = str(pick.iloc[0]["region"])

        ar = run_oaxaca_analysis(
            df=df,
            region_column=region_column,
            numerator_column=numerator_column,
            denominator_column=denominator_column,
            category_columns=list(fixed_cut),
            baseline_type=baseline_type,
            decomposition_method=decomposition_method,
            thresholds=thresholds,
        )
        title = _title_for_cut(fixed_cut, metric_name)
        s, payload = _exec_from_result(ar, region, title, fixed_cut, role="primary")
        slides.append(s)

        if "best_slice" not in payload:
            payload["best_slice"] = {
                "focus_region": region,
                "best_categories": list(fixed_cut),
                "score": None,
                "score_breakdown": {},
                "top_segments": None,
            }
        # Convert slides list to dictionary with cut names as keys
        slides_dict = {}
        for slide in slides:
            cut_name = _format_cut_name(slide['cut'])  # e.g., "Product", "Vertical", "Product × Vertical"
            # Remove the 'cut' and 'role' metadata, keep only slide content
            slides_dict[cut_name] = {
                'summary': slide['summary'],
                'slide_info': slide['slide_info']
            }
        
        return {"slides": slides_dict, "payload": payload}

    # ---------------------- AUTO MODE ----------------------------------------
    candidate_cuts = list(candidate_cuts)
    if not candidate_cuts:
        raise ValueError("Provide candidate_cuts for AUTO mode or fixed_cut for FIXED mode.")

    # 1) Pick best cut (and region if needed)
    best = auto_find_best_slice(
        df=df,
        candidate_cuts=candidate_cuts,
        focus_region=focus_region,
        region_column=region_column,
        numerator_column=numerator_column,
        denominator_column=denominator_column,
        baseline_type=baseline_type,
        decomposition_method=decomposition_method,
        thresholds=thresholds,
        portfolio_mode=(focus_region is None),
    )
    region = best.focus_region
    title = _title_for_cut(best.best_categories, metric_name)

    s, payload = _exec_from_result(best.analysis_result, region, title, best.best_categories, role="primary")
    # attach score info to the primary slide (no separate meta)
    s.update({"score": best.score, "score_breakdown": best.score_breakdown})
    slides.append(s)

    if "best_slice" not in payload:
        payload["best_slice"] = {
            "focus_region": region,
            "best_categories": list(best.best_categories),
            "score": best.score,
            "score_breakdown": best.score_breakdown,
            "top_segments": best.top_segments,
        }

    # 2) Rank and add alternatives
    if alternatives > 0:
        others = []
        for cols in candidate_cuts:
            cols = tuple(cols)
            if cols == best.best_categories:
                continue
            try:
                sc, br, ar, _ = _score_cut(
                    df=df, region=region, category_cols=cols, thresholds=thresholds,
                    baseline_type=baseline_type, numerator_column=numerator_column,
                    denominator_column=denominator_column, decomposition_method=decomposition_method,
                    region_column=region_column
                )
                others.append((sc, br, ar, cols))
            except Exception as e:
                logger.debug(f"[ALT] skipping {cols}: {e}")
                continue

        others.sort(key=lambda t: t[0], reverse=True)
        for sc, br, ar, cols in others[:alternatives]:
            alt_title = _title_for_cut(cols, metric_name)
            s_alt, _ = _exec_from_result(ar, region, alt_title, cols, role="alternative")
            s_alt.update({"score": sc, "score_breakdown": br})
            slides.append(s_alt)

    # Convert slides list to dictionary with cut names as keys
    slides_dict = {}
    for slide in slides:
        cut_name = _format_cut_name(slide['cut'])  # e.g., "Product", "Vertical", "Product × Vertical"
        # Remove the 'cut' and 'role' metadata, keep only slide content
        slides_dict[cut_name] = {
            'summary': slide['summary'],
            'slide_info': slide['slide_info']
        }
    
    return {"slides": slides_dict, "payload": payload}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_rates_and_mix_panel(result, region: str, top_n: int = 12, sort_by: str = "abs_gap"):
    """
    Left: slopechart of success rates (Region vs Peers) per category.
    Right: barh of mix % (Region vs Peers) for the same categories (aligned).
    """
    import matplotlib.pyplot as plt
    
    rows = result.decomposition_df[result.decomposition_df["region"] == region].copy()
    rows["rate_diff_pp"] = (rows["region_rate"] - rows["rest_rate"]) * 100
    rows["mix_diff_pp"]  = (rows["region_mix_pct"] - rows["rest_mix_pct"]) * 100
    rows["abs_gap"] = rows["net_gap"].abs()
    rows = rows.sort_values("abs_gap" if sort_by=="abs_gap" else "mix_diff_pp", ascending=False).head(top_n)
    rows = rows.iloc[::-1].copy()  # biggest at the top visually

    labels = rows.apply(lambda r: r.get("category_clean", r["category"]), axis=1)

    fig = plt.figure(figsize=(12, max(5, 0.45*len(rows))))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    axL = fig.add_subplot(gs[0,0])
    axR = fig.add_subplot(gs[0,1])

    y = range(len(rows))

    # Left: slopechart of rates
    axL.hlines(y, rows["rest_rate"]*100, rows["region_rate"]*100, alpha=0.6)
    axL.plot(rows["rest_rate"]*100, y, "o", label="Peers", alpha=0.8)
    axL.plot(rows["region_rate"]*100, y, "o", label="Region", alpha=0.8)
    for i, (_, r) in enumerate(rows.iterrows()):
        dx = (r["region_rate"] - r["rest_rate"]) * 100
        axL.text((r["rest_rate"]*100 + r["region_rate"]*100)/2, i,
                 f"{dx:+.1f}pp", va="center", ha="center", fontsize=9)
    axL.set_yticks(y, labels)
    axL.set_xlabel("Success rate (%)")
    axL.set_title(f"{region}: Rates by category")
    axL.grid(True, axis="x", alpha=0.3)
    axL.legend(loc="lower right")

    # Right: aligned mix percentages (barh, side-by-side)
    w = 0.4
    axR.barh([i - w/2 for i in y], rows["region_mix_pct"]*100, height=w, alpha=0.8, label="Region")
    axR.barh([i + w/2 for i in y], rows["rest_mix_pct"]*100,   height=w, alpha=0.6, label="Peers")
    axR.set_yticks(y, [""]*len(rows))  # align with left
    axR.set_xlabel("Mix (%)")
    axR.set_title("Exposure mix")
    axR.grid(True, axis="x", alpha=0.3)
    axR.legend(loc="lower right")

    fig.tight_layout()
    return fig, (axL, axR)


# =============================================================================
# SYNTHETIC DATA GENERATION FOR TESTING
# =============================================================================

def create_oaxaca_synthetic_data(target_region: str = "North America", target_gap_pp: float = -5.0) -> pd.DataFrame:
    """
    Generate synthetic conversion rate data where target_region underperforms Global
    by target_gap_pp percentage points, with clear category-level drivers.
    
    Creates realistic product × vertical combinations with Simpson's paradox potential.
    """
    import numpy as np
    
    np.random.seed(42)  # Reproducible results
    
    # Define categories that will drive the gap
    products = ["Premium", "Standard", "Basic"]
    verticals = ["Enterprise", "SMB", "Consumer"]
    regions = ["Global", "North America", "Europe", "Asia", "Latin America"]
    
    # Generate data with specific patterns
    data_rows = []
    
    # Strategy: North America has poor mix (too much low-converting stuff)
    # but decent execution within categories
    
    for region in regions:
        for product in products:
            for vertical in verticals:
                # Base conversion rates by segment (these are good rates)
                base_rates = {
                    ("Premium", "Enterprise"): 0.18,
                    ("Premium", "SMB"): 0.15,
                    ("Premium", "Consumer"): 0.12,
                    ("Standard", "Enterprise"): 0.14,
                    ("Standard", "SMB"): 0.11,
                    ("Standard", "Consumer"): 0.08,
                    ("Basic", "Enterprise"): 0.10,
                    ("Basic", "SMB"): 0.07,
                    ("Basic", "Consumer"): 0.05,
                }
                
                base_rate = base_rates[(product, vertical)]
                
                # Regional performance adjustments (execution differences)
                if region == "North America":
                    # NA is actually slightly BETTER at execution per segment
                    rate_multiplier = 1.05  # 5% better execution
                elif region == "Europe":
                    rate_multiplier = 1.02
                elif region == "Asia":  
                    rate_multiplier = 1.08
                elif region == "Latin America":
                    rate_multiplier = 0.95
                else:  # Global baseline
                    rate_multiplier = 1.0
                
                conversion_rate = base_rate * rate_multiplier
                
                # Volume/mix patterns (this is where NA gets hurt)
                if region == "North America":
                    # Bad mix: over-indexed on low-converting segments
                    if product == "Basic":
                        volume = np.random.randint(8000, 12000)  # Too much Basic
                    elif product == "Standard":
                        volume = np.random.randint(4000, 6000)   # Some Standard  
                    else:  # Premium
                        volume = np.random.randint(1000, 2000)   # Too little Premium
                        
                    if vertical == "Consumer":
                        volume = int(volume * 1.8)  # Over-indexed on Consumer
                    elif vertical == "SMB":
                        volume = int(volume * 1.0)
                    else:  # Enterprise
                        volume = int(volume * 0.6)  # Under-indexed on Enterprise
                        
                elif region == "Asia":
                    # Good mix: more premium and enterprise
                    if product == "Premium":
                        volume = np.random.randint(6000, 9000)
                    elif product == "Standard":
                        volume = np.random.randint(3000, 5000)
                    else:  # Basic
                        volume = np.random.randint(1000, 2000)
                        
                    if vertical == "Enterprise":
                        volume = int(volume * 1.5)
                    elif vertical == "SMB": 
                        volume = int(volume * 1.2)
                    else:  # Consumer
                        volume = int(volume * 0.8)
                        
                else:
                    # Other regions: more balanced
                    volume = np.random.randint(3000, 7000)
                    if vertical == "Enterprise":
                        volume = int(volume * 1.2)
                    elif vertical == "Consumer":
                        volume = int(volume * 1.1)
                
                # Calculate conversions
                conversions = int(volume * conversion_rate)
                
                data_rows.append({
                    'region': region,
                    'product': product,
                    'vertical': vertical, 
                    'visits': volume,
                    'conversions': conversions,
                    'conversion_rate': conversion_rate
                })
    
    df = pd.DataFrame(data_rows)
    
    # Verify we hit our target (approximately)
    global_rate = df[df['region'] == 'Global']['conversions'].sum() / df[df['region'] == 'Global']['visits'].sum()
    target_rate = df[df['region'] == target_region]['conversions'].sum() / df[df['region'] == target_region]['visits'].sum()
    actual_gap_pp = (target_rate - global_rate) * 100
    
    print(f"🎯 Synthetic data generated:")
    print(f"   Global conversion rate: {global_rate:.1%}")  
    print(f"   {target_region} conversion rate: {target_rate:.1%}")
    print(f"   Gap: {actual_gap_pp:+.1f}pp (target: {target_gap_pp:+.1f}pp)")
    
    return df


def analyze_oaxaca_metrics(
    config: Dict[str, Any],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    data_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Perform Oaxaca-Blinder analysis for all configured metrics and return results in unified format.
    
    Args:
        config: Configuration dictionary with metrics and their Oaxaca settings
        metric_anomaly_map: Dictionary mapping metric names to anomaly information
        data_df: DataFrame containing the data for analysis
    
    Returns:
        Dictionary in unified format: {metric_name: {'slides': {'oaxaca_primary': slide_data, 'oaxaca_alternative': slide_data}, 'payload': {...}}}
    """
    metrics_config = config.get('metrics', {})
    unified_results = {}
    
    # Analyze each metric
    for metric_name, metric_config in metrics_config.items():
        # Check if this metric has an anomaly detected
        if metric_name not in metric_anomaly_map:
            print(f"   ⏭️  Skipping {metric_name} - no anomaly detected")
            continue
            
        try:
            print(f"   📊 Processing Oaxaca-Blinder analysis for {metric_name}")
            
            # Get anomaly information
            anomaly = metric_anomaly_map[metric_name]
            target_region = anomaly['anomalous_region']
            
            # Get configuration
            region_column = metric_config.get('region_column', 'region')
            numerator_column = metric_config.get('numerator_column')
            denominator_column = metric_config.get('denominator_column')
            candidate_cuts = metric_config.get('candidate_cuts', [["product"]])
            max_charts = metric_config.get('max_charts', 2)
            alternatives = metric_config.get('alternatives', 1)
            
            if not numerator_column or not denominator_column:
                print(f"   ⚠️  Skipping {metric_name} - missing numerator/denominator column configuration")
                continue
            
            print(f"   📊 Testing candidate cuts: {candidate_cuts}")
            
            # Run Oaxaca-Blinder analysis using the unified driver function
            oaxaca_results = build_oaxaca_exec_pack(
                df=data_df,  
                candidate_cuts=candidate_cuts,
                focus_region=target_region,
                region_column=region_column,
                numerator_column=numerator_column,
                denominator_column=denominator_column,
                max_charts=max_charts,
                metric_name=metric_name,
                alternatives=alternatives
            )
            
            # Extract best cut information for logging
            best_categories = oaxaca_results['payload']['best_slice']['best_categories']
            best_score = oaxaca_results['payload']['best_slice']['score']
            print(f"   🏆 Best cut selected: {best_categories} (score: {best_score:.3f})")
            
            # Store results in unified format
            unified_results[metric_name] = oaxaca_results
            print(f"   ✅ Oaxaca-Blinder analysis completed for {metric_name}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to analyze {metric_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return unified_results