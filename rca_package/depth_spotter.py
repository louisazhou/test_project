"""
Depth Spotter Module for Sub-Region Analysis

This module provides tools for analyzing performance gaps at a deeper level
by examining sub-regions or slices within anomalous regions to identify
the specific contributors to performance differences.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
import logging
from jinja2 import Template

# Setup logging
logger = logging.getLogger(__name__)

# Font styling constants for consistent appearance
FONTS = {
    'title': {
        'size': 16,
        'weight': 'normal',
        'family': 'Arial'
    },
    'axis_label': {
        'size': 14,
        'weight': 'normal',
        'family': 'Arial'
    },
    'tick_label': {
        'size': 12,
        'weight': 'normal',
        'family': 'Arial'
    },
    'annotation': {
        'size': 12,
        'weight': 'bold',
        'family': 'Arial'
    }
}

# Color scheme for visualizations (matching hypothesis_scorer.py exactly)
COLORS = {
    'metric_negative': '#e74c3c',     # Red for bad metric anomalies
    'metric_positive': '#2ecc71',     # Green for good metric anomalies
    'default_bar': '#BDC3C7',         # Light gray for default bars
    'global_line': '#34495e',         # Dark blue-gray for reference line
}


def _build_rate_tracebacks(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    metric_name: str,
    row_rate: float,
    delta: float,
    higher_is_better: bool,
    focal_region: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Traceback tables for rate_contrib()."""
    # Re-derive pieces explicitly for transparency
    expected_total = df['expected'].sum()
    delta_ratio = abs(delta) / expected_total if expected_total > 0 else 0.0
    diff = df[numerator_col] - df['expected']
    pos_sum = diff.clip(lower=0).sum()
    neg_sum = -diff.clip(upper=0).sum()
    max_dev = diff.abs().max()
    max_potential_contrib = (max_dev / abs(delta)) if delta != 0 else 0.0
    mode = "two_sided" if ((delta_ratio < 0.05) or (max_potential_contrib > 1.0)) else "standard"

    # Per-slice walkthrough (explicit formulas + values)
    rows = []
    for _, r in df.reset_index(drop=True).iterrows():
        exp_formula = f"{denominator_col} × ROW_rate = {int(r[denominator_col]):,} × {row_rate:.6f}"
        if mode == "two_sided":
            rc_formula = ("diff/Σpos if diff>0; diff/Σ|neg| if diff<0  "
                          f"(diff={r[numerator_col]-r['expected']:.6f}, Σpos={pos_sum:.6f}, Σ|neg|={neg_sum:.6f})")
        else:
            rc_formula = f"(actual - expected)/Δ  (Δ={delta:.6f})"

        sign_rule = ("keep" if higher_is_better and (delta >= 0)
                     else "flip" if higher_is_better and (delta < 0)
                     else "flip" if (not higher_is_better) and (delta > 0)
                     else "keep")

        score_formula = "sqrt(|contrib| + max(|contrib| - coverage, 0) / max(coverage, 0.01))"

        rows.append({
            "slice": r["slice"] if "slice" in df.columns else getattr(r, "slice", ""),
            f"{denominator_col}": int(r[denominator_col]),
            f"{numerator_col}": int(r[numerator_col]),
            "ROW_rate": row_rate,
            "expected_formula": exp_formula,
            "expected_value": r["expected"],
            "diff": (r[numerator_col] - r["expected"]),
            "raw_contrib_formula": rc_formula,
            "raw_contrib_value": r.get("raw_contribution", np.nan),
            "sign_adjustment_rule": sign_rule,
            "contribution": r["contribution"],
            "coverage": r["coverage"],
            "score_formula": score_formula,
            "score_value": r["score"],
            "rate_display": r.get("display_value", "")
        })
    walkthrough = pd.DataFrame(rows)

    # Summary with focal + method choice + key numbers
    summary = pd.DataFrame([
        {"Component": "metric_type", "Formula": "rate", "Value": "rate"},
        {"Component": "metric_name", "Formula": "-", "Value": metric_name},
        {"Component": "focal_region", "Formula": "-", "Value": focal_region},
        {"Component": "higher_is_better", "Formula": "-", "Value": bool(higher_is_better)},
        {"Component": "ROW_rate", "Formula": "ROW_num/ROW_den", "Value": row_rate},
        {"Component": "Δ (total gap)", "Formula": "Σactual − Σexpected", "Value": delta},
        {"Component": "mode", "Formula": "two_sided if Δ/Σexp<5% or max_dev/|Δ|>1 else standard", "Value": mode},
        {"Component": "Σpos, Σ|neg|", "Formula": "from diff", "Value": f"{pos_sum:.6f}, {neg_sum:.6f}"},
    ])

    return walkthrough, summary


def _build_additive_tracebacks(
    df: pd.DataFrame,
    metric_col: str,
    row_total: float,
    delta: float,
    higher_is_better: bool,
    focal_region: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Traceback tables for additive_contrib()."""
    rows = []
    for _, r in df.reset_index(drop=True).iterrows():
        cov_formula = f"{metric_col} / Σ{metric_col}"
        exp_formula = f"coverage × ROW_total = {r['coverage']:.6f} × {row_total:,.0f}"
        rc_formula = f"(actual - expected)/Δ  (Δ={delta:.6f})"
        sign_rule = ("flip" if (higher_is_better and delta < 0) or ((not higher_is_better) and delta > 0) else "keep")
        score_formula = "sqrt(|contrib| + max(|contrib| - coverage, 0) / max(coverage, 0.01))"

        rows.append({
            "slice": r["slice"] if "slice" in df.columns else getattr(r, "slice", ""),
            metric_col: r[metric_col],
            "coverage_formula": cov_formula,
            "coverage_value": r["coverage"],
            "expected_formula": exp_formula,
            "expected_value": r["expected"],
            "diff": (r[metric_col] - r["expected"]),
            "raw_contrib_formula": rc_formula,
            "raw_contrib_value": r.get("raw_contribution", np.nan),
            "sign_adjustment_rule": sign_rule,
            "contribution": r["contribution"],
            "score_formula": score_formula,
            "score_value": r["score"],
        })
    walkthrough = pd.DataFrame(rows)

    summary = pd.DataFrame([
        {"Component": "metric_type", "Formula": "additive", "Value": "additive"},
        {"Component": "metric_name", "Formula": "-", "Value": metric_col},
        {"Component": "focal_region", "Formula": "-", "Value": focal_region},
        {"Component": "higher_is_better", "Formula": "-", "Value": bool(higher_is_better)},
        {"Component": "ROW_total", "Formula": "-", "Value": row_total},
        {"Component": "Δ (total gap)", "Formula": "Σactual − ROW_total", "Value": delta},
        {"Component": "mode", "Formula": "standard", "Value": "standard"},
    ])

    return walkthrough, summary


def _calculate_two_sided_contributions(actual_values: pd.Series, expected_values: pd.Series) -> pd.Series:
    """
    Calculate contributions using two-sided normalization for small delta cases.
    Positive contributions sum to +1, negative contributions sum to -1.
    This approach provides stable, interpretable results even when large deviations cancel out.
    
    Based on the formula:
    - diff = actual - expected (signed gap per slice)
    - pos_sum = Σ positive gaps
    - neg_sum = Σ |negative| gaps  
    - contribution_i = diff_i / pos_sum (if positive) or diff_i / neg_sum (if negative)
    
    Args:
        actual_values: Actual values for each slice
        expected_values: Expected values for each slice
        
    Returns:
        Series of contributions where positives sum to +1, negatives sum to -1
    """
    # Calculate signed gaps per slice
    diff = actual_values - expected_values
    
    # Calculate positive and negative sums
    pos_sum = diff.clip(lower=0).sum()  # Σ positive gaps
    neg_sum = -diff.clip(upper=0).sum()  # Σ |negative| gaps
    
    def _share(x):
        if x > 0 and pos_sum > 0:
            return x / pos_sum  # Normalize positives to sum to +1
        elif x < 0 and neg_sum > 0:
            return x / neg_sum  # Normalize negatives to sum to -1 (x is negative, neg_sum is positive)
        else:
            return 0.0
    
    contributions = diff.apply(_share)
    return contributions




def rate_contrib(
    df_slice: pd.DataFrame, 
    row_numerator: float, 
    row_denominator: float,
    denominator_col: str,
    numerator_col: str,
    metric_name: str,
    higher_is_better: bool = True,
    focal_region: Optional[str] = None
) -> Tuple[pd.DataFrame, float, float]:
    """
    Calculate contribution analysis for rate metrics (e.g., conversion rate).
    
    Args:
        df_slice: DataFrame containing slice-level data for the anomalous region
        row_numerator: Total numerator from rest-of-world (e.g., total conversions)
        row_denominator: Total denominator from rest-of-world (e.g., total visits)
        denominator_col: Name of the denominator column (e.g., visits, sessions)
        numerator_col: Name of the numerator column (e.g., conversions, orders)
        metric_name: Name of the metric (for the calculated rate column)
        higher_is_better: Whether higher values of this metric are better
    
    Returns:
        Tuple of (enhanced_df, delta, row_rate) where:
        - enhanced_df: DataFrame with contribution analysis columns added
        - delta: Total gap between actual and expected performance
        - row_rate: Rest-of-world rate used as benchmark
    """
    df = df_slice.copy()
    # Calculate rest-of-world rate
    row_rate = row_numerator / row_denominator if row_denominator > 0 else 0
    
    # Calculate expected conversions if each slice performed at ROW rate
    df['expected'] = df[denominator_col] * row_rate
    
    # Calculate total delta
    delta = df[numerator_col].sum() - df['expected'].sum()
    
    # Calculate coverage (share of total visits within the region)
    df['coverage'] = df[denominator_col] / df[denominator_col].sum()
    
    # Calculate contribution using appropriate method based on delta size and potential overflow
    expected_total = df['expected'].sum()
    delta_ratio = abs(delta) / expected_total if expected_total > 0 else 0
    
    # Check if standard approach would produce >100% contributions
    max_deviation = (df[numerator_col] - df['expected']).abs().max()
    max_potential_contrib = max_deviation / abs(delta) if delta != 0 else 0
    
    # Use two-sided normalization if delta is small OR if contributions would exceed 100%
    use_two_sided_normalization = (delta_ratio < 0.05) or (max_potential_contrib > 1.0)
    
    if delta != 0:
        if use_two_sided_normalization:
            # Two-sided normalization produces mathematically correct signs, but we need to
            # adjust for the higher_is_better context to get intuitive interpretation
            raw_contribution = _calculate_two_sided_contributions(df[numerator_col], df['expected'])
            
            # Store raw contribution for ranking drivers of delta
            df['raw_contribution'] = raw_contribution
            
            if higher_is_better:
                # For metrics where higher is better:
                # - actual > expected (positive raw) = good performance = hero (positive)
                # - actual < expected (negative raw) = bad performance = culprit (negative)
                df['contribution'] = raw_contribution  # Keep as-is
            else:
                # For metrics where lower is better:
                # - actual > expected (positive raw) = bad performance = culprit (negative)
                # - actual < expected (negative raw) = good performance = hero (positive)
                df['contribution'] = -raw_contribution  # Flip sign
            
        else:
            # Standard attribution for normal-sized deltas
            raw_contribution = (df[numerator_col] - df['expected']) / delta
            
            # Store raw contribution for ranking drivers of delta
            df['raw_contribution'] = raw_contribution
            
            # Adjust sign based on context for intuitive interpretation
            if higher_is_better:
                if delta < 0:  # Region underperforming - flip sign for intuitive interpretation
                    df['contribution'] = -raw_contribution
                else:
                    df['contribution'] = raw_contribution
            else:
                if delta > 0:  # Region underperforming (higher than desired) - flip sign
                    df['contribution'] = -raw_contribution
                else:
                    df['contribution'] = raw_contribution
    else:
        # If delta is exactly 0, all contributions are 0
        df['contribution'] = 0.0
        df['raw_contribution'] = 0.0
    
    # Calculate heuristic score: sqrt(|contribution| + max(|contribution| - coverage, 0) / coverage)
    # Floor coverage at 1% to avoid division by zero
    coverage_floored = df['coverage'].clip(lower=0.01)
    df['score'] = np.sqrt(
        np.abs(df['contribution']) + 
        np.maximum(np.abs(df['contribution']) - df['coverage'], 0) / coverage_floored
    )
    
    # Add actual rate using the proper metric name, handling zero denominators
    df[metric_name] = df[numerator_col] / df[denominator_col].replace(0, np.nan)
    
    # Create display string with ratio components
    def format_ratio_display(row):
        num = row[numerator_col]
        denom = row[denominator_col]
        rate = row[metric_name]
        
        if pd.isna(rate):
            return "N/A"
        ratio_str = f"{rate*100:.1f}%"
        if denom > 0:  # Only show components if denominator exists
            return f"{ratio_str} ({int(num):,}/{int(denom):,})"
        return ratio_str
    
    # Add display string column
    df['display_value'] = df.apply(format_ratio_display, axis=1)

    # === NEW: attach tracebacks for payload ===
    try:
        fw, fs = _build_rate_tracebacks(
            df=df,
            numerator_col=numerator_col,
            denominator_col=denominator_col,
            metric_name=metric_name,
            row_rate=row_rate,
            delta=delta,
            higher_is_better=higher_is_better,
            focal_region=focal_region
        )
        df.attrs["tracebacks"] = {
            "formulas_walkthrough": fw.to_dict("records"),
            "formulas_summary": fs.to_dict("records")
        }
    except Exception as _e:
        logger.debug(f"Traceback build (rate) skipped: {_e}")
    # =========================================
    
    return df, delta, row_rate


def additive_contrib(
    df_slice: pd.DataFrame, 
    row_total: float,
    metric_col: str,
    higher_is_better: bool = True,
    focal_region: Optional[str] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate contribution analysis for additive metrics (e.g., revenue, orders).
    
    Args:
        df_slice: DataFrame containing slice-level data for the anomalous region
        row_total: Total metric value from rest-of-world
        metric_col: Name of the metric column to analyze
        higher_is_better: Whether higher values of this metric are better
    
    Returns:
        Tuple of (enhanced_df, delta) where:
        - enhanced_df: DataFrame with contribution analysis columns added
        - delta: Total gap between actual and expected performance
    """
    df = df_slice.copy()
    
    # Calculate total delta
    delta = df[metric_col].sum() - row_total
    
    # Calculate coverage (share of total metric within the region) 
    df['coverage'] = df[metric_col] / df[metric_col].sum()
    
    # Calculate expected value based on proportional allocation
    df['expected'] = df['coverage'] * row_total
    
    # Calculate raw contribution for ranking drivers of delta
    raw_contribution = (df[metric_col] - df['expected']) / delta if delta != 0 else 0
    df['raw_contribution'] = raw_contribution
    
    # Adjust sign based on context for intuitive interpretation
    if higher_is_better:
        # For metrics where higher is better (revenue, orders)
        if delta < 0:  # Region underperforming
            # Positive raw contribution = harmful (making gap worse)
            # Negative raw contribution = helpful (reducing gap) 
            df['contribution'] = -raw_contribution  # Flip sign for intuitive interpretation
        else:  # Region overperforming
            # Positive raw contribution = helpful (making region better)
            # Negative raw contribution = harmful (reducing advantage)
            df['contribution'] = raw_contribution  # Keep original sign
    else:
        # For metrics where lower is better (cost, errors)
        if delta > 0:  # Region underperforming (higher than desired)
            # Positive raw contribution = harmful (making gap worse)
            # Negative raw contribution = helpful (reducing gap)
            df['contribution'] = -raw_contribution  # Flip sign for intuitive interpretation
        else:  # Region overperforming (lower than others)
            # Positive raw contribution = helpful (making region better)
            # Negative raw contribution = harmful (reducing advantage)
            df['contribution'] = raw_contribution  # Keep original sign
    
    # Calculate heuristic score: sqrt(|contribution| + max(|contribution| - coverage, 0) / coverage)
    # Floor coverage at 1% to avoid division by zero
    coverage_floored = df['coverage'].clip(lower=0.01)
    df['score'] = np.sqrt(
        np.abs(df['contribution']) + 
        np.maximum(np.abs(df['contribution']) - df['coverage'], 0) / coverage_floored
    )
    
    # === NEW: attach tracebacks for payload ===
    try:
        fw, fs = _build_additive_tracebacks(
            df=df,
            metric_col=metric_col,
            row_total=row_total,
            delta=delta,
            higher_is_better=higher_is_better,
            focal_region=focal_region
        )
        df.attrs["tracebacks"] = {
            "formulas_walkthrough": fw.to_dict("records"),
            "formulas_summary": fs.to_dict("records")
        }
    except Exception as _e:
        logger.debug(f"Traceback build (additive) skipped: {_e}")
    # =========================================
    
    return df, delta


def format_value(value: float, is_percent: bool) -> str:
    """Format a value as either percentage or regular number."""
    if is_percent:
        return f"{value*100:.1f}%"
    return f"{value:,.0f}"



def plot_subregion_bars(
    df_slice: pd.DataFrame,
    metric_col: str,
    title: str,
    row_value: Optional[float] = None,
    region_value: Optional[float] = None,
    highlight_slices: Optional[list] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a meaningful bar chart showing sub-regions color-coded by their contribution to the performance gap.
    
    Args:
        df_slice: DataFrame containing the slice-level data with contribution analysis
        metric_col: Name of the metric column to plot
        title: Chart title
        row_value: Rest-of-world value to show as reference line (optional)
        region_value: Region average value to show as reference line (optional)
        highlight_slices: List of slices to highlight (optional)
        figsize: Figure size as (width, height) tuple
    
    Returns:
        Matplotlib figure with the contribution-aware bar chart
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get values and labels, handling NaN values
    values = df_slice[metric_col].values
    
    # Replace NaN values with 0 for plotting
    values = np.nan_to_num(values, nan=0.0)
    
    # Handle case where 'slice' is either a column or the index
    if 'slice' in df_slice.columns:
        labels = df_slice['slice'].values
    else:
        # 'slice' is the index
        labels = df_slice.index.values
    
    # Get contribution and score values for color-coding
    contributions = df_slice['contribution'].values if 'contribution' in df_slice.columns else np.zeros(len(values))
    scores = df_slice['score'].values if 'score' in df_slice.columns else np.zeros(len(values))
    
    # Get display name for y-axis
    ylabel = metric_col.replace('_', ' ').title()
    
    # Format values based on metric name
    is_percent = ('_pct' in metric_col or '%' in metric_col or 'rate' in metric_col.lower())
    
    # Determine slices to highlight (problematic ones passed in)
    highlight_set = set(highlight_slices) if highlight_slices else set()

    # Color-code bars: highlight only problematic slices provided
    colors = []
    for i, val in enumerate(values):
        slice_name = labels[i]
        if slice_name in highlight_set and len(contributions) > i:
            contribution = contributions[i]
            if contribution > 0:
                colors.append(COLORS['metric_positive'])  # Green for positive contributors
            else:
                colors.append(COLORS['metric_negative'])  # Red for negative contributors
        else:
            colors.append(COLORS['default_bar'])
    
    # Create bars
    bars = ax.bar(range(len(values)), values, width=0.6, color=colors, linewidth=0.5)
    
    # Add value labels on bars
    original_values = df_slice[metric_col].values  # Keep original values for labeling
    for i, val in enumerate(values):
        # Position text above the bar
        max_val = max(values) if len(values) > 0 else 0
        text_y = val + (max_val * 0.02)
        
        # First check for NaN
        original_val = original_values[i]
        if pd.isna(original_val):
            label_text = "N/A"
        else:
            # Use pre-calculated display string if available
            if 'display_value' in df_slice.columns:
                label_text = df_slice.iloc[i]['display_value']
            else:
                # For non-rate metrics or when components aren't available
                label_text = format_value(original_val, is_percent)
        
        ax.text(i, text_y, label_text, ha="center", va="bottom", 
               fontweight=FONTS['annotation']['weight'], fontsize=FONTS['annotation']['size'])
    
    # Customize the plot
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONTS['tick_label']['size'])
    ax.set_ylabel(ylabel, fontsize=FONTS['axis_label']['size'])
    ax.set_title(title, fontsize=FONTS['title']['size'], fontweight=FONTS['title']['weight'])
    
    # Format y-axis
    if is_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    # Add reference lines
    legend_labels = []
    
    # Add rest-of-world line if provided
    if row_value is not None:
        ax.axhline(row_value, color=COLORS['global_line'], linestyle='--', linewidth=2, alpha=0.7)
        legend_labels.append('Rest-of-World')
    
    # Add region average line if provided
    if region_value is not None:
        ax.axhline(region_value, color='#FF6B35', linestyle='-', linewidth=2, alpha=0.8)
        legend_labels.append('Region Average')
    
    # Add legend if we have lines
    if legend_labels:
        ax.legend(legend_labels, fontsize=FONTS['tick_label']['size'])
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def analyze_region_depth(
    sub_df: pd.DataFrame,
    config: Dict[str, Any],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    slice_col: str = 'slice',
    region_col: str = 'region'
) -> Dict[str, Dict[str, Any]]:
    """
    Perform depth analysis and return results in unified format directly.
    No integration step needed - this returns the final format.
    
    Returns:
        Dictionary in unified format: {metric_name: {'slides': {'depth': slide_data}}}
    """
    metrics_config = config.get('metrics', {})
    unified_results = {}
    
    # Analyze each metric
    for metric_name, metric_config in metrics_config.items():
        # Check if this metric has an anomaly detected
        if metric_name not in metric_anomaly_map:
            logger.warning(f"No anomaly information found for metric: {metric_name}")
            continue
            
        # Get the anomalous region for this specific metric
        anomalous_region = metric_anomaly_map[metric_name].get('anomalous_region')
        if not anomalous_region:
            logger.warning(f"No anomalous region specified for metric: {metric_name}")
            continue
        
        try:
            metric_type = metric_config['type']
            template = metric_config.get('template', '')
            
            # Filter data for this metric's anomalous region and rest-of-world
            region_df = sub_df[sub_df[region_col] == anomalous_region].copy()
            row_df = sub_df[sub_df[region_col] != anomalous_region].copy()
            
            if region_df.empty:
                logger.warning(f"No data found for region: {anomalous_region} (metric: {metric_name})")
                continue
            
            if row_df.empty:
                logger.warning(f"No rest-of-world data found for comparison (metric: {metric_name})")
                continue
            
            # Perform contribution analysis based on metric type
            if metric_type == 'rate':
                numerator_col = metric_config['numerator_col']
                denominator_col = metric_config['denominator_col']
                row_numerator = row_df[numerator_col].sum()
                row_denominator = row_df[denominator_col].sum()
                
                contrib_df, delta, row_rate = rate_contrib(
                    region_df, row_numerator, row_denominator,
                    denominator_col, numerator_col, metric_name,
                    metric_anomaly_map[metric_name].get('higher_is_better', True),
                    focal_region=anomalous_region  # <-- NEW
                )
                
                region_rate = contrib_df[numerator_col].sum() / contrib_df[denominator_col].sum()
                plot_metric_col = metric_name
                row_value = row_rate
                
                # Template parameters for rate metrics
                template_params = {
                    'metric_name': metric_name,
                    'anomalous_region': anomalous_region,
                    'delta': delta,
                    'row_rate': row_rate,
                    'region_rate': region_rate
                }
                
            elif metric_type == 'additive':
                metric_col = metric_config['metric_col']
                row_total = row_df[metric_col].sum()
                
                contrib_df, delta = additive_contrib(
                    region_df, row_total, metric_col,
                    metric_anomaly_map[metric_name].get('higher_is_better', True),
                    focal_region=anomalous_region  # <-- NEW
                )
                
                region_total = contrib_df[metric_col].sum()
                plot_metric_col = metric_col
                row_value = row_total / len(row_df) if len(row_df) > 0 else None
                
                # Template parameters for additive metrics
                template_params = {
                    'metric_name': metric_name,
                    'anomalous_region': anomalous_region,
                    'delta': delta,
                    'row_total': row_total,
                    'region_total': region_total
                }
                
            else:
                logger.warning(f"Unknown metric type '{metric_type}' for metric '{metric_name}'")
                continue
            
            # Create full analysis data sorted by score (for chart - show ALL slices)
            contrib_df = contrib_df.sort_values('score', ascending=False)
            
            # Keep display_value separate from summary columns
            chart_cols = ['slice', plot_metric_col, 'contribution', 'coverage', 'score', 'raw_contribution']
            if 'display_value' in contrib_df.columns:
                chart_cols.append('display_value')
            
            # Create display table without display_value
            display_cols = ['slice', plot_metric_col, 'contribution', 'coverage', 'score']
            display_table = contrib_df[display_cols].head(3).copy().set_index('slice')
            
            # Format display table
            display_table['contribution'] = display_table['contribution'].apply(lambda x: f"{x:.1%}")
            display_table['coverage'] = display_table['coverage'].apply(lambda x: f"{x:.1%}")
            display_table['score'] = display_table['score'].apply(lambda x: f"{x:.2f}")
            
            if metric_type == 'rate':
                display_table[plot_metric_col] = display_table[plot_metric_col].apply(lambda x: f"{x:.1%}")
            else:
                display_table[plot_metric_col] = display_table[plot_metric_col].apply(lambda x: f"{int(x):,}")
            
            # Calculate absolute contribution for sorting
            contrib_df['abs_contribution'] = contrib_df['contribution'].abs()
            
            # Calculate region average for comparison
            if metric_type == 'rate':
                region_value = region_rate
            else:
                region_value = region_total / len(contrib_df) if len(contrib_df) > 0 else 0
            
            # Determine if we're looking for positive or negative contributions
            if (metric_anomaly_map[metric_name].get('higher_is_better', True) and delta > 0) or (not metric_anomaly_map[metric_name].get('higher_is_better', True) and delta < 0):
                # Looking for positive contributions (lifts)
                contrib_df['contributor_type'] = np.where(contrib_df['contribution'] > 0, 'lift', 'drag')
                top_contributors = contrib_df[contrib_df['contribution'] > 0].nlargest(3, 'abs_contribution')
                contributor_type = 'lifts'
            else:
                # Looking for negative contributions (drags)
                contrib_df['contributor_type'] = np.where(contrib_df['contribution'] < 0, 'drag', 'lift')
                top_contributors = contrib_df[contrib_df['contribution'] < 0].nlargest(3, 'abs_contribution')
                contributor_type = 'drags'
            
            # Format contribution percentages with actual values
            def format_contrib(row):
                contrib_pct = row['contribution'] * 100  # Use actual value
                return f"{row['slice']}"  # Just return the slice name
            
            def format_pct(x):
                return f"{x*100:+.1f}%"  # Format as percentage with sign
            
            # Generate more explanatory sentence comparing against region average
            if not top_contributors.empty:
                # Build lists of slices & contributions but only keep those that are problematic
                problematic_slices: list[str] = []
                problematic_contribs: list[str] = []
                for _, row in top_contributors.iterrows():
                    slice_name = row['slice']
                    slice_value = row[plot_metric_col]
                    contrib_formatted = format_pct(row['contribution'])

                    # For underperforming regions, problematic slices are those above region average
                    # For overperforming regions, problematic slices are those below region average
                    is_problematic = (delta > 0 and slice_value > region_value) or (delta < 0 and slice_value < region_value)

                    if is_problematic:
                        problematic_slices.append(slice_name)
                        problematic_contribs.append(contrib_formatted)

                # Fallback to top contributors if no problematic slice meets criteria
                if problematic_slices:
                    main_contributors_sentence = (
                        f"{', '.join(problematic_slices)} contribute {', '.join(problematic_contribs)} "
                        f"to the gap between {anomalous_region} and Rest-of-World"
                    )
                else:
                    # Use up to 3 top contributors regardless
                    fallback_slices = top_contributors['slice'].tolist()
                    fallback_contribs = top_contributors['contribution'].apply(format_pct).tolist()
                    main_contributors_sentence = (
                        f"{', '.join(fallback_slices)} contribute {', '.join(fallback_contribs)} "
                        f"to the gap between {anomalous_region} and Rest-of-World"
                    )
            else:
                main_contributors_sentence = f"No significant {contributor_type} identified in the gap between {anomalous_region} and Rest-of-World"
            
            # Create full analysis DataFrame for chart
            chart_cols.append('abs_contribution')  # Add abs_contribution to chart columns
            full_analysis_df = contrib_df[chart_cols].copy().set_index('slice')
            
            # Update template parameters with main contributors sentence
            template_params['main_contributors_sentence'] = main_contributors_sentence
            
            # Use main_contributors_sentence as summary_text
            summary_text = main_contributors_sentence

            # Return in unified format directly - no integration needed!
            analysis_type = 'Depth'  # Derived from config_depth.yaml
            unified_results[metric_name] = {
                'slides': {
                    analysis_type: {
                        'summary': {
                            'summary_text': summary_text
                        },
                        'slide_info': {
                            'title': f"{metric_name} - Depth Analysis",
                            'template_text': template,
                            'template_params': template_params,
                            'figure_generators': [
                                {
                                    "function": plot_subregion_bars,
                                    'title_suffix': "",
                                    'params': {
                                        'df_slice': full_analysis_df,  # Pass ALL slices to chart function
                                        'metric_col': plot_metric_col,
                                        'title': f"{metric_name} by sub-regions",
                                        'row_value': row_value,
                                        'region_value': region_value,
                                        'highlight_slices': problematic_slices
                                    }
                                }
                            ],
                            'dfs': {"summary_table": display_table},  # Top 3 only for table display
                            'layout_type': "text_tables_figure",
                            'total_hypotheses': 1  # Depth analysis counts as one hypothesis
                        }
                    }
                },
                'payload': {
                    'summary_df': full_analysis_df,  # Full analysis data for internal inspection
                    'delta': delta,
                    'row_value': row_value,
                    # --- NEW: always include the two traceback tables ---
                    'tracebacks': contrib_df.attrs.get('tracebacks', {})
                },
            }
                
        except KeyError as e:
            logger.error(f"Missing required configuration for metric '{metric_name}': {e}")
        except Exception as e:
            logger.error(f"Error analyzing metric '{metric_name}': {e}")
    
    return unified_results


def create_synthetic_data() -> pd.DataFrame:
    """
    Create synthetic sub-region data for testing depth analysis.
    
    Returns:
        DataFrame with synthetic slice-level data
    """
    regions = ["North America", "Europe", "Asia", "Latin America"]
    rows = []

    # Helper function to append a slice
    def add_slice(region, suffix, visits, conv_rate, orders, aov, surveys, csat):
        rows.append({
            "slice": f"{region}_{suffix}",
            "region": region,
            "visits": visits,
            "conversions": int(round(visits * conv_rate)),
            "orders": orders,
            "revenue": round(orders * aov, 2),
            "surveys": surveys,
            "sum_csat": round(surveys * csat, 1)
        })

    # North America (EXTREME variance - designed to force mixed contributions)
    # Strategy: Create huge disparity in AOV while keeping revenue proportional allocation challenged
    add_slice("North America", "a", 20_000, 0.05, 1000, 25, 900, 3.0)   # Massive volume, tiny AOV → underperforms expected
    add_slice("North America", "b", 18_000, 0.06, 1080, 30, 850, 3.2)   # High volume, low AOV → underperforms  
    add_slice("North America", "c", 15_000, 0.07, 1050, 35, 800, 3.1)   # High volume, low AOV → underperforms
    add_slice("North America", "d", 12_000, 0.08, 960, 40, 700, 3.3)    # Medium volume, low AOV → underperforms
    add_slice("North America", "e", 2_000, 0.20, 400, 200, 150, 4.8)    # Tiny volume, HUGE AOV → outperforms expected!
    add_slice("North America", "f", 1_500, 0.22, 330, 180, 120, 4.7)    # Tiny volume, HUGE AOV → outperforms expected!
    add_slice("North America", "g", 1_000, 0.25, 250, 160, 80, 4.6)     # Tiny volume, HUGE AOV → outperforms expected!

    # Europe
    add_slice("Europe", "a", 9_000, 0.12, 1800, 79, 850, 4.4)
    add_slice("Europe", "b", 10_000, 0.11, 1900, 80, 900, 4.3)
    add_slice("Europe", "c", 11_000, 0.10, 2000, 81, 950, 4.2)

    # Asia
    add_slice("Asia", "a", 11_000, 0.14, 2100, 86, 1000, 4.6)
    add_slice("Asia", "b", 12_000, 0.13, 2300, 85, 1050, 4.5)
    add_slice("Asia", "c", 10_000, 0.12, 2000, 84, 950, 4.4)

    # Latin America
    add_slice("Latin America", "a", 8_000, 0.11, 1700, 73, 800, 4.1)
    add_slice("Latin America", "b", 9_000, 0.10, 1800, 72, 820, 4.0)
    add_slice("Latin America", "c", 8_500, 0.09, 1700, 71, 780, 3.9)

    return pd.DataFrame(rows)


def main(output_dir: str = '.'):
    """
    Test the depth analysis functionality with synthetic data.
    
    Args:
        output_dir: Directory to save output files
    """
    print("Starting Depth Analysis Testing...")
    
    # Create synthetic data
    sub_df = create_synthetic_data()
    
    print(f"\nCreated synthetic data with {len(sub_df)} slices across {sub_df['region'].nunique()} regions")
    print("\nData Preview:")
    print(sub_df.head())
    
    # Analyze North America (the anomalous region)
    anomalous_region = "North America"
    
    # Define test configuration (no need for config file for testing)
    test_config = {
        'metrics': {
            'conversion_rate_pct': {
                'name': "Conversion Rate",
                'type': "rate",
                'numerator_col': "conversions",
                'denominator_col': "visits",
                'template': "{{ metric_name }} in {{ anomalous_region }} is {{ (region_rate*100)|round(1) }}% compared to {{ (row_rate*100)|round(1) }}% in the rest of world.\n\n{{ summary_table }}\n\n**Coverage** shows each sub-region's share of total volume. **Contribution** shows each sub-region's share of the performance gap."
            },
            'revenue': {
                'name': "Revenue",
                'type': "additive",
                'metric_col': "revenue",
                'template': "{{ metric_name }} in {{ anomalous_region }} totals {{ '{:,.0f}'.format(region_total) }} compared to {{ '{:,.0f}'.format(row_total) }} in the rest of world.\n\n{{ summary_table }}\n\n**Coverage** shows each sub-region's share of total volume. **Contribution** shows each sub-region's share of the performance gap."
            },
            'customer_satisfaction': {
                'name': "Customer Satisfaction",
                'type': "rate",
                'numerator_col': "sum_csat",
                'denominator_col': "surveys",
                'template': "{{ metric_name }} in {{ anomalous_region }} averages {{ region_rate|round(2) }} compared to {{ row_rate|round(2) }} in the rest of world.\n\n{{ summary_table }}\n\n**Coverage** shows each sub-region's share of total volume. **Contribution** shows each sub-region's share of the performance gap."
            }
        }
    }
    
    print(f"\nPerforming depth analysis for: {anomalous_region}")
    
    # Create test metric_anomaly_map for the test
    test_metric_anomaly_map = {
        'conversion_rate_pct': {
            'anomalous_region': anomalous_region,
            'higher_is_better': True
        },
        'revenue': {
            'anomalous_region': anomalous_region,
            'higher_is_better': True
        }, 
        'customer_satisfaction': {
            'anomalous_region': anomalous_region,
            'higher_is_better': True
        }
    }
    
    results = analyze_region_depth(
        sub_df=sub_df,
        config=test_config,
        metric_anomaly_map=test_metric_anomaly_map
    )
    
    # Print structured results with more detail
    print(f"\n{'='*60}")
    print("STRUCTURED RESULTS")
    print(f"{'='*60}")
    
    for metric_name, metric_result in results.items():
        print(f"\nMetric: {metric_name}")
        
        # Access the depth slide data
        if 'Depth' in metric_result['slides']:
            depth_data = metric_result['slides']['Depth']
            
            print(f"Summary: {depth_data['summary']['summary_text']}")
            
            # Print figure parameters
            if 'slide_info' in depth_data:
                slide_info = depth_data['slide_info']
                if 'figure_generators' in slide_info:
                    for i, gen in enumerate(slide_info['figure_generators']):
                        print(f"\nFigure {i+1} Parameters:")
                        print(f"Function: {gen['function'].__name__}")
                        print("Parameters:")
                        for k, v in gen['params'].items():
                            if isinstance(v, pd.DataFrame):
                                print(f"{k} (DataFrame):")
                                print(v.head())
                            else:
                                print(f"{k}: {v}")
        else:
            print("No Depth analysis data found")
    
    print(f"\nDepth analysis completed!")
    
    return results


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run depth analysis testing')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Run with provided arguments
    main(output_dir=args.output_dir) 