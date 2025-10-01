from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from jinja2 import Template
import scipy.stats as stats
import textwrap


def get_non_reference_regions(df: pd.DataFrame, reference_row: str = "Global") -> List[str]:
    """Get all regions excluding the reference row."""
    return [r for r in df.index if r != reference_row]


def get_all_regions_ordered(df: pd.DataFrame, reference_row: str = "Global") -> List[str]:
    """Get all regions with reference row first, then others."""
    non_ref_regions = get_non_reference_regions(df, reference_row)
    return [reference_row] + non_ref_regions


def _calculate_region_deltas_and_colors(
    df: pd.DataFrame,
    metric_col: str,
    hypo_col: str,
    regions: List[str],
    global_metric: float,
    global_hypo: float,
    metric_anomaly_info: Dict[str, Any],
    expected_direction: str,
    reference_row: str
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Calculate deltas and colors for all regions in one pass."""
    metric_deltas = []
    hypo_deltas = []
    valid_regions = []
    region_colors = {}
    region_text_colors = {}
    region_intensities = []  # Store raw intensities for normalization
    
    # Get direction logic once
    _, higher_is_better_hypo = calculate_hypothesis_direction_logic(metric_anomaly_info, expected_direction)
    
    for r in regions:
        m_val = df.loc[r, metric_col]
        h_val = df.loc[r, hypo_col]
        if pd.isna(m_val) or pd.isna(h_val) or pd.isna(global_metric) or pd.isna(global_hypo):
            # Skip region with missing data
            continue
        if global_metric == 0 or global_hypo == 0:
            continue
        
        metric_delta = (m_val - global_metric) / global_metric
        hypo_delta = (h_val - global_hypo) / global_hypo
        
        metric_deltas.append(metric_delta)
        hypo_deltas.append(hypo_delta)
        valid_regions.append(r)
        
        # Store raw intensity for normalization
        intensity = abs(hypo_delta)
        region_intensities.append((r, intensity, h_val > global_hypo))
    
    metric_deltas = np.array(metric_deltas)
    hypo_deltas = np.array(hypo_deltas)
    regions = valid_regions  # overwrite with valid regions
    
    # Calculate hypothesis colors using helper
    region_colors, region_text_colors = _normalize_and_calculate_colors(
        region_intensities, higher_is_better_hypo, include_text_colors=True
    )
    
    # Calculate metric colors using same helper
    metric_intensities = [(r, abs(metric_deltas[i]), metric_deltas[i] > 0) for i, r in enumerate(valid_regions)]
    higher_is_better_metric = metric_anomaly_info.get('higher_is_better', True)
    metric_colors = _normalize_and_calculate_colors(
        metric_intensities, higher_is_better_metric, include_text_colors=False
    )

    # Add reference row and missing regions
    all_regions_ordered = get_all_regions_ordered(df, reference_row)
    for region in all_regions_ordered:
        if region not in region_colors:
            region_colors[region] = 'white'
            region_text_colors[region] = '#808080'
        if region not in metric_colors:
            metric_colors[region] = 'white'
    
    return metric_deltas, hypo_deltas, regions, region_colors, region_text_colors, metric_colors


def _normalize_and_calculate_colors(
    region_intensities: List[Tuple[str, float, bool]], 
    higher_is_better: bool,
    include_text_colors: bool = True
) -> Dict[str, str]:
    """Normalize intensities and calculate colors for regions."""
    colors = {}
    text_colors = {} if include_text_colors else None
    
    max_intensity = max([intensity for _, intensity, _ in region_intensities]) if region_intensities else 1.0
    
    for r, raw_intensity, is_positive_delta in region_intensities:
        normalized_intensity = min(1.0, raw_intensity / max_intensity) if max_intensity > 0 else 0.0
        is_desirable = is_positive_delta if higher_is_better else not is_positive_delta
        
        bg_color, text_color = get_colors_from_intensity(normalized_intensity, is_desirable)
        colors[r] = bg_color
        if text_colors is not None:
            text_colors[r] = text_color
    
    return (colors, text_colors) if include_text_colors else colors


def get_colors_from_intensity(intensity: float, is_desirable: bool) -> Tuple[str, str]:
    """Get both background and text colors from intensity and desirability."""
    # Background color for high-score rows
    if intensity < 0.05:
        bg_color = 'white'
    elif is_desirable:
        if intensity < 0.3:
            bg_color = '#F0F8F0'  # Very light green
        elif intensity < 0.6:
            bg_color = '#C8E6C9'  # Light green  
        elif intensity < 0.8:
            bg_color = '#81C784'  # Medium green
        else:
            bg_color = '#4CAF50'  # Strong green
    else:
        if intensity < 0.3:
            bg_color = '#FFF0F0'  # Very light red
        elif intensity < 0.6:
            bg_color = '#FFCDD2'  # Light red
        elif intensity < 0.8:
            bg_color = '#E57373'  # Medium red
        else:
            bg_color = '#F44336'  # Strong red
    
    # Text color for low-score rows
    if intensity < 0.6:
        text_color = '#808080'  # Grey for low intensity
    else:
        text_color = '#2E7D32' if is_desirable else '#D32F2F'
    
    return bg_color, text_color


def calculate_hypothesis_direction_logic(
    metric_anomaly_info: Dict[str, Any], 
    expected_direction: str
) -> Tuple[bool, bool]:
    """
    Calculate the direction logic for hypothesis evaluation.
    
    Args:
        metric_anomaly_info: Anomaly information containing higher_is_better
        expected_direction: Expected relationship ('same' or 'opposite')
    
    Returns:
        Tuple of (metric_higher_is_better, higher_is_better_hypo)
    """
    metric_higher_is_better = metric_anomaly_info.get('higher_is_better', True)
    higher_is_better_hypo = metric_higher_is_better if expected_direction == 'same' else not metric_higher_is_better
    return metric_higher_is_better, higher_is_better_hypo


def format_number_for_display(val: float, col_name: str) -> str:
    """Centralized number formatting for consistent display across the application."""
    if pd.isna(val):
        return 'N/A'
    
    # Determine if it's a percentage column
    is_pct = any(substr in col_name.lower() for substr in ['pct', '%', 'rate'])
    
    if is_pct:
        return f"{val*100:.1f}%"
    elif abs(val) < 1000:
        return f"{val:.2f}"
    else:
        return f"{val:,.0f}"


def format_score(score: float) -> str:
    """Format hypothesis scores consistently."""
    if pd.isna(score):
        return 'N/A'
    return f"{score:.2f}"


def create_template_params(
    metric_anomaly_info: Dict[str, Any],
    metric_name: str,
    hypo_name: Optional[str] = None,
    hypo_result: Optional[Dict[str, Any]] = None,
    reference_row: str = "Global"
) -> Dict[str, Any]:
    """
    Create standardized template parameters for rendering.
    
    Args:
        metric_anomaly_info: Anomaly information
        metric_name: Name of the metric
        hypo_name: Name of the hypothesis (optional)
        hypo_result: Hypothesis result (optional)
    
    Returns:
        Dictionary of template parameters
    """
    template_params = {
        'region': metric_anomaly_info['anomalous_region'],
        'metric_name': metric_name,
        'metric_deviation': metric_anomaly_info.get('magnitude', '0%'),
        'metric_dir': metric_anomaly_info['direction'],
        'reference_row': reference_row
    }
    
    if hypo_name and hypo_result:
        template_params.update({
            'hypo_name': hypo_name,
            'hypo_dir': hypo_result['direction'],
            'hypo_delta': hypo_result['magnitude'],
            'ref_hypo_val': hypo_result['ref_hypo_val'],
            'score': hypo_result['final_score'],
            'explained_ratio': hypo_result['explained_ratio'] * 100
        })
    
    return template_params

# Color scheme for visualizations
COLORS: Dict[str, Union[str, Dict[str, str]]] = {
    'metric_negative': '#e74c3c',     # Red for bad metric anomalies
    'metric_positive': '#2ecc71',     # Green for good metric anomalies
    'hypo_highlight': '#5DADE2',      # Blue for highlighted hypothesis region
    'default_bar': '#BDC3C7',         # Light gray for default bars
    'global_line': '#34495e',         # Dark blue-gray for reference line
    'score_color': '#AF7AC5',         # Purple for final score
    'score_components': {
        'direction_alignment': '#3498DB',  # Blue 
        'consistency': '#4ECDC4',          # Teal
        'hypo_z_score_norm': '#FFC300',    # Yellow/Orange
        'explained_ratio': '#FF9F43'       # Orange
    },
    'sign_score_components': {
        'sign_agreement': '#3498DB',       # Blue (reusing direction_alignment color)
        'explained_ratio': '#FF9F43'       # Orange (reusing)
    }
}

# Font styling constants for consistent appearance
FONTS: Dict[str, Dict[str, Union[int, str]]] = {
    'title': {'size': 16, 'weight': 'normal',
        'family': 'Arial'},
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
    },
    'score_card': {
        'size': 12,
        'weight': 'normal',
        'family': 'Arial'
    },
    'legend': {
        'size': 12,
        'weight': 'normal',
        'family': 'Arial'
    }
}


def _build_formulas_tables(
    df: pd.DataFrame,
    metric_col: str,
    hypo_col: str,
    expected_direction: str,
    anomalous_region: str,
    reference_row: str = "Global"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build detailed walk-through tables showing formulas and calculations step-by-step.
    
    This creates two DataFrames that provide complete transparency into the scoring process:
    1. formulas_walkthrough: Shows per-region calculations with formulas
    2. formulas_summary: Shows final score components and their calculations
    
    Args:
        df: Input DataFrame with metric and hypothesis data
        metric_col: Name of the metric column
        hypo_col: Name of the hypothesis column  
        expected_direction: Expected relationship ('same' or 'opposite')
        anomalous_region: The focal region being analyzed
        reference_row: Reference row name (usually "Global")
        
    Returns:
        Tuple of (formulas_walkthrough_df, formulas_summary_df)
    """
    M_g = df.loc[reference_row, metric_col]
    H_g = df.loc[reference_row, hypo_col]
    regions = get_non_reference_regions(df, reference_row)

    rows, m_deltas, h_adj_deltas, sign_flags = [], [], [], []
    for r in regions:
        M_r = df.loc[r, metric_col]
        H_r = df.loc[r, hypo_col]
        dM = np.nan if (pd.isna(M_g) or M_g == 0) else (M_r - M_g) / M_g
        dH_raw = np.nan if (pd.isna(H_g) or H_g == 0) else (H_r - H_g) / H_g
        dH_adj = -dH_raw if expected_direction == "opposite" else dH_raw

        m_deltas.append(dM)
        h_adj_deltas.append(dH_adj)
        sign_flags.append(bool(np.sign(dM) == np.sign(dH_adj)))

        rows.append({
            "Region": r,
            "Metric (M_r)": M_r,
            "Δ_metric formula": f"({M_r} - {M_g}) / {M_g}",
            "Δ_metric value": dM,
            "Hypothesis (H_r)": H_r,
            "Δ_hypo_raw formula": f"({H_r} - {H_g}) / {H_g}",
            "Δ_hypo_raw value": dH_raw,
            "Δ_hypo_adj formula": ("-(" + f"({H_r} - {H_g}) / {H_g}" + ")") if expected_direction == "opposite"
                                   else f"({H_r} - {H_g}) / {H_g}",
            "Δ_hypo_adj value": dH_adj,
            "Sign match?": sign_flags[-1]
        })

    walk_df = pd.DataFrame(rows)

    m_arr = np.array(m_deltas, dtype=float)
    h_arr = np.array(h_adj_deltas, dtype=float)
    sign_agree = float(np.nanmean(np.array(sign_flags, dtype=float))) if len(sign_flags) else 0.0
    sigma_m = float(np.nanstd(m_arr)) if len(m_arr) else 0.0
    sigma_h = float(np.nanstd(h_arr)) if len(h_arr) else 0.0

    focal_idx = regions.index(anomalous_region) if anomalous_region in regions else None
    m_focal = m_arr[focal_idx] if focal_idx is not None else np.nan
    h_focal = h_arr[focal_idx] if focal_idx is not None else np.nan

    z_m = (m_focal / sigma_m) if sigma_m and sigma_m > 1e-6 else 0.0
    z_h = (h_focal / sigma_h) if sigma_h and sigma_h > 1e-6 else 0.0
    explained_ratio = 0.0 if (np.isnan(z_m) or np.isnan(z_h) or abs(z_m) < 1e-6) else min(abs(z_h)/abs(z_m), 1.0)
    focal_agrees = bool(sign_flags[focal_idx]) if (focal_idx is not None and len(sign_flags)) else False

    base_score = 0.6*sign_agree + 0.4*explained_ratio
    final_score = base_score if focal_agrees else base_score*0.3

    # Create summary showing all calculation steps with actual focal region name
    summary_df = pd.DataFrame([
        {"Component": "Expected direction", "Formula": expected_direction, "Value": expected_direction},
        {"Component": "Sign agreement", "Formula": f"{sum(sign_flags)}/{len(sign_flags)}", "Value": sign_agree},
        {"Component": "σ(metric Δ)", "Formula": f"std({m_arr.tolist()})", "Value": sigma_m},
        {"Component": "σ(hypo Δ, adj)", "Formula": f"std({h_arr.tolist()})", "Value": sigma_h},
        {"Component": f"z_m ({anomalous_region})", "Formula": f"{m_focal} / {sigma_m}", "Value": z_m},
        {"Component": f"z_h ({anomalous_region})", "Formula": f"{h_focal} / {sigma_h}", "Value": z_h},
        {"Component": "Explained ratio", "Formula": f"min(|{z_h}|/|{z_m}|, 1)", "Value": explained_ratio},
        {"Component": "Penalty rule", "Formula": f"×0.3 if {anomalous_region} sign mismatches",
         "Value": "applied" if not focal_agrees else "not applied"},
        {"Component": "Final score", "Formula": f"0.6*{sign_agree} + 0.4*{explained_ratio}", "Value": final_score},
    ])

    return walk_df, summary_df


def sign_based_score_hypothesis(
    df: pd.DataFrame,
    metric_anomaly_info: Dict[str, Any],
    expected_direction: str = 'same',
    reference_row: str = "Global"
) -> Dict[str, Any]:
    """
    Score hypothesis using sign-agreement and explained-ratio metrics,
    which is more robust with small sample sizes (4-5 regions).
    
    Args:
        df: DataFrame with two columns - metric and hypothesis values indexed by region
        metric_anomaly_info: Dictionary containing anomaly information
        expected_direction: Expected relationship between metric and hypothesis ('same' or 'opposite')
    
    Returns:
        Dictionary containing hypothesis evaluation and scoring results
    """
    # Extract metric and hypothesis columns
    metric_col, hypo_col = df.columns
    
    # Extract key info
    anomalous_region = metric_anomaly_info['anomalous_region']
    metric_val = metric_anomaly_info['metric_val']
    ref_metric_val = metric_anomaly_info['global_val']

    # Calculate deltas and colors for all regions
    regions = get_non_reference_regions(df, reference_row)
    global_metric = df.loc[reference_row, metric_col]
    global_hypo = df.loc[reference_row, hypo_col]
    
    metric_deltas, hypo_deltas, regions, region_colors, region_text_colors, metric_colors = _calculate_region_deltas_and_colors(
        df, metric_col, hypo_col, regions, global_metric, global_hypo, 
        metric_anomaly_info, expected_direction, reference_row
    )
    
    # Handle NaNs for focal/global values
    if pd.isna(metric_val) or pd.isna(ref_metric_val):
        metric_delta = np.nan
    else:
        metric_delta = (metric_val - ref_metric_val) / ref_metric_val if ref_metric_val != 0 else np.nan
    
    # Get hypothesis values for anomalous region
    hypo_val = df.loc[anomalous_region, hypo_col]
    ref_hypo_val = df.loc[reference_row, hypo_col]
    if pd.isna(hypo_val) or pd.isna(ref_hypo_val):
        hypo_delta = np.nan
    else:
        hypo_delta = (hypo_val - ref_hypo_val) / ref_hypo_val if ref_hypo_val != 0 else np.nan
    
    # Adjust sign for expected direction
    if expected_direction == 'opposite':
        # Flip the sign of hypothesis deltas if expecting opposite direction
        hypo_deltas = -hypo_deltas
    
    # Count how many regions have same sign
    if len(metric_deltas) == 0:
        sign_agreements = np.array([])
        sign_agreement_score = 0.0
    else:
        sign_agreements = (np.sign(metric_deltas) == np.sign(hypo_deltas))
        sign_agreement_score = sign_agreements.sum() / len(regions)
    
    # CORNER CASE FIX: Check if focal region agrees with hypothesis direction
    if anomalous_region in regions:
        anomalous_idx = regions.index(anomalous_region)
        focal_region_agrees = sign_agreements[anomalous_idx]
    else:
        # Missing data for focal region – treat as disagreement
        focal_region_agrees = False
    
    # Calculate binomial p-value (not used in score but included for reference)
    if len(regions) == 0:
        p_binom_pvalue = 1.0
    else:
        p_binom_pvalue = stats.binomtest(sign_agreements.sum(), n=len(regions), p=0.5).pvalue
    
    # Calculate explained ratio for the anomalous region using MAD-normalized z-scores
    # Compute robust spreads using MAD (Median Absolute Deviation)
    sigma_m = np.std(metric_deltas)  # ddof=0 population std
    sigma_h = np.std(hypo_deltas)
    
    # Convert focal region deltas to z-scores
    z_m = metric_delta / sigma_m if sigma_m > 1e-6 else 0
    z_h = hypo_delta / sigma_h if sigma_h > 1e-6 else 0
    
    # Explained-ratio using z-scores (cap at 1)
    if np.isnan(z_m) or np.isnan(z_h) or abs(z_m) < 1e-6:
        explained_ratio = 0.0
    else:
        explained_ratio = min(abs(z_h) / abs(z_m), 1.0)
    
    # Enhanced guardrail logic with focal region check
    meets_basic_guardrails = (sign_agreement_score >= 0.5 and 
                             explained_ratio >= 0.2)
    
    # CORNER CASE: Even if overall sign agreement is good, focal region must agree
    meets_focal_check = focal_region_agrees
    
    # Combined guardrail check (will also consider missing data later)
    meets_guardrails = meets_basic_guardrails and meets_focal_check
    
    # RANKING FIX: Integrate guardrails into final_score for proper ranking
    base_score = 0.6 * sign_agreement_score + 0.4 * explained_ratio
    penalty_factor = 0.3
    if not focal_region_agrees:
        final_score = base_score * penalty_factor  # Only penalise when focal region disagrees
    else:
        final_score = base_score
    
    # Determine failure reason if it doesn't explain
    failure_reason = ""
    if not meets_guardrails:
        reasons = []
        if not meets_basic_guardrails:
            if sign_agreement_score < 0.5:
                reasons.append("wrong hypothesis direction")
            if explained_ratio < 0.2:
                reasons.append("coincidental moves")
        if not meets_focal_check:
            reasons.append("focal region direction mismatch")
        if np.isnan(metric_delta) or np.isnan(hypo_delta):
            reasons.append("missing data")
        failure_reason = ", ".join(reasons)
    
    # Format magnitude based on column name 
    is_percent_column = '_pct' in hypo_col or '%' in hypo_col
    if pd.isna(hypo_val) or pd.isna(ref_hypo_val):
        magnitude_str = 'N/A'
    else:
        if is_percent_column:
            magnitude_value = abs(hypo_val - ref_hypo_val) * 100
            magnitude_unit = 'pp'
        else:
            magnitude_value = abs(hypo_delta) * 100
            magnitude_unit = '%'
        magnitude_str = f"{magnitude_value:.1f}{magnitude_unit}"
    
    # Build detailed traceback tables showing step-by-step calculations
    fw_df, fs_df = _build_formulas_tables(
        df=df,
        metric_col=metric_col,
        hypo_col=hypo_col,
        expected_direction=expected_direction,
        anomalous_region=anomalous_region,
        reference_row=reference_row
    )

    # Results
    return {
        'hypo_val': hypo_val,
        'direction': 'higher' if hypo_delta > 0 else 'lower',
        'ref_hypo_val': ref_hypo_val,
        'magnitude': magnitude_str,
        'expected_direction': expected_direction,
        'scores': {
            'sign_agreement': sign_agreement_score,
            'explained_ratio': explained_ratio,
            'focal_region_agrees': focal_region_agrees,
            'p_value': p_binom_pvalue,
            'final_score': final_score,
            'explains': meets_guardrails,
            'failure_reason': failure_reason
        },
        'colors': {
            'background_colors': region_colors,
            'text_colors': region_text_colors,
            'metric_colors': metric_colors
        },
        # JSON-serializable traceback payload with step-by-step calculations
        'traceback': {
            'formulas_walkthrough': fw_df.to_dict('records'),
            'formulas_summary': fs_df.to_dict('records')
        }
    }


def plot_bars(
    ax: plt.Axes, 
    df: pd.DataFrame, 
    metric_anomaly_info: Dict[str, Any], 
    hypo_result: Optional[Dict[str, Any]] = None, 
    plot_type: str = 'metric',
    highlight_region: bool = True,
    reference_row: str = "Global"
) -> plt.Axes:
    """
    Plot a bar chart for either metric or hypothesis data.
    
    Args:
        ax: Matplotlib axes to plot on
        df: DataFrame with values to plot (two columns for metric and hypothesis)
        metric_anomaly_info: Info about the metric anomaly
        hypo_result: Results from hypothesis scoring (needed for hypothesis plot)
        plot_type: 'metric' or 'hypothesis'
        highlight_region: Whether to highlight the anomalous region (for hypotheses)
    
    Returns:
        The modified axes
    """
    # Clear existing content
    ax.clear()
    
    # Set up the plot based on type
    if plot_type == 'metric':
        col_to_plot = df.columns[0]  # First column is metric
    else:
        col_to_plot = df.columns[1]  # Second column is hypothesis
    
    # Since we now use display names as column headers, col_to_plot is already the display name
    display_name = col_to_plot
    
    # Format y-axis as percentage if needed
    is_percent = ('pct' in col_to_plot.lower() or '%' in col_to_plot or 'rate' in col_to_plot.lower() or
                 (hypo_result and 'pp' in hypo_result['magnitude']))
    
    # Extract regions to plot as bars (exclude the reference row). The
    # reference value is shown as a dashed line for context.
    regions = get_non_reference_regions(df, reference_row)
    values_series = df.loc[regions, col_to_plot]
    values = values_series.values
    
    # Set up bar colors
    bar_colors = [COLORS['default_bar']] * len(regions)
    
    # Determine colors based on plot type
    anomalous_region = metric_anomaly_info['anomalous_region']
    if plot_type == 'metric':
        # Color anomalous region based on whether higher is better
        if anomalous_region in regions:
            idx = regions.index(anomalous_region)
            direction = metric_anomaly_info['direction']
            higher_is_better = metric_anomaly_info.get('higher_is_better', True)
            
            # Determine if this is a good or bad anomaly
            is_good = (direction == 'higher' and higher_is_better) or (direction == 'lower' and not higher_is_better)
            bar_colors[idx] = COLORS['metric_positive'] if is_good else COLORS['metric_negative']
    elif highlight_region:
        # For hypothesis, highlight the anomalous region only if highlight_region is True
        if anomalous_region in regions:
            idx = regions.index(anomalous_region)
            bar_colors[idx] = COLORS['hypo_highlight']
    
    # Prepare bar heights, handle NaN by setting height 0 and grey color
    plot_values = []
    for i, val in enumerate(values):
        if pd.isna(val):
            plot_values.append(0.0)
            bar_colors[i] = '#D3D3D3'  # light grey for missing data
        else:
            plot_values.append(val)
    
    # Plot bars
    x_positions = np.arange(len(regions))
    bars = ax.bar(x_positions, plot_values, color=bar_colors)
    
    # Set x-ticks with region names and adapt label style to avoid overlap
    ax.set_xticks(x_positions)
    n_bars = len(regions)
    # Dynamic rotation and font size for readability
    if n_bars <= 6:
        rotation = 0
        lbl_size = FONTS['tick_label']['size']
    elif n_bars <= 10:
        rotation = 30
        lbl_size = max(9, FONTS['tick_label']['size'] - 1)
    elif n_bars <= 14:
        rotation = 45
        lbl_size = max(8, FONTS['tick_label']['size'] - 2)
    else:
        rotation = 60
        lbl_size = max(8, FONTS['tick_label']['size'] - 3)
    ax.set_xticklabels(regions, rotation=rotation, ha='right' if rotation else 'center', fontsize=lbl_size)
    
    # Add reference line if reference row is in the data and value is not NaN
    if reference_row in df.index:
        try:
            ref_val = df.loc[reference_row, col_to_plot]
            if not pd.isna(ref_val):
                ax.axhline(ref_val, color=COLORS['global_line'], linestyle='--', linewidth=1)
        except KeyError:
            pass
    
    # Add values on top of bars or N/A
    for i, val in enumerate(values):
        if pd.isna(val):
            ax.text(i, 0, 'N/A', ha='center', va='bottom', fontsize=FONTS['tick_label']['size'])
        else:
            # Use centralized formatting
            formatted_val = format_number_for_display(val, col_to_plot)
            ax.text(i, val, formatted_val, ha='center', va='bottom', fontsize=FONTS['tick_label']['size'])
    
    # Set y-axis label using display name
    ax.set_ylabel(display_name, fontsize=FONTS['axis_label']['size'])
    
    # Format y-axis as percentage if needed
    if is_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    
    # Adjust y-axis to only use 80% of vertical space, leaving 20% at top for score components
    y_min, y_max = ax.get_ylim()
    if y_max == 0:  # All bars zero (likely NaNs)
        y_max = 1
    y_range = y_max - y_min
    
    # Set new limits that leave 20% at top empty
    ax.set_ylim(y_min, y_max + (y_range * 0.25))
    
    # Add score components for hypothesis plot
    if plot_type == 'hypothesis' and hypo_result and 'final_score' in hypo_result:
        # Handle flattened structure (from DataFrame rows)
        components = [
            ('final_score', hypo_result['final_score']),
            ('sign_agreement', hypo_result.get('sign_agreement', 0.0)),
            ('explained_ratio', hypo_result.get('explained_ratio', 0.0))
        ]
        
        # Calculate positions - place cards near the top of the axes
        card_width = 0.11
        card_height = 0.05
        spacing = 0.01
        total_width = len(components) * (card_width + spacing) - spacing
        start_x = 0.5 - (total_width / 2)
        
        # Position at 90% up the axes (in axis coordinates)
        y_pos = 0.9
        
        # Add each score component
        for i, (comp_name, comp_value) in enumerate(components):
            x_pos = start_x + i * (card_width + spacing)
            
            # Use score color for final score, component colors for others
            if comp_name == 'final_score':
                color = COLORS['score_color']
            else:
                color = COLORS['sign_score_components'].get(comp_name, COLORS['default_bar'])
            
            # Add background rectangle
            rect = plt.Rectangle(
                (x_pos, y_pos), card_width, card_height,
                facecolor=color, alpha=0.2, transform=ax.transAxes
            )
            ax.add_artist(rect)
            
            # Add value text
            ax.text(
                x_pos + card_width/2, y_pos + card_height/2,
                f"{comp_value:.2f}", ha='center', va='center',
                fontweight='bold', color=color, fontsize=FONTS['score_card']['size'],
                transform=ax.transAxes
            )
    
    return ax


def create_scatter_grid(
    df: pd.DataFrame, 
    metric_col: str,
    hypo_cols: List[str],
    metric_anomaly_info: Dict[str, Any],
    expected_directions: Dict[str, str]
) -> plt.Figure:
    """
    Create a grid of scatter plots with natural aspect ratios.
    Uses the original readable figsize approach.
    """
    n_plots = len(hypo_cols)
    if n_plots == 0:
        fig = plt.figure() 
        plt.close(fig)
        return fig

    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Original figsize calculations that were readable but adjusted for slide fitting
    if n_rows == 1:
        # Single row - make narrower to fit slides better
        figsize = (10, 5)
    elif n_rows == 2:
        # Two rows - make narrower
        figsize = (12, 6)
    else:
        # Three or more rows - keep height reasonable
        figsize = (12, n_rows * 2.5)

    # Create figure and subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=figsize, 
        squeeze=False
    )
    axes = axes.flatten()

    for i, hypo_col in enumerate(hypo_cols):
        ax = axes[i]
        temp_df = df[[metric_col, hypo_col]]
        expected_direction = expected_directions.get(hypo_col, 'same')
        plot_scatter(ax, temp_df, metric_anomaly_info, expected_direction)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Use tight_layout
    plt.tight_layout(pad=1.0)
    
    return fig

def calculate_tight_table_dimensions(
    all_regions: List[str], 
    formatted_row_labels: List[str], 
    table_rows: List[List[str]],
    metric_col: str
) -> Tuple[float, float]:
    """Calculate tight-fitting dimensions with minimal whitespace."""
    # More precise character width calculation
    char_width = 0.06  # Reduced from 0.08 - matplotlib tables are more compact
    
    # Calculate actual content widths more precisely
    col_headers = all_regions + ["Score", "Rank"]
    col_widths = []
    
    # More precise column width calculation
    for i, header in enumerate(col_headers):
        header_width = len(str(header)) * char_width
        
        # Check actual data content for this column
        data_width = header_width
        if table_rows:
            for row in table_rows[:3]:  # Check first few rows for accurate sizing
                if i < len(row):
                    data_width = max(data_width, len(str(row[i])) * char_width)
        
        # Minimum column width
        col_widths.append(max(data_width, 0.6))  # Min 0.6" per column
    
    # Row labels width - more conservative calculation
    max_row_label_length = max(len(str(label)) for label in formatted_row_labels + [metric_col])
    row_label_width = max_row_label_length * char_width
    
    # Tight table width calculation
    data_width = sum(col_widths)
    table_content_width = row_label_width + data_width
    
    # Minimal padding - just enough for readability
    padding = 0.2  # Very minimal padding
    total_table_width = table_content_width + padding
    
    # Compact height calculation
    num_hypotheses = len(formatted_row_labels)
    max_name_length = max(len(label) for label in formatted_row_labels) if formatted_row_labels else 20
    
    # More compact row heights
    base_row_height = 0.25  # Reduced from 0.35
    if max_name_length > 50:
        base_row_height = 0.35  # Only increase for very long names
    elif max_name_length > 30:
        base_row_height = 0.3
    
    # Compact total height
    metric_table_height = 0.8  # Space for metric table + title
    hypo_table_height = num_hypotheses * base_row_height + 0.6  # Hypotheses + title
    total_height = metric_table_height + hypo_table_height + 0.3  # Small gap
    
    return total_table_width, total_height


def create_hypothesis_delta_table(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    metric_col: str,
    fontsize: int = 10,
    reference_row: str = "Global"
) -> plt.Figure:
    """Create a compact table showing metric and hypothesis values for all regions.
    
    Uses unified sizing approach where both ax1 and ax2 tables are sized together
    with consistent width calculations and smart wrapping applied uniformly.
    """
    # Extract basic info
    all_regions = get_all_regions_ordered(df, reference_row)
    
    # Extract table data directly from DataFrame (use original metric name for filtering)
    metric_results = results_df[results_df['metric'] == metric_col].sort_values('rank')
    
    # Get wrapped metric name for display (if available)
    wrapped_metric_name = metric_results.iloc[0]['wrapped_metric'] if len(metric_results) > 0 and 'wrapped_metric' in metric_results.columns else textwrap.fill(metric_col.replace('_', ' '), width=25)
    
    table_rows = []
    formatted_row_labels = []
    
    for _, row in metric_results.iterrows():
        # Use hypothesis name from DataFrame (already wrapped)
        formatted_row_labels.append(row['hypothesis'])
        
        # Build row data: regional values + score + rank
        formatted_regional_values = row['formatted_regional_values']
        row_data = [formatted_regional_values[region] for region in all_regions]
        row_data.extend([row['score'], str(row['rank'])])
        table_rows.append(row_data)
    
    # TIGHT SIZING: Calculate minimal dimensions for maximum space utilization
    table_width, table_height = calculate_tight_table_dimensions(
        all_regions, formatted_row_labels, table_rows, metric_col
    )
    
    # Tight figure dimensions - minimal margins
    margin_space = 0.4  # Total margin space (much reduced)
    fig_width = table_width + margin_space  # No arbitrary cap - size to content
    fig_height = table_height + 0.3  # Minimal vertical padding
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Minimal margins for maximum table space utilization
    margin_left, margin_right = 0.02, 0.02  # Very tight side margins
    margin_top, margin_bottom = 0.02, 0.02  # Very tight top/bottom margins  
    gap_between_tables = 0.01  # Minimal gap
    
    # Calculate positions
    available_height = 1.0 - margin_top - margin_bottom - gap_between_tables
    metric_height = available_height * 0.2   # 20% for metric
    hypo_height = available_height * 0.8     # 80% for hypotheses
    
    # Create subplots
    metric_bottom = margin_bottom + hypo_height + gap_between_tables
    hypo_bottom = margin_bottom
    
    ax1 = fig.add_axes([margin_left, metric_bottom, 1.0 - margin_left - margin_right, metric_height])
    ax2 = fig.add_axes([margin_left, hypo_bottom, 1.0 - margin_left - margin_right, hypo_height])
    
    ax1.axis('off')
    ax2.axis('off')
    
    # === METRIC TABLE (simplified) ===
    metric_vals = df.loc[all_regions, metric_col].values

    metric_data = [[format_number_for_display(val, metric_col) for val in metric_vals]]
    
    metric_table = ax1.table(
        cellText=metric_data,
        colLabels=all_regions,
        rowLabels=[wrapped_metric_name],
        loc='center',
        cellLoc='center'
    )
    metric_table.auto_set_font_size(False)
    metric_table.set_fontsize(fontsize)
    
    # Apply TIGHT column width settings for space efficiency
    for i in range(len(all_regions)):
        metric_table.auto_set_column_width(i)
    metric_table.auto_set_column_width(-1)  # Row labels
    
    # More compact scaling for better space utilization
    metric_table.scale(1.0, 1.2)  # Reduced vertical scaling
    
    # Color metric cells using metric colors from first hypothesis result
    if len(metric_results) > 0:
        first_row = metric_results.iloc[0]
        # Get metric colors from the DataFrame
        if 'metric_colors' in first_row and isinstance(first_row['metric_colors'], dict):
            metric_colors = first_row['metric_colors']
            for i, region in enumerate(all_regions):
                color = metric_colors.get(region, 'white')
                metric_table[(1, i)].set_facecolor(color)
    
    # Bold headers
    for i in range(len(all_regions)):
        metric_table[(0, i)].set_text_props(weight='bold')
    metric_table[(1, -1)].set_text_props(weight='bold')
    
    ax1.text(0.5, 1.05, "Metric Values Across Regions", 
             fontsize=FONTS['title']['size'], ha='center', va='bottom',
             transform=ax1.transAxes, weight='normal')
    
    # === HYPOTHESIS TABLE (using data extracted above) ===
    hypo_table = ax2.table(
        cellText=table_rows,
        colLabels=all_regions + ["Score", "Rank"],
        rowLabels=formatted_row_labels,
        loc='center',
        cellLoc='left'
    )
    hypo_table.auto_set_font_size(False)
    hypo_table.set_fontsize(fontsize)
    
    # Apply UNIFIED column width settings to match metric table
    for i in range(len(all_regions) + 2):  # All data columns (regions + score + rank)
        hypo_table.auto_set_column_width(i)
    hypo_table.auto_set_column_width(-1)  # Row labels
    
    # Configure UNIFIED smart text wrapping for both table types
    def apply_smart_wrapping(table, table_type="hypothesis"):
        """Apply consistent smart wrapping to any table."""
        cell_dict = table.get_celld()
        for (row, col), cell in cell_dict.items():
            # Enable text wrapping and proper alignment for ALL cells
            cell.get_text().set_fontsize(fontsize)
            cell.get_text().set_wrap(True)
            
            # Smart height adjustment for row labels (works for both metric and hypothesis names)
            if col == -1 and row > 0:  # Row label cells
                text_content = cell.get_text().get_text()
                # Count newlines to determine wrapped lines
                estimated_lines = text_content.count('\n') + 1
                if estimated_lines > 1:
                    # Dynamically increase cell height for ALL cells in this row
                    factor = estimated_lines * 0.85
                    for (r2, _c2), row_cell in cell_dict.items():
                        if r2 == row:
                            current_height = row_cell.get_height()
                            row_cell.set_height(current_height * factor)
    
    # Apply smart wrapping to BOTH tables consistently
    apply_smart_wrapping(metric_table, "metric")
    apply_smart_wrapping(hypo_table, "hypothesis")
    
    # More compact scaling for better space utilization
    hypo_table.scale(1.0, 1.1)  # Reduced from 1.3 to 1.1 for tighter layout
    
    # Apply colors and styling using DataFrame data
    for i, (_, row) in enumerate(metric_results.iterrows()):
        score = row['final_score']
        background_colors = row['background_colors']
        text_colors = row['text_colors']
        
        # Build row colors: background colors for regions + white for score/rank
        row_colors = []
        if score >= 0.5:
            # Use background colors for high-score rows
            for region in all_regions:
                row_colors.append(background_colors.get(region, 'white'))
        else:
            # White background for low-score rows
            row_colors = ['white'] * len(all_regions)
        
        # Add white for score and rank columns
        row_colors.extend(['white', 'white'])
        
        # Apply background colors
        for j, color in enumerate(row_colors):
            hypo_table[(i+1, j)].set_facecolor(color)
        
        # For low-score rows: apply text colors from DataFrame (already computed)
        if score < 0.5:
            text_colors = row['text_colors']
            
            # Apply text colors to region columns
            for j, region in enumerate(all_regions):
                text_color = text_colors.get(region, '#808080')
                hypo_table[(i+1, j)].set_text_props(color=text_color)
            
            # Gray out Score and Rank columns for low-score rows
            for j in range(len(all_regions), len(all_regions) + 2):
                hypo_table[(i+1, j)].set_text_props(color='#808080')
    
    # Style headers and row labels
    for (row, col), cell in hypo_table.get_celld().items():
        if row == 0:  # Headers
            cell.set_text_props(weight='bold')
        elif col == -1 and row > 0:  # Row labels
            # Get score from DataFrame
            result_row = metric_results.iloc[row-1]
            score = result_row['final_score']
            is_low_score = score < 0.5
            
            if is_low_score:
                cell.set_text_props(weight='normal', color='#808080')
            else:
                cell.set_text_props(weight='bold')
    
    ax2.text(0.5, 0.98, "Hypothesis Values Across Regions (Ranked by Score)", 
             fontsize=FONTS['title']['size'], ha='center', va='bottom',
             transform=ax2.transAxes, weight='normal')
    
    return fig

def get_compact_view_decision(
    hypo_cols: List[str],
    max_hypos: int = 7,
    long_name_ref: str = "Avg CIs - Pitched/Committed -> Actioned",
    long_name_threshold: int = 2
) -> Tuple[bool, str]:
    """
    Decide whether to use compact view and get the reason why.
    
    Returns:
        Tuple of (needs_compact_view, reason_string)
    """
    long_limit = len(long_name_ref)
    long_count = sum(len(str(h)) > long_limit for h in hypo_cols)
    
    triggers = []
    if len(hypo_cols) > max_hypos:
        triggers.append(f"too many hypotheses ({len(hypo_cols)} > {max_hypos})")
    if long_count >= long_name_threshold:
        triggers.append(f"long hypothesis names ({long_count} hypotheses longer than {long_limit} chars)")
    
    needs_compact = len(triggers) > 0
    reason = " and ".join(triggers) if triggers else ""
    
    return needs_compact, reason

def create_multi_hypothesis_plot(
    df: pd.DataFrame,
    metric_col: str,
    hypo_cols: List[str],
    metric_anomaly_info: Dict[str, Any],
    ordered_hypos: List[Tuple[str, Dict[str, Any]]],
    figsize: Tuple[int, int] = (18, 10),
    include_score_formula: bool = True,
    is_conclusive: bool = True,
    reference_row: str = "Global"
) -> plt.Figure:
    """
    Create a multi-panel figure showing hypothesis analysis with bar charts.
    Automatically includes score formula unless disabled.
    
    Args:
        df: DataFrame containing the data
        metric_col: Name of the metric column
        hypo_cols: List of hypothesis column names
        metric_anomaly_info: Anomaly information for the metric
        ordered_hypos: Ordered list of hypotheses by score
        figsize: Figure size as (width, height)
        include_score_formula: Whether to add score formula to the figure
        is_conclusive: Whether the analysis is conclusive
        
    Returns:
        matplotlib Figure object
    """
    # Check if we need compact view and get the reason
    needs_compact, compact_reason = get_compact_view_decision(hypo_cols)
    if needs_compact:
        print(f"Using compact table view due to {compact_reason} - this avoids text overlap and improves readability")
        # Create mini results DataFrame for this metric (temporary solution)
        mini_results = []
        all_regions = get_all_regions_ordered(df, reference_row)
        
        for rank, (hypo_name, hypo_result) in enumerate(ordered_hypos, 1):
            # Apply wrapping to both metric and hypothesis names for display
            wrapped_metric = textwrap.fill(metric_col.replace('_', ' '), width=25)
            wrapped_hypothesis = textwrap.fill(hypo_name.replace('_', ' '), width=25)
            
            mini_results.append({
                'metric': metric_col,  # Original for filtering/building
                'wrapped_metric': wrapped_metric,  # Wrapped for display
                'hypothesis': wrapped_hypothesis,  # Wrapped for display
                'rank': rank,
                'score': format_score(hypo_result['final_score']),  # Formatted score
                'final_score': hypo_result['final_score'],  # Raw score for styling logic
                'background_colors': hypo_result['background_colors'],
                'text_colors': hypo_result['text_colors'],
                'metric_colors': hypo_result.get('metric_colors', {}),  # Add metric colors directly
                'formatted_regional_values': {
                    region: format_number_for_display(df.loc[region, hypo_name], hypo_name) 
                    for region in all_regions
                }
            })
        mini_df = pd.DataFrame(mini_results)
        
        fig = create_hypothesis_delta_table(
            df=df,
            results_df=mini_df,
            metric_col=metric_col,
            fontsize=10,
            reference_row=reference_row
        )
        return fig
    
    # Ensure reference row exists; if not, fall back to the first row to avoid KeyError
    if reference_row not in df.index and len(df.index) > 0:
        reference_row = df.index[0]

    # Create an empty figure
    fig = plt.figure(figsize=figsize)
    
    # Define figure layout with reserved space for top and bottom
    top_margin = 0.80  # Reserve top 20% for explanatory text
    bottom_margin = 0.15  # Reserve bottom 15% for formula
    
    # Determine which hypotheses to plot on the right
    if is_conclusive and ordered_hypos:
        best_hypo_name, best_hypo_result = ordered_hypos[0]
        hypos_on_right = ordered_hypos[1:]
    else:
        # For inconclusive, all hypos go on the right, none are "best"
        best_hypo_name, best_hypo_result = None, None
        hypos_on_right = ordered_hypos
        
    # Calculate grid dimensions based on number of hypotheses to show on the right
    num_hypos_on_right = len(hypos_on_right)
    
    # Determine grid dimensions
    if num_hypos_on_right <= 4:
        total_cols = 3  # 1 for metric/best + 2 for others
        total_rows = 2  # Always 2 rows
    else:
        other_cols = min(3, math.ceil(math.sqrt(num_hypos_on_right)))
        other_rows = math.ceil(num_hypos_on_right / other_cols)
        total_cols = other_cols + 1  # +1 for metric/best column
        total_rows = max(2, other_rows)
    
    # Create GridSpec with explicit positioning
    gs = gridspec.GridSpec(
        total_rows, total_cols,
        left=0.05, right=0.95,
        top=top_margin, bottom=bottom_margin,
        wspace=0.3, hspace=0.3
    )
    
    # Create axes for metric (top left) and best hypothesis (bottom left)
    ax_metric = fig.add_subplot(gs[0, 0])
    ax_best_hypo = fig.add_subplot(gs[1, 0])
    
    # Plot metric
    plot_bars(ax_metric, df[[metric_col]], metric_anomaly_info, plot_type='metric', reference_row=reference_row)
    ax_metric.set_title(f"Metric: {metric_col}", fontsize=FONTS['title']['size'])
    
    # Handle the "Best Hypothesis" plot area
    if is_conclusive and best_hypo_name is not None and best_hypo_result is not None:
        plot_bars(
            ax_best_hypo, 
            df[[metric_col, best_hypo_name]], 
            metric_anomaly_info, 
            best_hypo_result, 
            plot_type='hypothesis',
            highlight_region=True,
            reference_row=reference_row
        )
        
        # Check if metric name + "Best Hypothesis" is too long (more than ~25 characters)
        full_title = f"Best Hypothesis: {best_hypo_name}"
        if len(full_title) > 25:
            title = "Best Hypothesis"
        else:
            title = full_title
        
        ax_best_hypo.set_title(title, fontsize=FONTS['title']['size'])
    else:
        # For inconclusive cases, leave the "best hypothesis" area blank
        ax_best_hypo.set_visible(False)
        
    # Plot the other hypotheses on the right
    for i, (hypo_name, hypo_result) in enumerate(hypos_on_right):
        if i >= num_hypos_on_right:
            break  # Safety check
            
        # Calculate position in grid (skipping the first column)
        if num_hypos_on_right <= 4:
            row = i // 2
            col = (i % 2) + 1
        else:
            row = i // (total_cols - 1)
            col = (i % (total_cols - 1)) + 1
        
        # Create axis and plot
        if row < total_rows and col < total_cols:  # Safety check
            ax = fig.add_subplot(gs[row, col])
            plot_bars(
                ax, 
                df[[metric_col, hypo_name]], 
                metric_anomaly_info, 
                hypo_result, 
                plot_type='hypothesis',
                highlight_region=False,  # No highlight on the right side
                reference_row=reference_row
            )
            
            # Adjust title based on context
            title_prefix = f"Hypothesis {i+2}" if is_conclusive else f"Hypothesis {i+1}"
            
            if len(hypo_cols) > 4:
                ax.set_title(title_prefix, fontsize=FONTS['title']['size'])
            else:
                ax.set_title(f"{title_prefix}: {hypo_name}", fontsize=FONTS['title']['size'])
    
    # Add score formula if included (but not for compact view)
    needs_compact, _ = get_compact_view_decision(hypo_cols)
    if include_score_formula and not needs_compact:
        add_score_formula(fig)
    
    return fig


def score_all_hypotheses(
    df: pd.DataFrame,
    metric_col: str,
    hypo_cols: List[str],
    metric_anomaly_info: Dict[str, Any],
    expected_directions: Dict[str, str],
    reference_row: str = "Global"
) -> Dict[str, Dict[str, Any]]:
    """
    Score all hypotheses for a given metric using sign-based scoring.
    
    Args:
        df: DataFrame containing metric and hypothesis columns
        metric_col: Name of the metric column
        hypo_cols: List of hypothesis column names
        metric_anomaly_info: Info about the metric anomaly
        expected_directions: Dictionary mapping hypothesis names to their expected directions
        reference_row: Name of the reference row to use (default: "Global")
    
    Returns:
        Dictionary mapping hypothesis names to their score results
    """
    # Always use sign-based scoring
    score_func = sign_based_score_hypothesis
    
    # Score all hypotheses
    hypo_results = {}
    for hypo_col in hypo_cols:
        # Create a temporary dataframe with just the metric and this hypothesis
        temp_df = df[[metric_col, hypo_col]]
        
        # Get expected direction
        expected_direction = expected_directions.get(hypo_col, 'same')
        
        # Score the hypothesis
        hypo_results[hypo_col] = score_func(temp_df, metric_anomaly_info, expected_direction, reference_row)
    
    return hypo_results


def process_metrics(
    df: pd.DataFrame,
    metric_cols: List[str],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    expected_directions: Dict[str, str],
    metric_hypothesis_map: Dict[str, List[str]],
    reference_row: str = "Global"
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Process multiple metrics and their hypotheses, storing results for each.
    
    Args:
        df: DataFrame containing all metrics and hypotheses
        metric_cols: List of metric column names to process
        metric_anomaly_map: Dictionary mapping metric names to their anomaly info
        expected_directions: Dictionary mapping hypothesis names to their expected directions
        metric_hypothesis_map: Mapping from metrics to their relevant hypotheses
    
    Returns:
        Dictionary mapping metric names to dictionaries of hypothesis results
    """
    # Dictionary to store all results
    all_results = {}
    
    # Process each metric
    for metric_col in metric_cols:
        # Skip if no anomaly info for this metric
        if metric_col not in metric_anomaly_map:
            continue
        
        # Get anomaly info for this metric
        metric_anomaly_info = metric_anomaly_map[metric_col]
        
        # Skip metrics not in the mapping
        if metric_col not in metric_hypothesis_map or not metric_hypothesis_map[metric_col]:
            continue
        
        # Use metric-specific hypotheses from the map
        current_hypo_cols = metric_hypothesis_map[metric_col]
        
        # Score all hypotheses for this metric
        hypo_results = score_all_hypotheses(
            df, 
            metric_col, 
            current_hypo_cols, 
            metric_anomaly_info, 
            expected_directions,
            reference_row
        )
        
        # Store results for this metric
        all_results[metric_col] = hypo_results
    
    return all_results

def plot_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_anomaly_info: Dict[str, Any],
    expected_direction: str = 'same',
    reference_row: str = "Global"
) -> plt.Axes:
    """
    Create a scatter plot showing the relationship between metric and hypothesis values.
    
    Args:
        ax: Matplotlib axes to plot on
        df: DataFrame with metric and hypothesis values indexed by region
        metric_anomaly_info: Info about the metric anomaly
        expected_direction: Expected relationship between metric and hypothesis ('same' or 'opposite')
        
    Returns:
        The modified axes with the scatter plot
    """
    # Clear existing content
    ax.clear()
    
    # Extract metric and hypothesis columns
    metric_col, hypo_col = df.columns
    
    # Since we now use display names as column headers, these are already display names
    metric_display_name = metric_col
    hypo_display_name = hypo_col
    
    # Get all regions (including reference row)
    regions = df.index.tolist()
    
    # Get anomalous region info
    anomalous_region = metric_anomaly_info['anomalous_region']
    direction = metric_anomaly_info['direction']
    higher_is_better = metric_anomaly_info.get('higher_is_better', True)
    
    # Determine if the anomaly is good or bad
    is_good_anomaly = (direction == 'higher' and higher_is_better) or (direction == 'lower' and not higher_is_better)
    
    # Create scatter plot point colors
    colors = []
    sizes = []
    for region in regions:
        if region == anomalous_region:
            # Use green/red for anomalous region based on whether it's a good or bad anomaly
            colors.append(COLORS['metric_positive'] if is_good_anomaly else COLORS['metric_negative'])
            sizes.append(100)  # Make anomalous region point larger
        elif region == reference_row:
            colors.append(COLORS['global_line'])
            sizes.append(100)  # Make reference row point larger
        else:
            colors.append(COLORS['default_bar'])
            sizes.append(70)   # Regular size for other regions
    
    # Create scatter plot
    x = df[metric_col].values
    y = df[hypo_col].values
    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidths=1)
    
    # Annotate each point with region name, but only when data point < 7
    for i, region in enumerate(regions):
        # Only annotate if either x or y value is less than 7
        if x[i] < 7 or y[i] < 7:
            # Position the text with slight offset to avoid overlap with the point
            offset_x = 0.0
            offset_y = 0.002 * (max(y) - min(y))
            
            # Add region name as annotation
            ax.annotate(
                region,
                (x[i], y[i]),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=FONTS['tick_label']['size'],
                ha='center', 
                va='bottom'
            )
    
    # Draw trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x), max(x), 100)
    
    # Determine if the trendline aligns with expected direction
    slope = z[0]
    direction_match = (slope > 0 and expected_direction == 'same') or (slope < 0 and expected_direction == 'opposite')
    trendline_color = COLORS['metric_positive'] if direction_match else COLORS['metric_negative']
    
    # Plot the trendline
    ax.plot(x_line, p(x_line), '--', color=trendline_color, linewidth=1.5)
    
    # Calculate correlation
    corr = df[metric_col].corr(df[hypo_col])
    
    # Add correlation annotation
    ax.text(
        0.05, 0.95, 
        f"Correlation: {corr:.2f}\nExpected Direction: {expected_direction}",
        transform=ax.transAxes,
        fontsize=FONTS['annotation']['size'],
        verticalalignment='top',
        bbox={'boxstyle': 'round,pad=0.5', 'facecolor': 'white', 'alpha': 0.7}
    )
    
    # Set axis labels using display names
    ax.set_xlabel(metric_display_name, fontsize=FONTS['axis_label']['size'])
    ax.set_ylabel(hypo_display_name, fontsize=FONTS['axis_label']['size'])
    
    # Format axes as percentages if needed
    metric_is_percent = ('pct' in metric_col.lower() or '%' in metric_col or 'rate' in metric_col.lower())
    hypo_is_percent = ('pct' in hypo_col.lower() or '%' in hypo_col or 'rate' in hypo_col.lower())
    
    if metric_is_percent:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    
    if hypo_is_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax


def add_score_formula(fig: plt.Figure) -> None:
    """
    Add sign-based score formula at the bottom of the figure.
    
    Args:
        fig: The figure to add the formula to
    """
    # Sign-based scoring components (60% sign agreement + 40% explained ratio)
    components = [
        ('Score', None, COLORS['score_color']),
        ('Sign Agreement', 0.6, COLORS['sign_score_components']['sign_agreement']),
        ('Explained Ratio', 0.4, COLORS['sign_score_components']['explained_ratio'])
    ]
    
    # Calculate spacing - position in the reserved bottom area
    formula_y = 0.07  # Position at 7% from bottom (middle of the bottom 15%)
    card_width = 0.11
    card_height = 0.025
    spacing = 0.01
    
    # Calculate total width
    total_width = 0
    for i, (name, weight, _) in enumerate(components):
        if i == 0:  # Score
            total_width += card_width + spacing + 0.02  # width + spacing + "="
        else:
            if i > 1:  # Add "+" for components after the first one
                total_width += 0.02 + spacing
            total_width += 0.03 + card_width + spacing  # "0.Nx" + width + spacing
    
    # Center the formula
    start_x = 0.5 - (total_width / 2)
    curr_x = start_x
    
    # Add the components
    for i, (name, weight, color) in enumerate(components):
        if i == 0:  # Score component
            # Add "Score" card
            rect = plt.Rectangle(
                (curr_x, formula_y), card_width, card_height,
                facecolor=color, alpha=0.2, transform=fig.transFigure
            )
            fig.add_artist(rect)
            
            fig.text(
                curr_x + card_width/2, formula_y + card_height/2,
                "Score", ha='center', va='center',
                fontweight='bold', color=color, fontsize=FONTS['score_card']['size'],
                transform=fig.transFigure
            )
            
            curr_x += card_width + spacing
            
            # Add "=" sign
            fig.text(
                curr_x, formula_y + card_height/2,
                "=", ha='center', va='center',
                color=color, fontsize=FONTS['score_card']['size'],
                transform=fig.transFigure
            )
            
            curr_x += 0.02 + spacing
            
        else:  # Other components
            # Add "+" for components after the first
            if i > 1:
                fig.text(
                    curr_x, formula_y + card_height/2,
                    "+", ha='center', va='center',
                    color=color, fontsize=FONTS['score_card']['size'],
                    transform=fig.transFigure
                )
                curr_x += 0.02 + spacing
            
            # Add weight
            fig.text(
                curr_x, formula_y + card_height/2,
                f"{weight:.1f}×", ha='center', va='center',
                color=color, fontsize=FONTS['score_card']['size'],
                transform=fig.transFigure
            )
            curr_x += 0.03
            
            # Add component card
            rect = plt.Rectangle(
                (curr_x, formula_y), card_width, card_height,
                facecolor=color, alpha=0.2, transform=fig.transFigure
            )
            fig.add_artist(rect)
            
            fig.text(
                curr_x + card_width/2, formula_y + card_height/2,
                name, ha='center', va='center',
                fontweight='bold', color=color, fontsize=FONTS['score_card']['size'],
                transform=fig.transFigure
            )
            
            curr_x += card_width + spacing


def render_template_text(
    template: str, 
    metric_anomaly_info: Dict[str, Any],
    metric_col: str,
    best_hypo_name: Optional[str] = None,
    best_hypo_result: Optional[Dict[str, Any]] = None,
    reference_row: str = "Global"
) -> str:
    """
    Render a Jinja2 template with values from the results.
    
    Args:
        template: Text template with placeholders in {{variable}} using Jinja2 syntax
        metric_anomaly_info: Metric anomaly information
        metric_col: Name of the metric column
        best_hypo_name: Name of the best hypothesis
        best_hypo_result: Results from the best hypothesis
    
    Returns:
        Rendered text string
    """
    context = create_template_params(
        metric_anomaly_info=metric_anomaly_info,
        metric_name=metric_col,
        hypo_name=best_hypo_name,
        hypo_result=best_hypo_result,
        reference_row=reference_row
    )
    
    # Fill template with values using Jinja2 Template
    try:
        # Create Jinja2 Template object
        jinja_template = Template(template)
        # Render template with values
        filled_text = jinja_template.render(**context)
        return filled_text
    except Exception as e:
        return f"Error filling template: {str(e)}"


def add_template_text(
    fig: plt.Figure, 
    template: str, 
    metric_anomaly_info: Dict[str, Any],
    metric_col: str,
    best_hypo_name: Optional[str] = None,
    best_hypo_result: Optional[Dict[str, Any]] = None, 
    position: Tuple[float, float] = (0.05, 0.85),  # Moved down from 0.97 to 0.85 to reduce spacing
    max_width: float = 0.9,
    fontsize: int = 12
) -> None:
    """
    Add text to the figure using a Jinja2 template filled with values from the results.
    
    Args:
        fig: The figure to add the text to
        template: Text template with placeholders in {{variable}} using Jinja2 syntax
        metric_anomaly_info: Metric anomaly information
        metric_col: Name of the metric column
        best_hypo_name: Name of the best hypothesis
        best_hypo_result: Results from the best hypothesis
        position: (x, y) coordinates for the text (figure coordinates)
        max_width: Maximum width for the text as fraction of figure width
        fontsize: Font size for the text
    """
    # Render the template text
    filled_text = render_template_text(
        template=template,
        metric_anomaly_info=metric_anomaly_info,
        metric_col=metric_col,
        best_hypo_name=best_hypo_name,
        best_hypo_result=best_hypo_result
    )
    
    # Add text to figure
    fig.text(
        position[0], position[1], 
        filled_text, 
        ha='left', va='top', 
        fontsize=fontsize,
        wrap=True,
        transform=fig.transFigure,
        bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'round,pad=0.5'}
    )


def create_consolidated_results_dataframe(
    regional_df: pd.DataFrame,
    anomaly_map: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    reference_row: str = "Global"
) -> pd.DataFrame:
    """
    Create a consolidated DataFrame with all hypothesis scoring results.
    
    This is the single source of truth for all scoring results with proper
    ranking, conclusiveness determination, and template information.
    
    Returns:
        DataFrame with columns: metric, hypothesis, final_score, explains, 
        conclusive, rank, best_hypothesis, template, summary_template, etc.
    """
    metrics_config = config.get('metrics', {})
    
    # Extract metric columns and create mappings
    metric_cols = []
    metric_hypothesis_map = {}
    expected_directions = {}
    
    for metric_name, metric_config in metrics_config.items():
        if 'hypotheses' not in metric_config:
            continue
        if metric_name not in anomaly_map:
            continue
            
        metric_cols.append(metric_name)
        hypothesis_names = list(metric_config['hypotheses'].keys())
        metric_hypothesis_map[metric_name] = hypothesis_names
        
        # Get expected directions
        for hypo_name, hypo_config in metric_config['hypotheses'].items():
            expected_directions[hypo_name] = hypo_config.get('expected_direction', 'same')
    
    # Use existing process_metrics function
    all_results = process_metrics(
        df=regional_df,
        metric_cols=metric_cols,
        metric_anomaly_map=anomaly_map,
        expected_directions=expected_directions,
        metric_hypothesis_map=metric_hypothesis_map,
        reference_row=reference_row
    )
    
    # Create basic results DataFrame 
    records = []
    for metric_name, hypo_results in all_results.items():
        for hypo_name, result in hypo_results.items():
            scores = result['scores']
            record = {
                'metric': metric_name,
                'hypothesis': hypo_name,
                'final_score': scores['final_score'],
                'explains': scores['explains'],
                'failure_reason': scores['failure_reason'],
                'magnitude': result['magnitude'],
                'direction': result['direction'],
                'ref_hypo_val': result['ref_hypo_val'],
                'hypo_val': result['hypo_val'],
                'sign_agreement': scores['sign_agreement'],
                'explained_ratio': scores['explained_ratio'],
                'p_value': scores['p_value'],
                # Keep tracebacks attached for downstream access
                'traceback': result.get('traceback', {})
            }
            records.append(record)
    
    results_df = pd.DataFrame(records)
    
    # Add ranking and enhanced information directly
    enhanced_records = []
    for metric_name in metric_cols:
        if metric_name not in all_results:
            continue
            
        hypo_results = all_results[metric_name]
        metric_config = metrics_config[metric_name]
        
        # Sort hypotheses by final score (descending)
        sorted_hypos = sorted(hypo_results.items(), key=lambda x: x[1]['scores']['final_score'], reverse=True)
        
        # Determine conclusiveness (any hypothesis meets guardrails)
        qualified_hypos = [h for h, r in sorted_hypos if r['scores']['explains']]
        is_conclusive = len(qualified_hypos) > 0
        
        for rank, (hypo_name, hypo_result) in enumerate(sorted_hypos, 1):
            # Find the base record in results_df
            base_record = results_df[
                (results_df['metric'] == metric_name) & 
                (results_df['hypothesis'] == hypo_name)
            ].iloc[0].to_dict()
            
            # Add enhanced information and formatted columns
            hypo_config = metric_config['hypotheses'][hypo_name]
            base_record.update({
                'rank': rank,
                'conclusive': is_conclusive,
                'best_hypothesis': is_conclusive and rank == 1,
                'template': hypo_config.get('template', ''),
                'summary_template': hypo_config.get('summary_template', ''),
                'technical_name': hypo_config.get('technical_name', hypo_name),
                'expected_direction': hypo_config.get('expected_direction', 'same'),
                'background_colors': hypo_result.get('colors', {}).get('background_colors', {}),
                'text_colors': hypo_result.get('colors', {}).get('text_colors', {}),
                'metric_colors': hypo_result.get('colors', {}).get('metric_colors', {}),
                # Add formatted display columns
                'formatted_score': format_score(base_record['final_score']),
                'formatted_rank': str(rank),
                'formatted_hypo_val': format_number_for_display(base_record['hypo_val'], hypo_name),
                'formatted_ref_hypo_val': format_number_for_display(base_record['ref_hypo_val'], hypo_name),
                # Add formatted regional values
                'formatted_regional_values': {
                    region: format_number_for_display(regional_df.loc[region, hypo_name], hypo_name) 
                    for region in get_all_regions_ordered(regional_df, reference_row)
                },
                # Expose traceback tables at row level for convenient access
                'formulas_walkthrough': base_record.get('traceback', {}).get('formulas_walkthrough'),
                'formulas_summary': base_record.get('traceback', {}).get('formulas_summary'),
            })
            
            enhanced_records.append(base_record)
    
    # Create enhanced DataFrame
    enhanced_df = pd.DataFrame(enhanced_records)
    
    # Sort by metric, then by rank
    enhanced_df = enhanced_df.sort_values(['metric', 'rank']).reset_index(drop=True)
    
    return enhanced_df


def score_hypotheses_for_metrics(
    regional_df: pd.DataFrame,
    anomaly_map: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    reference_row: str = "Global"
) -> Dict[str, Any]:
    """
    Score hypotheses for multiple metrics and return results in unified format directly.
    
    Args:
        regional_df: DataFrame with regional data, indexed by region
        anomaly_map: Dictionary mapping metric names to anomaly information
        config: Full configuration dictionary
        region_col: Name of the region column
    
    Returns:
        Dictionary in unified format: {metric_name: {'slides': {'Directional': slide_data}}}
    """
    # Create consolidated results DataFrame - single source of truth
    results_df = create_consolidated_results_dataframe(
        regional_df=regional_df,
        anomaly_map=anomaly_map,
        config=config,
        reference_row=reference_row
    )
    
    # Convert to the unified format expected by downstream systems
    unified_results = {}
    
    # Process each metric using the consolidated DataFrame
    for metric_name in results_df['metric'].unique():
        metric_results = results_df[results_df['metric'] == metric_name]
        anomaly_info = anomaly_map[metric_name]
        
        # Get best hypothesis info
        best_hypo_row = metric_results[metric_results['best_hypothesis']]
        is_conclusive = len(best_hypo_row) > 0
        
        if is_conclusive:
            best_hypo_data = best_hypo_row.iloc[0]
            best_hypo_dict = best_hypo_data.to_dict()  # Convert once
            template = best_hypo_data['template']
            
            # Generate main template text 
            main_template_text = render_template_text(
                template=template,
                metric_anomaly_info=anomaly_info,
                metric_col=metric_name,
                best_hypo_name=best_hypo_data['hypothesis'],
                best_hypo_result=best_hypo_dict,
                reference_row=reference_row
            )
            
            # Extract summary text: only the part after "Root cause:" from template
            if "Root cause:" in main_template_text:
                summary_text = main_template_text.split("Root cause:", 1)[1].strip()
            else: # This shouldn't be triggered if the yaml file is set up correctly but just in case
                summary_text = f"{best_hypo_data['hypothesis']} is {best_hypo_data['direction']}"
            
            # Generate additional reasons from other high-scoring hypotheses
            additional_reasons = []
            other_good_hypos = metric_results[
                (metric_results['final_score'] > 0.5) & 
                (~metric_results['best_hypothesis'])
            ]
            
            if len(other_good_hypos) > 0:
                for idx, (_, row) in enumerate(other_good_hypos.iterrows(), 2):
                    if row['summary_template'] and row['summary_template'] != 'TBA':
                        additional_text = render_template_text(
                            template=row['summary_template'],
                            metric_anomaly_info=anomaly_info,
                            metric_col=metric_name,
                            best_hypo_name=row['hypothesis'],
                            best_hypo_result=row.to_dict()
                        )
                        additional_reasons.append(f"#{idx}: {additional_text}")
            
            # Create filled_text: complete template + additional reasons
            if additional_reasons:
                filled_text = main_template_text + "\n\nOther possible reasons are: " + ", ".join(additional_reasons)
            else:
                filled_text = main_template_text
            template_params = create_template_params(
                metric_anomaly_info=anomaly_info,
                metric_name=metric_name,
                hypo_name=best_hypo_data['hypothesis'],
                hypo_result=best_hypo_dict
            )
        else:
            template = "Analysis of {{metric_name}} in {{region}} shows {{metric_deviation}} {{metric_dir}} performance than {{reference_row}}. However, none of the provided hypotheses meet the minimum evidence thresholds for a confident root cause determination."
            filled_text = render_template_text(template, anomaly_info, metric_name, reference_row=reference_row)
            summary_text = "Inconclusive - no hypothesis meets minimum evidence thresholds."
            template_params = create_template_params(metric_anomaly_info=anomaly_info, metric_name=metric_name, reference_row=reference_row)
        
        # Create figure generators using DataFrame data
        hypothesis_names = metric_results['hypothesis'].tolist()
        ordered_hypos = [(row['hypothesis'], row.to_dict()) for _, row in metric_results.iterrows()]
        
        figure_generators = [{
            "function": create_multi_hypothesis_plot,
            "title_suffix": "",
            "params": {
                'df': regional_df,
                'metric_col': metric_name,
                'hypo_cols': hypothesis_names,
                'metric_anomaly_info': anomaly_info,
                'ordered_hypos': ordered_hypos,
                'is_conclusive': is_conclusive,
                'reference_row': reference_row
            }
        }]
        
        # Store in unified format
        unified_results[metric_name] = {
            'slides': {
                'Business': {
                    'summary': {'summary_text': summary_text},
                    'slide_info': {
                        'title': f"{metric_name} - Business Ops. Difference",
                        'template_text': filled_text,
                        'template_params': template_params,
                        'figure_generators': figure_generators,
                        'dfs': {},
                        'layout_type': "text_figure",
                        'total_hypotheses': len(hypothesis_names)
                    }
                }
            },
            'payload': {
                'best_hypothesis': best_hypo_data.to_dict() if is_conclusive else None,
                'all_results': metric_results.to_dict('records'),
                'total_hypotheses': len(hypothesis_names),
                # Per-hypothesis traceback tables for detailed inspection
                'tracebacks': {
                    row['hypothesis']: {
                        'formulas_walkthrough': row.get('formulas_walkthrough'),
                        'formulas_summary': row.get('formulas_summary')
                    }
                    for _, row in metric_results.iterrows()
                }
            }
        }
    
    return unified_results


def main(save_path: str = '.', results_path: Optional[str] = None, reference_row: str = "Global"):
    """
    Test the hypothesis scoring and visualization with multiple hypotheses.
    
    Args:
        save_path: Directory path to save generated figures
        results_path: Path to save DataFrame results (if None, results are not saved)
    """
    # Create test data
    np.random.seed(42)
    regions = [reference_row, "North America", "Europe", "Asia", "Latin America"]
    
    # Create test data with multiple metrics and hypotheses (18 hypotheses to test compact view)
    data = {
        # Metrics
        'conversion_rate_pct': np.array([0.12, 0.08, 0.11, 0.13, 0.10]),
        'avg_order_value': np.array([75.0, 65.0, 80.0, 85.0, 72.0]),
        'customer_satisfaction': np.array([4.2, 3.8, 4.3, 4.5, 4.0]),
        
        # Hypotheses (18 total to trigger compact view)
        'bounce_rate_pct': np.array([0.35, 0.45, 0.32, 0.28, 0.34]),
        'page_load_time': np.array([2.4, 3.8, 2.2, 1.9, 2.5]),
        'session_duration': np.array([180, 120, 190, 210, 175]),
        'pages_per_session': np.array([4.2, 3.1, 4.5, 4.8, 4.0]),
        'new_users_pct': np.array([0.25, 0.18, 0.28, 0.30, 0.23]),
        'cart_abandonment_rate': np.array([0.70, 0.85, 0.65, 0.60, 0.72]),
        'mobile_traffic_pct': np.array([0.60, 0.45, 0.65, 0.70, 0.58]),
        'search_usage_rate': np.array([0.40, 0.25, 0.45, 0.50, 0.38]),
        'email_open_rate': np.array([0.22, 0.15, 0.25, 0.28, 0.20]),
        'social_media_traffic': np.array([0.15, 0.08, 0.18, 0.20, 0.12]),
        'product_reviews_count': np.array([150, 80, 170, 200, 140]),
        'customer_service_calls': np.array([25, 45, 20, 15, 28]),
        'return_rate_pct': np.array([0.08, 0.15, 0.06, 0.05, 0.09]),
        'inventory_availability': np.array([0.95, 0.85, 0.97, 0.98, 0.93]),
        'shipping_speed_days': np.array([2.5, 4.2, 2.0, 1.8, 2.8]),
        'promotional_discount_pct': np.array([0.10, 0.05, 0.12, 0.15, 0.08]),
        'website_uptime_pct': np.array([0.999, 0.995, 0.9995, 0.9998, 0.998]),
        'payment_failure_rate': np.array([0.02, 0.08, 0.015, 0.01, 0.025]),
        'recommendation_ctr': np.array([0.12, 0.06, 0.15, 0.18, 0.10])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data, index=regions)
    
    # Define metric columns and hypothesis columns
    metric_cols = ['conversion_rate_pct', 'avg_order_value', 'customer_satisfaction']
    hypo_cols = ['bounce_rate_pct', 'page_load_time', 'session_duration', 'pages_per_session', 'new_users_pct',
                 'cart_abandonment_rate', 'mobile_traffic_pct', 'search_usage_rate', 'email_open_rate', 
                 'social_media_traffic', 'product_reviews_count', 'customer_service_calls', 'return_rate_pct',
                 'inventory_availability', 'shipping_speed_days', 'promotional_discount_pct', 'website_uptime_pct',
                 'payment_failure_rate', 'recommendation_ctr']
    
    # Import anomaly detector
    from rca_package.anomaly_detector import detect_snapshot_anomaly_for_column
    
    # Create metric anomaly map using the anomaly detector
    metric_anomaly_map = {}
    for metric_col in metric_cols:
        anomaly_info = detect_snapshot_anomaly_for_column(df, reference_row, column=metric_col)
        if anomaly_info:
            metric_anomaly_map[metric_col] = anomaly_info
    
    # Define expected directions for each hypothesis
    expected_directions = {
        'bounce_rate_pct': 'opposite',          # Higher bounce rate -> lower conversion
        'page_load_time': 'opposite',           # Higher load time -> lower conversion
        'session_duration': 'same',             # Higher session time -> higher conversion
        'pages_per_session': 'same',            # More pages viewed -> higher conversion
        'new_users_pct': 'opposite',            # New users tend to convert less
        'cart_abandonment_rate': 'opposite',    # Higher abandonment -> lower conversion
        'mobile_traffic_pct': 'same',           # Mobile traffic can convert well
        'search_usage_rate': 'same',            # Search users convert better
        'email_open_rate': 'same',              # Higher engagement -> better conversion
        'social_media_traffic': 'same',         # Social engagement -> conversion
        'product_reviews_count': 'same',        # More reviews -> trust -> conversion
        'customer_service_calls': 'opposite',   # More calls -> problems -> lower conversion
        'return_rate_pct': 'opposite',          # High returns -> lower satisfaction -> conversion
        'inventory_availability': 'same',       # Better stock -> higher conversion
        'shipping_speed_days': 'opposite',      # Faster shipping -> higher conversion
        'promotional_discount_pct': 'same',     # More discounts -> higher conversion
        'website_uptime_pct': 'same',           # Better uptime -> higher conversion
        'payment_failure_rate': 'opposite',     # Payment issues -> lower conversion
        'recommendation_ctr': 'same'            # Better recommendations -> higher conversion
    }
    
    # 1. Demonstrate scatter plot visualization
    print("\n===== SCATTER PLOT VISUALIZATION =====")
    metric_col = 'conversion_rate_pct'
    hypo_col = 'bounce_rate_pct'
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_scatter(
        ax=ax,
        df=df[[metric_col, hypo_col]],
        metric_anomaly_info=metric_anomaly_map[metric_col],
        expected_direction=expected_directions[hypo_col]
    )
    plt.tight_layout()
    
    # Save with descriptive filename
    filename = f"{save_path}/scatter_single_{metric_col}_{hypo_col}.png"
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Created single scatter plot: {filename}")
    
    # 2. Demonstrate scatter grid visualization for all hypotheses
    print("\n===== SCATTER GRID VISUALIZATION =====")
    scatter_grid = create_scatter_grid(
        df=df,
        metric_col=metric_col,
        hypo_cols=hypo_cols,
        metric_anomaly_info=metric_anomaly_map[metric_col],
        expected_directions=expected_directions
    )
    
    # Save with descriptive filename
    filename = f"{save_path}/scatter_grid_{metric_col}.png"
    scatter_grid.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(scatter_grid)
    print(f"Created scatter grid: {filename}")
    
    # 3. Demonstrate processing of multiple metrics
    print("\n===== MULTIPLE METRICS PROCESSING =====")

    
    # Create simple config for testing
    config = {
        'metrics': {
            'conversion_rate_pct': {'hypotheses': {h: {'expected_direction': expected_directions[h]} for h in hypo_cols}},
            'avg_order_value': {'hypotheses': {h: {'expected_direction': expected_directions[h]} for h in hypo_cols}},
            'customer_satisfaction': {'hypotheses': {h: {'expected_direction': expected_directions[h]} for h in hypo_cols}}
        }
    }
    
    # Save to DataFrame for analysis
    df_results = create_consolidated_results_dataframe(df, metric_anomaly_map, config)
    print("\nResults DataFrame Preview:")
    print(df_results[['metric', 'hypothesis', 'final_score', 'magnitude', 'direction', 'explains']].head(10))
    
    # Save results to file if path is provided
    if results_path is not None:
        df_results.to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")
    
    # Display top hypothesis for each metric using DataFrame
    print("\nTop Hypothesis for Each Metric:")
    for metric_col in metric_cols:
        metric_data = df_results[df_results['metric'] == metric_col]
        if len(metric_data) > 0:
            best_hypo = metric_data.iloc[0]  # Already sorted by rank
            if best_hypo['explains']:
                print(f"  {metric_col}: {best_hypo['hypothesis']} (Score: {format_score(best_hypo['final_score'])})")
            else:
                print(f"  {metric_col}: Inconclusive - no hypothesis meets guardrails")
    
    # 4. Create visualization using DataFrame results
    print("\n===== VISUALIZATION WITH PRE-SORTED RESULTS =====")
    metric_col = 'conversion_rate_pct'
    metric_data = df_results[df_results['metric'] == metric_col]
    if len(metric_data) == 0:
        print(f"No data found for {metric_col}")
        return
    
    conclusive_hypos = metric_data[metric_data['explains'] == True]
    is_conclusive_main = len(conclusive_hypos) > 0
    
    if not is_conclusive_main:
        # Handle inconclusive case
        print(f"Cannot create visualization for {metric_col}: Inconclusive")
        fig = plt.figure(figsize=(12, 8))
        fig.text(0.5, 0.5, f"{metric_col}\n\nInconclusive - no hypothesis meets guardrails", 
                ha='center', va='center', fontsize=16,
                bbox={'boxstyle': 'round,pad=1', 'facecolor': 'lightgray', 'alpha': 0.7})
        filename = f"{save_path}/inconclusive_{metric_col}.png"
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Created inconclusive visualization: {filename}")
    else:
        # Create visualization with qualified hypotheses
        # Create ordered_hypos from DataFrame
        ordered_hypos = [(row['hypothesis'], row.to_dict()) for _, row in metric_data.iterrows()]
        
        fig = create_multi_hypothesis_plot(
            df=df,
            metric_col=metric_col,
            hypo_cols=hypo_cols,
            metric_anomaly_info=metric_anomaly_map[metric_col],
            ordered_hypos=ordered_hypos,  # Always show all hypotheses
            include_score_formula=True,
            is_conclusive=is_conclusive_main
        )
        
        # Use the template with Jinja2 double-brace syntax
        template = "{{metric_name}} in {{region}} is {{metric_deviation}} {{metric_dir}} than reference mean.\n"\
                   "Root cause: {{region}} has {{hypo_delta}} {{hypo_dir}} of {{hypo_name}} than the reference mean ({{ref_hypo_val}}). "\
                   "This suggests Account Managers have different interaction volumes, potentially impacting their ability to "\
                   "effectively manage and prioritize CLIs in their portfolio."
        
        # Add template text and score formula
        best_hypo_name, best_hypo_result = ordered_hypos[0]
        add_template_text(
            fig, 
            template, 
            metric_anomaly_map[metric_col],
            metric_col,
            best_hypo_name,
            best_hypo_result, 
        )
        add_score_formula(fig)
        
        # Save with descriptive filename
        filename = f"{save_path}/bar_{metric_col}_sign_based.png"
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Created multi-metric visualization: {filename}")
    
    print("\nAll visualizations and analyses completed successfully.")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hypothesis scoring and visualization")
    parser.add_argument('--save_path', type=str, default='.', 
                        help='Directory to save generated figures')
    parser.add_argument('--results_path', type=str, default=None, 
                        help='Path to save DataFrame results (CSV)')
    args = parser.parse_args()
    
    # Run the main function with command line arguments
    main(save_path=args.save_path, results_path=args.results_path) 
