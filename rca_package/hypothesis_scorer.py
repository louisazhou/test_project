from typing import Dict, List, Tuple, Union, Optional, Any, TypedDict, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from jinja2 import Template
import scipy.stats as stats


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

class HypothesisScores(TypedDict):
    sign_agreement: float
    explained_ratio: float
    p_value: float
    final_score: float
    explains: bool
    failure_reason: str


class HypothesisResult(TypedDict):
    hypo_val: float
    direction: str
    ref_hypo_val: float
    magnitude: str
    scores: HypothesisScores


def sign_based_score_hypothesis(
    df: pd.DataFrame,
    metric_anomaly_info: Dict[str, Any],
    expected_direction: str = 'same'
) -> HypothesisResult:
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
    
    # Calculate deltas for anomalous region
    metric_delta = (metric_val - ref_metric_val) / ref_metric_val if ref_metric_val > 0 else 0
    
    # Get hypothesis values for anomalous region
    hypo_val = df.loc[anomalous_region, hypo_col]
    ref_hypo_val = df.loc["Global", hypo_col]
    hypo_delta = (hypo_val - ref_hypo_val) / ref_hypo_val if ref_hypo_val > 0 else 0
    
    # Calculate sign agreement for all regions
    regions = [r for r in df.index if r != "Global"]
    global_metric = df.loc["Global", metric_col]
    global_hypo = df.loc["Global", hypo_col]
    
    # Calculate deltas for all regions compared to global
    metric_deltas = np.array([(df.loc[r, metric_col] - global_metric) / global_metric if global_metric > 0 else 0 for r in regions])
    hypo_deltas = np.array([(df.loc[r, hypo_col] - global_hypo) / global_hypo if global_hypo > 0 else 0 for r in regions])
    
    # Adjust sign for expected direction
    if expected_direction == 'opposite':
        # Flip the sign of hypothesis deltas if expecting opposite direction
        hypo_deltas = -hypo_deltas
    
    # Count how many regions have same sign
    sign_agreements = (np.sign(metric_deltas) == np.sign(hypo_deltas))
    sign_agreement_score = sign_agreements.sum() / len(regions)
    
    # Calculate binomial p-value (not used in score but included for reference)
    p_binom = stats.binomtest(sign_agreements.sum(), n=len(regions), p=0.5)
    
    # Calculate explained ratio for the anomalous region using MAD-normalized z-scores
    # Compute robust spreads using MAD (Median Absolute Deviation)
    sigma_m = np.median(np.abs(metric_deltas - np.median(metric_deltas)))
    sigma_h = np.median(np.abs(hypo_deltas - np.median(hypo_deltas)))
    
    # Convert focal region deltas to z-scores
    z_m = metric_delta / sigma_m if sigma_m > 1e-6 else 0
    z_h = hypo_delta / sigma_h if sigma_h > 1e-6 else 0
    
    # Redefine explained ratio using z-scores (scale-free)
    explained_ratio = min(abs(z_h) / abs(z_m), 1.0) if abs(z_m) > 1e-6 else 0
    
    # Calculate final score: 60% sign agreement + 40% explained ratio
    final_score = 0.6 * sign_agreement_score + 0.4 * explained_ratio
    
    # Apply guardrail logic for 'explains' field
    meets_guardrails = (sign_agreement_score >= 0.5 and 
                       explained_ratio >= 0.2 and 
                       final_score >= 0.5)
    
    # Determine failure reason if it doesn't explain
    failure_reason = ""
    if not meets_guardrails:
        reasons = []
        if sign_agreement_score < 0.5:
            reasons.append("wrong hypothesis direction")
        if explained_ratio < 0.2:
            reasons.append("coincidental moves")
        if final_score < 0.5:
            reasons.append("final score <0.5")
        failure_reason = ", ".join(reasons)
    
    # Format magnitude based on column name (same as original function)
    is_percent_column = '_pct' in hypo_col or '%' in hypo_col
    if is_percent_column:
        magnitude_value = abs(hypo_val - ref_hypo_val) * 100
        magnitude_unit = 'pp'
    else:
        magnitude_value = abs(hypo_delta) * 100
        magnitude_unit = '%'
    
    # Results
    return {
        'hypo_val': hypo_val,
        'direction': 'higher' if hypo_delta > 0 else 'lower',
        'ref_hypo_val': ref_hypo_val,
        'magnitude': f"{magnitude_value:.1f}{magnitude_unit}",
        'scores': {
            'sign_agreement': sign_agreement_score,
            'explained_ratio': explained_ratio,
            'p_value': p_binom.pvalue,
            'final_score': final_score,
            'explains': meets_guardrails,
            'failure_reason': failure_reason
        }
    }


def plot_bars(
    ax: plt.Axes, 
    df: pd.DataFrame, 
    metric_anomaly_info: Dict[str, Any], 
    hypo_result: Optional[HypothesisResult] = None, 
    plot_type: str = 'metric',
    highlight_region: bool = True
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
    
    # Extract regions and values (excluding Global)
    regions = [r for r in df.index.tolist() if r != "Global"]
    values = df.loc[regions, col_to_plot].values
    
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
    
    # Plot bars
    x_positions = np.arange(len(regions))
    bars = ax.bar(x_positions, values, color=bar_colors)
    
    # Set x-ticks with region names
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regions, rotation=0, ha='center', fontsize=FONTS['tick_label']['size'])
    
    # Add global reference line if 'Global' in the data
    if 'Global' in df.index:
        global_val = df.loc['Global', col_to_plot]
        ax.axhline(global_val, color=COLORS['global_line'], linestyle='--', linewidth=1)
    
    # Add values on top of bars
    for i, val in enumerate(values):
        display_val = val * 100 if is_percent else val
        format_str = '{:.1f}%' if is_percent else '{:.1f}'
        ax.text(i, val, format_str.format(display_val), ha='center', va='bottom', fontsize=FONTS['tick_label']['size'])
    
    # Set y-axis label using display name
    ax.set_ylabel(display_name, fontsize=FONTS['axis_label']['size'])
    
    # Format y-axis as percentage if needed
    if is_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    
    # Adjust y-axis to only use 80% of vertical space, leaving 20% at top for score components
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Set new limits that leave 20% at top empty
    ax.set_ylim(y_min, y_max + (y_range * 0.25))  # Add 25% to the top (which becomes 20% of the new range)
    
    # Add score components for hypothesis plot
    if plot_type == 'hypothesis' and hypo_result and hypo_result['scores']:
        scores = hypo_result['scores']
        
        # Always use sign-based scoring
        components = [
            ('final_score', scores['final_score']),
            ('sign_agreement', scores['sign_agreement']),
            ('explained_ratio', scores['explained_ratio'])
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


def create_multi_hypothesis_plot(
    df: pd.DataFrame,
    metric_col: str,
    hypo_cols: List[str],
    metric_anomaly_info: Dict[str, Any],
    ordered_hypos: List[Tuple[str, HypothesisResult]],
    figsize: Tuple[int, int] = (18, 10),
    include_score_formula: bool = True,
    is_conclusive: bool = True
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
    # Create an empty figure
    fig = plt.figure(figsize=figsize)
    
    # Define figure layout with reserved space for top and bottom
    top_margin = 0.80  # Reserve top 20% for explanatory text
    bottom_margin = 0.15  # Reserve bottom 15% for formula
    
    # Determine which hypotheses to plot on the right side
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
    plot_bars(ax_metric, df[[metric_col]], metric_anomaly_info, plot_type='metric')
    ax_metric.set_title(f"Metric: {metric_col}", fontsize=FONTS['title']['size'])
    
    # Handle the "Best Hypothesis" plot area
    if is_conclusive and best_hypo_name is not None and best_hypo_result is not None:
        plot_bars(
            ax_best_hypo, 
            df[[metric_col, best_hypo_name]], 
            metric_anomaly_info, 
            best_hypo_result, 
            plot_type='hypothesis',
            highlight_region=True
        )
        ax_best_hypo.set_title(f"Best Hypothesis: {best_hypo_name}", fontsize=FONTS['title']['size'])
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
                highlight_region=False  # No highlight on the right side
            )
            
            # Adjust title based on context
            title_prefix = f"Hypothesis {i+2}" if is_conclusive else f"Hypothesis {i+1}"
            
            if len(hypo_cols) > 4:
                ax.set_title(title_prefix, fontsize=FONTS['title']['size'])
            else:
                ax.set_title(f"{title_prefix}: {hypo_name}", fontsize=FONTS['title']['size'])
    
    # Add score formula if included
    if include_score_formula:
        add_score_formula(fig)
    
    return fig


def score_all_hypotheses(
    df: pd.DataFrame,
    metric_col: str,
    hypo_cols: List[str],
    metric_anomaly_info: Dict[str, Any],
    expected_directions: Dict[str, str]
) -> Dict[str, HypothesisResult]:
    """
    Score all hypotheses for a given metric using sign-based scoring.
    
    Args:
        df: DataFrame containing metric and hypothesis columns
        metric_col: Name of the metric column
        hypo_cols: List of hypothesis column names
        metric_anomaly_info: Info about the metric anomaly
        expected_directions: Dictionary mapping hypothesis names to their expected directions
    
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
        hypo_results[hypo_col] = score_func(temp_df, metric_anomaly_info, expected_direction)
    
    return hypo_results


def process_metrics(
    df: pd.DataFrame,
    metric_cols: List[str],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    expected_directions: Dict[str, str],
    metric_hypothesis_map: Dict[str, List[str]]
) -> Dict[str, Dict[str, HypothesisResult]]:
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
            expected_directions
        )
        
        # Store results for this metric
        all_results[metric_col] = hypo_results
    
    return all_results


def get_ranked_hypotheses(hypo_results: Dict[str, HypothesisResult]) -> List[Tuple[str, HypothesisResult]]:
    """
    Rank hypotheses by their final score in descending order.
    
    Args:
        hypo_results: Dictionary mapping hypothesis names to their score results
    
    Returns:
        List of (hypothesis_name, result) tuples sorted by final score in descending order
    """
    # Convert to list of tuples for sorting
    hypo_list = [(name, result) for name, result in hypo_results.items()]
    
    # Sort by final score in descending order
    hypo_list.sort(key=lambda x: x[1]['scores']['final_score'], reverse=True)
    
    return hypo_list


def rank_with_guardrail(hypo_results: Dict[str, HypothesisResult]) -> Union[List[Tuple[str, HypothesisResult]], str]:
    """
    Apply guardrail filtering before ranking hypotheses.
    
    Args:
        hypo_results: Dictionary mapping hypothesis names to their score results
    
    Returns:
        List of (hypothesis_name, result) tuples sorted by final score, or "Inconclusive" if none pass guardrails
    """
    # Apply guardrail filter by checking the 'explains' flag
    qualified = {h: r for h, r in hypo_results.items() if r['scores']['explains']}
    
    if not qualified:
        return "Inconclusive – no hypothesis meets guardrails"
    else:
        return get_ranked_hypotheses(qualified)


def save_results_to_dataframe(all_results: Dict[str, Dict[str, HypothesisResult]]) -> pd.DataFrame:
    """
    Convert scoring results to a pandas DataFrame for easy analysis and storage.
    
    Args:
        all_results: Dictionary mapping metrics to dictionaries of hypothesis results
        
    Returns:
        DataFrame with metrics, hypotheses, and scores
    """
    # Create empty lists to store data
    records = []
    
    # Iterate through all metrics and hypotheses
    for metric_name, hypo_results in all_results.items():
        for hypo_name, result in hypo_results.items():
            # Extract scores
            scores = result['scores']
            
            # Create record with common fields
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
            }
              
            record.update({
                    'sign_agreement': scores['sign_agreement'],
                    'explained_ratio': scores['explained_ratio'],
                    'p_value': scores['p_value']
                })
              
            # Add record to list
            records.append(record)
    
    # Convert records to DataFrame
    df_results = pd.DataFrame(records)
    
    # Sort by metric and final_score (descending)
    df_results = df_results.sort_values(['metric', 'final_score'], ascending=[True, False])
    
    return df_results


def plot_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_anomaly_info: Dict[str, Any],
    expected_direction: str = 'same'
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
    
    # Get all regions (including Global)
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
        elif region == 'Global':
            colors.append(COLORS['global_line'])
            sizes.append(100)  # Make Global point larger
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
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
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
    best_hypo_result: Optional[HypothesisResult] = None
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
    # Prepare template values - only include what's actually in the template
    context: Dict[str, Any] = {
        'region': metric_anomaly_info['anomalous_region'],
        'metric_name': metric_col,
        'metric_deviation': metric_anomaly_info.get('magnitude', '0%'),  # Use string magnitude directly
        'metric_dir': metric_anomaly_info['direction'],
    }
    
    if best_hypo_name and best_hypo_result:
        context.update({
            'hypo_name': best_hypo_name,
            'hypo_dir': best_hypo_result['direction'],
            'hypo_delta': best_hypo_result['magnitude'],
            'ref_hypo_val': best_hypo_result['ref_hypo_val'],
            'score': best_hypo_result['scores']['final_score'],
            'explained_ratio': best_hypo_result['scores']['explained_ratio'] * 100
        })
    
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
    best_hypo_result: Optional[HypothesisResult] = None, 
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
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
    )


def score_hypotheses_for_metrics(
    regional_df: pd.DataFrame,
    anomaly_map: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    region_col: str = 'region'
) -> Dict[str, Any]:
    """
    Score hypotheses for multiple metrics and return results in unified format directly.
    
    Args:
        regional_df: DataFrame with regional data
        anomaly_map: Dictionary mapping metric names to anomaly information
        config: Full configuration dictionary
        region_col: Name of the region column
    
    Returns:
        Dictionary in unified format: {metric_name: {'slides': {'Directional': slide_data}}}
    """
    metrics_config = config.get('metrics', {})
    unified_results = {}
    
    for metric_name, metric_config in metrics_config.items():
        # Skip metrics without hypotheses
        if 'hypotheses' not in metric_config:
            continue
            
        # Skip metrics not in anomaly map
        if metric_name not in anomaly_map:
            continue
            
        try:
            # Get anomaly information
            anomaly_info = anomaly_map[metric_name]
            anomalous_region = anomaly_info.get('anomalous_region')
            
            if not anomalous_region:
                continue
            
            # Get hypotheses for this metric
            hypothesis_names = list(metric_config['hypotheses'].keys())
            
            # Get expected directions from config
            expected_directions = {}
            for hypo_name, hypo_config in metric_config['hypotheses'].items():
                expected_directions[hypo_name] = hypo_config.get('expected_direction', 'same')
            
            # Score all hypotheses using existing function
            hypothesis_results = score_all_hypotheses(
                df=regional_df,
                metric_col=metric_name,
                hypo_cols=hypothesis_names,
                metric_anomaly_info=anomaly_info,
                expected_directions=expected_directions
            )
            
            if not hypothesis_results:
                continue
            
            # Get all hypotheses ranked by score (regardless of guardrail)
            all_ranked_hypotheses = get_ranked_hypotheses(hypothesis_results)
            
            # Check if any hypothesis meets guardrail criteria
            qualified_hypotheses = rank_with_guardrail(hypothesis_results)
            is_conclusive = not isinstance(qualified_hypotheses, str)
            
            if is_conclusive:
                # Conclusive case - we have qualified hypotheses
                best_hypo_name, best_hypo_result = qualified_hypotheses[0]
                template = metric_config['hypotheses'][best_hypo_name].get('template', '')
                # summary_template = metric_config['hypotheses'][best_hypo_name].get('summary_template', '')
                
                # Prepare template parameters for SlideContent rendering (only template variables!)
                template_params = {
                    'region': anomaly_info['anomalous_region'],
                    'metric_name': metric_name,
                    'metric_deviation': anomaly_info.get('magnitude', '0%'),
                    'metric_dir': anomaly_info['direction'],
                    'hypo_name': best_hypo_name,
                    'hypo_dir': best_hypo_result['direction'],
                    'hypo_delta': best_hypo_result['magnitude'],
                    'ref_hypo_val': best_hypo_result['ref_hypo_val'],
                    'score': best_hypo_result['scores']['final_score'],
                    'explained_ratio': best_hypo_result['scores']['explained_ratio'] * 100
                }
                
                # Generate summary text from template
                filled_text = render_template_text(
                    template=template,
                    metric_anomaly_info=anomaly_info,
                    metric_col=metric_name,
                    best_hypo_name=best_hypo_name,
                    best_hypo_result=best_hypo_result
                )
                
                # Extract root cause portion for summary
                if "Root cause:" in filled_text:
                    summary_text = filled_text.split("Root cause:")[1].strip()
                else:
                    summary_text = f"{best_hypo_name} is {best_hypo_result['direction']}"
                
                slide_title = f"{metric_name} - Root Cause"
                
            else:
                # Inconclusive case - no hypothesis meets guardrails, but still show all hypotheses
                inconclusive_template = "Analysis of {{metric_name}} in {{region}} shows {{metric_deviation}} {{metric_dir}} performance than the global mean. However, none of the provided hypotheses meet the minimum evidence thresholds for a confident root cause determination."
                
                template_params = {
                    'region': anomaly_info['anomalous_region'],
                    'metric_name': metric_name,
                    'metric_deviation': anomaly_info.get('magnitude', '0%'),
                    'metric_dir': anomaly_info['direction']
                }
                
                summary_text = f"Inconclusive - no hypothesis meets minimum evidence thresholds."
                template = inconclusive_template
                slide_title = f"{metric_name} - Inconclusive to determine root cause"
                # Set these to None since no hypothesis qualifies as "best"
                best_hypo_name = None
                best_hypo_result = None
            
            # Store functions directly in figure_generators (much simpler!)
            figure_generators = [
                {
                    "function": create_multi_hypothesis_plot,
                    "title_suffix": "",
                    "params": {
                        'df': regional_df,
                        'metric_col': metric_name,
                        'hypo_cols': hypothesis_names,
                        'metric_anomaly_info': anomaly_info,
                        'ordered_hypos': all_ranked_hypotheses,  # Always show all hypotheses ranked by score
                        'is_conclusive': is_conclusive
                    }
                },
                # {
                #     "function": create_scatter_grid, 
                #     "title_suffix": " (Part 2)",
                #     "params": {
                #         'df': regional_df,
                #         'metric_col': metric_name,
                #         'hypo_cols': hypothesis_names,
                #         'metric_anomaly_info': anomaly_info,
                #         'expected_directions': expected_directions
                #     }
                # }
            ]
            
            # Create slide data in unified format
            analysis_type = 'Directional'  # Changed from 'scorer' to 'Directional'
            unified_results[metric_name] = {
                'slides': {
                    analysis_type: {
                        'summary': {
                            'summary_text': summary_text
                        },
                        'slide_info': {
                            'title': slide_title,
                            'template_text': template,
                            'template_params': template_params,
                            'figure_generators': figure_generators,
                            'dfs': {},
                            'layout_type': "text_figure"
                        }
                    }
                },
                'payload': {
                    'best_hypothesis': best_hypo_result,
                    'all_results': hypothesis_results
                },
            }
            
        except Exception as e:
            print(f"Error processing metric '{metric_name}': {e}")
            continue
    
    return unified_results


def main(save_path: str = '.', results_path: Optional[str] = None):
    """
    Test the hypothesis scoring and visualization with multiple hypotheses.
    
    Args:
        save_path: Directory path to save generated figures
        results_path: Path to save DataFrame results (if None, results are not saved)
    """
    # Create test data
    np.random.seed(42)
    regions = ["Global", "North America", "Europe", "Asia", "Latin America"]
    
    # Create test data with multiple metrics and hypotheses
    data = {
        # Metrics
        'conversion_rate_pct': np.array([0.12, 0.08, 0.11, 0.13, 0.10]),
        'avg_order_value': np.array([75.0, 65.0, 80.0, 85.0, 72.0]),
        'customer_satisfaction': np.array([4.2, 3.8, 4.3, 4.5, 4.0]),
        
        # Hypotheses
        'bounce_rate_pct': np.array([0.35, 0.45, 0.32, 0.28, 0.34]),
        'page_load_time': np.array([2.4, 3.8, 2.2, 1.9, 2.5]),
        'session_duration': np.array([180, 120, 190, 210, 175]),
        'pages_per_session': np.array([4.2, 3.1, 4.5, 4.8, 4.0]),
        'new_users_pct': np.array([0.25, 0.18, 0.28, 0.30, 0.23])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data, index=regions)
    
    # Define metric columns and hypothesis columns
    metric_cols = ['conversion_rate_pct', 'avg_order_value', 'customer_satisfaction']
    hypo_cols = ['bounce_rate_pct', 'page_load_time', 'session_duration', 'pages_per_session', 'new_users_pct']
    
    # Import anomaly detector
    from rca_package.anomaly_detector import detect_snapshot_anomaly_for_column
    
    # Create metric anomaly map using the anomaly detector
    metric_anomaly_map = {}
    for metric_col in metric_cols:
        anomaly_info = detect_snapshot_anomaly_for_column(df, 'Global', column=metric_col)
        if anomaly_info:
            metric_anomaly_map[metric_col] = anomaly_info
    
    # Define expected directions for each hypothesis
    expected_directions = {
        'bounce_rate_pct': 'opposite',  # Higher bounce rate -> lower conversion
        'page_load_time': 'opposite',   # Higher load time -> lower conversion
        'session_duration': 'same',     # Higher session time -> higher conversion
        'pages_per_session': 'same',    # More pages viewed -> higher conversion
        'new_users_pct': 'opposite'     # New users tend to convert less
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
    # Process for all metrics using sign-based scoring
    all_results = process_metrics(
        df=df,
        metric_cols=metric_cols,
        metric_anomaly_map=metric_anomaly_map,
        expected_directions=expected_directions,
        metric_hypothesis_map={
            'conversion_rate_pct': hypo_cols,
            'avg_order_value': hypo_cols,
            'customer_satisfaction': hypo_cols
        }
    )
    
    # Save to DataFrame for analysis
    df_results = save_results_to_dataframe(all_results)
    print("\nResults DataFrame Preview:")
    print(df_results[['metric', 'hypothesis', 'final_score', 'magnitude', 'direction', 'explains']].head(10))
    
    # Save results to file if path is provided
    if results_path is not None:
        df_results.to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")
    
    # Display top hypothesis for each metric
    print("\nTop Hypothesis for Each Metric:")
    for metric_col in metric_cols:
        hypo_results = all_results[metric_col]
        ranked_hypos = rank_with_guardrail(hypo_results)
        if isinstance(ranked_hypos, str):
            print(f"  {metric_col}: {ranked_hypos}")
        else:
            best_hypo_name, best_hypo_result = ranked_hypos[0]
            score = best_hypo_result['scores']['final_score']
            print(f"  {metric_col}: {best_hypo_name} (Score: {score:.2f})")
    
    # 4. Create visualization using pre-sorted results
    print("\n===== VISUALIZATION WITH PRE-SORTED RESULTS =====")
    metric_col = 'conversion_rate_pct'
    hypo_results = all_results[metric_col]
    ranked_hypos = rank_with_guardrail(hypo_results)
    
    if isinstance(ranked_hypos, str):
        # Handle inconclusive case
        print(f"Cannot create visualization for {metric_col}: {ranked_hypos}")
        fig = plt.figure(figsize=(12, 8))
        fig.text(0.5, 0.5, f"{metric_col}\n\n{ranked_hypos}", 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.7))
        filename = f"{save_path}/inconclusive_{metric_col}.png"
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Created inconclusive visualization: {filename}")
    else:
        # Create visualization with qualified hypotheses
        all_ranked = get_ranked_hypotheses(hypo_results)
        is_conclusive_main = not isinstance(ranked_hypos, str)
        
        fig = create_multi_hypothesis_plot(
            df=df,
            metric_col=metric_col,
            hypo_cols=hypo_cols,
            metric_anomaly_info=metric_anomaly_map[metric_col],
            ordered_hypos=all_ranked,  # Always show all hypotheses
            include_score_formula=True,
            is_conclusive=is_conclusive_main
        )
        
        # Use the template with Jinja2 double-brace syntax
        template = "{{metric_name}} in {{region}} is {{metric_deviation}} {{metric_dir}} than Global mean.\n"\
                   "Root cause: {{region}} has {{hypo_delta}} {{hypo_dir}} of {{hypo_name}} than the global mean ({{ref_hypo_val}}). "\
                   "This suggests Account Managers have different interaction volumes, potentially impacting their ability to "\
                   "effectively manage and prioritize CLIs in their portfolio."
        
        # Add template text and score formula
        best_hypo_name, best_hypo_result = ranked_hypos[0]
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