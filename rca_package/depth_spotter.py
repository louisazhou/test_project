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
    higher_is_better: bool = True
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
    
    return df, delta, row_rate


def additive_contrib(
    df_slice: pd.DataFrame, 
    row_total: float,
    metric_col: str,
    higher_is_better: bool = True
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
    
    return df, delta


def plot_subregion_bars(
    df_slice: pd.DataFrame,
    metric_col: str,
    title: str,
    row_value: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a meaningful bar chart showing sub-regions color-coded by their contribution to the performance gap.
    
    Args:
        df_slice: DataFrame containing the slice-level data with contribution analysis
        metric_col: Name of the metric column to plot
        title: Chart title
        row_value: Rest-of-world value to show as reference line (optional)
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
    if is_percent:
        value_formatter = lambda x: f"{x*100:.1f}%"
    else:
        value_formatter = lambda x: f"{x:,.0f}"
    
    # Identify top-3 by score
    top_3_indices = set(np.argsort(scores)[-3:]) if len(scores) > 0 else set()
    
    # Color-code bars: only highlight top-3, with INTUITIVE LOGIC (positive = helpful = green)
    colors = []
    for i, contrib in enumerate(contributions):
        if i in top_3_indices:  # Only highlight top-3 by score
            # SIMPLIFIED LOGIC: Contribution calculation is now intuitive
            # Positive contribution = helpful = GREEN
            # Negative contribution = harmful = RED
            if contrib > 0:
                colors.append(COLORS['metric_positive'])  # Green (helpful)
            else:
                colors.append(COLORS['metric_negative'])  # Red (harmful)
        else:  # Not in top-3
            colors.append(COLORS['default_bar'])  # Gray
    
    # Create bars
    bars = ax.bar(range(len(values)), values, width=0.6, color=colors, linewidth=0.5)
    
    # Add value labels on bars
    original_values = df_slice[metric_col].values  # Keep original values for labeling
    for i, val in enumerate(values):
        # Position text above the bar
        max_val = max(values) if len(values) > 0 else 0
        text_y = val + (max_val * 0.02)
        
        # Format and add text, handling NaN in original values
        original_val = original_values[i]
        if pd.isna(original_val):
            label_text = "N/A"
        else:
            label_text = value_formatter(original_val)
        
        ax.text(i, text_y, label_text, ha="center", va="bottom", 
               fontweight=FONTS['annotation']['weight'], fontsize=FONTS['annotation']['size'])
    
    # Customize the plot
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONTS['tick_label']['size'])
    ax.set_ylabel(ylabel, fontsize=FONTS['axis_label']['size'])
    ax.set_title(title, fontsize=FONTS['title']['size'], fontweight=FONTS['title']['weight'])
    
    # Format y-axis to match value annotations
    if is_percent:
        # Format y-axis as percentages
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
    else:
        # Format y-axis as regular numbers with commas
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    # Add reference line if row_value is provided
    if row_value is not None:
        ax.axhline(row_value, color=COLORS['global_line'], linestyle='--', linewidth=2, alpha=0.7, label='Rest-of-World')
        ax.legend(fontsize=FONTS['tick_label']['size'])
    
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
                    metric_anomaly_map[metric_name].get('higher_is_better', True)
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
                
                contrib_df, delta = additive_contrib(region_df, row_total, metric_col, metric_anomaly_map[metric_name].get('higher_is_better', True))
                
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
            summary_cols = ['slice', plot_metric_col, 'contribution', 'coverage', 'score', 'raw_contribution']
            full_analysis_df = contrib_df[summary_cols].copy().set_index('slice')
            
            # Create formatted display table (top 3 only for table display)
            display_table = full_analysis_df.drop(['raw_contribution'], axis=1).head(3).copy()
            display_table['contribution'] = display_table['contribution'].apply(lambda x: f"{x:.1%}")
            display_table['coverage'] = display_table['coverage'].apply(lambda x: f"{x:.1%}")
            display_table['score'] = display_table['score'].apply(lambda x: f"{x:.2f}")
            
            if metric_type == 'rate':
                display_table[plot_metric_col] = display_table[plot_metric_col].apply(lambda x: f"{x:.1%}")
            else:
                display_table[plot_metric_col] = display_table[plot_metric_col].apply(lambda x: f"{int(x):,}")
            
            # Generate concise summary text showing top 2 drivers of the delta
            # Use raw contribution (before sign flipping) to rank actual drivers of delta
            # Raw contribution represents: (actual - expected) / delta
            # So biggest absolute raw contributions are the biggest drivers regardless of delta sign
            contrib_df['abs_raw_contribution'] = contrib_df['raw_contribution'].abs()
            top_drivers = contrib_df.nlargest(2, 'abs_raw_contribution')
            
            summary_parts = []
            for _, row in top_drivers.iterrows():
                name = row['slice']
                contrib = row['contribution'] * 100
                summary_parts.append(f"{name} ({contrib:+.0f}%)")
            
            summary_text = ", ".join(summary_parts) if summary_parts else f"Depth analysis completed for {metric_name}"

            # Return in unified format directly - no integration needed!
            analysis_type = 'depth'  # Derived from config_depth.yaml
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
                                    "title_suffix": "",
                                    "params": {
                                        'df_slice': full_analysis_df,  # Pass ALL slices to chart function
                                        'metric_col': plot_metric_col,
                                        'title': f"{metric_name} by sub-regions",
                                        'row_value': row_value
                                    }
                                }
                            ],
                            'dfs': {"summary_table": display_table},  # Top 3 only for table display
                            'layout_type': "text_tables_figure"
                        }
                    }
                },
                'payload': {
                            'summary_df': full_analysis_df,  # Full analysis data for internal inspection
                            'delta': delta,
                            'row_value': row_value
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
        'conversion_rate_pct': {'anomalous_region': anomalous_region},
        'revenue': {'anomalous_region': anomalous_region}, 
        'customer_satisfaction': {'anomalous_region': anomalous_region}
    }
    
    results = analyze_region_depth(
        sub_df=sub_df,
        config=test_config,
        metric_anomaly_map=test_metric_anomaly_map
    )
    
    # Print structured results
    print(f"\n{'='*60}")
    print("STRUCTURED RESULTS")
    print(f"{'='*60}")
    
    for metric_name, metric_result in results.items():
        print(f"\nMetric: {metric_name}")
        
        # Access the depth slide data
        depth_data = metric_result['slides']['depth']['data']
        slide_info = metric_result['slides']['depth']['slide_info']
        
        print(f"Name: {depth_data['name']}")
        print(f"Type: {metric_result['slides']['depth']['type']}")
        print(f"Summary: {depth_data['summary_text']}")
        
        # Show rendered template text
        if slide_info['template_text']:
            print(f"Template Text:\n{slide_info['template_text'][:200]}...")  # Truncate for readability
    
    # Additional analysis: Show data summary
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    
    # Regional summary
    regional_summary = sub_df.groupby('region').agg({
        'visits': 'sum',
        'conversions': 'sum',
        'revenue': 'sum',
        'surveys': 'sum',
        'sum_csat': 'sum'
    }).round(2)
    
    # Calculate rates
    regional_summary['conversion_rate'] = regional_summary['conversions'] / regional_summary['visits']
    regional_summary['avg_csat'] = regional_summary['sum_csat'] / regional_summary['surveys']
    
    print("\nRegional Summary:")
    print(regional_summary[['visits', 'conversion_rate', 'revenue', 'avg_csat']].round(3))
    
    print(f"\nDepth analysis completed successfully!")
    print(f"Results would be saved to: {output_dir}")
    
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