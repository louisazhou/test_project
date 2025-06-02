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


def rate_contrib(
    df_slice: pd.DataFrame, 
    row_numerator: float, 
    row_denominator: float,
    visits_col: str = 'visits',
    conversions_col: str = 'conversions',
    metric_name: str = 'conversion_rate_pct'
) -> Tuple[pd.DataFrame, float, float]:
    """
    Calculate contribution analysis for rate metrics (e.g., conversion rate).
    
    Args:
        df_slice: DataFrame containing slice-level data for the anomalous region
        row_numerator: Total numerator from rest-of-world (e.g., total conversions)
        row_denominator: Total denominator from rest-of-world (e.g., total visits)
        visits_col: Name of the visits/denominator column
        conversions_col: Name of the conversions/numerator column
        metric_name: Name of the metric (for the calculated rate column)
    
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
    df['expected'] = df[visits_col] * row_rate
    
    # Calculate total delta
    delta = df[conversions_col].sum() - df['expected'].sum()
    
    # Calculate coverage (share of total visits within the region)
    df['coverage'] = df[visits_col] / df[visits_col].sum()
    
    # Calculate contribution to the gap
    df['contribution'] = (df[conversions_col] - df['expected']) / delta if delta != 0 else 0
    
    # Calculate heuristic score: sqrt(|contribution| + max(|contribution| - coverage, 0) / coverage)
    # Floor coverage at 1% to avoid division by zero
    coverage_floored = df['coverage'].clip(lower=0.01)
    df['score'] = np.sqrt(
        np.abs(df['contribution']) + 
        np.maximum(np.abs(df['contribution']) - df['coverage'], 0) / coverage_floored
    )
    
    # Add actual rate using the proper metric name
    df[metric_name] = df[conversions_col] / df[visits_col]
    
    return df, delta, row_rate


def additive_contrib(
    df_slice: pd.DataFrame, 
    row_total: float,
    metric_col: str
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate contribution analysis for additive metrics (e.g., revenue, orders).
    
    Args:
        df_slice: DataFrame containing slice-level data for the anomalous region
        row_total: Total metric value from rest-of-world
        metric_col: Name of the metric column to analyze
    
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
    
    # Calculate contribution to the gap
    df['contribution'] = (df[metric_col] - df['expected']) / delta if delta != 0 else 0
    
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
    
    # Get values and labels
    values = df_slice[metric_col].values
    
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
    
    # Color-code bars: only highlight top-3, color by contribution sign
    colors = []
    for i, contrib in enumerate(contributions):
        if i in top_3_indices:  # Only highlight top-3 by score
            if contrib < 0:  # Negative contribution (making problem worse)
                colors.append(COLORS['metric_negative'])  # Red
            else:  # Positive contribution (helping performance)
                colors.append(COLORS['metric_positive'])  # Green
        else:  # Not in top-3
            colors.append(COLORS['default_bar'])  # Gray
    
    # Create bars
    bars = ax.bar(range(len(values)), values, width=0.6, color=colors, linewidth=0.5)
    
    # Add value labels on bars
    for i, val in enumerate(values):
        # Position text above the bar
        max_val = max(values) if len(values) > 0 else 0
        text_y = val + (max_val * 0.02)
        
        # Format and add text
        ax.text(i, text_y, value_formatter(val), ha="center", va="bottom", 
               fontweight=FONTS['annotation']['weight'], fontsize=FONTS['annotation']['size'])
    
    # Customize the plot
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONTS['tick_label']['size'])
    ax.set_ylabel(ylabel, fontsize=FONTS['axis_label']['size'])
    ax.set_title(title, fontsize=FONTS['title']['size'], fontweight=FONTS['title']['weight'])
    
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
    anomalous_region: str,
    config: Dict[str, Any],
    slice_col: str = 'slice',
    region_col: str = 'region'
) -> Dict[str, Dict[str, Any]]:
    """
    Perform depth analysis for an anomalous region across multiple metrics.
    
    Args:
        sub_df: DataFrame containing sub-region/slice level data
        anomalous_region: Name of the region to analyze
        config: Configuration dictionary
        slice_col: Name of the column containing slice identifiers
        region_col: Name of the column containing region identifiers
    
    Returns:
        Dictionary containing analysis results for each metric
    """
    metrics_config = config.get('metrics', {})
    
    results = {}
    
    # Filter data for the anomalous region and rest-of-world
    region_df = sub_df[sub_df[region_col] == anomalous_region].copy()
    row_df = sub_df[sub_df[region_col] != anomalous_region].copy()
    
    if region_df.empty:
        logger.warning(f"No data found for region: {anomalous_region}")
        return results
    
    if row_df.empty:
        logger.warning("No rest-of-world data found for comparison")
        return results
    
    # Analyze each metric
    for metric_name, config in metrics_config.items():
        try:
            metric_type = config['type']
            display_name = config.get('name', metric_name)
            template = config.get('template', '')
            
            if metric_type == 'rate':
                # Rate metric analysis
                numerator_col = config['numerator_col']
                denominator_col = config['denominator_col']
                
                # Calculate rest-of-world totals
                row_numerator = row_df[numerator_col].sum()
                row_denominator = row_df[denominator_col].sum()
                
                # Perform contribution analysis
                contrib_df, delta, row_rate = rate_contrib(
                    region_df, row_numerator, row_denominator,
                    denominator_col, numerator_col, metric_name
                )
                
                region_rate = contrib_df[numerator_col].sum() / contrib_df[denominator_col].sum()
                # For rate metrics, we plot the calculated rate using the proper metric name
                plot_metric_col = metric_name
                
            elif metric_type == 'additive':
                # Additive metric analysis
                metric_col = config['metric_col']
                
                # Calculate rest-of-world total
                row_total = row_df[metric_col].sum()
                
                # Perform contribution analysis
                contrib_df, delta = additive_contrib(region_df, row_total, metric_col)
                
                region_total = contrib_df[metric_col].sum()
                # For additive metrics, we plot the original metric column
                plot_metric_col = metric_col
                
            else:
                logger.warning(f"Unknown metric type '{metric_type}' for metric '{metric_name}'")
                continue
            
            # Prepare data for markdown table (top 3 contributors)
            sorted_contrib = contrib_df.sort_values('score', ascending=False)
            
            # Create summary table with slice, metric value, contribution, coverage, and score
            summary_cols = ['slice', plot_metric_col, 'contribution', 'coverage', 'score']
            summary_df = sorted_contrib[summary_cols].head(3).copy()
            
            # Set slice as index for cleaner table display
            summary_df = summary_df.set_index('slice')
            
            # Prepare template parameters
            template_params = {
                'metric_name': display_name,
                'anomalous_region': anomalous_region,
                'delta': delta,
                'summary_table': summary_df.to_markdown(index=True, floatfmt='.3f')
            }
            
            # Add metric-specific parameters
            if metric_type == 'rate':
                template_params.update({
                    'row_rate': row_rate,
                    'region_rate': region_rate
                })
            else:
                template_params.update({
                    'row_total': row_total,
                    'region_total': region_total
                })
            
            # Render template immediately with actual values
            if template:
                template_obj = Template(template)
                rendered_text = template_obj.render(**template_params)
            else:
                rendered_text = f"{display_name} analysis for {anomalous_region}"
            
            # Create structured result with slide-ready content
            depth_hypothesis_name = f"{metric_name}_in_subregion"
            results[depth_hypothesis_name] = {
                "hypothesis": depth_hypothesis_name,
                "name": display_name,
                "type": "depth_spotter",
                "selected": True,
                "rendered_text": rendered_text,
                "summary_df": summary_df,  # Only the relevant table data
                "payload": {
                    'contrib_df': contrib_df,
                    'delta': delta,
                    'metric_col': plot_metric_col,
                    'row_value': row_rate if metric_type == 'rate' else row_total / len(row_df) if len(row_df) > 0 else None
                },
                # Ready-to-use slide content
                "slide": {
                    "title": f"{display_name} - Depth Analysis",
                    "text": rendered_text,
                    "table_df": summary_df,  # Clean table with slice as index
                    "layout_type": "text_table_figure"
                }
            }
                
        except KeyError as e:
            logger.error(f"Missing required configuration for metric '{metric_name}': {e}")
        except Exception as e:
            logger.error(f"Error analyzing metric '{metric_name}': {e}")
    
    return results


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

    # North America (anomalously low metrics)
    add_slice("North America", "a", 10_000, 0.09, 2000, 66, 900, 3.9)
    add_slice("North America", "b", 11_000, 0.07, 2200, 64, 950, 3.7)
    add_slice("North America", "c", 12_000, 0.08, 2100, 65, 1000, 3.8)

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
    
    results = analyze_region_depth(
        sub_df=sub_df,
        anomalous_region=anomalous_region,
        config=test_config
    )
    
    # Print structured results
    print(f"\n{'='*60}")
    print("STRUCTURED RESULTS")
    print(f"{'='*60}")
    
    for hypo_name, result in results.items():
        print(f"\nHypothesis: {hypo_name}")
        print(f"Name: {result['name']}")
        print(f"Type: {result['type']}")
        
        # Render template
        if result['rendered_text']:
            print(f"Template Text:\n{result['rendered_text']}")
    
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