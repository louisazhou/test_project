"""
Depth Spotter Module for Sub-Region Analysis

This module provides tools for analyzing performance gaps at a deeper level
by examining sub-regions or slices within anomalous regions to identify
the specific contributors to performance differences.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import logging
from jinja2 import Template

# Setup logging
logger = logging.getLogger(__name__)

# Colors for visualization
COLORS = {
    'positive': '#2ecc71',      # Green for positive contributions
    'negative': '#e74c3c',      # Red for negative contributions
    'neutral': '#95a5a6',       # Gray for neutral/small contributions
    'baseline': '#34495e'       # Dark blue-gray for baseline
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
    
    return df, delta


def plot_subregion_bars(
    df_slice: pd.DataFrame,
    metric_col: str,
    title: str,
    config: Dict[str, Any] = None,
    get_display_name_func: callable = None,
    row_value: float = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Create a bar chart showing actual data values for sub-regions.
    
    Args:
        df_slice: DataFrame containing the slice-level data
        metric_col: Name of the metric column to plot
        title: Chart title
        config: Configuration dictionary (optional, for display names)
        get_display_name_func: Function to get display names (optional)
        row_value: Rest-of-world value to show as reference line (optional)
        figsize: Figure size as (width, height) tuple
    
    Returns:
        Matplotlib figure with the bar chart
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get values and labels
    values = df_slice[metric_col].values
    labels = df_slice['slice'].values
    
    # Get display name for y-axis
    if config and get_display_name_func:
        ylabel = get_display_name_func(config, metric_col)
    else:
        ylabel = metric_col.replace('_', ' ').title()
    
    # Format values based on metric name
    is_percent = ('_pct' in metric_col or '%' in metric_col or 'rate' in metric_col.lower())
    if is_percent:
        value_formatter = lambda x: f"{x*100:.1f}%"
    else:
        value_formatter = lambda x: f"{x:,.0f}"
    
    # Use consistent colors
    colors = [COLORS['baseline']] * len(values)
    
    # Create bars
    bars = ax.bar(range(len(values)), values, width=0.6, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for i, val in enumerate(values):
        # Position text above the bar
        max_val = max(values) if len(values) > 0 else 0
        text_y = val + (max_val * 0.02)
        
        # Format and add text
        ax.text(i, text_y, value_formatter(val), ha="center", va="bottom", 
               fontweight='bold', fontsize=9)
    
    # Customize the plot
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add reference line if row_value is provided
    if row_value is not None:
        ax.axhline(row_value, color=COLORS['negative'], linestyle='--', linewidth=2, alpha=0.7, label='Rest-of-World')
        ax.legend()
    
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
                scoring_method = 'depth_rate'
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
                scoring_method = 'depth_additive'
                # For additive metrics, we plot the original metric column
                plot_metric_col = metric_col
                
            else:
                logger.warning(f"Unknown metric type '{metric_type}' for metric '{metric_name}'")
                continue
            
            # Prepare data for markdown table (top 3 contributors)
            sorted_contrib = contrib_df.sort_values('contribution', key=abs, ascending=False)
            
            # Create summary table with slice, metric value, contribution, and coverage
            summary_cols = ['slice', plot_metric_col, 'contribution', 'coverage']
            summary_df = sorted_contrib[summary_cols].head(3).copy()
            
            # Prepare template parameters
            template_params = {
                'metric_name': display_name,
                'anomalous_region': anomalous_region,
                'delta': delta,
                'summary_table': summary_df.to_markdown(index=False, floatfmt='.3f')
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
            
            # Create structured result similar to hypothesis scorer format
            depth_hypothesis_name = f"{metric_name}_in_subregion"
            results[depth_hypothesis_name] = {
                "hypothesis": depth_hypothesis_name,
                "name": display_name,
                "type": "depth_spotter",
                "selected": True,  # Always selected since it's the depth analysis
                "template": template,
                "parameters": template_params,
                "payload": {
                    'contrib_df': contrib_df,
                    'delta': delta,
                    'region_df': contrib_df if metric_type == 'rate' else region_df,
                    'metric_col': plot_metric_col,
                    'row_value': row_rate if metric_type == 'rate' else row_total / len(row_df) if len(row_df) > 0 else None
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


def main(output_dir: str = '.', save_figures: bool = True):
    """
    Test the depth analysis functionality with synthetic data.
    
    Args:
        output_dir: Directory to save output files
        save_figures: Whether to save generated figures
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
        if result['template']:
            template = Template(result['template'])
            rendered_text = template.render(**result['parameters'])
            print(f"Template Text:\n{rendered_text}")
    

    
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
    
    if save_figures:
        print(f"Figures saved to: {output_dir}")
    
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
    parser.add_argument('--no-save', action='store_true', help='Do not save figures to disk')
    
    args = parser.parse_args()
    
    # Run with provided arguments
    main(
        output_dir=args.output_dir,
        save_figures=not args.no_save
    ) 