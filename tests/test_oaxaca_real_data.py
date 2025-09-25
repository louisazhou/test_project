#!/usr/bin/env python3
"""
Test script for Oaxaca-Blinder analysis with real Simpson's paradox data.

This script:
1. Reads product.csv and vertical.csv from tests/
2. Converts them to the format expected by run_oaxaca_analysis()
3. Runs the analysis and prints narratives
4. Saves figures and tables to output/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to sys.path to import rca_package
sys.path.insert(0, str(Path(__file__).parent.parent))

from rca_package.oaxaca_blinder import run_oaxaca_analysis, plot_rates_and_mix_panel, quadrant_prompts


def create_synthetic_data_from_rates(rates_df, region_name="LATAM"):
    """
    Convert rate/mix data to synthetic wins/total data for Oaxaca analysis.
    Focus on LATAM vs rest-of-world comparison only.
    
    Args:
        rates_df: DataFrame with columns [category, region_mix_pct, rest_mix_pct, region_rate, rest_rate]
        region_name: Name for the target region (LATAM)
    
    Returns:
        DataFrame suitable for run_oaxaca_analysis()
    """
    synthetic_data = []
    
    # Total volume to distribute (large enough for stable rates)
    total_volume = 100000
    
    # Calculate rest-of-world total volume (sum of all rest mix percentages)
    rest_total_volume = int(total_volume * rates_df['rest_mix_pct'].sum())
    
    for _, row in rates_df.iterrows():
        category = row['category']
        
        # LATAM data - use region mix percentages
        region_volume = int(total_volume * row['region_mix_pct'])
        if region_volume > 0:
            region_wins = int(region_volume * row['region_rate'])
            synthetic_data.append({
                'region': region_name,
                'category': category,
                'wins': region_wins,
                'total': region_volume
            })
        
        # Rest-of-world data - normalize rest mix to create proper baseline
        rest_volume = int(rest_total_volume * row['rest_mix_pct'] / rates_df['rest_mix_pct'].sum())
        if rest_volume > 0:
            rest_wins = int(rest_volume * row['rest_rate'])
            synthetic_data.append({
                'region': 'REST_OF_WORLD',
                'category': category,
                'wins': rest_wins,
                'total': rest_volume
            })
    
    return pd.DataFrame(synthetic_data)


def save_analysis_outputs(result, analysis_name, output_dir):
    """Save all analysis outputs to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save only the essential decomposition data
    decomp_file = output_path / f"{analysis_name}_LATAM_decomposition.csv"
    result.decomposition_df.to_csv(decomp_file, index=False)
    print(f"✅ Saved decomposition data: {decomp_file}")
    
    # Also save full decomposition_df so user can see all columns
    full_decomp_file = output_path / f"{analysis_name}_full_decomposition_df.csv"
    result.decomposition_df.to_csv(full_decomp_file, index=False)
    print(f"✅ Saved full decomposition_df: {full_decomp_file}")
    
    # Save narrative decisions and supporting tables - LATAM only
    if 'LATAM' in result.narrative_decisions:
        decision = result.narrative_decisions['LATAM']
        
        # Supporting table
        support_file = output_path / f"{analysis_name}_LATAM_supporting_table.csv"
        decision.supporting_table_df.to_csv(support_file, index=False)
        print(f"✅ Saved supporting table for LATAM: {support_file}")
        
        # Key metrics
        metrics_file = output_path / f"{analysis_name}_LATAM_metrics.csv"
        metrics_df = pd.DataFrame([decision.key_metrics])
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✅ Saved key metrics for LATAM: {metrics_file}")
    else:
        print("⚠️  No LATAM decision found to save")


def run_analysis_and_save(csv_file, analysis_name):
    """Run Oaxaca analysis on a CSV file and save all outputs."""
    print(f"\n🔍 ANALYZING: {analysis_name.upper()}")
    print("=" * 60)
    
    # Read the rates data
    rates_df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(rates_df)} categories from {csv_file}")
    
    # Convert to synthetic data
    synthetic_df = create_synthetic_data_from_rates(rates_df)
    print(f"📊 Created synthetic data with {len(synthetic_df)} rows")
    
    # Calculate overall rates for context
    region_data = synthetic_df[synthetic_df['region'] == 'LATAM']
    rest_data = synthetic_df[synthetic_df['region'] == 'REST_OF_WORLD']
    
    region_overall = region_data['wins'].sum() / region_data['total'].sum()
    rest_overall = rest_data['wins'].sum() / rest_data['total'].sum()
    gap_pp = (region_overall - rest_overall) * 100
    
    print(f"📈 LATAM overall rate: {region_overall:.1%}")
    print(f"📈 REST_OF_WORLD overall rate: {rest_overall:.1%}")
    print(f"📈 Gap: {gap_pp:+.1f}pp")
    
    # Run Oaxaca analysis
    print("\n🚀 Running Oaxaca-Blinder analysis...")
    result = run_oaxaca_analysis(
        df=synthetic_df,
        region_column='region',
        numerator_column='wins',
        denominator_column='total',
        category_columns=['category']
    )
    
    # Print narratives - focus on LATAM only
    print("\n📝 NARRATIVE RESULTS:")
    print("-" * 60)
    
    latam_decision = result.narrative_decisions.get('LATAM')
    if latam_decision:
        print(f"\n🌎 LATAM ANALYSIS:")
        print(f"   Root cause: {latam_decision.root_cause_type.value}")
        print(f"   Direction: {latam_decision.performance_direction.value}")
        print(f"\n📖 FULL NARRATIVE:")
        print(f"   {latam_decision.narrative_text}")
        
        # Print supporting table info
        table = latam_decision.supporting_table_df
        print(f"\n📊 SUPPORTING TABLE: {len(table)} rows")
        print(f"   Columns: {list(table.columns)}")
        
        # Show top 5 contributors with more detail
        print(f"\n🏆 TOP 5 CONTRIBUTORS:")
        for i, (_, row) in enumerate(table.head(5).iterrows()):
            status_icon = "🔴" if "Problem" in row['Status'] else "🟢" if "Strength" in row['Status'] else "⚪"
            print(f"     {i+1}. {status_icon} {row['Category']}: {row['Net Impact_pp']}")
            print(f"        Rate: {row['Region Rate %']} vs {row['Baseline Rate %']}")
            print(f"        Share: {row['Region Share %']} vs {row['Baseline Share %']}")
    else:
        print("   ❌ No LATAM decision found")
    
    # Print paradox detection results
    if hasattr(result, 'paradox_report') and result.paradox_report.paradox_detected:
        print(f"\n🎭 SIMPSON'S PARADOX DETECTED!")
        print(f"   Affected regions: {result.paradox_report.affected_regions}")
        print(f"   Business impact: {result.paradox_report.business_impact.value}")
        # Paradox type removed - paradox_detected boolean is sufficient
        
        # Show Simpson's columns in decomposition for verification
        decomp_cols = result.decomposition_df.columns.tolist()
        simpson_cols = [col for col in decomp_cols if 'donated' in col or 'acquired' in col]
        print(f"   Simpson's columns in decomposition: {simpson_cols}")
        
        # Show sample of affected region's data
        if result.paradox_report.affected_regions:
            region = result.paradox_report.affected_regions[0]
            region_data = result.decomposition_df[result.decomposition_df['region'] == region]
            print(f"   {region} decomposition columns: {region_data.columns.tolist()}")
    else:
        print(f"\n✅ No Simpson's paradox detected")
    
    # Generate and save figures
    print(f"\n📊 GENERATING FIGURES...")
    output_path = Path("output")
    try:
        # Create rates and mix panel figure for LATAM
        fig_result = plot_rates_and_mix_panel(
            result=result,
            region="LATAM",
            top_n=16,
            metric_name="Conversion Rate",
            category_name="Category"
        )
        
        # Handle different return types (could be tuple or single figure)
        if isinstance(fig_result, tuple):
            fig = fig_result[0] if fig_result[0] is not None else fig_result[1]
        else:
            fig = fig_result
            
        if fig is not None:
            fig_path = output_path / f"{analysis_name}_LATAM_rates_mix_panel.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Free memory
            print(f"✅ Saved figure: {fig_path}")
        else:
            print(f"⚠️  No figure generated")
    except Exception as e:
        print(f"⚠️  Figure generation failed: {e}")
    
    # Generate quadrant chart
    try:
        print(f"📊 GENERATING QUADRANT CHART...")
        quad_fig = quadrant_prompts(result, region="LATAM", annotate_top_n=5)
        
        if quad_fig is not None:
            quad_path = output_path / f"{analysis_name}_LATAM_quadrant_chart.png"
            quad_fig.savefig(quad_path, dpi=300, bbox_inches='tight')
            plt.close(quad_fig)  # Free memory
            print(f"✅ Saved quadrant chart: {quad_path}")
        else:
            print(f"⚠️  No quadrant chart generated")
    except Exception as e:
        print(f"⚠️  Quadrant chart generation failed: {e}")
    
    # Save math walkthrough tracebacks
    try:
        print(f"\n📋 SAVING MATH WALKTHROUGH...")
        # Get tracebacks from the slide payload for LATAM region
        slide_spec, payload = result.present_executive_pack_for_slides("LATAM")
        tracebacks = payload.get('tracebacks', {})
        
        if tracebacks.get('formulas_walkthrough'):
            walkthrough_df = pd.DataFrame(tracebacks['formulas_walkthrough'])
            walkthrough_file = output_path / f"{analysis_name}_LATAM_math_walkthrough.csv"
            walkthrough_df.to_csv(walkthrough_file, index=False)
            print(f"✅ Saved math walkthrough: {walkthrough_file}")
            
        if tracebacks.get('formulas_summary'):
            summary_df = pd.DataFrame(tracebacks['formulas_summary'])
            summary_file = output_path / f"{analysis_name}_LATAM_math_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Saved math summary: {summary_file}")
            
        # Print key summary info
        if tracebacks.get('formulas_summary'):
            print(f"\n📊 MATH SUMMARY HIGHLIGHTS:")
            summary_data = {row['Component']: row['Value'] for row in tracebacks['formulas_summary']}
            print(f"   Focus region: {summary_data.get('focus_region', 'N/A')}")
            print(f"   Total gap: {summary_data.get('total_gap', 'N/A')}")
            print(f"   Simpson's detected: {summary_data.get('simpson_detected', 'N/A')}")
            print(f"   Business conclusion: {summary_data.get('business_conclusion', 'N/A')}")
            print(f"   Ranking method: {summary_data.get('ranking_method', 'N/A')}")
            
    except Exception as e:
        print(f"⚠️  Math walkthrough save failed: {e}")
    
    # Save all outputs
    print(f"\n💾 SAVING OUTPUTS...")
    save_analysis_outputs(result, analysis_name, "output")
    
    return result


def main():
    """Main function to run all analyses."""
    print("🧪 OAXACA-BLINDER REAL DATA TESTING")
    print("=" * 60)
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Run product analysis (using vertical.csv - the data with Simpson's paradox)
    product_result = run_analysis_and_save(
        csv_file="tests/vertical.csv",
        analysis_name="product"
    )
    
    # Run vertical analysis (using product.csv - the standard case)
    vertical_result = run_analysis_and_save(
        csv_file="tests/product.csv", 
        analysis_name="vertical"
    )
    
    # Run vertical2 analysis (new dataset)
    vertical2_result = run_analysis_and_save(
        csv_file="tests/vertical2.csv", 
        analysis_name="vertical2"
    )
    
    # Run vertical3 analysis (new dataset)
    vertical3_result = run_analysis_and_save(
        csv_file="tests/vertical3.csv", 
        analysis_name="vertical3"
    )
    
    # Run product3 analysis (new dataset with potential issue)
    product3_result = run_analysis_and_save(
        csv_file="tests/product3.csv", 
        analysis_name="product3"
    )
    
    print("\n" + "=" * 60)
    print("✨ ANALYSIS COMPLETE!")
    print("📁 All outputs saved to output/ directory")
    print("🔍 Check the CSV files and narratives above for impact distribution issues")


if __name__ == "__main__":
    main()
