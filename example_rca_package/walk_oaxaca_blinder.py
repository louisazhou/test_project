#!/usr/bin/env python3
"""
Walk-through Example: Oaxaca-Blinder Decomposition with Math Transparency

This script demonstrates the comprehensive math walkthrough system in the Oaxaca-Blinder
decomposition, showing exactly what calculations happen under the hood for complete
transparency and auditability.

The walkthrough system provides detailed tables:
1. math_walkthrough: Per-category calculations with explicit formulas and step-by-step rebalancing
2. math_summary: Analysis methodology, thresholds, and business conclusions
3. supporting_table: Business-ready evidence table
4. narrative: Human-readable explanation of findings

Run this script to see complete transparency into the Oaxaca-Blinder analysis process.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import rca_package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rca_package.oaxaca_blinder import (
    analyze_oaxaca_metrics, create_oaxaca_synthetic_data
)
from rca_package.yaml_processor import load_config


def create_demo_data():
    """Create realistic demo data for Oaxaca-Blinder walkthrough."""
    print("🎯 Creating synthetic data for Oaxaca-Blinder demonstration...")
    
    # Create data that will trigger Simpson's Paradox for educational purposes
    # This mimics real-world scenarios where aggregate performance contradicts segment performance
    
    # Scenario: North America underperforms in conversion rate despite having good individual segments
    # This happens when they have poor allocation (too much share in low-performing segments)
    
    conversion_data = create_oaxaca_synthetic_data(
        target_region="North America",
        target_gap_pp=-3.0  # North America underperforms by 3.0pp
    )
    
    print(f"✅ Created synthetic dataset with {len(conversion_data)} rows")
    print(f"📊 Regions: {conversion_data['region'].unique().tolist()}")
    print(f"📊 Columns: {conversion_data.columns.tolist()}")
    
    # Check if we have product or vertical as category columns
    category_cols = [col for col in conversion_data.columns if col in ['product', 'vertical', 'category']]
    if category_cols:
        for col in category_cols:
            print(f"📊 {col.title()}s: {conversion_data[col].nunique()} unique values")
    else:
        print("📊 Categories: Column structure detected")
    
    return conversion_data


def demonstrate_basic_oaxaca_analysis():
    """Demonstrate basic Oaxaca-Blinder analysis with math walkthrough."""
    
    print("\n" + "="*80)
    print("🔍 BASIC OAXACA-BLINDER ANALYSIS WITH MATH WALKTHROUGH")
    print("="*80)
    
    # Create demo data
    data = create_demo_data()
    
    # Load configuration (following example_analysis.py pattern)
    config_path = Path(__file__).parent.parent / "configs" / "config_oaxaca.yaml"
    oaxaca_config = load_config(str(config_path))
    
    print(f"\n📋 Loading configuration from: {config_path}")
    
    # Create a synthetic anomaly map (following example_analysis.py pattern)
    # This simulates detecting an anomaly in conversion rate for North America
    metric_anomaly_map = {
        'Conversion Rate': {
            'anomalous_region': 'North America',
            'direction': 'underperforms',
            'magnitude': '3.0pp',
            'metric_val': 0.084,  # 8.4% from our synthetic data
            'global_val': 0.114,  # 11.4% from our synthetic data
            'higher_is_better': True
        }
    }
    
    print(f"📋 Simulated anomaly detection: {metric_anomaly_map['Conversion Rate']}")
    
    # Run the analysis exactly as in example_analysis.py
    print("\n🚀 Running Oaxaca-Blinder analysis...")
    
    results = analyze_oaxaca_metrics(
        config=oaxaca_config,
        metric_anomaly_map=metric_anomaly_map,
        data_df=data
    )
    
    print("✅ Analysis completed!")
    print(f"📊 Generated results for metrics: {list(results.keys())}")
    
    return results


def display_math_walkthrough(results, region="North America", metric_name="Conversion Rate"):
    """Display the detailed math walkthrough for transparency."""
    
    print("\n" + "="*80)
    print(f"🧮 MATH WALKTHROUGH FOR {region.upper()} - {metric_name}")
    print("="*80)
    
    # Get the results for the specified metric
    try:
        if metric_name not in results:
            print(f"❌ No results found for metric: {metric_name}")
            return
            
        metric_results = results[metric_name]
        
        # Get payload from the primary slide
        payload = metric_results.get('payload', {})
        
        # Display the math walkthrough table
        if 'tracebacks' in payload and 'formulas_walkthrough' in payload['tracebacks']:
            walkthrough_data = payload['tracebacks']['formulas_walkthrough']
            
            # Check if it's a DataFrame or needs to be converted
            if isinstance(walkthrough_data, list):
                # Convert list of dicts to DataFrame if needed
                if walkthrough_data and isinstance(walkthrough_data[0], dict):
                    walkthrough_df = pd.DataFrame(walkthrough_data)
                else:
                    print("❌ Walkthrough data format not supported")
                    return
            else:
                walkthrough_df = walkthrough_data
            
            print(f"\n📊 STEP-BY-STEP CALCULATIONS ({len(walkthrough_df)} categories):")
            print("-" * 80)
            
            # Display key columns for transparency
            key_columns = [
                'category', 'E_c_formula', 'E_c_value', 'M_c_formula', 'M_c_value', 
                'I₀', 'E (pp)', 'M₀ (pp)', 'I₀ (pp)', 'pipeline_formula', 'I_final (pp)'
            ]
            
            available_columns = [col for col in key_columns if col in walkthrough_df.columns]
            display_df = walkthrough_df[available_columns]
            
            # Show first few categories in detail
            print("First 5 categories (detailed view):")
            for i, (_, row) in enumerate(display_df.head(5).iterrows()):
                print(f"\n{i+1}. {row['category']}:")
                if 'E_c_formula' in row:
                    print(f"   Rate Effect: {row['E_c_formula']}")
                    print(f"   Rate Value: {row['E_c_value']:.4f} ({row.get('E (pp)', 0):+.1f}pp)")
                if 'M_c_formula' in row:
                    print(f"   Mix Effect:  {row['M_c_formula']}")
                    print(f"   Mix Value:  {row['M_c_value']:.4f} ({row.get('M₀ (pp)', 0):+.1f}pp)")
                if 'pipeline_formula' in row:
                    print(f"   Pipeline:   {row['pipeline_formula']}")
                    print(f"   Final Impact: {row.get('I_final (pp)', 0):+.1f}pp")
            
            # Show totals row if available
            totals_row = walkthrough_df[walkthrough_df['category'] == '=== TOTALS ===']
            if not totals_row.empty:
                print(f"\n📈 CONSERVATION CHECK (Totals):")
                row = totals_row.iloc[0]
                if 'E' in row:
                    print(f"   Total Rate Effect: {row['E']:.4f} ({row.get('E (pp)', 0):+.1f}pp)")
                if 'M₀' in row:
                    print(f"   Total Mix Effect:  {row['M₀']:.4f} ({row.get('M₀ (pp)', 0):+.1f}pp)")
                if 'I_final' in row:
                    print(f"   Total Net Impact:  {row['I_final']:.4f} ({row.get('I_final (pp)', 0):+.1f}pp)")
            
        else:
            print("❌ Math walkthrough not available in results")
            
    except Exception as e:
        print(f"❌ Error displaying math walkthrough: {e}")
        import traceback
        traceback.print_exc()


def display_business_summary(results, region="North America", metric_name="Conversion Rate"):
    """Display the business summary and methodology."""
    
    print("\n" + "="*80)
    print(f"📋 BUSINESS SUMMARY FOR {region.upper()} - {metric_name}")
    print("="*80)
    
    try:
        if metric_name not in results:
            print(f"❌ No results found for metric: {metric_name}")
            return
            
        metric_results = results[metric_name]
        
        # Get slide spec and payload from the primary slide
        slides = metric_results.get('slides', {})
        payload = metric_results.get('payload', {})
        
        # Get the primary slide (usually 'oaxaca_primary')
        primary_slide_key = next(iter(slides.keys())) if slides else None
        slide_spec = slides.get(primary_slide_key, {}).get('slide_info', {}) if primary_slide_key else {}
        
        # Display narrative
        if 'narrative' in payload:
            print(f"\n📝 BUSINESS NARRATIVE:")
            print("-" * 40)
            print(f"   {payload['narrative']}")
        elif 'template_text' in slide_spec:
            print(f"\n📝 BUSINESS NARRATIVE:")
            print("-" * 40)
            print(f"   {slide_spec['template_text']}")
        
        # Display methodology summary
        if 'tracebacks' in payload and 'formulas_summary' in payload['tracebacks']:
            summary_data = payload['tracebacks']['formulas_summary']
            
            # Check if it's a DataFrame or needs to be converted
            if isinstance(summary_data, list):
                # Convert list of dicts to DataFrame if needed
                if summary_data and isinstance(summary_data[0], dict):
                    summary_df = pd.DataFrame(summary_data)
                else:
                    print("❌ Summary data format not supported")
                    return
            else:
                summary_df = summary_data
            
            print(f"\n🔧 ANALYSIS METHODOLOGY:")
            print("-" * 40)
            
            # Show key methodology components
            key_components = [
                'total_gap', 'business_conclusion', 'simpson_detected', 
                'ranking_method', 'within_group_priority'
            ]
            
            for component in key_components:
                component_row = summary_df[summary_df['Component'] == component]
                if not component_row.empty:
                    row = component_row.iloc[0]
                    print(f"   {component}: {row.get('Value', 'N/A')}")
                    if component in ['ranking_method', 'within_group_priority']:
                        formula = row.get('Formula', '')
                        if formula and len(formula) < 100:  # Only show short formulas
                            print(f"      → {formula}")
        
        # Display supporting evidence table
        if 'supporting_evidence' in payload:
            evidence_df = payload['supporting_evidence']
            print(f"\n📊 SUPPORTING EVIDENCE (Top 5 categories by impact):")
            print("-" * 60)
            
            # Show key columns
            display_cols = ['Category', 'Region Rate %', 'Region Share %', 'Net Impact_pp', 'Band']
            available_cols = [col for col in display_cols if col in evidence_df.columns]
            
            if available_cols:
                top_evidence = evidence_df[available_cols].head(5)
                for _, row in top_evidence.iterrows():
                    impact = row.get('Net Impact_pp', 'N/A')
                    rate = row.get('Region Rate %', 'N/A')
                    share = row.get('Region Share %', 'N/A')
                    band = row.get('Band', 'N/A')
                    print(f"   {row['Category'][:25]:25} | {rate:>8} | {share:>8} | {impact:>10} | {band}")
        
    except Exception as e:
        print(f"❌ Error displaying business summary: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_simpson_paradox_detection(results, metric_name="Conversion Rate"):
    """Demonstrate Simpson's Paradox detection and explanation."""
    
    print("\n" + "="*80)
    print("🔄 SIMPSON'S PARADOX DETECTION")
    print("="*80)
    
    try:
        if metric_name not in results:
            print(f"❌ No results found for metric: {metric_name}")
            return
            
        metric_results = results[metric_name]
        payload = metric_results.get('payload', {})
        
        # Check if Simpson's paradox was detected in the payload
        paradox_detected = payload.get('simpson_detected', False)
        
        if paradox_detected:
            print("✅ SIMPSON'S PARADOX DETECTED!")
            
            # Get additional details from tracebacks if available
            if 'tracebacks' in payload and 'formulas_summary' in payload['tracebacks']:
                summary_data = payload['tracebacks']['formulas_summary']
                
                # Convert to DataFrame if needed
                if isinstance(summary_data, list):
                    if summary_data and isinstance(summary_data[0], dict):
                        summary_df = pd.DataFrame(summary_data)
                        simpson_row = summary_df[summary_df['Component'] == 'simpson_detected']
                        if not simpson_row.empty:
                            print(f"   Detection details: {simpson_row.iloc[0].get('Value', 'Yes')}")
                else:
                    summary_df = summary_data
                    simpson_row = summary_df[summary_df['Component'] == 'simpson_detected']
                    if not simpson_row.empty:
                        print(f"   Detection details: {simpson_row.iloc[0].get('Value', 'Yes')}")
            
            print(f"\n📚 WHAT THIS MEANS:")
            print(f"   Simpson's Paradox occurs when aggregate performance contradicts")
            print(f"   segment-level performance due to composition effects.")
            print(f"   ")
            print(f"   In this case: The region may have good individual category performance")
            print(f"   but poor overall performance due to suboptimal allocation of resources")
            print(f"   across categories (too much share in underperforming segments).")
            
        else:
            print("ℹ️  No Simpson's Paradox detected in this dataset")
            print("   This means segment-level performance aligns with aggregate performance")
            
    except Exception as e:
        print(f"❌ Error checking Simpson's paradox: {e}")
        import traceback
        traceback.print_exc()


def save_walkthrough_files(results, output_dir="output", metric_name="Conversion Rate"):
    """Save the math walkthrough files for external review."""
    
    print("\n" + "="*80)
    print("💾 SAVING MATH WALKTHROUGH FILES")
    print("="*80)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if metric_name not in results:
            print(f"❌ No results found for metric: {metric_name}")
            return
            
        metric_results = results[metric_name]
        payload = metric_results.get('payload', {})
        
        # Save walkthrough files
        metric_safe = metric_name.replace(' ', '_').replace('/', '_')
        
        # Save math walkthrough
        if 'tracebacks' in payload and 'formulas_walkthrough' in payload['tracebacks']:
            walkthrough_data = payload['tracebacks']['formulas_walkthrough']
            walkthrough_path = f"{output_dir}/{metric_safe}_math_walkthrough.csv"
            
            # Convert to DataFrame if needed
            if isinstance(walkthrough_data, list):
                if walkthrough_data and isinstance(walkthrough_data[0], dict):
                    pd.DataFrame(walkthrough_data).to_csv(walkthrough_path, index=False)
                    print(f"✅ Saved math walkthrough: {walkthrough_path}")
            else:
                walkthrough_data.to_csv(walkthrough_path, index=False)
                print(f"✅ Saved math walkthrough: {walkthrough_path}")
        
        # Save methodology summary
        if 'tracebacks' in payload and 'formulas_summary' in payload['tracebacks']:
            summary_data = payload['tracebacks']['formulas_summary']
            summary_path = f"{output_dir}/{metric_safe}_math_summary.csv"
            
            # Convert to DataFrame if needed
            if isinstance(summary_data, list):
                if summary_data and isinstance(summary_data[0], dict):
                    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
                    print(f"✅ Saved methodology summary: {summary_path}")
            else:
                summary_data.to_csv(summary_path, index=False)
                print(f"✅ Saved methodology summary: {summary_path}")
        
        # Save supporting evidence (if available in payload)
        if 'supporting_evidence' in payload:
            evidence_path = f"{output_dir}/{metric_safe}_supporting_evidence.csv"
            payload['supporting_evidence'].to_csv(evidence_path, index=False)
            print(f"✅ Saved supporting evidence: {evidence_path}")
        
        # Also save slide data if available
        slides = metric_results.get('slides', {})
        for slide_name, slide_data in slides.items():
            slide_info = slide_data.get('slide_info', {})
            dfs = slide_info.get('dfs', {})
            for df_name, df in dfs.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df_path = f"{output_dir}/{metric_safe}_{slide_name}_{df_name}.csv"
                    df.to_csv(df_path, index=False)
                    print(f"✅ Saved slide data: {df_path}")
                
    except Exception as e:
        print(f"❌ Error saving walkthrough files: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main walkthrough demonstration."""
    
    print("🎯 OAXACA-BLINDER DECOMPOSITION WALKTHROUGH")
    print("=" * 60)
    print("This script demonstrates the complete math transparency")
    print("available in the Oaxaca-Blinder decomposition analysis.")
    print()
    
    # Step 1: Run basic analysis
    results = demonstrate_basic_oaxaca_analysis()
    
    if not results:
        print("❌ No results generated - exiting")
        return
    
    metric_name = "Conversion Rate"  # The metric we're analyzing
    region = "North America"  # The focus region
    
    # Step 2: Show detailed math walkthrough
    display_math_walkthrough(results, region=region, metric_name=metric_name)
    
    # Step 3: Show business summary
    display_business_summary(results, region=region, metric_name=metric_name)
    
    # Step 4: Demonstrate Simpson's paradox detection
    demonstrate_simpson_paradox_detection(results, metric_name=metric_name)
    
    # Step 5: Save files for external review
    save_walkthrough_files(results, metric_name=metric_name)
    
    print("\n" + "="*80)
    print("✨ WALKTHROUGH COMPLETE!")
    print("="*80)
    print("Key takeaways:")
    print("• Math walkthrough provides complete transparency into calculations")
    print("• Each category shows rate effect (E_c) and mix effect (M_c) formulas")
    print("• Pipeline shows step-by-step rebalancing: I₀ → pool → project → I_final")
    print("• Conservation is verified: sum of parts equals total gap")
    print("• Simpson's paradox detection explains aggregate vs segment contradictions")
    print("• All calculations are saved as CSV files for external audit")
    print()
    print("📁 Check the 'output/' directory for detailed CSV files!")


if __name__ == "__main__":
    main()
