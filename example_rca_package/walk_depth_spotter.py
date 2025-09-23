#!/usr/bin/env python3
"""
Walk-through Example: Depth Spotter with Minimal Patches Traceback

This script demonstrates the minimal patches traceback system for depth spotter,
showing exactly what calculations happen under the hood for sub-region analysis.

The traceback system provides two detailed tables:
1. formulas_walkthrough: Per-slice calculations with explicit formulas
2. formulas_summary: Analysis methodology and key parameters

Run this script to see complete transparency into the depth analysis process.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to Python path to import rca_package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rca_package.depth_spotter import (
    analyze_region_depth, rate_contrib, additive_contrib, create_synthetic_data
)


def create_demo_data():
    """Create focused demo data for walkthrough."""
    # Create smaller, more focused dataset for clear demonstration
    rows = []
    
    # North America slices with clear contribution patterns
    rows.extend([
        {"slice": "NA_high_traffic", "region": "North America", "visits": 10000, "conversions": 800, "revenue": 40000},
        {"slice": "NA_medium_traffic", "region": "North America", "visits": 8000, "conversions": 400, "revenue": 20000},
        {"slice": "NA_low_traffic", "region": "North America", "visits": 5000, "conversions": 100, "revenue": 5000},
    ])
    
    # Rest of world for comparison
    rows.extend([
        {"slice": "ROW_slice_1", "region": "Europe", "visits": 8000, "conversions": 640, "revenue": 32000},
        {"slice": "ROW_slice_2", "region": "Asia", "visits": 7000, "conversions": 560, "revenue": 28000},
        {"slice": "ROW_slice_3", "region": "Latin America", "visits": 6000, "conversions": 480, "revenue": 24000},
    ])
    
    df = pd.DataFrame(rows)
    df['conversion_rate'] = df['conversions'] / df['visits']
    df['avg_order_value'] = df['revenue'] / df['conversions']
    
    return df


def demo_rate_traceback_minimal_patches():
    """Demonstrate minimal patches traceback for rate metrics."""
    print("=" * 80)
    print("RATE METRIC TRACEBACK - MINIMAL PATCHES")
    print("=" * 80)
    
    df = create_demo_data()
    print("📊 Sample Data:")
    print(df[['slice', 'region', 'visits', 'conversions', 'conversion_rate']].round(3))
    print()
    
    # Filter for North America (focal region)
    na_df = df[df['region'] == 'North America'].copy()
    row_df = df[df['region'] != 'North America']
    
    # Calculate rest-of-world totals
    row_conversions = row_df['conversions'].sum()
    row_visits = row_df['visits'].sum()
    
    print(f"🔍 Analyzing North America Conversion Rate (focal_region='North America')")
    print(f"Rest-of-World: {row_conversions:,}/{row_visits:,} = {row_conversions/row_visits:.1%}")
    print()
    
    # Run rate contribution analysis with focal region
    contrib_df, delta, row_rate = rate_contrib(
        df_slice=na_df,
        row_numerator=row_conversions,
        row_denominator=row_visits,
        denominator_col='visits',
        numerator_col='conversions',
        metric_name='conversion_rate',
        higher_is_better=True,
        focal_region='North America'  # Minimal patches addition
    )
    
    # Access traceback from DataFrame attrs
    if hasattr(contrib_df, 'attrs') and 'tracebacks' in contrib_df.attrs:
        tracebacks = contrib_df.attrs['tracebacks']
        
        print("📋 FORMULAS WALKTHROUGH:")
        print("-" * 60)
        walkthrough_df = pd.DataFrame(tracebacks['formulas_walkthrough'])
        
        # Show key columns for clarity
        key_cols = ['slice', 'visits', 'conversions', 'expected_value', 'diff', 'contribution', 'score_value']
        if all(col in walkthrough_df.columns for col in key_cols):
            display_df = walkthrough_df[key_cols].round(3)
            print(display_df.to_string(index=False))
        else:
            print(walkthrough_df.round(3).to_string(index=False))
        print()
        
        print("📊 FORMULAS SUMMARY:")
        print("-" * 60)
        summary_df = pd.DataFrame(tracebacks['formulas_summary'])
        print(summary_df.to_string(index=False))
        print()
    else:
        print("❌ No tracebacks found in DataFrame attrs")
    
    print("🎯 FINAL RESULTS:")
    result_cols = ['slice', 'conversion_rate', 'contribution', 'coverage', 'score']
    if all(col in contrib_df.columns for col in result_cols):
        result_df = contrib_df[result_cols].round(3)
        print(result_df.to_string(index=False))
    print(f"\nTotal Delta: {delta:.1f} conversions")
    print()


def demo_additive_traceback_minimal_patches():
    """Demonstrate minimal patches traceback for additive metrics."""
    print("=" * 80)
    print("ADDITIVE METRIC TRACEBACK - MINIMAL PATCHES")
    print("=" * 80)
    
    df = create_demo_data()
    print("📊 Sample Data:")
    print(df[['slice', 'region', 'revenue']].round(0))
    print()
    
    # Filter for North America (focal region)
    na_df = df[df['region'] == 'North America'].copy()
    row_df = df[df['region'] != 'North America']
    
    # Calculate rest-of-world total
    row_total = row_df['revenue'].sum()
    
    print(f"🔍 Analyzing North America Revenue (focal_region='North America')")
    print(f"Rest-of-World Total: ${row_total:,.0f}")
    print()
    
    # Run additive contribution analysis with focal region
    contrib_df, delta = additive_contrib(
        df_slice=na_df,
        row_total=row_total,
        metric_col='revenue',
        higher_is_better=True,
        focal_region='North America'  # Minimal patches addition
    )
    
    # Access traceback from DataFrame attrs
    if hasattr(contrib_df, 'attrs') and 'tracebacks' in contrib_df.attrs:
        tracebacks = contrib_df.attrs['tracebacks']
        
        print("📋 FORMULAS WALKTHROUGH:")
        print("-" * 60)
        walkthrough_df = pd.DataFrame(tracebacks['formulas_walkthrough'])
        
        # Show key columns for clarity
        key_cols = ['slice', 'revenue', 'coverage_value', 'expected_value', 'diff', 'contribution', 'score_value']
        if all(col in walkthrough_df.columns for col in key_cols):
            display_df = walkthrough_df[key_cols].round(3)
            print(display_df.to_string(index=False))
        else:
            print(walkthrough_df.round(3).to_string(index=False))
        print()
        
        print("📊 FORMULAS SUMMARY:")
        print("-" * 60)
        summary_df = pd.DataFrame(tracebacks['formulas_summary'])
        print(summary_df.to_string(index=False))
        print()
    else:
        print("❌ No tracebacks found in DataFrame attrs")
    
    print("🎯 FINAL RESULTS:")
    result_cols = ['slice', 'revenue', 'contribution', 'coverage', 'score']
    if all(col in contrib_df.columns for col in result_cols):
        result_df = contrib_df[result_cols].round(3)
        print(result_df.to_string(index=False))
    print(f"\nTotal Delta: ${delta:,.0f}")
    print()


def demo_unified_system_with_tracebacks():
    """Demonstrate unified depth analysis system with minimal patches tracebacks."""
    print("=" * 80)
    print("UNIFIED DEPTH ANALYSIS - MINIMAL PATCHES TRACEBACK")
    print("=" * 80)
    
    # Use comprehensive synthetic dataset
    df = create_synthetic_data()
    print(f"📊 Using synthetic data: {len(df)} slices across {df['region'].nunique()} regions")
    print()
    
    # Configuration
    config = {
        'metrics': {
            'conversion_rate_pct': {
                'type': 'rate',
                'numerator_col': 'conversions',
                'denominator_col': 'visits',
            },
            'revenue': {
                'type': 'additive',
                'metric_col': 'revenue',
            }
        }
    }
    
    # Anomaly map
    anomaly_map = {
        'conversion_rate_pct': {
            'anomalous_region': 'North America',
            'higher_is_better': True
        },
        'revenue': {
            'anomalous_region': 'North America',
            'higher_is_better': True
        }
    }
    
    print("🔄 Running Unified Analysis with Focal Region Tracebacks")
    print()
    
    # Run unified analysis
    results = analyze_region_depth(df, config, anomaly_map)
    
    # Show traceback integration for each metric
    for metric_name, metric_results in results.items():
        print(f"📊 {metric_name.upper()} TRACEBACK INTEGRATION:")
        print("-" * 50)
        
        payload = metric_results['payload']
        if 'tracebacks' in payload and payload['tracebacks']:
            tracebacks = payload['tracebacks']
            
            # Show key summary components
            if 'formulas_summary' in tracebacks:
                summary_df = pd.DataFrame(tracebacks['formulas_summary'])
                key_components = summary_df[summary_df['Component'].isin([
                    'focal_region', 'metric_type', 'Δ (total gap)', 'mode'
                ])]
                print("Key Components:")
                for _, row in key_components.iterrows():
                    print(f"  {row['Component']}: {row['Value']}")
                print()
            
            # Show sample slice calculations
            if 'formulas_walkthrough' in tracebacks:
                walkthrough_df = pd.DataFrame(tracebacks['formulas_walkthrough'])
                if not walkthrough_df.empty:
                    print("Sample Slice Calculations (Top 2):")
                    
                    if metric_name == 'conversion_rate_pct':
                        sample_cols = ['slice', 'visits', 'conversions', 'expected_value', 'contribution']
                    else:
                        sample_cols = ['slice', 'revenue', 'coverage_value', 'expected_value', 'contribution']
                    
                    if all(col in walkthrough_df.columns for col in sample_cols):
                        sample_df = walkthrough_df[sample_cols].head(2)
                        print(sample_df.round(3).to_string(index=False))
                    print()
        else:
            print("  No traceback data available")
            print()


if __name__ == "__main__":
    print("🚀 DEPTH SPOTTER MINIMAL PATCHES TRACEBACK WALKTHROUGH")
    print("Demonstrates the minimal patches system for transparent depth analysis\n")
    
    # Run all demos
    demo_rate_traceback_minimal_patches()
    demo_additive_traceback_minimal_patches()
    demo_unified_system_with_tracebacks()
    
    print("=" * 80)
    print("✅ MINIMAL PATCHES TRACEBACK COMPLETE")
    print("=" * 80)
    print("The minimal patches system provides:")
    print("  1. Two helper functions for building traceback tables")
    print("  2. Focal region awareness in all calculations")
    print("  3. DataFrame attrs attachment with JSON-serializable data")
    print("  4. Unified payload integration with tracebacks field")
    print("  5. Complete backward compatibility - no behavioral changes")
    print()
    print("Key benefits:")
    print("  • Explicit formulas for every calculation step")
    print("  • Focal region properly identified in all outputs")
    print("  • Method selection transparency (two_sided vs standard)")
    print("  • Sign adjustment rules clearly documented")
    print("  • Score formula breakdown with actual values")
    print("  • JSON-ready format for downstream consumption")