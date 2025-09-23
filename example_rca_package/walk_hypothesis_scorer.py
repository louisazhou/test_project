#!/usr/bin/env python3
"""
Walk-through Example: Hypothesis Scorer with Complete Traceback

This script demonstrates the comprehensive traceback system in the hypothesis scorer,
showing exactly what calculations happen under the hood for transparency and debugging.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to Python path to import rca_package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rca_package.hypothesis_scorer import (
    sign_based_score_hypothesis, create_consolidated_results_dataframe
)


def create_demo_data():
    """Create realistic demo data for walkthrough."""
    np.random.seed(42)
    regions = ["Global", "North America", "Europe", "Asia", "Latin America"]
    
    # Create realistic business scenario: conversion rate analysis
    data = {
        'conversion_rate_pct': [0.12, 0.11, 0.13, 0.14, 0.08],  # Latin America: underperforming
        'page_load_time': [2.1, 2.3, 1.9, 1.8, 3.2],          # Higher is bad (Latin America: slow)
        'mobile_traffic_pct': [0.65, 0.60, 0.70, 0.75, 0.45],   # Lower mobile usage in LatAm
    }
    
    return pd.DataFrame(data, index=regions)


def demo_single_hypothesis_traceback():
    """Demonstrate detailed traceback for a single hypothesis."""
    print("=" * 80)
    print("SINGLE HYPOTHESIS TRACEBACK DEMO")
    print("=" * 80)
    
    df = create_demo_data()
    print("📊 Sample Data:")
    print(df.round(3))
    print()
    
    # Create mock anomaly info for Latin America
    metric_anomaly_info = {
        'anomalous_region': 'Latin America',
        'metric_val': 0.08,
        'global_val': 0.12,
        'direction': 'lower',
        'higher_is_better': True
    }
    
    # Test hypothesis: page load time (expecting opposite relationship)
    temp_df = df[['conversion_rate_pct', 'page_load_time']]
    
    print("🔍 Testing Hypothesis: Page Load Time")
    print(f"Expected Direction: opposite (higher load time → lower conversion)")
    print(f"Anomalous Region: {metric_anomaly_info['anomalous_region']}")
    print()
    
    # Score the hypothesis with traceback
    result = sign_based_score_hypothesis(
        df=temp_df,
        metric_anomaly_info=metric_anomaly_info,
        expected_direction='opposite'
    )
    
    print("📋 FORMULAS WALKTHROUGH (Per-Region Calculations):")
    print("-" * 60)
    walkthrough_df = pd.DataFrame(result['traceback']['formulas_walkthrough'])
    print(walkthrough_df.to_string(index=False))
    print()
    
    print("📊 FORMULAS SUMMARY (Score Component Calculations):")
    print("-" * 60)
    summary_df = pd.DataFrame(result['traceback']['formulas_summary'])
    print(summary_df.to_string(index=False))
    print()
    
    print("🎯 FINAL RESULTS:")
    print(f"  Final Score: {result['scores']['final_score']:.3f}")
    print(f"  Explains Anomaly: {result['scores']['explains']}")
    print(f"  Sign Agreement: {result['scores']['sign_agreement']:.3f}")
    print(f"  Explained Ratio: {result['scores']['explained_ratio']:.3f}")
    print(f"  Focal Region Agrees: {result['scores']['focal_region_agrees']}")
    if result['scores']['failure_reason']:
        print(f"  Failure Reason: {result['scores']['failure_reason']}")
    print()


if __name__ == "__main__":
    print("🚀 HYPOTHESIS SCORER TRACEBACK WALKTHROUGH")
    print("This demo shows the complete transparency of the scoring system\n")
    
    demo_single_hypothesis_traceback()
    
    print("=" * 80)
    print("✅ TRACEBACK WALKTHROUGH COMPLETE")
    print("=" * 80)
    print("The traceback system provides complete transparency into:")
    print("  1. Per-region delta calculations with explicit formulas")
    print("  2. Sign agreement computation across all regions")
    print("  3. Standard deviation calculations for z-score normalization") 
    print("  4. Explained ratio computation for the focal region")
    print("  5. Penalty application when focal region contradicts hypothesis")
    print("  6. Final score assembly from weighted components")