#!/usr/bin/env python3
"""
Diagnostic script to quantify Oaxaca-Blinder impact distribution issues.

This script analyzes the supporting tables to check:
1. Sign coherence: better performers should have positive impacts
2. Uniformity index: how many rows show artificial ±0.5pp plateaus  
3. Share monotonicity: larger share should correlate with larger impact magnitude
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_sign_coherence(df, eta=0.005, delta=0.0005):
    """
    Check sign coherence: better performers should have positive impacts.
    
    Args:
        df: DataFrame with supporting table data
        eta: Near-tie threshold for rate differences (0.5pp)
        delta: Minimum impact threshold for violations (0.05pp)
    
    Returns:
        dict with coherence metrics
    """
    # Calculate rate differences
    df['rate_diff'] = df['Region Rate %'].str.rstrip('%').astype(float) / 100 - \
                      df['Baseline Rate %'].str.rstrip('%').astype(float) / 100
    
    # Parse net impact
    df['net_impact'] = df['Net Impact_pp'].str.rstrip('pp').astype(float) / 100
    
    # Find violations
    better_but_negative = (df['rate_diff'] > eta) & (df['net_impact'] < -delta)
    worse_but_positive = (df['rate_diff'] < -eta) & (df['net_impact'] > delta)
    
    total_rows = len(df)
    violations = better_but_negative.sum() + worse_but_positive.sum()
    
    return {
        'total_rows': total_rows,
        'better_but_negative': better_but_negative.sum(),
        'worse_but_positive': worse_but_positive.sum(),
        'total_violations': violations,
        'violation_rate': violations / total_rows if total_rows > 0 else 0,
        'violating_categories': df[better_but_negative | worse_but_positive]['Category'].tolist()
    }

def analyze_uniformity_index(df, epsilon=0.005):
    """
    Check uniformity: how many rows show artificial ±ε plateaus.
    
    Args:
        df: DataFrame with supporting table data
        epsilon: Projection threshold (0.5pp)
    
    Returns:
        dict with uniformity metrics
    """
    # Parse net impact
    df['net_impact'] = df['Net Impact_pp'].str.rstrip('pp').astype(float) / 100
    
    # Count rows near ±epsilon
    near_pos_epsilon = np.abs(df['net_impact'] - epsilon) < 0.0001  # Within 0.01pp
    near_neg_epsilon = np.abs(df['net_impact'] + epsilon) < 0.0001  # Within 0.01pp
    
    total_rows = len(df)
    plateau_rows = near_pos_epsilon.sum() + near_neg_epsilon.sum()
    
    return {
        'total_rows': total_rows,
        'near_pos_epsilon': near_pos_epsilon.sum(),
        'near_neg_epsilon': near_neg_epsilon.sum(),
        'plateau_rows': plateau_rows,
        'uniformity_index': plateau_rows / total_rows if total_rows > 0 else 0,
        'plateau_categories': df[near_pos_epsilon | near_neg_epsilon]['Category'].tolist()
    }

def analyze_share_monotonicity(df):
    """
    Check share monotonicity: within performance bands, larger share should correlate with larger impact.
    
    Args:
        df: DataFrame with supporting table data
    
    Returns:
        dict with monotonicity metrics
    """
    # Parse shares and impacts
    df['region_share'] = df['Region Share %'].str.rstrip('%').astype(float) / 100
    df['net_impact_abs'] = df['Net Impact_pp'].str.rstrip('pp').astype(float).abs() / 100
    
    # Group by band and calculate correlations
    correlations = {}
    for band in df['Band'].unique():
        band_df = df[df['Band'] == band]
        if len(band_df) > 2:  # Need at least 3 points for meaningful correlation
            corr = band_df['region_share'].corr(band_df['net_impact_abs'])
            correlations[band] = corr
    
    # Overall correlation
    overall_corr = df['region_share'].corr(df['net_impact_abs'])
    
    return {
        'band_correlations': correlations,
        'overall_correlation': overall_corr,
        'positive_correlations': sum(1 for c in correlations.values() if c > 0.4),
        'total_bands': len(correlations),
        'monotonicity_score': overall_corr if pd.notna(overall_corr) else 0
    }

def diagnose_dataset(csv_path, dataset_name):
    """
    Run full diagnostic analysis on a supporting table CSV.
    """
    print(f"\n{'='*60}")
    print(f"🔍 DIAGNOSING: {dataset_name}")
    print(f"📁 File: {csv_path}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} rows")
    
    # 1. Sign Coherence Analysis
    print(f"\n1️⃣ SIGN COHERENCE ANALYSIS")
    print(f"{'─'*40}")
    coherence = analyze_sign_coherence(df)
    print(f"   Total rows: {coherence['total_rows']}")
    print(f"   Better performers with negative impact: {coherence['better_but_negative']}")
    print(f"   Worse performers with positive impact: {coherence['worse_but_positive']}")
    print(f"   Total violations: {coherence['total_violations']}")
    print(f"   Violation rate: {coherence['violation_rate']:.1%}")
    
    if coherence['violating_categories']:
        print(f"   🚨 Violating categories:")
        for cat in coherence['violating_categories'][:5]:  # Show first 5
            print(f"      - {cat}")
    
    # 2. Uniformity Index Analysis  
    print(f"\n2️⃣ UNIFORMITY INDEX ANALYSIS")
    print(f"{'─'*40}")
    uniformity = analyze_uniformity_index(df)
    print(f"   Total rows: {uniformity['total_rows']}")
    print(f"   Rows near +0.5pp: {uniformity['near_pos_epsilon']}")
    print(f"   Rows near -0.5pp: {uniformity['near_neg_epsilon']}")
    print(f"   Total plateau rows: {uniformity['plateau_rows']}")
    print(f"   Uniformity index: {uniformity['uniformity_index']:.1%}")
    
    if uniformity['plateau_categories']:
        print(f"   📊 Plateau categories:")
        for cat in uniformity['plateau_categories'][:5]:  # Show first 5
            print(f"      - {cat}")
    
    # 3. Share Monotonicity Analysis
    print(f"\n3️⃣ SHARE MONOTONICITY ANALYSIS")
    print(f"{'─'*40}")
    monotonicity = analyze_share_monotonicity(df)
    print(f"   Overall correlation (share vs |impact|): {monotonicity['overall_correlation']:.3f}")
    print(f"   Bands with positive correlation (>0.4): {monotonicity['positive_correlations']}/{monotonicity['total_bands']}")
    print(f"   Monotonicity score: {monotonicity['monotonicity_score']:.3f}")
    
    print(f"   📈 Per-band correlations:")
    for band, corr in monotonicity['band_correlations'].items():
        status = "✅" if corr > 0.4 else "❌"
        print(f"      {status} {band}: {corr:.3f}")
    
    # 4. Summary Assessment
    print(f"\n📋 SUMMARY ASSESSMENT")
    print(f"{'─'*40}")
    
    # Calculate overall health score
    sign_score = 1.0 - coherence['violation_rate']
    uniformity_score = 1.0 - uniformity['uniformity_index'] 
    monotonicity_score = max(0, monotonicity['monotonicity_score'])
    
    overall_score = (sign_score + uniformity_score + monotonicity_score) / 3
    
    print(f"   Sign coherence score: {sign_score:.3f} ({'✅' if sign_score > 0.9 else '❌'})")
    print(f"   Uniformity score: {uniformity_score:.3f} ({'✅' if uniformity_score > 0.8 else '❌'})")
    print(f"   Monotonicity score: {monotonicity_score:.3f} ({'✅' if monotonicity_score > 0.4 else '❌'})")
    print(f"   Overall health: {overall_score:.3f} ({'✅' if overall_score > 0.7 else '❌'})")
    
    return {
        'dataset': dataset_name,
        'coherence': coherence,
        'uniformity': uniformity,
        'monotonicity': monotonicity,
        'scores': {
            'sign': sign_score,
            'uniformity': uniformity_score,
            'monotonicity': monotonicity_score,
            'overall': overall_score
        }
    }

def main():
    """Run diagnostics on all supporting table CSVs."""
    print("🧪 OAXACA-BLINDER DIAGNOSTIC ANALYSIS")
    print("="*60)
    
    output_dir = Path("output")
    datasets = [
        ("vertical2_LATAM_supporting_table.csv", "Vertical2 (LATAM)"),
        ("vertical3_LATAM_supporting_table.csv", "Vertical3 (LATAM)"),
        ("product_LATAM_supporting_table.csv", "Product (LATAM)"),
        ("vertical_LATAM_supporting_table.csv", "Vertical (LATAM)")
    ]
    
    results = []
    
    for csv_file, dataset_name in datasets:
        csv_path = output_dir / csv_file
        if csv_path.exists():
            result = diagnose_dataset(csv_path, dataset_name)
            results.append(result)
        else:
            print(f"⚠️  File not found: {csv_path}")
    
    # Summary comparison
    if results:
        print(f"\n{'='*60}")
        print("📊 COMPARATIVE SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'Dataset':<20} {'Sign':<6} {'Uniform':<8} {'Monoton':<8} {'Overall':<8}")
        print(f"{'─'*20} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
        
        for result in results:
            scores = result['scores']
            print(f"{result['dataset']:<20} "
                  f"{scores['sign']:.3f}  "
                  f"{scores['uniformity']:.3f}    "
                  f"{scores['monotonicity']:.3f}    "
                  f"{scores['overall']:.3f}")

if __name__ == "__main__":
    main()
