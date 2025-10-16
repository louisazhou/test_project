#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regional RCA Analysis Workflow - North America Markets
This script demonstrates regional analysis with North America as reference and 4 sub-markets.
"""

# %% [markdown]
# # Regional RCA Analysis - North America Markets
# 
# This notebook demonstrates regional analysis with North America as reference:
# - **Regional focus**: North America as reference region with 4 sub-markets
# - **Market-level analysis**: North America 1, 2, 3, 4 as comparison markets
# - **Team-level depth**: Each market has multiple teams for depth analysis
# - **Directional + Depth only**: Focused on core RCA analysis methods

# %% [setup]
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# Auto-detect working directory and adjust paths accordingly
current_dir = Path.cwd()
if current_dir.name == 'notebooks':
    # Running from notebooks directory (cell-by-cell)
    base_dir = current_dir.parent
    os.chdir(base_dir)  # Change to parent directory
    print(f"Detected notebook execution. Changed working directory to: {base_dir}")
else:
    # Running from parent directory (as script)
    base_dir = current_dir
    print(f"Detected script execution from: {base_dir}")

# Add package to path
sys.path.append(str(base_dir))

from rca_package.anomaly_detector import detect_snapshot_anomaly_for_column
from rca_package.yaml_processor import (
    load_config, get_all_metrics, get_metric_hypothesis_map,
    get_metric_info, convert_dataframe_to_display_names
)
from rca_package.hypothesis_scorer import (
    score_hypotheses_for_metrics
)
from rca_package.depth_spotter import (
    analyze_region_depth, create_synthetic_data
)
from rca_package.funnel_reason import (
    analyze_funnel_reasons, analyze_funnel_reasons_all_regions, create_funnel_synthetic_data
)
from rca_package.make_slides import SlideLayouts, dual_output, SlideContent
from rca_package.slice_directional import inject_slice_directional_after_depth

# Control figure display - set to True if you want to see figures inline
# Check if running as script (__name__ == '__main__') or in interactive environment
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules or 'IPython' in sys.modules:
    # Running in interactive environment (Jupyter, IPython, etc.)
    SHOW_FIGURES = True
else:
    # Running as script
    SHOW_FIGURES = False

print(f"Figure display mode: {'Enabled (interactive)' if SHOW_FIGURES else 'Disabled (script mode)'}")

# Create slide builder once for the entire analysis
slides = SlideLayouts()

print("=" * 65)

# %% [markdown]
# ## 1. Data Preparation

# %% [data_prep]
print("\n📊 PREPARING TEST DATA")

# Create sample data using technical names (as they come from real data sources)
np.random.seed(42)
# Regional analysis: North America as reference, with 4 sub-markets
regions = ["North America", "North America 1", "North America 2", "North America 3", "North America 4"]

# Create test data with technical names as column headers (realistic scenario)
# Note: conversion_rate_pct will be replaced by Oaxaca-Blinder synthetic data
data = {
    # Metrics (using technical names from config)
    # North America as reference, sub-markets show different performance patterns
    'conversion_rate_pct': np.array([0.084, 0.072, 0.095, 0.078, 0.089]),  # NA1 underperforms, NA2 overperforms
    'avg_order_value': np.array([65.0, 58.0, 72.0, 61.0, 68.0]),  # NA1 low AOV, NA2 high AOV
    'customer_satisfaction': np.array([3.8, 3.5, 4.1, 3.6, 3.9]),  # NA1 low CSAT, NA2 high CSAT
    'revenue': np.array([650000, 520000, 780000, 580000, 720000]),  # Corresponding revenue patterns
    
    # Funnel metrics (direct data instead of function call)
    'technical_demo_conversion_rate': np.array([0.45, 0.38, 0.52, 0.41, 0.48]),  # NA1 underperforms, NA2 overperforms
    'technical_demo_pipeline_value': np.array([25000000, 20000000, 30000000, 22000000, 28000000]),
    'business_proposal_success_rate': np.array([0.42, 0.35, 0.48, 0.38, 0.45]),  # NA1 underperforms, NA2 overperforms
    'business_proposal_pipeline_value': np.array([22000000, 18000000, 26000000, 20000000, 24000000]),
    'contract_negotiation_close_rate': np.array([0.55, 0.48, 0.62, 0.51, 0.58]),  # NA1 underperforms, NA2 overperforms
    'contract_negotiation_pipeline_value': np.array([18000000, 15000000, 21000000, 17000000, 19500000]),
    'total_lost_rate': np.array([0.25, 0.32, 0.18, 0.28, 0.22]),  # NA1 high loss rate, NA2 low loss rate
    
    # Hypotheses (using technical names from config)
    # Patterns that explain the performance differences between NA markets
    'bounce_rate_pct': np.array([0.45, 0.52, 0.38, 0.48, 0.42]),  # NA1 high bounce, NA2 low bounce
    'page_load_time': np.array([3.8, 4.5, 3.2, 4.0, 3.6]),  # NA1 slow loading, NA2 fast loading
    'session_duration': np.array([120, 95, 145, 110, 130]),  # NA1 short sessions, NA2 long sessions
    'pages_per_session': np.array([3.1, 2.8, 3.5, 3.0, 3.3]),  # NA1 fewer pages, NA2 more pages
    'new_users_pct': np.array([0.18, 0.15, 0.22, 0.16, 0.20]),  # NA1 fewer new users, NA2 more new users
    'cart_abandonment_rate': np.array([0.85, 0.92, 0.78, 0.88, 0.82]),  # NA1 high abandonment, NA2 low abandonment
    'mobile_traffic_pct': np.array([0.45, 0.38, 0.52, 0.42, 0.48]),  # NA1 low mobile, NA2 high mobile
    'search_usage_rate': np.array([0.25, 0.18, 0.32, 0.22, 0.28]),  # NA1 low search, NA2 high search
    'email_open_rate': np.array([0.15, 0.12, 0.18, 0.14, 0.16]),  # NA1 low email engagement, NA2 high
    'social_media_traffic': np.array([0.08, 0.05, 0.12, 0.07, 0.10]),  # NA1 low social, NA2 high social
    'product_reviews_count': np.array([80, 60, 100, 70, 90]),  # NA1 fewer reviews, NA2 more reviews
    'customer_service_calls': np.array([45, 55, 35, 50, 40]),  # NA1 more calls, NA2 fewer calls
    'return_rate_pct': np.array([0.15, 0.18, 0.12, 0.16, 0.14]),  # NA1 high returns, NA2 low returns
    'inventory_availability': np.array([0.85, 0.80, 0.90, 0.82, 0.88]),  # NA1 low inventory, NA2 high inventory
    'shipping_speed_days': np.array([4.2, 5.0, 3.5, 4.5, 3.8]),  # NA1 slow shipping, NA2 fast shipping
    'promotional_discount_pct': np.array([0.05, 0.03, 0.08, 0.04, 0.07]),  # NA1 low discounts, NA2 high discounts
    'website_uptime_pct': np.array([0.995, 0.992, 0.998, 0.994, 0.997]),  # NA1 low uptime, NA2 high uptime
    'payment_failure_rate': np.array([0.08, 0.12, 0.05, 0.10, 0.06]),  # NA1 high failures, NA2 low failures
    'recommendation_ctr': np.array([0.06, 0.04, 0.09, 0.05, 0.08])  # NA1 low CTR, NA2 high CTR
}

# Create DataFrame with technical names (as it comes from data sources)
df_technical = pd.DataFrame(data, index=regions)
print("✅ Test data created with technical column names")

# Create granular synthetic data aligned with top-line data
def create_granular_synthetic_from_topline(df_topline: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Create team-level synthetic data that aggregates back to the market topline.

    Produces technical-schema columns matching depth + hypotheses usage and a
    parent_territory column that maps to the topline region index.
    """
    rng = np.random.default_rng(seed)

    # For regional analysis, use all markets (including reference North America)
    regions = [r for r in df_topline.index]

    rows = []
    # Choose 3 teams per market
    for region in regions:
        # Pull topline metric/hypothesis values
        conv_rate = float(df_topline.loc[region, 'conversion_rate_pct'])
        revenue_total = float(df_topline.loc[region, 'revenue'])
        csat = float(df_topline.loc[region, 'customer_satisfaction'])

        # Hypotheses columns (if present, otherwise default ranges)
        def _get(name, default):
            return float(df_topline.loc[region, name]) if name in df_topline.columns else default

        bounce = _get('bounce_rate_pct', 0.35)
        page_time = _get('page_load_time', 2.5)
        session_duration = _get('session_duration', 180)
        pages_per_sess = _get('pages_per_session', 4.2)
        new_users = _get('new_users_pct', 0.2)
        cart_abandon = _get('cart_abandonment_rate', 0.6)
        mobile_pct = _get('mobile_traffic_pct', 0.55)
        search_rate = _get('search_usage_rate', 0.4)
        email_open = _get('email_open_rate', 0.2)
        social_traffic = _get('social_media_traffic', 0.12)
        prod_reviews = _get('product_reviews_count', 120)
        cust_calls = _get('customer_service_calls', 20)
        return_rate = _get('return_rate_pct', 0.08)
        inventory = _get('inventory_availability', 0.95)
        ship_days = _get('shipping_speed_days', 2.5)
        promo_disc = _get('promotional_discount_pct', 0.1)
        uptime = _get('website_uptime_pct', 0.999)
        payment_fail = _get('payment_failure_rate', 0.02)
        reco_ctr = _get('recommendation_ctr', 0.12)

        # Allocate visits across 3 slices and solve conversions to match regional rate
        # Keep visits large enough for stability
        visit_shares = rng.dirichlet([2, 2, 2])
        total_visits = 30_000  # per region
        visits = (visit_shares * total_visits).round().astype(int)
        # Target total conversions
        target_conversions = int(round(total_visits * conv_rate))
        # Create slice-level rates around conv_rate and adjust last to match total exactly
        slice_rates = np.clip(conv_rate + rng.normal(0, 0.01, size=3), 0.01, 0.95)
        convs = (visits * slice_rates).round().astype(int)
        diff = target_conversions - int(convs.sum())
        # Adjust the largest-volume slice to close rounding gap
        if visits.sum() > 0 and diff != 0:
            idx_big = int(np.argmax(visits))
            convs[idx_big] = max(0, convs[idx_big] + diff)

        # Revenue: split to sum to topline
        rev_shares = rng.dirichlet([2, 2, 2])
        rev_values = (rev_shares * revenue_total).round(2)
        # Adjust rounding to hit exact topline
        rev_diff = revenue_total - float(rev_values.sum())
        rev_values[0] = round(float(rev_values[0] + rev_diff), 2)

        # Orders and AOV plausible: orders ~ conversions; aov within a band
        orders = (convs * (1.0 + rng.normal(0.0, 0.05, size=3))).round().astype(int)
        aov = np.maximum(20, rev_values / np.maximum(1, orders))

        # CSAT: allocate surveys and sum_csat to match regional average
        surveys = (rng.uniform(600, 1200, size=3)).round().astype(int)
        target_sum_csat = csat * float(surveys.sum())
        # Start with per-slice csat around regional value
        slice_csats = np.clip(csat + rng.normal(0, 0.1, size=3), 1.0, 5.0)
        sum_csat = (surveys * slice_csats).round(1)
        csat_diff = round(target_sum_csat - float(sum_csat.sum()), 1)
        if len(sum_csat) > 0 and abs(csat_diff) > 0:
            sum_csat[0] = round(float(sum_csat[0] + csat_diff), 1)

        # Hypotheses at slice-level around regional levels
        def jitter(val, scale):
            return float(max(0, val + rng.normal(0, scale)))

        for i, suffix in enumerate(['a', 'b', 'c']):
            rows.append({
                'slice': f"{region}_team_{suffix}",
                'parent_territory': region,
                'region': region,  # for depth module
                'visits': int(visits[i]),
                'conversions': int(convs[i]),
                'orders': int(orders[i]),
                'revenue': float(rev_values[i]),
                'surveys': int(surveys[i]),
                'sum_csat': float(sum_csat[i]),
                # Derived technical metrics to match topline schema
                'conversion_rate_pct': float(0 if int(visits[i]) == 0 else int(convs[i]) / max(1, int(visits[i]))),
                'avg_order_value': float(0 if int(orders[i]) == 0 else float(rev_values[i]) / max(1, int(orders[i]))),
                'customer_satisfaction': float(0 if int(surveys[i]) == 0 else float(sum_csat[i]) / max(1, int(surveys[i]))),
                # Hypotheses (technical names)
                'bounce_rate_pct': jitter(bounce, 0.03),
                'page_load_time': jitter(page_time, 0.3),
                'session_duration': jitter(session_duration, 10.0),
                'pages_per_session': jitter(pages_per_sess, 0.3),
                'new_users_pct': jitter(new_users, 0.03),
                'cart_abandonment_rate': jitter(cart_abandon, 0.05),
                'mobile_traffic_pct': jitter(mobile_pct, 0.05),
                'search_usage_rate': jitter(search_rate, 0.04),
                'email_open_rate': jitter(email_open, 0.03),
                'social_media_traffic': jitter(social_traffic, 0.03),
                'product_reviews_count': int(max(0, round(jitter(prod_reviews, 10.0)))),
                'customer_service_calls': int(max(0, round(jitter(cust_calls, 3.0)))),
                'return_rate_pct': jitter(return_rate, 0.02),
                'inventory_availability': min(1.0, jitter(inventory, 0.01)),
                'shipping_speed_days': max(0.5, jitter(ship_days, 0.3)),
                'promotional_discount_pct': jitter(promo_disc, 0.02),
                'website_uptime_pct': min(0.9999, jitter(uptime, 0.0005)),
                'payment_failure_rate': jitter(payment_fail, 0.01),
                'recommendation_ctr': jitter(reco_ctr, 0.02),
            })

    return pd.DataFrame(rows)


# Build granular synthetic team-level table (technical schema)
slice_df_technical = create_granular_synthetic_from_topline(df_technical)
print(f"✅ Created granular team-level data with {len(slice_df_technical)} rows")
# %% [markdown]
slice_df_technical

# %%
df_technical

# %% [markdown]
# ## 2. Config-Driven Processing

# %% [config_processing]

# Load the main config that drives everything
config = load_config('configs/config_scorer.yaml')
print("✅ Loaded config_scorer.yaml")

# Convert data from technical to display names using config
df = convert_dataframe_to_display_names(df_technical, config)
print("✅ Converted technical → display names using config")

# Also create a display-named slice DataFrame for directional work later
slice_df = convert_dataframe_to_display_names(slice_df_technical, config)
print("✅ Converted slice-level technical → display names using config")

# Get all metrics from config and detect anomalies
metric_names = get_all_metrics(config)
print(f"📋 Found {len(metric_names)} metrics in config: {metric_names}")

# Detect anomalies for ALL metrics in config (using North America as reference)
metric_anomaly_map = {}
for metric_name in metric_names:
    anomaly_info = detect_snapshot_anomaly_for_column(df, 'North America', column=metric_name)
    if anomaly_info:
        # Add higher_is_better from config
        metric_info = get_metric_info(config, metric_name)
        anomaly_info['higher_is_better'] = metric_info.get('higher_is_better', True)
        metric_anomaly_map[metric_name] = anomaly_info

print(f"✅ Detected anomalies for {len(metric_anomaly_map)} metrics:")
for metric, info in metric_anomaly_map.items():
    print(f"   • {metric}: {info['anomalous_region']} is {info['direction']} by {info['magnitude']}")

# %% [markdown]
# ## 3. Title Slide

# %% [title_slide]
print("\n📋 CREATING TITLE SLIDE")

@dual_output(console=True, slide=True, slide_builder=slides, layout_type='text', show_figures=SHOW_FIGURES)
def create_title_slide():
    return SlideContent(
        title="RCA Analysis Results",
        text_template="""
Analysis Summary:
• {{ num_metrics }} metrics analyzed
• {{ num_anomalies }} anomalies detected
• Automatic slide generation in config order

Generated on: {{ timestamp }}
        """,
        dfs={},
        template_params={
            'num_metrics': len(metric_names),
            'num_anomalies': len(metric_anomaly_map),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    )

title_content, title_results = create_title_slide()
print("✅ Title slide created")

# %% [markdown]
# ## 4. Automated Analysis Workflow

# %% [analysis_workflow]
print("\n🚀 RUNNING AUTOMATED ANALYSIS WORKFLOW")

def process_analysis_results(analysis_results, unified_results):
    """
    Generic processor for any type of analysis results.
    Updates unified_results in place.
    """
    for metric_name, result in analysis_results.items():
        if metric_name not in unified_results:
            unified_results[metric_name] = {'slides': {}, 'payload': {}}
            
        unified_results[metric_name]['slides'].update(result['slides'])
        unified_results[metric_name]['payload'].update(result['payload'])

def generate_analysis_slide(metric_name, analysis_type, slide_data, total_hypotheses, current_count, increment_count=True):
    """
    Generate a single analysis slide with proper formatting and titles.
    
    Args:
        metric_name: Name of the metric being analyzed
        analysis_type: Type of analysis being performed
        slide_data: Data for the slide
        total_hypotheses: Total number of hypotheses for this metric
        current_count: Current hypothesis count
        increment_count: Whether this slide should increment the hypothesis count (default True)
                       Set to False for continuation slides that are part of the same hypothesis
    """
    # Get the number of hypotheses in this analysis
    hypotheses_in_analysis = slide_data['slide_info'].get('total_hypotheses', 0) if increment_count else 0
    
    # Calculate the starting hypothesis number for this slide
    if increment_count:
        start_num = current_count - hypotheses_in_analysis + 1
    else:
        # For continuation slides, use the current count as both start and end
        start_num = current_count - hypotheses_in_analysis
    
    # If this slide contains multiple hypotheses, show the range, otherwise show single number
    if hypotheses_in_analysis > 1:
        range_text = f"{start_num}-{current_count}"
    else:
        range_text = str(current_count)
    
    slide_data['slide_info']['title'] = f"{slide_data['slide_info']['title']} ({range_text}/{total_hypotheses})"
    
    @dual_output(console=True, slide=True, slide_builder=slides, 
                layout_type=slide_data['slide_info'].get('layout_type', 'text_figure'), 
                show_figures=SHOW_FIGURES)
    def create_analysis_slide():
        template_params = slide_data['slide_info'].get('template_params', {}).copy()
        figure_generators = slide_data['slide_info'].get('figure_generators', [])
        template_params['total_hypotheses'] = total_hypotheses
        dfs = slide_data['slide_info'].get('dfs', {}).copy()
        
        return SlideContent(
            title=slide_data['slide_info']['title'],
            text_template=slide_data['slide_info'].get('template_text', ''),
            dfs=dfs,
            template_params=template_params,
            figure_generators=figure_generators,
            show_figures=SHOW_FIGURES
        )
    
    content, results = create_analysis_slide()
    print(f"✅ Generated {analysis_type} slide for {metric_name}")

# Initialize results containers
unified_results = {}
total_hypotheses_per_metric = {}

# Get metric hypothesis mapping from config
metric_hypo_map = get_metric_hypothesis_map(config)
print(f"📋 Metrics with hypotheses: {list(metric_hypo_map.keys())}")

# Process directional analysis (using North America as reference)
metrics_with_hypotheses = [m for m in metric_names if m in metric_hypo_map and metric_hypo_map[m]]
if metrics_with_hypotheses:
    print(f"🔄 Processing directional analysis for {len(metrics_with_hypotheses)} metrics...")
    try:
        results = score_hypotheses_for_metrics(
            regional_df=df,
            anomaly_map=metric_anomaly_map,
            config=config,
            reference_row="North America"  # Use North America as reference
        )
        process_analysis_results(results, unified_results)
    except Exception as e:
        print(f"⚠️  Directional analysis skipped: {e}")

# Process depth analysis
try:
    print("🔄 Processing depth analysis...")
    results = analyze_region_depth(
        sub_df=slice_df_technical,
        config=load_config('configs/config_depth.yaml'),
        metric_anomaly_map=metric_anomaly_map
    )
    process_analysis_results(results, unified_results)
except Exception as e:
    print(f"⚠️  Depth analysis skipped: {e}")

# Skip Oaxaca-Blinder and Funnel analysis for regional focus
print("ℹ️  Skipping Oaxaca-Blinder and Funnel analysis - focusing on directional and depth only")

# Calculate total hypotheses per metric once
for metric_name in unified_results:
    total_hypotheses_per_metric[metric_name] = sum(
        slide_data['slide_info'].get('total_hypotheses', 0)
        for slide_data in unified_results[metric_name]['slides'].values()
    )

# Generate summary slide (will be second after title slide)
# Create a new DataFrame with only the metrics we want to show
metrics_to_show = [m for m in metric_names if m in df.columns]
summary_df = df[metrics_to_show].copy()

# Add the hypotheses count column - per metric
hypotheses_counts = {}
for metric_name in metrics_to_show:
    if metric_name in unified_results:
        hypotheses_counts[metric_name] = sum(
            slide_data['slide_info'].get('total_hypotheses', 0)
            for slide_data in unified_results[metric_name]['slides'].values()
        )
    else:
        hypotheses_counts[metric_name] = 0

# Add the hypotheses count as a new row
summary_df.loc['# of Hypotheses'] = pd.Series(hypotheses_counts).astype(int)

# Collect summary texts from slide data (preserve YAML metric order)
summary_texts = {}
for metric_name in metric_names:  # Use original YAML order
    if metric_name in unified_results:
        result = unified_results[metric_name]
        texts = []
        for analysis_type, slide_data in result['slides'].items():
            if 'summary' in slide_data and 'summary_text' in slide_data['summary']:
                texts.append(f"{analysis_type}: {slide_data['summary']['summary_text']}")
        if texts:
            summary_texts[metric_name] = "\n".join(texts)

# Create summary slide with all metrics
slides.create_metrics_summary_slide(
    df=summary_df,
    metrics_text=summary_texts,  # Only metrics with analysis will have summary text
    metric_anomaly_map=metric_anomaly_map,  # Only metrics with anomalies will be highlighted
    title="Analysis Summary"
)
print("✅ Summary slide created (will appear as second slide)")

# Skip comprehensive funnel slides for regional focus
print("ℹ️  Skipping comprehensive funnel slides - focusing on directional and depth only")

try:
    print("\n🔧 Adding team-level Directional slides after Depth via module...")
    unified_results = inject_slice_directional_after_depth(
        unified_results=unified_results,
        regional_df=df,
        slice_df=slice_df,
        config=config,
        metric_anomaly_map=metric_anomaly_map,
        k=2,
        # only_metrics=['Conversion Rate','Average Order Value','Customer Satisfaction'],
        verbose=True,
    )
    print("✅ Added team-level Directional slides where applicable")
except Exception as e:
    print(f"⚠️  Failed to add team-level Directional slides: {e}")

# Generate detailed analysis slides (preserve YAML order)
for metric_name in metric_names:
    if metric_name not in unified_results:
        continue
        
    print(f"\n🔄 Generating detailed slides for metric: {metric_name}")
    
    # Add a divider slide for this metric
    slides._create_divider_slide(metric_name)
    
    # Get total hypotheses for this metric
    total_hypotheses = sum(
        slide_data['slide_info'].get('total_hypotheses', 0)
        for slide_data in unified_results[metric_name]['slides'].values()
    )
    
    # Generate slides with running hypothesis count
    current_count = 0
    for analysis_type, slide_data in unified_results[metric_name]['slides'].items():
        hypotheses_in_analysis = slide_data['slide_info'].get('total_hypotheses', 0)
        current_count += hypotheses_in_analysis
        generate_analysis_slide(
            metric_name, 
            analysis_type, 
            slide_data, 
            total_hypotheses, 
            current_count
        )

# Skip comprehensive funnel slides generation for regional focus
print("ℹ️  Skipping comprehensive funnel slides generation - focusing on directional and depth only")

# %% [markdown]
# ## 7. Save and Upload Presentation

# %% [save_upload]
print("\n💾 SAVING AND UPLOADING PRESENTATION")

# Create output directory
output_dir = os.path.abspath('output')
os.makedirs(output_dir, exist_ok=True)

# Save the presentation
presentation_path = slides.save("RCA_Analysis_Regional_NA", output_dir)
print(f"✅ Presentation saved: {presentation_path}")

# Optional: Upload to Google Drive
from rca_package.google_drive_utils import upload_to_google_drive
try:
    upload_result = upload_to_google_drive(
        file_path=presentation_path,
        credentials_path="credentials.json",
        token_path="token.json"
    )
    print(f"✅ Uploaded to Google Drive: {upload_result['gdrive_url']}")
except Exception as e:
    print(f"⚠️  Upload to Google Drive failed: {e}")
    print("   (This is normal if credentials are not configured)")

# %% [markdown]
# ## 8. Final Summary

# %% [analysis_summary]
print(f"\n📊 ANALYSIS RESULTS SUMMARY")

print(f"\n🎯 METRICS PROCESSED:")

# Count total metrics with any analysis
total_metrics = len(unified_results)

for metric_name in metric_names:
    if metric_name in unified_results:
        print(f"\n📈 {metric_name}:")
        print(f"   Available slide types: {list(unified_results[metric_name]['slides'].keys())}")
        print(f"   Generated slides: {len(unified_results[metric_name]['slides'])}")

print(f"\n📊 OVERALL STATISTICS:")
print(f"   • Total metrics analyzed: {total_metrics}")
print(f"   • Total slides generated: {sum(len(result['slides']) for result in unified_results.values())}")
print(f"   • Figure display: {'Enabled' if SHOW_FIGURES else 'Disabled'}")
print(f"   • Architecture: Direct analysis → Direct slide generation (no unnecessary layers!)")

# %% [final_summary]
print(f"\n📄 PRESENTATION DETAILS:")
print(f"   • File: {presentation_path}")
file_size = os.path.getsize(presentation_path) / 1024  # KB
print(f"   • Size: {file_size:.1f} KB")

# Calculate total slides
total_content_slides = sum(len(result['slides']) for result in unified_results.values())
print(f"   • Slides: {total_content_slides} content slides + title + summary")

print(f"\n✨ SUCCESS! Regional North America RCA analysis complete!")

# %% [cleanup]
# Close all figures to prevent display spam
plt.close('all')
