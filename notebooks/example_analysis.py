#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example RCA Analysis Workflow - Simplified Config-Driven Approach
This script demonstrates a simplified approach that uses the config to drive everything.
"""

# %% [markdown]
# # Example RCA Analysis Workflow - Simplified
# 
# This notebook demonstrates a simplified config-driven approach:
# - **Single config source**: Uses config_scorer.yaml to drive everything
# - **Automatic data conversion**: Technical → Display names from config
# - **Automatic anomaly detection**: For all metrics in config
# - **Config-driven slide ordering**: Order based on config structure
# - **Complete automation**: From config loading to slide generation

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
    analyze_funnel_reasons, create_funnel_synthetic_data
)
from rca_package.make_slides import SlideLayouts, dual_output, SlideContent

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
regions = ["Global", "North America", "Europe", "Asia", "Latin America"]

# Create test data with technical names as column headers (realistic scenario)
# Note: conversion_rate_pct will be replaced by Oaxaca-Blinder synthetic data
data = {
    # Metrics (using technical names from config)
    'conversion_rate_pct': np.array([0.114, 0.084, 0.114, 0.149, 0.102]),  # From Oaxaca synthetic data
    'avg_order_value': np.array([75.0, 65.0, 80.0, 85.0, 72.0]),
    'customer_satisfaction': np.array([4.2, 3.8, 4.3, 4.5, 4.0]),
    'revenue': np.array([1000000, 650000, 950000, 1100000, 850000]),
    
    # Funnel metrics (direct data instead of function call)
    'technical_demo_conversion_rate': np.array([0.65, 0.45, 0.68, 0.72, 0.66]),  # NA underperforms
    'technical_demo_pipeline_value': np.array([45000000, 25000000, 48000000, 35000000, 42000000]),
    'business_proposal_success_rate': np.array([0.58, 0.62, 0.42, 0.61, 0.59]),  # Europe underperforms
    'business_proposal_pipeline_value': np.array([38000000, 32000000, 22000000, 28000000, 35000000]),
    'contract_negotiation_close_rate': np.array([0.72, 0.75, 0.78, 0.55, 0.74]),  # Asia underperforms
    'contract_negotiation_pipeline_value': np.array([28000000, 31000000, 35000000, 18000000, 29000000]),
    'total_lost_rate': np.array([0.15, 0.25, 0.16, 0.14, 0.17]),  # North America underperforms overall
    
    # Hypotheses (using technical names from config)
    'bounce_rate_pct': np.array([0.35, 0.45, 0.32, 0.28, 0.34]),
    'page_load_time': np.array([2.4, 3.8, 2.2, 1.9, 2.5]),
    'session_duration': np.array([180, 120, 190, 210, 175]),
    'pages_per_session': np.array([4.2, 3.1, 4.5, 4.8, 4.0]),
    'new_users_pct': np.array([0.25, 0.18, 0.28, 0.30, 0.23]),
    'cart_abandonment_rate': np.array([0.70, 0.85, 0.65, 0.60, 0.72]),
    'mobile_traffic_pct': np.array([0.60, 0.45, 0.65, 0.70, 0.58]),
    'search_usage_rate': np.array([0.40, 0.25, 0.45, 0.50, 0.38]),
    'email_open_rate': np.array([0.22, 0.15, 0.25, 0.28, 0.20]),
    'social_media_traffic': np.array([0.15, 0.08, 0.18, 0.20, 0.12]),
    'product_reviews_count': np.array([150, 80, 170, 200, 140]),
    'customer_service_calls': np.array([25, 45, 20, 15, 28]),
    'return_rate_pct': np.array([0.08, 0.15, 0.06, 0.05, 0.09]),
    'inventory_availability': np.array([0.95, 0.85, 0.97, 0.98, 0.93]),
    'shipping_speed_days': np.array([2.5, 4.2, 2.0, 1.8, 2.8]),
    'promotional_discount_pct': np.array([0.10, 0.05, 0.12, 0.15, 0.08]),
    'website_uptime_pct': np.array([0.999, 0.995, 0.9995, 0.9998, 0.998]),
    'payment_failure_rate': np.array([0.02, 0.08, 0.015, 0.01, 0.025]),
    'recommendation_ctr': np.array([0.12, 0.06, 0.15, 0.18, 0.10])
}

# Create DataFrame with technical names (as it comes from data sources)
df_technical = pd.DataFrame(data, index=regions)
print("✅ Test data created with technical column names")

# %% [markdown]
# ## 2. Config-Driven Processing

# %% [config_processing]

# Load the main config that drives everything
config = load_config('configs/config_scorer.yaml')
print("✅ Loaded config_scorer.yaml")

# Convert data from technical to display names using config
df = convert_dataframe_to_display_names(df_technical, config)
print("✅ Converted technical → display names using config")

# Get all metrics from config and detect anomalies
metric_names = get_all_metrics(config)
print(f"📋 Found {len(metric_names)} metrics in config: {metric_names}")

# Detect anomalies for ALL metrics in config
metric_anomaly_map = {}
for metric_name in metric_names:
    anomaly_info = detect_snapshot_anomaly_for_column(df, 'Global', column=metric_name)
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

# Process directional analysis
metrics_with_hypotheses = [m for m in metric_names if m in metric_hypo_map and metric_hypo_map[m]]
if metrics_with_hypotheses:
    print(f"🔄 Processing directional analysis for {len(metrics_with_hypotheses)} metrics...")
    try:
        results = score_hypotheses_for_metrics(
            regional_df=df,
            anomaly_map=metric_anomaly_map,
            config=config
        )
        process_analysis_results(results, unified_results)
    except Exception as e:
        print(f"⚠️  Directional analysis skipped: {e}")

# Process depth analysis
try:
    print("🔄 Processing depth analysis...")
    results = analyze_region_depth(
        sub_df=create_synthetic_data(),
        config=load_config('configs/config_depth.yaml'),
        metric_anomaly_map=metric_anomaly_map
    )
    process_analysis_results(results, unified_results)
except Exception as e:
    print(f"⚠️  Depth analysis skipped: {e}")

# Process Oaxaca-Blinder analysis
try:
    print("🔄 Processing Oaxaca-Blinder decomposition analysis...")
    from rca_package.oaxaca_blinder import analyze_oaxaca_metrics, create_oaxaca_synthetic_data
    
    # For demonstration, generate synthetic data for Conversion Rate if anomaly detected
    oaxaca_config = load_config('configs/config_oaxaca.yaml')
    if 'Conversion Rate' in metric_anomaly_map:
        anomaly = metric_anomaly_map['Conversion Rate']
        target_region = anomaly['anomalous_region']
        target_gap_pp = (anomaly['metric_val'] - anomaly['global_val']) * 100
        
        print(f"   🎯 Generating synthetic data for {target_region} with {target_gap_pp:+.1f}pp gap")
        conversion_data = create_oaxaca_synthetic_data(
            target_region=target_region, 
            target_gap_pp=target_gap_pp
        )
        
        results = analyze_oaxaca_metrics(
            config=oaxaca_config,
            metric_anomaly_map=metric_anomaly_map,
            data_df=conversion_data
        )
        process_analysis_results(results, unified_results)
    else:
        print("   ℹ️  No conversion rate anomaly detected - skipping Oaxaca-Blinder")
    
except Exception as e:
    print(f"⚠️  Oaxaca-Blinder analysis skipped: {e}")
    import traceback
    traceback.print_exc()

# Process Funnel analysis
try:
    print("🔄 Processing funnel analysis...")
    
    # Generate synthetic funnel data
    df_lost, df_blocked = create_funnel_synthetic_data()
    
    # Define column mappings for lost and blocked data (as required kwargs)
    lost_columns = {
        'lost_territory': "territory_l4_name",
        'lost_stage': "Stage Before Closed", 
        'lost_reason': "Closed Lost Sub Reason",
        'lost_count': "# of Solutions Lost",
        'lost_amount': "$ PRC Lost",
        'lost_current_count': "# of Solutions Currently in Stage",
        'lost_current_amount': "$ PRC Currently in Stage",
        'lost_total_count': "Total # of Solutions (Lost + Current)",
        'lost_total_amount': "Total $ PRC (Lost + Current)"
    }
    
    blocked_columns = {
        'blocked_territory': "territory_l4_name",
        'blocked_stage': "Stage Being Blocked",
        'blocked_reason': "Blocked Sub Reason", 
        'blocked_count': "# of Solutions Blocked",
        'blocked_amount': "$ PRC Blocked",
        'blocked_current_count': "# of Solutions Currently in Stage",
        'blocked_current_amount': "$ PRC Currently in Stage"
    }
    
    funnel_config = load_config('configs/config_funnel.yaml')
    results = analyze_funnel_reasons(
        df_lost=df_lost,
        df_block=df_blocked,
        config=funnel_config,
        metric_anomaly_map=metric_anomaly_map,
        lost_columns=lost_columns,
        blocked_columns=blocked_columns
    )
    process_analysis_results(results, unified_results)
except Exception as e:
    print(f"⚠️  Funnel analysis skipped: {e}")
    import traceback
    traceback.print_exc()

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

# %% [markdown]
# ## 7. Save and Upload Presentation

# %% [save_upload]
print("\n💾 SAVING AND UPLOADING PRESENTATION")

# Create output directory
output_dir = os.path.abspath('output')
os.makedirs(output_dir, exist_ok=True)

# Save the presentation
presentation_path = slides.save("RCA_Analysis_Simplified", output_dir)
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

print(f"\n✨ SUCCESS! Simplified config-driven RCA analysis complete!")

# %% [cleanup]
# Close all figures to prevent display spam
plt.close('all')
