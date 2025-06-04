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

def get_function_by_name(func_name: str):
    # Look in current module's globals first
    if func_name in globals():
        return globals()[func_name]
    
    # Look in imported modules
    import rca_package.depth_spotter as depth_module
    import rca_package.hypothesis_scorer as scorer_module
    
    for module in [depth_module, scorer_module]:
        if hasattr(module, func_name):
            return getattr(module, func_name)
    
    raise ValueError(f"Function '{func_name}' not found in available modules")

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
    load_config, get_all_metrics, 
    get_expected_directions, get_metric_hypothesis_map,
    get_metric_info, convert_dataframe_to_display_names
)
from rca_package.hypothesis_scorer import (
    score_hypotheses_for_metrics
)
from rca_package.depth_spotter import (
    analyze_region_depth, create_synthetic_data
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

print("🎯 RCA ANALYSIS - SIMPLIFIED CONFIG-DRIVEN APPROACH", SHOW_FIGURES)
print("=" * 65)

# %% [markdown]
# ## 1. Data Preparation

# %% [data_prep]
print("\n📊 PREPARING TEST DATA")

# Create sample data using technical names (as they come from real data sources)
np.random.seed(42)
regions = ["Global", "North America", "Europe", "Asia", "Latin America"]

# Create test data with technical names as column headers (realistic scenario)
data = {
    # Metrics (using technical names from config)
    'conversion_rate_pct': np.array([0.12, 0.08, 0.11, 0.13, 0.10]),
    'avg_order_value': np.array([75.0, 65.0, 80.0, 85.0, 72.0]),
    'customer_satisfaction': np.array([4.2, 3.8, 4.3, 4.5, 4.0]),
    'revenue': np.array([1000000, 650000, 950000, 1100000, 850000]),
    
    # Hypotheses (using technical names from config)
    'bounce_rate_pct': np.array([0.35, 0.45, 0.32, 0.28, 0.34]),
    'page_load_time': np.array([2.4, 3.8, 2.2, 1.9, 2.5]),
    'session_duration': np.array([180, 120, 190, 210, 175]),
    'pages_per_session': np.array([4.2, 3.1, 4.5, 4.8, 4.0]),
    'new_users_pct': np.array([0.25, 0.18, 0.28, 0.30, 0.23])
}

# Create DataFrame with technical names (as it comes from data sources)
df_technical = pd.DataFrame(data, index=regions)
print("✅ Test data created with technical column names")

# %% [markdown]
# ## 2. Config-Driven Processing

# %% [config_processing]
print("\n⚙️  CONFIG-DRIVEN PROCESSING")

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
        title="RCA Analysis Results - Config-Driven",
        text_template="""
Root Cause Analysis using simplified config-driven approach.

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

# Get metric hypothesis mapping from config
metric_hypo_map = get_metric_hypothesis_map(config)

print(f"📋 Metrics with hypotheses: {list(metric_hypo_map.keys())}")

# Process scorer config - only metrics with hypotheses
metrics_with_hypotheses = [
    m for m in metric_names 
    if m in metric_hypo_map and metric_hypo_map[m]  # Non-empty hypothesis list
]

scorer_results = {}
if metrics_with_hypotheses:
    print(f"🔄 Processing {len(metrics_with_hypotheses)} metrics with hypotheses...")
    scorer_results = score_hypotheses_for_metrics(
        regional_df=df,
        anomaly_map=metric_anomaly_map,
        config=config  # Just pass the full config
    )
    print(f"✅ Processed scorer results for: {list(scorer_results.keys())}")

# Process depth config
depth_results = {}
try:
    from rca_package import load_config as load_depth_config
    depth_config = load_depth_config('configs/config_depth.yaml')
    sub_regional_data = create_synthetic_data()
    
    print("🔄 Processing depth analysis...")
    depth_results = analyze_region_depth(
        sub_df=sub_regional_data,
        config=depth_config,
        metric_anomaly_map=metric_anomaly_map
    )
    print(f"✅ Processed depth results for: {list(depth_results.keys())}")
except Exception as e:
    print(f"⚠️  Depth analysis skipped: {e}")

# Process slides in correct order with proper slide generation
for metric_name, metric_config in config['metrics'].items():  # ← Get both key and value
    
    print(f"\n🔄 Generating all slides for metric: {metric_name}")
    
    # Process all analysis types for this metric
    all_results = {}
    if scorer_results and metric_name in scorer_results:  # ← Use metric_name (the key)
        all_results.update(scorer_results[metric_name]['slides'])
    if depth_results and metric_name in depth_results:  # ← Use metric_name (the key)
        all_results.update(depth_results[metric_name]['slides'])
    
    # Generate slides for each analysis type using proper slide system
    for analysis_type, slide_data in all_results.items():
        slide_info = slide_data['slide_info']
        
        # Create proper slide using dual_output decorator
        @dual_output(console=True, slide=True, slide_builder=slides, 
                    layout_type=slide_info.get('layout_type', 'text_figure'), 
                    show_figures=SHOW_FIGURES)
        def create_analysis_slide():
            # Extract clean components
            template_params = slide_info.get('template_params', {}).copy()
            figure_generators = slide_info.get('figure_generators', [])
            
            return SlideContent(
                title=slide_info['title'],
                text_template=slide_info.get('template_text', ''),
                dfs=slide_info.get('dfs', {}),
                template_params=template_params,  # Only template variables!
                figure_generators=figure_generators,  # Functions stored directly inside!
                show_figures=SHOW_FIGURES
            )
        
        # Generate the slide 
        content, results = create_analysis_slide()
        
        print(f"✅ Generated {analysis_type} slide for {metric_name}")

# %% [markdown]
# ## 6. Summary Slide Creation

# %% [summary_slide]
print("\n📊 CREATING SUMMARY SLIDE")

# Generic summary text extraction - works for ANY number of analysis types!
def extract_summary_texts(all_results: Dict) -> Dict[str, str]:
    """Extract summary texts generically from any analysis results"""
    summary_texts = {}
    
    for metric_name, metric_data in all_results.items():
        summary_parts = []
        
        # Process all analysis types generically
        for analysis_type, slide_data in metric_data.get('slides', {}).items():
            summary_text = slide_data.get('summary', {}).get('summary_text', '')
            if summary_text:
                summary_parts.append(f"{analysis_type.title()}: {summary_text}")
        
        # Combine all summary parts for this metric
        if summary_parts:
            summary_texts[metric_name] = "\n".join(summary_parts)
    
    return summary_texts

# Combine all results for summary extraction
combined_results = {}
if scorer_results:
    combined_results.update(scorer_results)
if depth_results:
    # Merge depth results into existing metrics or create new entries
    for metric_name, metric_data in depth_results.items():
        if metric_name in combined_results:
            combined_results[metric_name]['slides'].update(metric_data['slides'])
        else:
            combined_results[metric_name] = metric_data

# Extract summaries generically
summary_texts = extract_summary_texts(combined_results)
print(f"📋 Summary texts collected for {len(summary_texts)} metrics")

if summary_texts:
    slides.create_metrics_summary_slide(
        df=df,
        metrics_text=summary_texts,
        metric_anomaly_map=metric_anomaly_map,
        title="Analysis Summary"
    )
    print(f"✅ Summary slide created for {len(summary_texts)} metrics")
else:
    print("⚠️  No summary information available - skipping summary slide")

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

# Count slides by analysis type
scorer_count = len(scorer_results) if scorer_results else 0
depth_count = len(depth_results) if depth_results else 0

for metric_name in metric_names:
    if metric_name in scorer_results:
        print(f"\n📈 {metric_name}:")
        print(f"   Available slide types: ['scorer']")
        print(f"   Generated slides: 1")

for metric_name in metric_names:
    if metric_name in depth_results:
        print(f"\n🔍 {metric_name} (Depth):")
        print(f"   Available slide types: ['depth']")
        print(f"   Generated slides: 1")

print(f"\n📊 OVERALL STATISTICS:")
print(f"   • Total metrics analyzed: {scorer_count + depth_count}")
print(f"   • Total slides generated: {scorer_count + depth_count}")
print(f"   • Figure display: {'Enabled' if SHOW_FIGURES else 'Disabled'}")
print(f"   • Architecture: Direct analysis → Direct slide generation (no unnecessary layers!)")

# %% [final_summary]
print(f"\n📄 PRESENTATION DETAILS:")
print(f"   • File: {presentation_path}")
file_size = os.path.getsize(presentation_path) / 1024  # KB
print(f"   • Size: {file_size:.1f} KB")

# Calculate total slides
total_content_slides = 0
if scorer_results:
    total_content_slides += len(scorer_results)
if depth_results:
    total_content_slides += len(depth_results)

print(f"   • Slides: {total_content_slides} content slides + title + summary")

print(f"\n✨ SUCCESS! Simplified config-driven RCA analysis complete!")

# %% [cleanup]
# Close all figures to prevent display spam
plt.close('all')
