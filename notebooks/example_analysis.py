#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example RCA Analysis Workflow
This script demonstrates how to use the RCA package to analyze metrics and create presentations.
"""

# %% [markdown]
# # Example RCA Analysis Workflow
# 
# This notebook demonstrates how to use the RCA package to analyze metrics and create presentations.
# 
# ## Figure Display Options
# The `@dual_output` decorator supports two modes for figure display:
# - `show_figures=True` (default): Display figures inline in notebooks/console using plt.show()
# - `show_figures=False`: Show figure file paths instead of displaying inline
# 
# This is useful for different environments:
# - Jupyter notebooks: Use `show_figures=True` for inline display
# - Terminal/console: Use `show_figures=False` to see where figures are saved
# - Mixed usage: Different slides can use different modes as demonstrated below

# %% [setup]
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

from rca_package import (
    load_config,
    get_metric_info,
    get_hypothesis_info,
    get_all_metrics,
    get_relevant_hypotheses,
    get_expected_directions,
    get_metric_hypothesis_map,
    get_template,
    convert_dataframe_to_display_names,
    get_technical_name
)
from rca_package.hypothesis_scorer import (
    process_metrics_with_structured_results,
    get_ranked_hypotheses,
    create_multi_hypothesis_plot,
    create_scatter_grid,
    add_score_formula,
    add_template_text,
    render_template_text
)
from rca_package.make_slides import (
    SlideLayouts, 
    SlideContent, 
    dual_output, 
    create_bar_chart,
    upload_to_google_drive,
    create_metrics_summary_slide
)
from rca_package.depth_spotter import (
    create_synthetic_data,
    analyze_region_depth,
    plot_subregion_bars
)

# Create slide builder once for the entire analysis
slides = SlideLayouts()

# %% [markdown]
# ## 1. Load Configuration

# %% [load_config]
# Load configuration
config_path = 'configs/config_scorer.yaml'
config = load_config(config_path)
print("Loaded configuration successfully")

# %% [markdown]
# ## 2. Create Sample Data

# %% [create_data]
# Create sample data using technical names as column headers (as they come from real data sources)
np.random.seed(42)
regions = ["Global", "North America", "Europe", "Asia", "Latin America"]

# Create test data with technical names as column headers (realistic scenario)
data = {
    # Metrics (using technical names)
    'conversion_rate_pct': np.array([0.12, 0.08, 0.11, 0.13, 0.10]),
    'avg_order_value': np.array([75.0, 65.0, 80.0, 85.0, 72.0]),
    'customer_satisfaction': np.array([4.2, 3.8, 4.3, 4.5, 4.0]),
    
    # Hypotheses (using technical names)
    'bounce_rate_pct': np.array([0.35, 0.45, 0.32, 0.28, 0.34]),
    'page_load_time': np.array([2.4, 3.8, 2.2, 1.9, 2.5]),
    'session_duration': np.array([180, 120, 190, 210, 175]),
    'pages_per_session': np.array([4.2, 3.1, 4.5, 4.8, 4.0]),
    'new_users_pct': np.array([0.25, 0.18, 0.28, 0.30, 0.23])
}

# Create DataFrame with technical names (as it comes from data sources)
df_technical = pd.DataFrame(data, index=regions)
print("\nOriginal Data (Technical Names):")
print(df_technical)

# Convert to display names for processing
df = convert_dataframe_to_display_names(df_technical, config)
print("\nConverted Data (Display Names):")
print(df)

# %% [markdown]
# ## 3. Prepare Analysis Parameters

# %% [prepare_analysis]
# Get all metrics and their hypotheses
metric_names = get_all_metrics(config)
metric_hypo_map = get_metric_hypothesis_map(config)
expected_directions = get_expected_directions(config)

# Create metric anomaly map with higher_is_better from config
metric_anomaly_map = {}
for metric_name in metric_names:
    anomaly_info = detect_snapshot_anomaly_for_column(df, 'Global', column=metric_name)
    if anomaly_info:
        # Add higher_is_better from config
        metric_info = get_metric_info(config, metric_name)
        anomaly_info['higher_is_better'] = metric_info.get('higher_is_better', True)
        metric_anomaly_map[metric_name] = anomaly_info

print("\nMetric Anomaly Map:")
for metric, info in metric_anomaly_map.items():
    print(f"\n{metric}:")
    for key, value in info.items():
        print(f"  {key}: {value}")

# %% [markdown]
# ## 4. Score Hypotheses

# %% [score_hypotheses]
# Score hypotheses for each metric using sign-based scoring - support multiple config types
all_metric_results = {}

# Load multiple config types (scorer does ranking, others are direct 1:1 mappings)
configs = [
    {
        'name': 'scorer',
        'config': config,  # Main scoring config
        'type': 'ranking',  # This one does winner selection among multiple hypotheses
        'get_template_func': get_template
    },
    {
        'name': 'depth',
        'config': load_config('configs/config_depth.yaml'),
        'type': 'direct',  # 1:1 mapping between metric and template
        'get_template_func': get_template  
    }
    # Add more config types here as needed
]

# Process each config type for each metric
for config_info in configs:
    print(f"\nProcessing config type: {config_info['name']}")
    
    if config_info['type'] == 'ranking':
        # Scorer type - does ranking among multiple hypotheses per metric
        hypo_results = process_metrics_with_structured_results(
            df=df,
            metric_cols=metric_names,
            metric_anomaly_map=metric_anomaly_map,
            expected_directions=expected_directions,
            metric_hypothesis_map=metric_hypo_map,
            config=config_info['config'],
            get_template_func=config_info['get_template_func']
        )
        
        # Store results from ranking config
        for metric_name, metric_result in hypo_results.items():
            if metric_name not in all_metric_results:
                all_metric_results[metric_name] = {'config_results': {}}
            all_metric_results[metric_name]['config_results'][config_info['name']] = metric_result
            
    elif config_info['type'] == 'direct':
        # Direct type - 1:1 mapping, use depth analysis function
        if config_info['name'] == 'depth':
            # Get synthetic sub-regional data for depth analysis
            sub_df = create_synthetic_data()
            
            depth_results = analyze_region_depth(
                sub_df=sub_df,
                config=config_info['config'],
                metric_anomaly_map=metric_anomaly_map
            )
            
            # Store depth results by matching to metrics
            for hypo_name, result in depth_results.items():
                if result['type'] == 'depth_spotter':
                    # Extract metric name from hypothesis name (e.g., "conversion_rate_pct_in_subregion" -> "conversion_rate_pct")
                    depth_metric_name = hypo_name.replace('_in_subregion', '')
                    if depth_metric_name in metric_names:
                        if depth_metric_name not in all_metric_results:
                            all_metric_results[depth_metric_name] = {'config_results': {}}
                        all_metric_results[depth_metric_name]['config_results'][config_info['name']] = result

print("\nAll Config Results Summary:")
for metric_name in all_metric_results:
    print(f"\n{metric_name}:")
    for config_name, result in all_metric_results[metric_name]['config_results'].items():
        if config_name == 'scorer':
            best_hypo = result['best_hypothesis']
            score = result['hypotheses'][best_hypo]['payload']['scores']['final_score']
            print(f"  {config_name}: {best_hypo} (Score: {score:.2f})")
        else:
            print(f"  {config_name}: {result.get('name', 'Analysis')}")

# Extract raw results for visualization (from scorer config only)
all_results = {}
for metric_name in all_metric_results:
    if 'scorer' in all_metric_results[metric_name]['config_results']:
        scorer_result = all_metric_results[metric_name]['config_results']['scorer']
        all_results[metric_name] = {
            hypo_name: hypo_info['payload'] 
            for hypo_name, hypo_info in scorer_result['hypotheses'].items()
        }

# %% [markdown]
# ## 5. Figure Generation Functions

# %% [figure_generators]
def create_hypothesis_chart(df: pd.DataFrame, **params):
    """Generate hypothesis analysis chart (returns figure object)."""
    metric_name = params['metric_name']
    hypo_names = params['hypo_names']
    metric_anomaly_info = params['metric_anomaly_info']
    hypo_results = params['hypo_results']
    ranked_hypos = params['ranked_hypos']
    
    fig = create_multi_hypothesis_plot(
        df=df,
        metric_col=metric_name,
        hypo_cols=hypo_names,
        metric_anomaly_info=metric_anomaly_info,
        hypo_results=hypo_results,
        ordered_hypos=ranked_hypos
    )
    
    # Add score formula (always sign-based)
    add_score_formula(fig)
    
    return fig

def create_scatter_chart(df: pd.DataFrame, **params):
    """Generate scatter plot chart (returns figure object)."""
    metric_name = params['metric_name']
    hypo_names = params['hypo_names']
    metric_anomaly_info = params['metric_anomaly_info']
    expected_directions = params['expected_directions']
    
    fig = create_scatter_grid(
        df=df,
        metric_col=metric_name,
        hypo_cols=hypo_names,
        metric_anomaly_info=metric_anomaly_info,
        expected_directions=expected_directions
    )
    
    return fig

def create_depth_chart(df: pd.DataFrame, **params):
    """Generate depth analysis chart (returns figure object)."""
    metric_col = params['metric_col']
    title = params['title']
    row_value = params.get('row_value')
    summary_df = params.get('summary_df')  # Get the raw numeric data for plotting
    
    # Use the raw summary_df for plotting (has numeric contribution values)
    fig = plot_subregion_bars(
        df_slice=summary_df,
        metric_col=metric_col,
        title=title,
        row_value=row_value,
        figsize=(12, 6)
    )
    
    return fig

# %% [markdown]
# ## 6. Integrated Analysis with Live Preview and Slide Creation

# %% [integrated_analysis]
print("\n" + "="*80)
print("INTEGRATED ANALYSIS WITH LIVE PREVIEW AND SLIDE CREATION")
print("="*80)

# Add title slide
@dual_output(console=True, slide=True, slide_builder=slides, layout_type='text', show_figures=True)
def create_title_slide():
    return SlideContent(
        title="RCA Analysis Results",
        text_template="""
Root Cause Analysis completed for {{ num_metrics }} metrics across {{ num_regions }} regions.

Analysis Summary:
• Hypothesis scoring using sign-based method
• {{ num_anomalies }} anomalies detected
• Best performing hypotheses identified
• Depth analysis included for sub-regional insights

Generated on: {{ timestamp }}
        """,
        template_params={
            'num_metrics': len(metric_names),
            'num_regions': len(regions) - 1,  # Exclude Global
            'num_anomalies': len(metric_anomaly_map),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    )

print("\n📋 CREATING TITLE SLIDE:")
title_content, title_results = create_title_slide()
print(title_results['console'])

# NOTE: Summary slide will be created AFTER all processing is complete
# because we need to process all hypothesis types first to know which have summary templates

# %% [markdown]
# ## 7. Depth Analysis with Integrated Preview

# %% [depth_analysis_integrated]
# Note: Depth analysis is now integrated into the config processing above
print("\n" + "="*80)
print("DEPTH ANALYSIS INTEGRATED INTO CONFIG PROCESSING")
print("="*80)
print("Depth analysis has been processed as part of the 'depth' config type above")

# %% [markdown]
# ## 8. Process Each Metric with All Its Slides Together

# %% [process_metrics_integrated]
# Process each metric with all its slides together from all config types
for metric_name in all_metric_results:
    print(f"\n" + "="*60)
    print(f"ANALYZING METRIC: {metric_name}")
    print("="*60)
    
    metric_configs = all_metric_results[metric_name]['config_results']
    
    # Process each config type for this metric
    for config_name, config_result in metric_configs.items():
        
        if config_name == 'scorer':
            # Scorer config - create hypothesis and scatter slides
            slides_data = config_result['slides']
            best_hypo = config_result['best_hypothesis']
            
            print(f"Processing scorer config - best hypothesis: {best_hypo}")
            
            # 1. Hypothesis Analysis Slide
            @dual_output(console=True, slide=True, slide_builder=slides, layout_type=slides_data['hypothesis']['layout_type'], show_figures=True)
            def create_hypothesis_slide():
                return SlideContent(
                    title=slides_data['hypothesis']['title'],
                    text_template=slides_data['hypothesis']['text'],  # Already rendered
                    df=df,
                    figure_generator=create_hypothesis_chart,
                    template_params={
                        'metric_name': metric_name,
                        'hypo_names': metric_hypo_map[metric_name],
                        'metric_anomaly_info': metric_anomaly_map[metric_name],
                        'hypo_results': all_results[metric_name],
                        'ranked_hypos': get_ranked_hypotheses(all_results[metric_name])
                    }
                )
            
            print(f"\n📊 HYPOTHESIS ANALYSIS FOR {metric_name}:")
            hyp_content, hyp_results = create_hypothesis_slide()
            
            # 2. Scatter Plot Analysis Slide
            @dual_output(console=True, slide=True, slide_builder=slides, layout_type=slides_data['scatter']['layout_type'], show_figures=False)
            def create_scatter_slide():
                return SlideContent(
                    title=slides_data['scatter']['title'],
                    text_template=slides_data['scatter']['text'],  # Empty for scatter
                    df=df,
                    figure_generator=create_scatter_chart,
                    template_params={
                        'metric_name': metric_name,
                        'hypo_names': metric_hypo_map[metric_name],
                        'metric_anomaly_info': metric_anomaly_map[metric_name],
                        'expected_directions': expected_directions
                    }
                )
            
            print(f"\n📈 SCATTER ANALYSIS FOR {metric_name} (Path Mode):")
            scatter_content, scatter_results = create_scatter_slide()
            
        elif config_name == 'depth':
            # Depth config - create depth analysis slide
            slide_info = config_result['slide']
            
            @dual_output(console=True, slide=True, slide_builder=slides, layout_type=slide_info['layout_type'], show_figures=True)
            def create_depth_slide():
                return SlideContent(
                    title=slide_info['title'],
                    text_template=slide_info['text'],  # Already rendered
                    df=slide_info['table_df'],  # Clean summary table with slice as index
                    figure_generator=create_depth_chart,
                    template_params={
                        'metric_col': config_result['payload']['metric_col'],
                        'title': f"{config_result['name']} by sub-regions",
                        'row_value': config_result['payload'].get('row_value'),
                        'summary_df': config_result['payload'].get('summary_df')
                    }
                )
            
            print(f"\n🔍 DEPTH ANALYSIS FOR {config_result['name']}:")
            depth_content, depth_results_output = create_depth_slide()

# %% [markdown] 
# ## 9. Create Summary Slide (After All Processing)

# %% [create_summary_after_processing]
print("\n" + "="*80)
print("CREATING SUMMARY SLIDE (AFTER ALL PROCESSING)")
print("="*80)

# Collect summary texts from ALL config types
summary_texts = {}
for metric_name in all_metric_results:
    config_results = all_metric_results[metric_name]['config_results']
    
    # Combine summaries from all config types for this metric
    metric_summaries = []
    
    for config_name, config_result in config_results.items():
        if config_name == 'scorer':
            # For scorer: only show the best hypothesis summary
            summary_text = config_result.get('summary_text')
            if summary_text:
                metric_summaries.append(f"{summary_text}")
        else:
            # For other types (depth, etc.): show their summary text
            summary_text = config_result.get('summary_text')
            if summary_text:
                metric_summaries.append(f"{config_name.title()}: {summary_text}")
    
    # Combine all summaries for this metric
    if metric_summaries:
        summary_texts[metric_name] = "\n\n".join(metric_summaries)

print(f"Found summary templates for {len(summary_texts)} out of {len(all_metric_results)} metrics")
for metric_name, combined_summary in summary_texts.items():
    print(f"\n{metric_name} summary sources:")
    # Count how many config types contributed
    source_count = combined_summary.count('\n\n') + 1 if combined_summary else 0
    print(f"  Combined from {source_count} config type(s)")

# Create summary slide using the specialized function
if summary_texts:  # Only create if we have any summary texts
    create_metrics_summary_slide(
        slide_layouts=slides,
        df=df,
        metrics_text=summary_texts,
        metric_anomaly_map=metric_anomaly_map,
        title="Metrics Summary"
    )
    print("📊 CREATED METRICS SUMMARY SLIDE WITH MULTI-CONFIG INSIGHTS")
else:
    print("⚠️  No summary templates found - skipping summary slide")

# %% [markdown]
# ## 10. Save and Upload Presentation

# %% [save_and_upload]
print("\n" + "="*80)
print("SAVING AND UPLOADING PRESENTATION")
print("="*80)

# Create output directory
output_dir = os.path.abspath('output')
os.makedirs(output_dir, exist_ok=True)

# Save the presentation
presentation_path = slides.save("RCA_Analysis_Results", output_dir)
print(f"\n✅ Presentation saved locally: {presentation_path}")

# Optional: Upload to Google Drive
try:
    upload_result = upload_to_google_drive(
        file_path=presentation_path,
        # user_email="your.email@company.com",  # Replace with actual email
        credentials_path="credentials.json",  # Optional: for local auth
        token_path="token.json"  # Optional: for local auth
    )
    print(f"✅ Uploaded to Google Drive: {upload_result['gdrive_url']}")
    if 'folder_url' in upload_result:
        print(f"📁 User folder: {upload_result['folder_url']}")
except Exception as e:
    print(f"⚠️  Upload to Google Drive failed: {e}")
    print("   (This is normal if credentials are not configured)")

# %% [markdown]
# ## 11. Summary

# %% [summary]
print("\n" + "="*80)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*80)

print(f"\n📊 METRICS ANALYZED: {len(metric_names)}")
for metric in metric_names:
    if metric in all_results:  # Only metrics with scorer results
        best_hypo = max(all_results[metric].items(), key=lambda x: x[1]['scores']['final_score'])
        print(f"   • {metric}: Best hypothesis = {best_hypo[0]} (Score: {best_hypo[1]['scores']['final_score']:.2f})")

print(f"\n🔧 CONFIG TYPES PROCESSED: {len(configs)}")
for config_info in configs:
    config_type_desc = "Ranking (winner selection)" if config_info['type'] == 'ranking' else "Direct (1:1 mapping)"
    print(f"   • {config_info['name']}: {config_type_desc}")

print(f"\n📄 PRESENTATION: {presentation_path}")
file_size = os.path.getsize(presentation_path) / 1024  # KB
print(f"   • File size: {file_size:.1f} KB")
print(f"   • Slides created with integrated preview workflow")

print(f"\n🎯 KEY FEATURES DEMONSTRATED:")
print(f"   • @dual_output decorator for preview-then-create workflow")
print(f"   • Flexible figure display: show_figures=True/False for different environments")
print(f"   • Multi-config architecture: scorer (ranking) + depth (direct) + others")
print(f"   • Simplified uniform processing for all config types")
print(f"   • Rich templates from config files properly rendered")
print(f"   • Figure generators return objects instead of saving files")
print(f"   • Proper table layout for depth analysis")
print(f"   • Jinja2 templating for dynamic content")
print(f"   • Integrated analysis without intermediate file creation")
print(f"   • Content-driven slide layouts")
print(f"   • Sign-based scoring for robust hypothesis evaluation")

print(f"\n✨ SUCCESS! Complete RCA analysis with integrated slide creation!")

# %% [cleanup]
# Close any remaining plots
plt.close('all') 
