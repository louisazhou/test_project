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

# %% [setup]
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from jinja2 import Template
import json

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
    get_scoring_method,
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
from rca_package.make_slides import create_flexible_presentation
from rca_package.depth_spotter import (
    create_synthetic_data,
    analyze_region_depth,
    plot_subregion_bars
)



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
scoring_method = get_scoring_method(config)

# Create metric anomaly map
metric_anomaly_map = {}
for metric_name in metric_names:
    anomaly_info = detect_snapshot_anomaly_for_column(df, 'Global', column=metric_name)
    if anomaly_info:
        metric_anomaly_map[metric_name] = anomaly_info

print("\nMetric Anomaly Map:")
for metric, info in metric_anomaly_map.items():
    print(f"\n{metric}:")
    for key, value in info.items():
        print(f"  {key}: {value}")

# %% [markdown]
# ## 4. Score Hypotheses

# %% [score_hypotheses]
# Score hypotheses for each metric using the new streamlined approach
all_metric_results = process_metrics_with_structured_results(
    df=df,
    metric_cols=metric_names,
    hypo_cols=[],  # Not used since we have metric_hypothesis_map
    metric_anomaly_map=metric_anomaly_map,
    expected_directions=expected_directions,
    scoring_method=scoring_method,
    metric_hypothesis_map=metric_hypo_map,
    config=config,
    get_template_func=get_template
)

print("\nHypothesis Scores:")
for metric, metric_result in all_metric_results.items():
    print(f"\n{metric}:")
    for hypo_name, hypo_info in metric_result['hypotheses'].items():
        score = hypo_info['payload']['scores']['final_score']
        print(f"  {hypo_name}: {score:.2f}")

# Extract raw results for visualization
all_results = {}
for metric_name, metric_result in all_metric_results.items():
    all_results[metric_name] = {
        hypo_name: hypo_info['payload'] 
        for hypo_name, hypo_info in metric_result['hypotheses'].items()
    }

# %% [markdown]
# ## 5. Create Visualizations

# %% [create_visualizations]
# Create output directory with absolute path (outside notebooks)
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
print(f"\nCreating output directory at: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Create visualizations for each metric and add figure paths to results
for metric_name in metric_names:
    # Get technical name for file naming
    metric_technical_name = get_technical_name(config, metric_name, 'metric')
    
    hypo_names = metric_hypo_map[metric_name]
    hypo_results = all_results[metric_name]
    ranked_hypos = get_ranked_hypotheses(hypo_results)
    best_hypo_name, best_hypo_result = ranked_hypos[0]
    
    # Create multi-hypothesis plot (bar chart)
    fig = create_multi_hypothesis_plot(
        df=df,
        metric_col=metric_name,
        hypo_cols=hypo_names,
        metric_anomaly_info=metric_anomaly_map[metric_name],
        hypo_results=hypo_results,
        ordered_hypos=ranked_hypos
    )
    
    # Add template text and score formula
    template = get_template(config, metric_name, best_hypo_name, 'template')
    # Comment out add_template_text - we'll add text to slide instead
    # if template:
    #     add_template_text(
    #         fig=fig,
    #         template=template,
    #         best_hypo_name=best_hypo_name,
    #         best_hypo_result=best_hypo_result,
    #         metric_anomaly_info=metric_anomaly_map[metric_name],
    #         metric_col=metric_name
    #     )
    
    scoring_method = get_scoring_method(config)
    # Add score formula
    add_score_formula(fig, is_sign_based=(scoring_method == 'sign_based'))
    
    # Save bar chart using technical name for filename
    bar_figure_path = os.path.join(output_dir, f"bar_{metric_technical_name}_{scoring_method}.png")
    print(f"\nSaving bar chart to: {bar_figure_path}")
    fig.savefig(bar_figure_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    # Render template text for slide
    template_text = ""
    if template:
        template_text = render_template_text(
            template=template,
            best_hypo_name=best_hypo_name,
            best_hypo_result=best_hypo_result,
            metric_anomaly_info=metric_anomaly_map[metric_name],
            metric_col=metric_name
        )
    
    # Create scatter plot
    scatter_fig = create_scatter_grid(
        df=df,
        metric_col=metric_name,
        hypo_cols=hypo_names,
        metric_anomaly_info=metric_anomaly_map[metric_name],
        expected_directions=get_expected_directions(config)
    )
    
    # Save scatter plot using technical name for filename
    scatter_figure_path = os.path.join(output_dir, f"scatter_{metric_technical_name}.png")
    print(f"Saving scatter plot to: {scatter_figure_path}")
    scatter_fig.savefig(scatter_figure_path, dpi=120, bbox_inches='tight')
    plt.close(scatter_fig)
    
    # Add figure paths to the existing structured results
    figure_paths = [
        {
            'path': bar_figure_path,
            'title': metric_name,
            'text': template_text  # Add rendered template text
        },
        {
            'path': scatter_figure_path,
            'title': metric_name
        }
    ]
    
    # Add figure paths to the existing metric result
    all_metric_results[metric_name]['figure_paths'] = figure_paths

print("\nCreated visualizations:")
for metric, result in all_metric_results.items():
    print(f"\n{metric}:")
    for fig_info in result['figure_paths']:
        print(f"  {fig_info['title']}: {fig_info['path']}")
        if os.path.exists(fig_info['path']):
            print(f"  ✓ File exists")
        else:
            print(f"  ✗ File not found")

# %% [markdown]
# ## 6. Depth Analysis

# %% [depth_analysis]
# Create synthetic sub-regional data for depth analysis
print("\n" + "="*60)
print("DEPTH ANALYSIS")
print("="*60)

# Get synthetic sub-regional data
sub_df = create_synthetic_data()
print(f"\nCreated synthetic sub-regional data with {len(sub_df)} slices")
print("\nSub-regional data preview:")
print(sub_df.head())

# Run depth analysis for the anomalous region
anomalous_region = "North America"  # This matches our synthetic data
depth_config_path = 'configs/config_depth.yaml'

print(f"\nRunning depth analysis for: {anomalous_region}")

# Load depth config using yaml_processor
depth_config = load_config(depth_config_path)

depth_results = analyze_region_depth(
    sub_df=sub_df,
    anomalous_region=anomalous_region,
    config=depth_config
)

print(f"\nDepth analysis found {len(depth_results)} metrics")

# Create bar charts for depth analysis and integrate into results
for hypo_name, result in depth_results.items():
    if result['type'] == 'depth_spotter':
        payload = result['payload']
        region_df = payload['region_df']
        metric_col = payload['metric_col']
        display_name = result['name']
        
        # Get technical name for file naming
        metric_technical_name = get_technical_name(config, display_name, 'metric')
        
        # Create actual data bar chart
        title = f"{display_name} by {anomalous_region} sub-regions"
        
        fig = plot_subregion_bars(
            df_slice=region_df,
            metric_col=metric_col,
            title=title,
            config=depth_config,
            row_value=payload.get('row_value'),
            figsize=(12, 6)
        )
        
        # Save figure using technical name for filename
        filename = f"{output_dir}/subregion_{metric_technical_name}.png"
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # Render template text
        rendered_text = ""
        if result.get('template') and result.get('parameters'):
            template = Template(result['template'])
            rendered_text = template.render(**result['parameters'])
        
        # Add to the corresponding metric's results
        metric_name = hypo_name.replace('_in_subregion', '')
        for metric_col, metric_result in all_metric_results.items():
            if metric_col == metric_name:
                # Add depth hypothesis to this metric's hypotheses
                metric_result['hypotheses'][hypo_name] = result
                
                # Add subregion figure with rendered text to figure paths
                subregion_figure = {
                    'path': filename,
                    'title': f"{display_name} - Depth Analysis",
                    'text': rendered_text  # Store rendered text for slides
                }
                metric_result['figure_paths'].append(subregion_figure)
                break

print("\nDepth analysis completed and integrated into results")

# Print structured depth results
print(f"\n{'-'*40}")
print("DEPTH ANALYSIS STRUCTURED RESULTS")
print(f"{'-'*40}")

for hypo_name, result in depth_results.items():
    print(f"\nHypothesis: {hypo_name}")
    print(f"Name: {result['name']}")
    print(f"Type: {result['type']}")
    
    # Render template
    if result['template']:
        template = Template(result['template'])
        rendered_text = template.render(**result['parameters'])
        print(f"Template Text:\n{rendered_text}")

# %% [markdown]
# ## 7. Save Results 

# %% [save_results]
# Save results to JSON
results_file = os.path.join(output_dir, f"analysis_results.json")
print(f"\nSaving analysis results to: {results_file}")

# Convert numpy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_): 
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='index')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Convert and save results
json_results = convert_numpy_types(all_metric_results)
with open(results_file, 'w') as f:
    json.dump(json_results, f, indent=2)

print(f"Successfully saved results to {results_file}")

# %% [markdown]
# ## 8. Create Presentation

# %% [create_presentation]
# Prepare metrics text using summary templates from best hypotheses
metrics_text = {}
for metric_name, metric_result in all_metric_results.items():
    # Find the selected (best) hypothesis
    best_hypo = next((h for h in metric_result['hypotheses'].values() if h.get('selected', False)), None)
    if best_hypo and best_hypo.get('summary_template'):
        # Use the metric name directly as the key (no conversion needed!)
        metrics_text[metric_name] = Template(best_hypo['summary_template']).render(**best_hypo['parameters'])

# Create presentation
builder = create_flexible_presentation(
    output_dir=output_dir,
    ppt_filename="Metrics_Analysis.pptx",
    upload_to_gdrive=True,
    gdrive_credentials_path="credentials.json",  # Explicit path required
    gdrive_token_path="token.json"  # Explicit path required
)

# Add metrics summary slide
builder['add_summary_slide'](
    df=df,
    metrics_text=metrics_text,
    metric_anomaly_map=metric_anomaly_map,
    title="Metrics Summary"
)

# Add figure slides
figure_paths = [path for metric in all_metric_results.values() for path in metric['figure_paths']]
for fig_info in figure_paths:
    figure_path = fig_info.get('path')
    title = fig_info.get('title')
    text = fig_info.get('text')  # Get rendered text if available
    
    # Check if this is a chart with text (like depth analysis or bar charts)
    if text:
        # Determine slide type based on content
        if 'Depth Analysis' in title:
            slide_type = 'depth_analysis'
        elif 'bar_' in figure_path:
            slide_type = 'bar_chart'
        else:
            slide_type = 'standard'
            
        # Create figure with text slide - use 'top' position with appropriate slide type
        builder['add_figure_with_text_slide'](
            title=title,
            figure_path=figure_path,
            text=text,
            text_position='top',  # Changed from 'bottom' to 'top'
            slide_type=slide_type  # Use dynamic slide type
        )
    else:
        # Add regular figure slide
        builder['add_figure_slide'](figure_path=figure_path, title=title)

# Save and upload
result = builder['save_and_upload']()

print("\nPresentation Results:")
print(f"Local path: {result['local_path']}")
if 'gdrive_url' in result:
    print(f"Google Drive URL: {result['gdrive_url']}")

# %% [markdown]
# ## 9. Cleanup (Optional)

# %% [cleanup]
# Close any remaining plots
plt.close('all') 
