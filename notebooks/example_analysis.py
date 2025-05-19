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

# Add package to path
sys.path.append(str(Path.cwd().parent))

from rca_package import (
    load_config,
    get_metric_info,
    get_hypothesis_info,
    get_all_metrics,
    get_relevant_hypotheses,
    get_expected_directions,
    get_metric_hypothesis_map,
    get_template,
    get_display_name,
    get_scoring_method
)
from rca_package.hypothesis_scorer import (
    score_all_hypotheses,
    get_ranked_hypotheses,
    create_multi_hypothesis_plot,
    create_scatter_grid,
    add_score_formula
)
from rca_package.make_slides import create_metrics_presentation

# %% [markdown]
# ## 1. Load Configuration

# %% [load_config]
# Load configuration
config_path = '../configs/config_scorer.yaml'
config = load_config(config_path)
print("Loaded configuration successfully")

# %% [markdown]
# ## 2. Create Sample Data

# %% [create_data]
# Create sample data
np.random.seed(42)
regions = ["Global", "North America", "Europe", "Asia", "Latin America"]

# Create test data with multiple metrics and hypotheses
data = {
    # Metrics
    'conversion_rate_pct': np.array([0.12, 0.08, 0.11, 0.13, 0.10]),
    'avg_order_value': np.array([75.0, 65.0, 80.0, 85.0, 72.0]),
    'customer_satisfaction': np.array([4.2, 3.8, 4.3, 4.5, 4.0]),
    
    # Hypotheses
    'bounce_rate_pct': np.array([0.35, 0.45, 0.32, 0.28, 0.34]),
    'page_load_time': np.array([2.4, 3.8, 2.2, 1.9, 2.5]),
    'session_duration': np.array([180, 120, 190, 210, 175]),
    'pages_per_session': np.array([4.2, 3.1, 4.5, 4.8, 4.0]),
    'new_users_pct': np.array([0.25, 0.18, 0.28, 0.30, 0.23])
}

# Create DataFrame
df = pd.DataFrame(data, index=regions)
print("\nSample Data:")
print(df)

# %% [markdown]
# ## 3. Prepare Analysis Parameters

# %% [prepare_analysis]
# Get all metrics and their hypotheses
metric_cols = get_all_metrics(config)
metric_hypo_map = get_metric_hypothesis_map(config)
expected_directions = get_expected_directions(config)

# Create metric anomaly map
metric_anomaly_map = {}
for metric_col in metric_cols:
    metric_info = get_metric_info(config, metric_col)
    anomalous_region = 'North America'  # For this example
    
    metric_anomaly_map[metric_col] = {
        'anomalous_region': anomalous_region,
        'metric_val': df.loc[anomalous_region, metric_col],
        'global_val': df.loc['Global', metric_col],
        'direction': 'higher' if df.loc[anomalous_region, metric_col] > df.loc['Global', metric_col] else 'lower',
        'magnitude': abs((df.loc[anomalous_region, metric_col] - df.loc['Global', metric_col]) / df.loc['Global', metric_col] * 100),
        'higher_is_better': metric_info.get('higher_is_better', True)
    }

print("\nMetric Anomaly Map:")
for metric, info in metric_anomaly_map.items():
    print(f"\n{metric}:")
    for key, value in info.items():
        print(f"  {key}: {value}")

# %% [markdown]
# ## 4. Score Hypotheses

# %% [score_hypotheses]
# Score hypotheses for each metric
all_results = {}
for metric_col in metric_cols:
    hypo_cols = metric_hypo_map[metric_col]
    metric_info = get_metric_info(config, metric_col)
    scoring_method = metric_info.get('scoring_method', 'standard')
    
    hypo_results = score_all_hypotheses(
        df=df,
        metric_col=metric_col,
        hypo_cols=hypo_cols,
        metric_anomaly_info=metric_anomaly_map[metric_col],
        expected_directions=expected_directions,
        scoring_method=scoring_method
    )
    all_results[metric_col] = hypo_results

print("\nHypothesis Scores:")
for metric, results in all_results.items():
    print(f"\n{metric}:")
    for hypo, result in results.items():
        print(f"  {hypo}: {result['scores']['final_score']:.2f}")

all_results

# %% [markdown]
# ## 5. Create Visualizations

def structure_metric_results(
    metric_col: str,
    metric_anomaly_info: Dict[str, Any],
    hypo_results: Dict[str, Dict[str, Any]],
    figure_paths: List[Dict[str, str]],
    config: Dict[str, Any],
    best_hypo_name: str = None
) -> Dict[str, Any]:
    """
    Structure the results for a metric in the requested format.
    """
    # Get all hypotheses for this metric
    hypotheses = {}
    for hypo_name, hypo_result in hypo_results.items():
        # Get templates
        template = get_template(config, metric_col, hypo_name, 'template')
        summary_template = get_template(config, metric_col, hypo_name, 'summary_template')
        
        # Prepare parameters for templates
        parameters = {
            'region': metric_anomaly_info['anomalous_region'],
            'metric_name': get_display_name(config, metric_col),
            'metric_deviation': metric_anomaly_info['magnitude'],
            'metric_dir': metric_anomaly_info['direction'],
            'hypo_name': get_display_name(config, hypo_name, 'hypothesis'),
            'hypo_dir': hypo_result['direction'],
            'hypo_delta': hypo_result['magnitude'],
            'ref_hypo_val': hypo_result['ref_hypo_val'],
            'score': hypo_result['scores']['final_score'],
            'explained_ratio': hypo_result['scores']['explained_ratio'] * 100
        }
        
        # Store hypothesis info
        hypotheses[hypo_name] = {
            "hypothesis": hypo_name,
            "name": get_display_name(config, hypo_name, 'hypothesis'),
            "type": "directional",
            "scoring_method": hypo_result.get('scoring_method', 'standard'),
            "score": hypo_result['scores']['final_score'],
            "selected": hypo_name == best_hypo_name,
            "template": template,
            "summary_template": summary_template,
            "parameters": parameters,
            "payload": hypo_result
        }
    
    # Create the complete metric result structure
    metric_result = {
        **metric_anomaly_info,  # Include all fields from metric_anomaly_info
        "figure_paths": figure_paths,
        "hypotheses": hypotheses
    }
    
    return metric_result

# %% [create_visualizations]
# Create output directory with absolute path (outside notebooks)
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
print(f"\nCreating output directory at: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Store all metric results
all_metric_results = {}

# Create visualizations for each metric
for metric_col in metric_cols:
    hypo_cols = metric_hypo_map[metric_col]
    hypo_results = all_results[metric_col]
    ranked_hypos = get_ranked_hypotheses(hypo_results)
    best_hypo_name, best_hypo_result = ranked_hypos[0]
    
    # Get metric display name
    metric_display_name = get_display_name(config, metric_col)
    
    # Create multi-hypothesis plot (bar chart)
    fig = create_multi_hypothesis_plot(
        df=df,
        metric_col=metric_col,
        hypo_cols=hypo_cols,
        metric_anomaly_info=metric_anomaly_map[metric_col],
        hypo_results=hypo_results,
        ordered_hypos=ranked_hypos
    )
    
    # Add title and axis labels using YAML config
    fig.suptitle(f"Hypothesis Analysis for {metric_display_name}", fontsize=14)
    ax = fig.axes[0]
    ax.set_xlabel(get_display_name(config, metric_col, 'metric'), fontsize=12)
    ax.set_ylabel("Hypothesis Score", fontsize=12)
    
    # Add score cards and template text
    template = get_template(config, metric_col, best_hypo_name, 'template')
    if template:
        context = {
            'region': metric_anomaly_map[metric_col]['anomalous_region'],
            'metric_name': metric_display_name,
            'metric_deviation': metric_anomaly_map[metric_col]['magnitude'],
            'metric_dir': metric_anomaly_map[metric_col]['direction'],
            'hypo_name': get_display_name(config, best_hypo_name, 'hypothesis'),
            'hypo_dir': best_hypo_result['direction'],
            'hypo_delta': best_hypo_result['magnitude'],
            'ref_hypo_val': best_hypo_result['ref_hypo_val'],
            'score': best_hypo_result['scores']['final_score'],
            'explained_ratio': best_hypo_result['scores']['explained_ratio'] * 100
        }
        template_text = Template(template).render(**context)
        fig.text(0.5, 0.95, template_text, ha='center', va='top', fontsize=10, wrap=True)
    
    scoring_method = get_scoring_method(config)
    # Add score formula
    add_score_formula(fig, is_sign_based=(scoring_method == 'sign_based'))
    
    # Save bar chart
    bar_figure_path = os.path.join(output_dir, f"bar_{metric_col}.png")
    print(f"\nSaving bar chart to: {bar_figure_path}")
    fig.savefig(bar_figure_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    # Create scatter plot
    scatter_fig = create_scatter_grid(
        df=df,
        metric_col=metric_col,
        hypo_cols=hypo_cols,
        metric_anomaly_info=metric_anomaly_map[metric_col],
        expected_directions=get_expected_directions(config)
    )
    
    # Style scatter plot
    scatter_fig.suptitle(f"Scatter Analysis for {metric_display_name}", fontsize=14)
    for ax in scatter_fig.axes:
        ax.set_xlabel(get_display_name(config, metric_col, 'metric'), fontsize=12)
        ax.set_ylabel(get_display_name(config, ax.get_title(), 'hypothesis'), fontsize=12)
    
    # Save scatter plot
    scatter_figure_path = os.path.join(output_dir, f"scatter_{metric_col}.png")
    print(f"Saving scatter plot to: {scatter_figure_path}")
    scatter_fig.savefig(scatter_figure_path, dpi=120, bbox_inches='tight')
    plt.close(scatter_fig)
    
    # Store figure paths
    figure_paths = [
        {
            'path': bar_figure_path,
            'title': f"Bar Analysis for {metric_display_name}"
        },
        {
            'path': scatter_figure_path,
            'title': f"Scatter Analysis for {metric_display_name}"
        }
    ]
    
    # Structure and store the results
    metric_result = structure_metric_results(
        metric_col=metric_col,
        metric_anomaly_info=metric_anomaly_map[metric_col],
        hypo_results=hypo_results,
        figure_paths=figure_paths,
        config=config,
        best_hypo_name=best_hypo_name
    )
    all_metric_results[metric_col] = metric_result

# Save results to JSON
results_file = os.path.join(output_dir, f"analysis_results_{scoring_method}.json")
print(f"\nSaving analysis results to: {results_file}")

# Convert numpy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):  # Handle numpy boolean type
        return bool(obj)
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
# ## 6. Create Presentation

# %% [create_presentation]
# Prepare metrics text using summary templates from best hypotheses
metrics_text = {}
for metric_col, metric_result in all_metric_results.items():
    # Find the selected (best) hypothesis
    best_hypo = next((h for h in metric_result['hypotheses'].values() if h['selected']), None)
    if best_hypo and best_hypo['summary_template']:
        # Use summary template and parameters from the best hypothesis
        metrics_text[metric_col] = Template(best_hypo['summary_template']).render(**best_hypo['parameters'])

# Create presentation
result = create_metrics_presentation(
    df=df,
    metrics_text=metrics_text,
    metric_anomaly_map=metric_anomaly_map,
    figure_paths=[path for metric in all_metric_results.values() for path in metric['figure_paths']],
    output_dir=output_dir,
    ppt_filename="Metrics_Analysis.pptx",
    upload_to_gdrive=True,
    gdrive_credentials_path="../credentials.json",
    use_oauth=True
)

print("\nPresentation Results:")
print(f"Local path: {result['local_path']}")
if 'gdrive_url' in result:
    print(f"Google Drive URL: {result['gdrive_url']}")

# %% [markdown]
# ## 7. Cleanup (Optional)

# %% [cleanup]
# Close any remaining plots
plt.close('all') 
