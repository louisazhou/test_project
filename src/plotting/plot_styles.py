"""
Central styling for plots.

This module provides consistent styling for all plots in the RCA system.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Centralized style settings
STYLE = {
    'colors': {
        'anomaly_positive': '#2ecc71',  # Green
        'anomaly_negative': '#e74c3c',  # Red
        'global_line': '#34495e',       # Dark blue-gray
        'confidence_band': '#AED6F1',   # Lighter blue
        'default_bar': '#BDC3C7',       # Lighter gray
        'highlight': '#5DADE2',         # Blue highlight
        'primary': '#3498DB',           # Primary blue
        'reference': '#34495e',         # Reference line
        'text': '#2c3e50',              # Dark text
        'highlight_text': 'dodgerblue', # Highlight text for [selected] / [Anomaly Detected in KPI]
        'score_color': '#AF7AC5',       # Purple for score
        'score_components': {
            'direction_alignment': '#3498DB',  # Blue
            'consistency': '#4ECDC4',         # Teal
            'hypo_z_score_norm': '#FFC300',   # Yellow/Orange
            'explained_ratio': '#FF9F43'      # Orange
        }
    },
    'score_components': {
        'direction_alignment': {'weight': 0.3, 'name': 'Dir. Align'},
        'consistency': {'weight': 0.3, 'name': 'Consistency'},
        'hypo_z_score_norm': {'weight': 0.2, 'name': 'Hypo Z-Score'},
        'explained_ratio': {'weight': 0.2, 'name': 'Expl. Ratio'}
    },
    'score_component_order': [
        'direction_alignment',
        'consistency',
        'hypo_z_score_norm',
        'explained_ratio'
    ],
    'anomaly_band_alpha': 0.2
}

def setup_style():
    """Set consistent style for all visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.grid": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "font.family": "sans-serif",
        "font.size": 10
    }) 