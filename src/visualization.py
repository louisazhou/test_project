import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional
import numpy as np
import logging
import functools # For decorator

# --- Decorator for Percentage Formatting ---
def format_yaxis_percentage(func):
    @functools.wraps(func)
    def wrapper(instance, ax, *args, **kwargs):
        # Extract metric/hypothesis name or identifier to check for '_pct'
        # This assumes the name is usually passed or inferable from data
        # Let's check kwargs first, then args if needed
        identifier = kwargs.get('metric_name') or kwargs.get('hypothesis_name')
        if not identifier and len(args) > 1:
             # Attempt to get identifier from args based on typical function signature
             # plot_metric: args[1] is metric_name
             # plot_hypothesis: args[1] is hypothesis_name
             if func.__name__ in ['plot_metric', 'plot_hypothesis']:
                 identifier = args[1] 

        is_percent = False
        if identifier and isinstance(identifier, str):
            is_percent = "_pct" in identifier.lower()
        # Additionally, check if y_label explicitly contains '%'
        y_label = kwargs.get('y_label', '')
        if not is_percent and '%' in y_label:
             is_percent = True
             
        # Call the original plotting function
        result = func(instance, ax, *args, **kwargs)
        
        # Apply percentage formatting if needed
        if is_percent:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
            current_ylabel = ax.get_ylabel()
            if '%' not in current_ylabel:
                 ax.set_ylabel(f"{current_ylabel} (%)")
                 
        return result
    return wrapper
# --- End Decorator ---

class RCAVisualizer:
    """Visualizer for Root Cause Analysis results.
    
    This class provides methods to create consistent, themed visualizations
    for RCA results. All plots follow a common style and can be combined
    flexibly like building blocks.
    """
    
    def __init__(self, cmap: str = "RdBu_r", anomaly_band_alpha: float = 0.2):
        """Create a reusable visualizer for all RCA visual output.
        
        Args:
            cmap: Colormap for heatmaps and gradients (default: "coolwarm")
            anomaly_band_alpha: Transparency for anomaly confidence bands (default: 0.2)
        """
        self.cmap = cmap
        self.anomaly_band_alpha = anomaly_band_alpha
        
        # Set default style
        self._set_style()
    
    def _set_style(self) -> None:
        """Set consistent style for all visualizations."""
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Set default font sizes
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
        
        # Set default colors
        self.colors = {
            'anomaly_positive': '#2ecc71',  # Green
            'anomaly_negative': '#e74c3c',  # Red
            'global_line': '#34495e',       # Dark blue-gray
            'confidence_band': '#AED6F1',   # Lighter blue for band
            'default_bar': '#BDC3C7',       # Lighter gray for default bars
            'highlight': '#5DADE2',         # Slightly different blue highlight
            'text': '#2c3e50',              # Dark blue-gray
            'score_color': '#AF7AC5',       # Purple for score
            'score_components': {
                'direction_alignment': '#3498DB', # Blue
                'consistency': '#4ECDC4',        # Teal
                'hypo_z_score_norm': '#FFC300',  # Yellow/Orange
                'explained_ratio': '#FF9F43'     # Orange
            }
        }
        
        # Score component weights and descriptions
        self.score_components = {
            'direction_alignment': {'weight': 0.3, 'name': 'Dir. Align'},
            'consistency': {'weight': 0.3, 'name': 'Consistency'},
            'hypo_z_score_norm': {'weight': 0.2, 'name': 'Hypo Z-Score'},
            'explained_ratio': {'weight': 0.2, 'name': 'Expl. Ratio'}
        }
        self.score_component_order = ['direction_alignment', 'consistency', 'hypo_z_score_norm', 'explained_ratio']

    def _create_scorecard(self, obj, x, y, value, component_name, color, is_figure=False, show_text=False, fontsize=None):
        """Create a colored scorecard with value or component name.
        
        Args:
            obj: The axis or figure to draw on
            x: x-position in axis/figure coordinates
            y: y-position in axis/figure coordinates
            value: The value to display (if show_text=False)
            component_name: Name to display (if show_text=True)
            color: Color for this component
            is_figure: Whether rendering on figure (True) or axis (False)
            show_text: Whether to show component name (True) or value (False)
            fontsize: Optional specific font size for the text inside the card
        """
        transform = obj.transFigure if is_figure else obj.transAxes
        # Use a much smaller base width - adjust as needed
        card_width = 0.09 if is_figure else 0.11 
        card_height = 0.025 if is_figure else 0.08
        
        # Create colored rectangle background
        rect = plt.Rectangle(
            (x, y), card_width, card_height, 
            facecolor=color, alpha=0.2, transform=transform
        )
        obj.add_artist(rect)
        
        # Add either value or component name, centered in the rectangle
        text_content = component_name if show_text else f"{value:.2f}"
        obj.text(
            x + card_width/2, y + card_height/2, text_content, 
            ha='center', va='center',
            transform=transform,
            fontweight='bold',
            color=color,
            fontsize=fontsize # Use the provided fontsize
        )
        
        return card_width  # Return width for positioning next element

    def _add_score_components_compact(self, ax, start_x, y, components_dict):
        """Add just the score component boxes without operators for non-root-cause axes."""
        card_spacing = 0.06  # Increased spacing slightly
        card_width = 0.1    # Keep smaller width
        card_height = 0.08   # Standard height
        text_size = 10        
        
        # Use the standard order defined in __init__
        ordered_components = [(comp, components_dict.get(comp, 0)) for comp in self.score_component_order]
        
        # Calculate total width needed (including space for the score value)
        num_items = len(ordered_components) + 1 # +1 for the score
        total_width = num_items * card_width + (num_items - 1) * card_spacing
        
        # Center the components
        x_pos = 0.5 - (total_width / 2)
        x_pos = max(0.01, x_pos) # Ensure not too far left
        
        # Add Score Value first (without the 'Score' text)
        score_value = components_dict.get('score', 0) # Get score from the dict
        score_color = self.colors.get('score_color', '#AF7AC5') # Use defined score color
        rect = plt.Rectangle(
            (x_pos, y), card_width, card_height, 
            facecolor=score_color, alpha=0.2, transform=ax.transAxes
        )
        ax.add_artist(rect)
        ax.text(
            x_pos + card_width/2, y + card_height/2, f"{score_value:.2f}", 
            ha='center', va='center',
            transform=ax.transAxes,
            fontweight='bold', color=score_color, fontsize=text_size
        )
        x_pos += card_width + card_spacing
        
        # Add component boxes
        for i, (component, value) in enumerate(ordered_components):
            if component not in self.colors['score_components']:
                continue # Skip if component color is not defined
            color = self.colors['score_components'][component]
            
            # Create scorecard
            rect = plt.Rectangle(
                (x_pos, y), card_width, card_height, 
                facecolor=color, alpha=0.2, transform=ax.transAxes
            )
            ax.add_artist(rect)
            
            # Add value
            ax.text(
                x_pos + card_width/2, y + card_height/2, f"{value:.2f}", 
                ha='center', va='center',
                transform=ax.transAxes,
                fontweight='bold', color=color, fontsize=text_size
            )
            
            x_pos += card_width + card_spacing

    def _add_score_components(self, obj, start_x, y, score_value, components_dict, is_figure=False, show_score=True, show_text=False):
        """Add score components in a consistent way for both axes and figure."""
        transform = obj.transFigure if is_figure else obj.transAxes
        card_spacing = 0.01 if is_figure else 0.02 
        text_size = 12 if is_figure else 11
        base_card_width = 0.09 if is_figure else 0.11 
        operator_spacing = 0.04 if is_figure else 0.05 
        score_card_height = 0.025 if is_figure else 0.08 # Height for score cards
        
        x_pos = start_x
        
        # Use the standard order defined in __init__
        ordered_components = [(comp, components_dict.get(comp, 0)) for comp in self.score_component_order]
        
        # For the figure bottom formula or root cause hypothesis
        if is_figure or show_score:
            score_label_color = self.colors.get('score_color', '#AF7AC5')
            
            if is_figure:
                # Figure bottom formula - uses text "Score"
                card_width = self._create_scorecard(
                    obj, x_pos, y, 0, "Score", 
                    score_label_color, is_figure=is_figure, show_text=True,
                    fontsize=text_size
                )
                x_pos += card_width + card_spacing/2
                obj.text(
                    x_pos, y + score_card_height/2,
                    " = ", ha='center', va='center',
                    color=score_label_color, fontsize=text_size, transform=transform
                )
                x_pos += operator_spacing
            else:
                # Root cause axis - uses text "Score (value)"
                label_text = f"Score ({score_value:.2f})"
                # Estimate a slightly wider card for the score + value text
                first_card_width_estimate = base_card_width * 1.15 
                # Draw the first card using the base width (text centering handles it)
                # but advance x_pos using the estimated *wider* width
                self._create_scorecard(
                    obj, x_pos, y, 0, label_text, 
                    score_label_color, is_figure=is_figure, show_text=True,
                    fontsize=text_size
                )
                x_pos += first_card_width_estimate + card_spacing 
        
        # Add remaining components based on the standard order
        for i, (component, value) in enumerate(ordered_components):
            if component not in self.score_components or component not in self.colors['score_components']:
                logging.warning(f"Skipping undefined score component: {component}")
                continue
                
            component_config = self.score_components[component]
            color = self.colors['score_components'][component]
            weight = component_config['weight']
            
            # Add "+" operator
            if i > 0 or not show_score: 
                obj.text(
                    x_pos, y + score_card_height/2,
                    " + ", ha='center', va='center',
                    color=color, fontsize=text_size, transform=transform 
                )
                x_pos += operator_spacing
            
            # Add weight and multiplication symbol
            obj.text(
                x_pos, y + score_card_height/2,
                f"{weight:.1f}×", ha='center', va='center',
                color=color, fontsize=text_size, transform=transform 
            )
            x_pos += 0.04 # Space after weight
            
            # Add component scorecard using the standard base_card_width
            component_text_size = text_size * 0.9 if not is_figure and not show_text else text_size
            card_width = self._create_scorecard(
                obj, x_pos, y, value, 
                component_config['name'],
                color, is_figure=is_figure, show_text=show_text,
                fontsize=component_text_size
            )
            x_pos += card_width + card_spacing
        
        return x_pos

    @format_yaxis_percentage # Apply decorator
    def plot_metric(
        self,
        ax: plt.Axes,
        metric_anomaly_data: pd.DataFrame, # Expects anomaly_df filtered for ONE metric
        metric_name: str,
        region_column: str, # Name of the region column used for index
        z_score_threshold: float = 1.0,
        title: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> plt.Axes:
        """Plot metric values with anomaly highlighting and confidence band."""
        if metric_anomaly_data.empty:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title(f"Metric: {title or metric_name} - No Data")
            return ax

        # Clear existing axes content to avoid double axes issues
        ax.clear()

        # Ensure data is indexed by the specified region column
        if metric_anomaly_data.index.name != region_column:
            # Check if region_column is in the columns (not in index)
            if region_column in metric_anomaly_data.columns:
                metric_anomaly_data = metric_anomaly_data.set_index(region_column)
            else:
                # Fall back to using the first column if region_column not found
                ax.text(0.5, 0.5, f"Error: '{region_column}' not found in data", ha='center', va='center')
                return ax
                
        # Sort by region for consistent order
        metric_anomaly_data = metric_anomaly_data.sort_index()

        # Get reference value and std
        ref_metric_val = metric_anomaly_data['ref_metric_val'].iloc[0]
        std_val = metric_anomaly_data['std'].iloc[0]

        # Create bar colors based on anomaly flags
        bar_colors = []
        for _, row in metric_anomaly_data.iterrows():
            if row.get('bad_anomaly', False):
                bar_colors.append(self.colors['anomaly_negative'])
            elif row.get('good_anomaly', False):
                bar_colors.append(self.colors['anomaly_positive'])
            else:
                bar_colors.append(self.colors['default_bar'])

        # Plot bars with regions as x-ticks
        regions = metric_anomaly_data.index.tolist()
        x_positions = np.arange(len(regions))
        bars = ax.bar(x_positions, metric_anomaly_data['metric_val'], color=bar_colors)
        
        # Set x-ticks explicitly using region names - centered under bars
        ax.set_xticks(x_positions)
        ax.set_xticklabels(regions, rotation=0)
        plt.setp(ax.get_xticklabels(), ha='center', fontsize=9)

        # Add reference line and confidence band
        ref_metric_val = metric_anomaly_data['ref_metric_val'].iloc[0]
        std_val = metric_anomaly_data['std'].iloc[0]
        
        ax.axhline(ref_metric_val, color=self.colors['global_line'],
                  linestyle='--', linewidth=1)

        # Add confidence band centered around the mean (WITH LABEL)
        if std_val > 0:
            ax.axhspan(ref_metric_val - std_val * z_score_threshold,
                       ref_metric_val + std_val * z_score_threshold,
                       color=self.colors['confidence_band'],
                       alpha=self.anomaly_band_alpha,
                       label=f'±{z_score_threshold:.1f} Std Dev') 
            # Only add legend if there's a band
            ax.legend(loc='upper right', fontsize=8)

        # --- Percentage Formatting --- 
        is_percent = "_pct" in metric_name.lower()
        value_format = '{:.1f}%' if is_percent else '{:.2f}'
        z_score_format = '(z={:.2f})'

        # Add value and z-score labels
        for i, (idx, row) in enumerate(metric_anomaly_data.iterrows()):
            val = row['metric_val']
            z_score = row['z_score']
            display_val = val * 100 if is_percent else val
            label_text = f"{value_format.format(display_val)}\n{z_score_format.format(z_score)}"
            ax.text(i, val, label_text, 
                    ha='center', va='bottom', color=self.colors['text'], fontsize=8)

        # Set title and labels (Decorator handles y-axis % formatting)
        ax.set_title(title or f"Metric: {metric_name}", fontsize=11)
        y_axis_label = y_label or "Metric Value"
        ax.set_ylabel(y_axis_label, fontsize=10)
        
        # Scale y-axis to use only 80% of vertical space for bars
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_min + y_range * 1.25)  # Add 25% extra space
        
        # Update x-axis labels to be horizontal
        ax.set_xticklabels(regions, rotation=0, ha='right')
        plt.setp(ax.get_xticklabels(), ha='right', fontsize=9)
        
        return ax

    @format_yaxis_percentage # Apply decorator
    def plot_hypothesis(
        self,
        ax: plt.Axes,
        hypothesis_data: pd.DataFrame, # Expects final_df filtered for ONE metric-hypothesis pair
        hypothesis_name: str,
        region_column: str, # Name of the region column used for index
        explaining_region: Optional[str] = None, # Region explained by *this* hypothesis (if it's the root cause)
        primary_anomaly_region: Optional[str] = None, # Primary anomaly region for the *metric*
        title: Optional[str] = None,
        y_label: Optional[str] = None,
        show_score_components: bool = True,
        **kwargs # Accept kwargs to allow decorator checks
    ) -> plt.Axes:
        """Plot hypothesis values, highlighting the explaining region if applicable."""
        if hypothesis_data.empty:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title(f"Hypothesis: {title or hypothesis_name} - No Data")
            return ax

        # Clear existing axes content
        ax.clear()

        # Ensure data is indexed by the specified region column
        if hypothesis_data.index.name != region_column:
            # Check if region_column is in the columns (not in index)
            if region_column in hypothesis_data.columns:
                hypothesis_data = hypothesis_data.set_index(region_column)
            else:
                # Fall back to using the first column if region_column not found
                ax.text(0.5, 0.5, f"Error: '{region_column}' not found in data", ha='center', va='center')
                return ax
                
        # Sort by region for consistent order
        hypothesis_data = hypothesis_data.sort_index()

        # Get reference value (should be consistent for the hypothesis)
        ref_hypo_val = hypothesis_data['ref_hypo_val'].iloc[0] if 'ref_hypo_val' in hypothesis_data.columns else np.nan

        # Create bar colors with highlighting for explaining region
        regions = hypothesis_data.index.tolist()
        bar_colors = [self.colors['default_bar']] * len(regions)
        
        # Find index of explaining region if provided
        explaining_idx = -1
        if explaining_region and explaining_region in regions:
            explaining_idx = regions.index(explaining_region)
            bar_colors[explaining_idx] = self.colors['highlight']

        # Plot bars with regions as x-ticks
        x_positions = np.arange(len(regions))
        bars = ax.bar(x_positions, hypothesis_data['hypothesis_val'], color=bar_colors)
        
        # Set x-ticks explicitly using region names - centered under bars
        ax.set_xticks(x_positions)
        ax.set_xticklabels(regions, rotation=0, ha='center')
        plt.setp(ax.get_xticklabels(), fontsize=9)

        # Add reference line (NO LABEL)
        if not np.isnan(ref_hypo_val):
            ax.axhline(ref_hypo_val, color=self.colors['global_line'],
                      linestyle='--', linewidth=1)

        # --- Percentage Formatting --- 
        value_col_name = 'hypothesis_val' 
        # Check name and also y_label passed in kwargs
        is_percent = "_pct" in hypothesis_name.lower() # Rely on name check from decorator mostly
        value_format = '{:.1f}%' if is_percent else '{:.1f}'
        
        # Add value labels
        for i, (idx, row) in enumerate(hypothesis_data.iterrows()):
            val = row[value_col_name]
            display_val = val * 100 if is_percent else val
            ax.text(i, val, value_format.format(display_val), ha='center', va='bottom',
                   color=self.colors['text'], fontsize=8)

        # Scale y-axis to use only 75% of vertical space for bars
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_min + y_range / 0.75)  # This ensures top 25% is free for score cards
        
        # Add score components if requested
        if show_score_components:
            # Determine if this hypothesis is the root cause for the explaining_region
            is_root_cause = explaining_region is not None and explaining_region == primary_anomaly_region
            
            # Fetch score data for the primary_anomaly_region if possible
            score_region_to_display = primary_anomaly_region
            if score_region_to_display is None or score_region_to_display not in hypothesis_data.index:
                # Fallback: if primary region missing for this hypo, use highest score region for this hypo
                if 'score' in hypothesis_data.columns and not hypothesis_data.empty:
                     score_region_to_display = hypothesis_data['score'].idxmax()
                else: # Or if no score/data, set to None to skip score display
                     score_region_to_display = None
            
            # Proceed only if we have a region to display scores for
            if score_region_to_display and score_region_to_display in hypothesis_data.index:
                score_row = hypothesis_data.loc[score_region_to_display]
                components_dict = {
                    'direction_alignment': score_row.get('direction_alignment', 0), 
                    'consistency': score_row.get('consistency', 0),
                    'hypo_z_score_norm': score_row.get('hypo_z_score_norm', 0),
                    'explained_ratio': score_row.get('explained_ratio', 0),
                    'score': score_row.get('score', 0) # Ensure score is included
                }
                score_value = components_dict['score'] # Use the score from the dict
                
                if is_root_cause:
                    # For root cause: show full equation with operators
                    self._add_score_components(
                        ax, 0.02, 0.85, score_value, components_dict,
                        is_figure=False, 
                        show_score=True,
                        show_text=False
                    )
                else:
                    # For other hypotheses: show just the component boxes, more compact
                    # Pass the full components_dict which now includes the score
                    self._add_score_components_compact(ax, 0.5, 0.85, components_dict)

        # Set title and labels (Decorator handles y-axis % formatting)
        plot_title = title if title else f"Hypothesis: {hypothesis_name}"
        if explaining_region and explaining_region in regions:
            plot_title = f"Root Cause: {title or hypothesis_name}"
        ax.set_title(plot_title, fontsize=11)
        
        y_axis_label = y_label or "Hypothesis Value"
        ax.set_ylabel(y_axis_label, fontsize=10)
        
        return ax
    
    def create_score_formula(self, fig):
        """Create a colored score formula at the bottom of the figure."""
        y_pos = 0.02
        
        # Components dictionary with dummy values (0.0) just for structure/names
        components_dict = {comp: 0.0 for comp in self.score_component_order}
        score_value = 0.0 
        
        # --- Recalculate start_x for centering --- 
        is_figure=True
        show_score=True
        show_text=True
        card_spacing = 0.01 # Matching tighter spacing used in _add_score_components
        text_size = 7 # Matching smaller size used in _add_score_components for figure
        base_card_width = 0.11 # Matching smaller width used in _create_scorecard
        
        # Calculate approximate width based on elements rendered by _add_score_components
        num_components = len(self.score_component_order)
        approx_total_width = 0
        # Score card
        if show_score:
             approx_total_width += base_card_width + card_spacing 
             approx_total_width += 0.03 # Space for '=' 
             
        # Components part
        for i in range(num_components):
            if i > 0 or not show_score: # Add space for '+'
                 approx_total_width += 0.02 + card_spacing # Minimal space for op
            approx_total_width += 0.03 # Minimal space for '0.Nx'
            approx_total_width += base_card_width + card_spacing
            
        # Remove last spacing
        if approx_total_width > 0: 
            approx_total_width -= card_spacing 
        
        # Center horizontally
        start_x = (1.0 - approx_total_width) / 2
        start_x = max(0.01, start_x) # Ensure it doesn't start too far left
        # --- End Recalculation --- 
        
        # Use shared method with show_text=True to display names
        self._add_score_components(
            fig, start_x, y_pos, score_value, components_dict, 
            is_figure=True, 
            show_score=True,
            show_text=True
        )

    def plot_color_legend(self, ax: plt.Axes):
        """Draws the bar color and line style legend into a given Axes.

        Args:
            ax: The matplotlib Axes to draw the legend into.
        """
        ax.set_axis_off() # Turn off axis lines and ticks

        legend_items = [
            (self.colors['anomaly_negative'], "Bad Anomaly (Significant Negative Deviation)"),
            (self.colors['anomaly_positive'], "Good Anomaly (Significant Positive Deviation)"),
            (self.colors['highlight'], "Candidate Root Cause Region"),
            (self.colors['default_bar'], "Non-Anomalous / Other Region"),
            (self.colors['global_line'], "Benchmark / Global Average (Dashed Line)")
        ]

        y_pos = 0.95
        x_pos_rect = 0.05
        x_pos_text = 0.15
        rect_height = 0.1
        rect_width = 0.08
        spacing = 0.18 # Vertical spacing between items

        ax.set_title("Legend", fontsize=10, loc='left', pad=10)

        for color, text in legend_items:
            if "Benchmark" in text: # Handle line legend differently
                 ax.plot([x_pos_rect, x_pos_rect + rect_width], [y_pos - rect_height/2, y_pos - rect_height/2],
                         color=color, linestyle='--', linewidth=1.5)
            else: # Draw colored rectangle for bar colors
                 rect = plt.Rectangle((x_pos_rect, y_pos - rect_height),
                                    rect_width, rect_height,
                                    facecolor=color, transform=ax.transAxes)
                 ax.add_patch(rect)

            ax.text(x_pos_text, y_pos - rect_height/2, text, 
                    ha='left', va='center', fontsize=8, wrap=True, 
                    transform=ax.transAxes)
            y_pos -= spacing

        # Adjust limits to ensure text fits
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

    def plot_score_component_explanations(self, ax: plt.Axes):
        """Draws explanations for score components with colored backgrounds.

        Args:
            ax: The matplotlib Axes to draw the explanations into.
        """
        ax.set_axis_off()

        # Get score color for title and final note
        score_color = self.colors.get('score_color', '#AF7AC5')
        
        # Setup spacing variables
        x_pos = 0.05
        card_width = 0.90
        card_height = 0.17
        spacing = 0.03
        name_y_offset = 0.25  # Controls how far down name starts (smaller = higher)
        expl_y_offset = 0.75  # Controls how far down explanation starts (larger = lower)
        
        # Start position for first title card
        y_pos = 0.95
        
        # Create background rectangle for title
        title_rect = plt.Rectangle((x_pos, y_pos - card_height), card_width, card_height,
                           facecolor=score_color, alpha=0.2, transform=ax.transAxes, clip_on=False)
        ax.add_patch(title_rect)
        
        # Add title (same format as component names)
        ax.text(x_pos + 0.03, y_pos - card_height * name_y_offset, "Explanation Score",
                ha='left', va='top', fontsize=9, fontweight='bold', color=score_color,
                transform=ax.transAxes)
                
        # Add subtitle (same format as explanations)
        ax.text(x_pos + 0.03, y_pos - card_height * expl_y_offset, "Weighted value of the following:",
                ha='left', va='bottom', fontsize=8, transform=ax.transAxes)
        
        # Move down for first component
        y_pos -= (card_height + spacing)

        # Use the order and definitions from __init__
        for component_key in self.score_component_order:
            if component_key not in self.score_components:
                continue

            comp_info = self.score_components[component_key]
            color = self.colors['score_components'].get(component_key, '#DDDDDD')
            weight_pct = int(comp_info['weight'] * 100)
            name = comp_info['name']

            # Updated, more intuitive explanations
            explanation = ""
            if component_key == 'direction_alignment':
                explanation = f"Metric/Hypo direction move as expected? ({weight_pct}%)"
            elif component_key == 'consistency':
                explanation = f"Metric/Hypo correlated across region? ({weight_pct}%)"
            elif component_key == 'hypo_z_score_norm':
                explanation = f"How unusual is the Hypo value? ({weight_pct}%)"
            elif component_key == 'explained_ratio':
                explanation = f"Hypo Δ explains Δ of metric? ({weight_pct}%)"

            # Draw background card
            rect = plt.Rectangle((x_pos, y_pos - card_height), card_width, card_height,
                                 facecolor=color, alpha=0.2, transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)

            # Add text (Component Name + Explanation)
            # Apply bold weight and component color to the name
            ax.text(x_pos + 0.03, y_pos - card_height * name_y_offset, name,
                    ha='left', va='top', fontsize=7.5, fontweight='bold', color=color,
                    transform=ax.transAxes)
            # Keep explanation text default color (black/dark)
            ax.text(x_pos + 0.03, y_pos - card_height * expl_y_offset, explanation,
                    ha='left', va='bottom', fontsize=7.5, wrap=True,
                    transform=ax.transAxes)

            y_pos -= (card_height + spacing)

        # Add final note about threshold - use score color, no background box
        ax.text(x_pos + card_width/2, y_pos, "Score > 50 indicates likely explanation",
                ha='center', va='top', fontsize=7, style='italic', color=score_color,
                transform=ax.transAxes)

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)