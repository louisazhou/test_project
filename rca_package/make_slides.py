import os
import logging
import pandas as pd
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from typing import Dict, Any, List, Optional, Tuple
import datetime
import json
import argparse

# Setup logging
logger = logging.getLogger(__name__)

# Define Google API scopes
SCOPES = ['https://www.googleapis.com/auth/presentations',
          'https://www.googleapis.com/auth/drive.file',
          'https://www.googleapis.com/auth/drive']

# Standardized layout constants for 16:9 slides
LAYOUT = {
    'title': {
        'left': 0.3,      # Left-aligned, not centered
        'top': 0.2,       # Higher up to save space
        'width': 9.0,
        'height': 0.4,    # Reduced from 0.6 to 0.4 to minimize title box height
        'font_size': 18,  # Updated to size 18
        'font_bold': True
    },
    'content': {
        'top': 0.7,       # Reduced from 1.0 to 0.7 (half the previous spacing)
        'available_height': 6.3  # Fixed: 7.3125 total - 0.7 top - 0.3 bottom margin = 6.3125
    },
    'text': {
        'font_size': 11,  # Consistent text size
        'line_spacing': 1.2
    }
}

def add_bw_table_to_slide_with_colors(
    slide,
    df: pd.DataFrame,
    highlight_cells: Optional[Dict[Tuple[str, str], str]] = None,
    left: float = 0.5,
    top: float = 1.0,
    width: float = 9.0,
    height: float = 3.0
):
    """Adds a styled B/W table with support for red/green/custom color highlighting.

    Args:
        slide: pptx Slide object to which the table will be added.
        df: DataFrame to visualize.
        highlight_cells: Dict of (row_label, column_name) → color ('red', 'green', or RGB tuple).
        left, top, width, height: Placement of the table in inches.

    Returns:
        The pptx Table object.
    """
    highlight_cells = highlight_cells or {}

    include_index = not isinstance(df.index, pd.RangeIndex)
    num_rows = df.shape[0] + 1
    num_cols = df.shape[1] + (1 if include_index else 0)

    table_shape = slide.shapes.add_table(
        num_rows, num_cols,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    table = table_shape.table

    # Calculate intelligent column widths
    if include_index:
        # Calculate index column width based on content length
        max_index_length = max(len(str(idx)) for idx in df.index)
        max_column_length = max(len(str(col)) for col in df.columns)
        
        # Also consider the length of "Index" or longest index value vs column headers
        effective_index_length = max(max_index_length, 5)  # Minimum for readability
        effective_column_length = max(max_column_length, 8)  # Minimum for readability
        
        # Base widths (as fraction of total width) - more nuanced calculation
        if effective_index_length > 20:  # Very long index names
            index_width_fraction = 0.45  # 45% for index
        elif effective_index_length > 15:  # Long index names
            index_width_fraction = 0.4   # 40% for index
        elif effective_index_length > 10:  # Medium index names
            index_width_fraction = 0.3   # 30% for index
        elif effective_index_length > 6:   # Short-medium index names
            index_width_fraction = 0.25  # 25% for index
        else:  # Very short index names
            index_width_fraction = 0.2   # 20% for index
        
        # Adjust based on number of columns - fewer columns can have wider index
        if df.shape[1] <= 3:
            index_width_fraction = min(index_width_fraction + 0.05, 0.4)  # Add 5% but cap at 40%
        
        # Remaining width distributed equally among data columns
        data_width_fraction = (1.0 - index_width_fraction) / df.shape[1]
        
        # Set column widths
        table.columns[0].width = Inches(width * index_width_fraction)
        for i in range(1, num_cols):
            table.columns[i].width = Inches(width * data_width_fraction)
    else:
        # Equal width for all columns when no index
        col_width = width / num_cols
        for i in range(num_cols):
            table.columns[i].width = Inches(col_width)

    # Header
    for col_idx in range(num_cols):
        cell = table.cell(0, col_idx)
        if include_index and col_idx == 0:
            cell.text = ""
        else:
            actual_col_idx = col_idx - 1 if include_index else col_idx
            cell.text = df.columns[actual_col_idx]
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0, 0, 0)
        for paragraph in cell.text_frame.paragraphs:
            paragraph.alignment = PP_ALIGN.CENTER
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(12)
                run.font.color.rgb = RGBColor(255, 255, 255)

    # Data
    for row_idx, row_label in enumerate(df.index):
        for col_idx in range(num_cols):
            cell = table.cell(row_idx + 1, col_idx)
            if include_index and col_idx == 0:
                val = str(row_label)
            else:
                actual_col_idx = col_idx - 1 if include_index else col_idx
                cell_value = df.iloc[row_idx, actual_col_idx]
                # Format the value (handle percentage detection)
                if isinstance(cell_value, (int, float)) and not np.isnan(cell_value):
                    col_name = df.columns[actual_col_idx]
                    row_name = str(row_label)
                    
                    # Check if this is a percentage based on column or row name
                    is_percent = ('_pct' in col_name or '%' in col_name or 'pct' in col_name.lower() or
                                 '_pct' in row_name or '%' in row_name or 'pct' in row_name.lower() or
                                 'rate' in row_name.lower() or 'ratio' in row_name.lower())
                    
                    if is_percent:
                        val = f"{cell_value*100:.0f}%"
                    else:
                        val = f"{cell_value:.2f}"
                else:
                    val = "N/A"
            cell.text = val
            cell.fill.solid()
            shade = RGBColor(242, 244, 248) if row_idx % 2 == 0 else RGBColor(255, 255, 255)
            cell.fill.fore_color.rgb = shade

            if include_index and col_idx == 0:
                color_rgb = RGBColor(0, 0, 0)
            else:
                key = (row_label, df.columns[actual_col_idx])
                color = highlight_cells.get(key)
                if color == "red":
                    color_rgb = RGBColor(255, 0, 0)
                elif color == "green":
                    color_rgb = RGBColor(0, 153, 0)
                elif isinstance(color, tuple) and len(color) == 3:
                    color_rgb = RGBColor(*color)
                else:
                    color_rgb = RGBColor(0, 0, 0)

            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(12)
                    run.font.color.rgb = color_rgb

    return table

def _create_highlight_cells_map(
    df: pd.DataFrame,
    metric_anomaly_map: Dict[str, Dict[str, Any]]
) -> Dict[Tuple[str, str], str]:
    """
    Create a highlight_cells dictionary for the table based on anomaly information.
    
    Args:
        df: DataFrame containing the data (metrics as rows, regions as columns) using display names
        metric_anomaly_map: Map of metrics to their anomaly information
        
    Returns:
        Dictionary mapping (row_label, column_name) to color ('red' or 'green')
    """
    highlight_cells = {}
    
    for metric_name, metric_info in metric_anomaly_map.items():
        # Since we now use display names as column headers, metric_name is already the display name
        if metric_name in df.index:
            anomalous_region = metric_info.get('anomalous_region')
            direction = metric_info.get('direction')
            higher_is_better = metric_info.get('higher_is_better', True)
            
            if anomalous_region and direction and anomalous_region in df.columns:
                # Determine if this is good or bad
                is_good = (direction == 'higher' and higher_is_better) or \
                         (direction == 'lower' and not higher_is_better)
                
                color = 'green' if is_good else 'red'
                # For table: (metric_name, anomalous_region)
                highlight_cells[(metric_name, anomalous_region)] = color
    
    return highlight_cells

def create_metrics_summary_slide(
    prs: Presentation,
    df: pd.DataFrame,
    metrics_text: Dict[str, str],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    title: str = "Metrics Summary"
) -> None:
    """
    Create a summary slide with a styled table of metrics data and pre-formatted text explanations.
    
    Args:
        prs: PowerPoint presentation object
        df: DataFrame containing the data with display names as column headers (index is region)
        metrics_text: Dictionary mapping metric display names to their pre-formatted explanation text
        metric_anomaly_map: Map of metrics to their anomaly information
        title: Title for the slide
    """
    # Use a blank slide layout
    slide_layout = prs.slide_layouts[6]  # Use a blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Add standardized title
    add_standardized_title(slide, title)
    
    # Filter DataFrame to only include metrics we have text for
    # Since we now use display names as column headers, this is straightforward
    metrics = list(metrics_text.keys())
    metrics_df = df[metrics]
    
    # Transpose the DataFrame so regions are columns and metrics are rows
    metrics_df_transposed = metrics_df.T
    
    # Create highlight cells map for transposed data
    highlight_cells = _create_highlight_cells_map(metrics_df_transposed, metric_anomaly_map)
    
    # Use standardized layout positions
    content_top = LAYOUT['content']['top']
    
    # Calculate table height based on number of rows (metrics + header)
    num_rows = len(metrics_df_transposed) + 1  # +1 for header
    row_height = 0.4  # Height per row in inches
    table_height = num_rows * row_height
    
    # Calculate appropriate table width based on content
    # Estimate width needed: index column + data columns
    max_index_length = max(len(str(idx)) for idx in metrics_df_transposed.index)
    max_column_length = max(len(str(col)) for col in metrics_df_transposed.columns)
    
    # Estimate table width (rough calculation based on character count)
    # Each character ≈ 0.08 inches, plus padding
    index_width_estimate = max(max_index_length * 0.08, 1.3)  # Minimum 1.3 inches
    data_width_estimate = max_column_length * 0.08 * len(metrics_df_transposed.columns)
    total_table_width = min(index_width_estimate + data_width_estimate + 0.5, 6.0)  # Cap at 6 inches
    
    # Add styled table on the left side of the slide
    table = add_bw_table_to_slide_with_colors(
        slide=slide,
        df=metrics_df_transposed,
        highlight_cells=highlight_cells,
        left=0.5,
        top=content_top,
        width=total_table_width,  # Use calculated width instead of fixed 5.0
        height=table_height  # Use calculated height instead of full available height
    )
    
    # Create text area on the right side for explanation - positioned close to right edge
    text_left = Inches(0.5 + total_table_width + 0.3)  # Table left + width + small gap
    text_width = Inches(13.0 - (0.5 + total_table_width + 0.3) - 0.2)  # Remaining space minus right margin
    text_top = Inches(content_top)
    text_height = Inches(LAYOUT['content']['available_height'])  # Text can use full available height
    
    # Add text box for explanations
    text_box = slide.shapes.add_textbox(
        text_left, text_top, 
        text_width, text_height
    )
    text_frame = text_box.text_frame
    text_frame.word_wrap = True
    
    # Add explanations for each metric with consistent formatting
    for metric, explanation_text in metrics_text.items():
        # Add header for this metric
        p = text_frame.add_paragraph()
        p.text = f"{metric}:"
        p.font.bold = True
        p.font.size = Pt(LAYOUT['text']['font_size'])
        p.space_after = Pt(6)
        
        # Add the explanation
        p = text_frame.add_paragraph()
        p.text = explanation_text
        p.font.size = Pt(LAYOUT['text']['font_size'])
        p.space_after = Pt(12)

def add_figure_to_slide(
    slide,
    figure_path: str,
    left: float = 0.1,
    top: float = 1.5,
    height: float = 6.0
) -> None:
    """
    Add a figure to an existing slide.
    
    Args:
        slide: pptx Slide object to add the figure to
        figure_path: Path to the figure file
        left: Left position in inches
        top: Top position in inches
        height: Height in inches
    """
    # Add picture
    if os.path.exists(figure_path):
        try:
            slide.shapes.add_picture(figure_path, Inches(left), Inches(top), height=Inches(height))
        except Exception as e:
            logger.error(f"Error adding figure {figure_path}: {e}")
            # Add a text box with the error message
            error_box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(8.0), Inches(1.0))
            error_box.text_frame.text = f"Error loading figure: {figure_path}\n{str(e)}"
    else:
        # Add a text box indicating the figure wasn't found
        error_box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(8.0), Inches(1.0))
        error_box.text_frame.text = f"Figure not found: {figure_path}"


def add_standardized_title(slide, title: str) -> None:
    """
    Add a standardized title to a slide using consistent formatting.
    
    Args:
        slide: pptx Slide object
        title: Title text
    """
    title_shape = slide.shapes.add_textbox(
        Inches(LAYOUT['title']['left']), 
        Inches(LAYOUT['title']['top']), 
        Inches(LAYOUT['title']['width']), 
        Inches(LAYOUT['title']['height'])
    )
    title_text_frame = title_shape.text_frame
    title_text_frame.text = title
    
    # Apply consistent formatting
    paragraph = title_text_frame.paragraphs[0]
    paragraph.font.size = Pt(LAYOUT['title']['font_size'])
    paragraph.font.bold = LAYOUT['title']['font_bold']
    paragraph.alignment = PP_ALIGN.LEFT  # Always left-aligned


def add_text_to_slide(
    slide,
    text: str,
    left: float = 0.5,
    top: float = 7.0,
    width: float = 9.0,
    height: float = 1.5,
    fontsize: int = None,  # Will use default from LAYOUT if None
    alignment: str = 'left'
) -> None:
    """
    Add text to an existing slide.
    
    Args:
        slide: pptx Slide object to add the text to
        text: Text content to add
        left: Left position in inches
        top: Top position in inches
        width: Width in inches
        height: Height in inches
        fontsize: Font size
        alignment: Text alignment ('left', 'center', 'right')
    """
    # Use default font size if not specified
    if fontsize is None:
        fontsize = LAYOUT['text']['font_size']
    
    # Add text box
    text_box = slide.shapes.add_textbox(
        Inches(left), Inches(top), 
        Inches(width), Inches(height)
    )
    text_frame = text_box.text_frame
    text_frame.word_wrap = True
    
    # Set text content
    text_frame.text = text
    
    # Style the text with consistent formatting
    for paragraph in text_frame.paragraphs:
        paragraph.font.size = Pt(fontsize)
        paragraph.space_after = Pt(6)  # Consistent spacing
        
        # Set alignment
        if alignment == 'center':
            paragraph.alignment = PP_ALIGN.CENTER
        elif alignment == 'right':
            paragraph.alignment = PP_ALIGN.RIGHT
        else:  # left
            paragraph.alignment = PP_ALIGN.LEFT


def create_slide_with_title(prs: Presentation, title: str):
    """
    Create a new slide with a standardized title.
    
    Args:
        prs: PowerPoint presentation object
        title: Title for the slide
        
    Returns:
        The created slide object
    """
    # Use a blank slide layout
    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)
    
    # Add standardized title
    add_standardized_title(slide, title)
    
    return slide


def calculate_dynamic_layout(text: str, slide_type: str = 'standard') -> Dict[str, float]:
    """
    Calculate dynamic layout proportions based on text content and slide type.
    
    Args:
        text: The text content to be displayed
        slide_type: Type of slide ('standard', 'depth_analysis', 'bar_chart')
    
    Returns:
        Dictionary with layout proportions and dimensions
    """
    available_height = LAYOUT['content']['available_height']
    
    # Estimate text height based on content
    if not text:
        text_lines = 0
    else:
        # Count actual lines (including \n) and estimate wrapped lines
        explicit_lines = text.count('\n') + 1
        
        # More accurate character wrapping estimation
        # Consider that PowerPoint text boxes are ~9 inches wide with 11pt font
        # At 11pt, approximately 100-110 characters fit per line in a 9-inch box
        chars_per_line = 100
        
        # Split by explicit line breaks and calculate wrapping for each line
        lines = text.split('\n')
        total_wrapped_lines = 0
        for line in lines:
            if len(line) <= chars_per_line:
                total_wrapped_lines += 1
            else:
                # This line will wrap
                wrapped_count = (len(line) + chars_per_line - 1) // chars_per_line  # Ceiling division
                total_wrapped_lines += wrapped_count
        
        text_lines = total_wrapped_lines
    
    # Calculate required text height (line height ~0.18 inches at 11pt with 1.2 spacing)
    line_height = 0.18
    text_height_needed = text_lines * line_height + 0.4  # +0.4 for padding
    
    # Define layout themes based on slide type
    if slide_type == 'depth_analysis':
        # Depth analysis needs more text space, smaller figure
        min_text_height = 2.2  # Minimum for tables and explanations
        max_text_height = 4.5  # Maximum to leave space for figure
        text_height = min(max(text_height_needed, min_text_height), max_text_height)
        figure_height = available_height - text_height
        
        # Ensure figure isn't too small
        if figure_height < 2.0:
            figure_height = 2.0
            text_height = available_height - figure_height
            
    elif slide_type == 'bar_chart':
        # Bar charts need less text space, more figure space
        min_text_height = 0.6  # Minimum for short explanations
        max_text_height = 2.0  # Maximum to prioritize figure
        text_height = min(max(text_height_needed, min_text_height), max_text_height)
        figure_height = available_height - text_height
        
        # Ensure figure gets priority
        if figure_height < 4.5:
            figure_height = 4.5
            text_height = available_height - figure_height
            
    else:  # 'standard'
        # Balanced layout
        min_text_height = 1.0
        max_text_height = 3.5
        text_height = min(max(text_height_needed, min_text_height), max_text_height)
        figure_height = available_height - text_height
    
    # Calculate proportions
    text_proportion = text_height / available_height
    figure_proportion = figure_height / available_height
    
    # Debug logging
    logger.debug(f"Layout calculation for {slide_type}:")
    logger.debug(f"  Text lines: {text_lines}, needed height: {text_height_needed:.2f}")
    logger.debug(f"  Final text height: {text_height:.2f}, figure height: {figure_height:.2f}")
    logger.debug(f"  Proportions - text: {text_proportion:.1%}, figure: {figure_proportion:.1%}")
    
    return {
        'text_height': text_height,
        'figure_height': figure_height,
        'text_proportion': text_proportion,
        'figure_proportion': figure_proportion,
        'estimated_lines': text_lines,
        'content_top': LAYOUT['content']['top']
    }


def create_figure_with_text_slide(
    prs: Presentation,
    title: str,
    figure_path: str,
    text: str,
    text_position: str = 'bottom',
    slide_type: str = 'standard'
) -> None:
    """
    Create a slide with a figure and text using dynamic layout.
    
    Args:
        prs: PowerPoint presentation object
        title: Title for the slide
        figure_path: Path to the figure file
        text: Text content to add
        text_position: Position of text relative to figure ('bottom', 'top', 'right')
        slide_type: Type of slide for layout optimization ('standard', 'depth_analysis', 'bar_chart')
    """
    slide = create_slide_with_title(prs, title)
    
    # Calculate dynamic layout
    layout = calculate_dynamic_layout(text, slide_type)
    content_top = layout['content_top']
    
    if text_position == 'bottom':
        # Figure on top, text at bottom
        add_figure_to_slide(slide, figure_path, left=0.1, top=content_top, height=layout['figure_height'])
        add_text_to_slide(slide, text, left=0.5, top=content_top + layout['figure_height'], 
                         width=9.0, height=layout['text_height'], alignment='left')
    elif text_position == 'top':
        # Text on top, figure at bottom
        add_text_to_slide(slide, text, left=0.5, top=content_top, 
                         width=9.0, height=layout['text_height'], alignment='left')
        add_figure_to_slide(slide, figure_path, left=0.1, top=content_top + layout['text_height'], 
                           height=layout['figure_height'])
    elif text_position == 'right':
        # Figure on left, text on right - use full height for both
        add_figure_to_slide(slide, figure_path, left=0.1, top=content_top, height=LAYOUT['content']['available_height'])
        add_text_to_slide(slide, text, left=5.5, top=content_top, 
                         width=4.0, height=LAYOUT['content']['available_height'], alignment='left')
    else:
        # Default to bottom with dynamic layout
        add_figure_to_slide(slide, figure_path, left=0.1, top=content_top, height=layout['figure_height'])
        add_text_to_slide(slide, text, left=0.5, top=content_top + layout['figure_height'], 
                         width=9.0, height=layout['text_height'], alignment='left')

def add_section_slide(prs: Presentation, title: str, subtitle: str = None) -> None:
    """
    Add a section divider slide with title and optional subtitle.
    
    Args:
        prs: PowerPoint presentation object
        title: Main title for the section
        subtitle: Optional subtitle
    """
    # Use title slide layout
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    
    # Set title
    slide.shapes.title.text = title
    
    # Set subtitle if provided
    if subtitle and len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle


def create_flexible_presentation(
    output_dir: str = '.',
    ppt_filename: str = None,
    upload_to_gdrive: bool = False,
    gdrive_folder_id: str = None,
    gdrive_credentials_path: str = None,
    gdrive_token_path: str = None
) -> Dict[str, Any]:
    """
    Create a flexible presentation builder that returns a presentation object
    and helper functions for easy slide creation.
    
    Args:
        output_dir: Directory to save the presentation
        ppt_filename: Filename for the presentation
        upload_to_gdrive: Whether to upload to Google Drive
        gdrive_folder_id: Google Drive folder ID
        gdrive_credentials_path: Path to credentials.json file (required for upload)
        gdrive_token_path: Path to token.json file (required for upload)
        
    Returns:
        Dictionary containing:
        - 'prs': PowerPoint presentation object
        - 'save_and_upload': Function to save and optionally upload
        - 'add_summary_slide': Function to add metrics summary slide
        - 'add_figure_slide': Function to add figure slide
        - 'add_figure_with_text_slide': Function to add figure with text
        - 'add_section_slide': Function to add section divider
        - 'add_custom_slide': Function to add custom slide with callback
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if ppt_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ppt_filename = f"Presentation_{timestamp}.pptx"
    
    # Create presentation with 16:9 aspect ratio
    prs = Presentation()
    # Set slide size to 16:9 (13 inches wide, 7.3125 inches tall)
    prs.slide_width = Inches(13)
    prs.slide_height = Inches(7.3125)
    
    def save_and_upload():
        """Save the presentation and optionally upload to Google Drive."""
        # Save the presentation
        ppt_path = os.path.join(output_dir, ppt_filename)
        prs.save(ppt_path)
        logger.info(f"Presentation saved to {ppt_path}")
        
        result = {
            'local_path': ppt_path,
            'filename': ppt_filename
        }
        
        # Upload to Google Drive if requested
        if upload_to_gdrive:
            try:
                gdrive_results = upload_to_google_drive(
                    file_path=ppt_path,
                    folder_id=gdrive_folder_id,
                    credentials_path=gdrive_credentials_path,
                    token_path=gdrive_token_path
                )
                result.update(gdrive_results)
                logger.info(f"Uploaded to Google Drive: {gdrive_results.get('gdrive_url', 'Unknown URL')}")
            except Exception as e:
                logger.error(f"Failed to upload to Google Drive: {e}")
                result['gdrive_error'] = str(e)
        
        return result
    
    def add_summary_slide(df, metrics_text, metric_anomaly_map, title="Metrics Summary"):
        """Add a metrics summary slide."""
        create_metrics_summary_slide(
            prs=prs,
            df=df,
            metrics_text=metrics_text,
            metric_anomaly_map=metric_anomaly_map,
            title=title
        )
    
    def add_figure_slide(figure_path, title=None):
        """Add a slide with just a figure."""
        # Use a blank slide layout
        blank_slide_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_slide_layout)
        
        # Add title if provided
        if title:
            add_standardized_title(slide, title)
            top = LAYOUT['content']['top']
            height = LAYOUT['content']['available_height']
        else:
            top = 0.75
            height = 6.0
        
        # Add figure to slide
        add_figure_to_slide(slide, figure_path, left=0.1, top=top, height=height)
    
    def add_figure_with_text_slide_func(title, figure_path, text, text_position='bottom', slide_type='standard'):
        """Add a slide with figure and text using dynamic layout."""
        create_figure_with_text_slide(
            prs=prs,
            title=title,
            figure_path=figure_path,
            text=text,
            text_position=text_position,
            slide_type=slide_type
        )
    
    def add_section_slide_func(title, subtitle=None):
        """Add a section divider slide."""
        add_section_slide(prs, title, subtitle)
    
    def add_custom_slide(slide_builder_func, *args, **kwargs):
        """Add a custom slide using a user-provided function."""
        # Create a blank slide
        blank_slide_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_slide_layout)
        
        # Call the user's slide builder function
        slide_builder_func(slide, *args, **kwargs)
    
    return {
        'prs': prs,
        'save_and_upload': save_and_upload,
        'add_summary_slide': add_summary_slide,
        'add_figure_slide': add_figure_slide,
        'add_figure_with_text_slide': add_figure_with_text_slide_func,
        'add_section_slide': add_section_slide_func,
        'add_custom_slide': add_custom_slide
    }

def get_credentials(credentials_path: str = None, token_path: str = None):
    """
    Get Google API credentials using OAuth2 flow.
    
    Args:
        credentials_path: Path to credentials.json file (required if no token.json exists)
        token_path: Path to token.json file (required if no credentials.json exists)
    
    Returns:
        Google API credentials object
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        logger.error(
            "Google Drive upload requires additional packages. "
            "Install with: pip install google-api-python-client google-auth google-auth-oauthlib"
        )
        raise
    
    # Check if we have either credentials or token path
    if not credentials_path and not token_path:
        raise ValueError("Either credentials_path or token_path must be provided for Google Drive upload")
    
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if token_path and os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Need credentials.json for OAuth flow
            if not credentials_path:
                raise ValueError("credentials_path is required when token.json doesn't exist or is invalid")
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run (only if token_path is provided)
        if token_path:
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
    
    return creds


def upload_to_google_drive(
    file_path: str, 
    folder_id: str = None,
    credentials_path: str = None,
    token_path: str = None
) -> Dict[str, str]:
    """
    Upload a file to Google Drive using OAuth2 credentials.
    
    Args:
        file_path: Path to the file to upload
        folder_id: Google Drive folder ID to upload to (optional)
        credentials_path: Path to credentials.json file (optional, defaults to base directory)
        token_path: Path to token.json file (optional, defaults to base directory)
        
    Returns:
        Dictionary with:
        - 'gdrive_id': Google Drive file ID
        - 'gdrive_url': Google Drive URL
    """
    if not file_path or not os.path.exists(file_path):
        logger.error(f"File not found or path is invalid: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Import the necessary libraries only when needed
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        
        # Get credentials using OAuth2 flow
        creds = get_credentials(credentials_path=credentials_path, token_path=token_path)
        if not creds:
            raise ValueError("Failed to get Google credentials. Cannot upload.")

        # Build the Drive service
        drive_service = build('drive', 'v3', credentials=creds)
        file_name = os.path.basename(file_path)

        # Prepare metadata for the file on Google Drive
        file_metadata = {
            'name': file_name,
            # Optionally convert to Google Slides format on upload
            'mimeType': 'application/vnd.google-apps.presentation'
        }
        if folder_id:
            file_metadata['parents'] = [folder_id]

        # Prepare the media for upload
        media = MediaFileUpload(
            file_path, 
            mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation', 
            resumable=True
        )

        # Perform the upload
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'  # Request fields needed
        ).execute()

        file_id = file.get('id')
        web_link = file.get('webViewLink')

        logger.info(f"Successfully uploaded '{file_name}' to Google Drive.")
        logger.info(f"File ID: {file_id}")
        logger.info(f"Web Link: {web_link}")

        return {
            'gdrive_id': file_id,
            'gdrive_url': web_link
        }

    except ImportError:
        logger.warning(
            "Google Drive upload requires additional packages. "
            "Install with: pip install google-api-python-client google-auth google-auth-oauthlib"
        )
        raise
    except Exception as e:
        logger.error(f"Error uploading file '{file_path}' to Google Drive: {e}")
        raise


def main(output_dir='.', upload_to_gdrive=False, gdrive_folder_id=None, 
        gdrive_credentials_path=None, gdrive_token_path=None):
    """
    Example function demonstrating usage with dummy data.
    
    Args:
        output_dir: Directory for output files
        upload_to_gdrive: Whether to upload to Google Drive
        gdrive_folder_id: Google Drive folder ID to upload to
        gdrive_credentials_path: Path to credentials.json file
        gdrive_token_path: Path to token.json file
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data with regions as index
    data = {
        'conversion_rate_pct': [0.12, 0.08, 0.11, 0.13, 0.10],
        'avg_order_value': [75.0, 65.0, 80.0, 85.0, 72.0],
        'bounce_rate_pct': [0.35, 0.45, 0.32, 0.28, 0.34],
        'session_duration': [180, 120, 190, 210, 175]
    }
    regions = ['Global', 'North America', 'Europe', 'Asia', 'Latin America']
    df = pd.DataFrame(data, index=regions)
    
    # Pre-formatted explanation texts
    metrics_text = {
        'conversion_rate_pct': 'Conversion Rate is 33.3% lower in North America compared to global average. Root cause: North America has 10pp higher Bounce Rate than global average, explaining 75% of the difference.',
        'avg_order_value': 'Average Order Value is 6.7% higher in Europe. Root cause: Session Duration is 5.6% higher than global average.'
    }
    
    # Example metric anomaly map
    metric_anomaly_map = {
        'conversion_rate_pct': {
            'anomalous_region': 'North America',
            'metric_val': df.loc['North America', 'conversion_rate_pct'],
            'global_val': df.loc['Global', 'conversion_rate_pct'],
            'direction': 'lower',
            'magnitude': 33.3,
            'higher_is_better': True
        },
        'avg_order_value': {
            'anomalous_region': 'Europe',
            'metric_val': df.loc['Europe', 'avg_order_value'],
            'global_val': df.loc['Global', 'avg_order_value'],
            'direction': 'higher',
            'magnitude': 6.7,
            'higher_is_better': True
        }
    }
    
    # Create dummy figure if it doesn't exist (just for testing)
    dummy_figure_path = os.path.join(output_dir, "dummy_figure.png")
    if not os.path.exists(dummy_figure_path):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')
            plt.title("Sample Figure")
            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.savefig(dummy_figure_path)
            plt.close()
        except ImportError:
            logger.warning("Matplotlib not available, can't create dummy figure")
            # Create an empty file
            with open(dummy_figure_path, 'w') as f:
                f.write("Dummy file")
    
    # Example figure paths
    figure_paths = [
        {'path': dummy_figure_path, 'title': 'Sample Figure #1'},
        {'path': dummy_figure_path, 'title': 'Sample Figure #2'}
    ]
    
    # Generate the presentation
    result = create_metrics_presentation(
        df=df,
        metrics_text=metrics_text,
        metric_anomaly_map=metric_anomaly_map,
        figure_paths=figure_paths,
        output_dir=output_dir,
        ppt_filename="Metrics_Summary_Example.pptx",
        upload_to_gdrive=upload_to_gdrive,
        gdrive_folder_id=gdrive_folder_id,
        gdrive_credentials_path=gdrive_credentials_path
    )
    
    logger.info(f"Example presentation created: {result['local_path']}")
    
    if upload_to_gdrive and 'gdrive_url' in result:
        logger.info(f"Uploaded to Google Drive: {result['gdrive_url']}")
        
    return result

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create metrics presentation and optionally upload to Google Drive')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--upload', action='store_true', help='Upload presentation to Google Drive')
    parser.add_argument('--folder-id', help='Google Drive folder ID to upload to')
    parser.add_argument('--credentials', help='Path to credentials.json file')
    parser.add_argument('--token', help='Path to token.json file')
    
    args = parser.parse_args()
    
    # Run with provided arguments
    main(
        output_dir=args.output_dir,
        upload_to_gdrive=args.upload,
        gdrive_folder_id=args.folder_id,
        gdrive_credentials_path=args.credentials,
        gdrive_token_path=args.token
    ) 