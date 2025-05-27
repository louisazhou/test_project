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

# Standardized layout constants
LAYOUT = {
    'title': {
        'left': 0.3,      # Left-aligned, not centered
        'top': 0.2,       # Higher up to save space
        'width': 9.0,
        'height': 0.6,
        'font_size': 20,  # Smaller than before
        'font_bold': True
    },
    'content': {
        'top': 1.0,       # Start content below title
        'available_height': 6.5  # Available space for content
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
    available_height = LAYOUT['content']['available_height']
    
    # Add styled table on the left side of the slide
    table = add_bw_table_to_slide_with_colors(
        slide=slide,
        df=metrics_df_transposed,
        highlight_cells=highlight_cells,
        left=0.5,
        top=content_top,
        width=5.0,
        height=available_height
    )
    
    # Create text area on the right side for explanation
    text_left = Inches(6.0)  # 0.5 + 5.0 + 0.5
    text_width = Inches(4.0)
    text_top = Inches(content_top)
    text_height = Inches(available_height)
    
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


def create_figure_with_text_slide(
    prs: Presentation,
    title: str,
    figure_path: str,
    text: str,
    text_position: str = 'bottom'
) -> None:
    """
    Create a slide with a figure and text.
    
    Args:
        prs: PowerPoint presentation object
        title: Title for the slide
        figure_path: Path to the figure file
        text: Text content to add
        text_position: Position of text relative to figure ('bottom', 'top', 'right')
    """
    slide = create_slide_with_title(prs, title)
    
    # Use standardized layout positions
    content_top = LAYOUT['content']['top']
    available_height = LAYOUT['content']['available_height']
    
    if text_position == 'bottom':
        # Figure on top, text at bottom
        figure_height = available_height * 0.7  # 70% for figure
        text_height = available_height * 0.3   # 30% for text
        add_figure_to_slide(slide, figure_path, left=0.1, top=content_top, height=figure_height)
        add_text_to_slide(slide, text, left=0.5, top=content_top + figure_height, 
                         width=9.0, height=text_height, alignment='left')
    elif text_position == 'top':
        # Text on top, figure at bottom
        text_height = available_height * 0.25   # 25% for text
        figure_height = available_height * 0.75 # 75% for figure
        add_text_to_slide(slide, text, left=0.5, top=content_top, 
                         width=9.0, height=text_height, alignment='left')
        add_figure_to_slide(slide, figure_path, left=0.1, top=content_top + text_height, height=figure_height)
    elif text_position == 'right':
        # Figure on left, text on right
        add_figure_to_slide(slide, figure_path, left=0.1, top=content_top, height=available_height)
        add_text_to_slide(slide, text, left=5.5, top=content_top, 
                         width=4.0, height=available_height, alignment='left')
    else:
        # Default to bottom
        figure_height = available_height * 0.7
        text_height = available_height * 0.3
        add_figure_to_slide(slide, figure_path, left=0.1, top=content_top, height=figure_height)
        add_text_to_slide(slide, text, left=0.5, top=content_top + figure_height, 
                         width=9.0, height=text_height, alignment='left')

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
        gdrive_credentials_path: Path to credentials.json file (optional, defaults to base directory)
        gdrive_token_path: Path to token.json file (optional, defaults to base directory)
        
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
    
    # Create presentation
    prs = Presentation()
    
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
    
    def add_figure_with_text_slide_func(title, figure_path, text, text_position='bottom'):
        """Add a slide with figure and text."""
        create_figure_with_text_slide(
            prs=prs,
            title=title,
            figure_path=figure_path,
            text=text,
            text_position=text_position
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
        credentials_path: Path to credentials.json file. If None, looks in base directory.
        token_path: Path to token.json file. If None, looks in base directory.
    
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
    
    # Set default paths if not provided
    if credentials_path is None or token_path is None:
        # Get the base directory (RCA_automation folder)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if credentials_path is None:
            credentials_path = os.path.join(base_dir, 'credentials.json')
        if token_path is None:
            token_path = os.path.join(base_dir, 'token.json')
    
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
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