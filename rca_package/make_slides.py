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

def _style_cell(cell, value: Any, is_anomalous: bool = False, 
               direction: str = None, higher_is_better: bool = True, 
               font_size: int = 10, alignment: str = 'center') -> None:
    """
    Style a table cell with the given parameters.
    
    Args:
        cell: Cell object to style
        value: Value to display in the cell
        is_anomalous: Whether this cell represents an anomalous value
        direction: Direction of anomaly ('higher' or 'lower')
        higher_is_better: Whether higher values are better for this metric
        font_size: Font size to use
        alignment: Text alignment ('center', 'left', or 'right')
    """
    # Format the value (handle percentage columns)
    if isinstance(value, (int, float)) and not np.isnan(value):
        is_percent = False
        if isinstance(value, str):
            is_percent = '_pct' in value or '%' in value
        if is_percent:
            value_text = f"{value*100:.1f}%" 
        else:
            value_text = f"{value:.2f}" 
    else:
        value_text = "N/A"
        
    # Add value to cell
    cell.text = value_text
    cell.text_frame.paragraphs[0].font.size = Pt(font_size)
    
    # Set alignment
    if alignment == 'center':
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    elif alignment == 'left':
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    elif alignment == 'right':
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
        
    # Color the cell if it's anomalous
    if is_anomalous and direction:
        # Determine if this is good or bad
        is_good = (direction == 'higher' and higher_is_better) or \
                  (direction == 'lower' and not higher_is_better)
        
        # Apply cell styling
        cell.fill.solid()
        if is_good:
            cell.fill.fore_color.rgb = RGBColor(200, 255, 200)  # Light green
        else:
            cell.fill.fore_color.rgb = RGBColor(255, 200, 200)  # Light red

def create_metrics_summary_slide(
    prs: Presentation,
    df: pd.DataFrame,
    metrics_text: Dict[str, str],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    title: str = "Metrics Summary",
) -> None:
    """
    Create a summary slide with a table of metrics data and pre-formatted text explanations.
    
    Args:
        prs: PowerPoint presentation object
        df: DataFrame containing the data (index is region)
        metrics_text: Dictionary mapping metric names to their pre-formatted explanation text
        metric_anomaly_map: Map of metrics to their anomaly information
        title: Title for the slide
    """
    # Use a blank slide layout
    slide_layout = prs.slide_layouts[5]  # Use a blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_shape = slide.shapes.title
    if title_shape is None:
        title_shape = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9.0), Inches(0.75))
    title_text_frame = title_shape.text_frame
    title_text_frame.text = title
    title_text_frame.paragraphs[0].font.size = Pt(28)
    title_text_frame.paragraphs[0].font.bold = True
    
    # Get metrics and regions for the table
    metrics = list(metrics_text.keys())
    regions = df.index
    
    # Calculate table dimensions
    table_rows = len(metrics) + 1  # +1 for header
    table_cols = len(regions) + 1  # +1 for metric names
    
    # Create table on the left side of the slide
    table_width = Inches(5.0)  # Adjust as needed
    table_height = Inches(5.0)  # Adjust as needed
    table_left = Inches(0.5)
    table_top = Inches(1.5)
    
    # Create the table with no style
    table = slide.shapes.add_table(
        table_rows, table_cols, 
        table_left, table_top, 
        table_width, table_height
    ).table
    
    # Set column widths
    metric_col_width = Inches(1.5)  # Width for metric names column
    region_width_inches = (5.0 - 1.5) / len(regions)  # Calculate in inches
    region_col_width = Inches(region_width_inches)  # Convert to Inches object
    
    table.columns[0].width = metric_col_width
    for i in range(1, table_cols):
        table.columns[i].width = region_col_width
    
    # Set header row
    header_cells = table.rows[0].cells
    header_cells[0].text = "Metric"
    
    # Add region names to header
    for i, region in enumerate(regions):
        header_cells[i+1].text = region
    
    # Style header row with minimal formatting
    for cell in header_cells:
        # Remove background fill
        cell.fill.background()
        
        # Style text
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.font.bold = True
        paragraph.font.size = Pt(11)
        paragraph.alignment = PP_ALIGN.CENTER
    
    # Add data rows
    for row_idx, metric in enumerate(metrics):
        row = table.rows[row_idx + 1].cells
        
        # Add metric name in first column
        row[0].text = metric
        row[0].text_frame.paragraphs[0].font.size = Pt(10)
        row[0].text_frame.paragraphs[0].font.bold = True
        
        # Get anomaly info for this metric if available
        metric_info = metric_anomaly_map.get(metric, {})
        anomalous_region = metric_info.get('anomalous_region')
        higher_is_better = metric_info.get('higher_is_better', True)
        direction = metric_info.get('direction')
        
        # Add region values
        for col_idx, region in enumerate(regions):
            col = col_idx + 1  # +1 for metric names column
            
            # Get the value for this metric and region
            try:
                value = df.loc[region, metric]
            except (IndexError, KeyError):
                value = np.nan
            
            # Style the cell
            is_anomalous = anomalous_region and region == anomalous_region
            _style_cell(
                row[col], 
                value, 
                is_anomalous=is_anomalous,
                direction=direction,
                higher_is_better=higher_is_better
            )
    
    # Create text area on the right side for explanation
    text_left = table_left + table_width + Inches(0.5)
    text_width = Inches(4.0)
    text_top = table_top
    text_height = table_height
    
    # Add text box for explanations
    text_box = slide.shapes.add_textbox(
        text_left, text_top, 
        text_width, text_height
    )
    text_frame = text_box.text_frame
    text_frame.word_wrap = True
    
    # Add explanations for each metric
    for metric, explanation_text in metrics_text.items():
        # Add header for this metric
        p = text_frame.add_paragraph()
        p.text = f"{metric}:"
        p.font.bold = True
        p.font.size = Pt(12)
        p.space_after = Pt(6)
        
        # Add the explanation
        p = text_frame.add_paragraph()
        p.text = explanation_text
        p.font.size = Pt(10)
        p.space_after = Pt(12)

def add_figure_slide(
    prs: Presentation,
    figure_path: str,
    title: str = None
) -> None:
    """
    Add a slide with a figure.
    
    Args:
        prs: PowerPoint presentation object
        figure_path: Path to the figure file
        title: Optional title for the slide
    """
    # Use a blank slide layout
    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)
    
    # Add title if provided
    if title:
        title_shape = slide.shapes.title
        if title_shape is None:
            title_shape = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9.0), Inches(0.75))
        title_text_frame = title_shape.text_frame
        title_text_frame.text = title
        title_text_frame.paragraphs[0].font.size = Pt(28)
        title_text_frame.paragraphs[0].font.bold = True
        
        # Adjust figure position if there's a title
        top = Inches(1.5)
    else:
        # Center figure if no title
        top = Inches(0.75)
    
    # Add figure to slide (centered)
    left = Inches(0.1)
    height = Inches(6.0)  # Adjust as needed
    
    # Add picture
    if os.path.exists(figure_path):
        try:
            slide.shapes.add_picture(figure_path, left, top, height=height)
        except Exception as e:
            logger.error(f"Error adding figure {figure_path}: {e}")
            # Add a text box with the error message
            error_box = slide.shapes.add_textbox(left, top, Inches(8.0), Inches(1.0))
            error_box.text_frame.text = f"Error loading figure: {figure_path}\n{str(e)}"
    else:
        # Add a text box indicating the figure wasn't found
        error_box = slide.shapes.add_textbox(left, top, Inches(8.0), Inches(1.0))
        error_box.text_frame.text = f"Figure not found: {figure_path}"

def create_metrics_presentation(
    df: pd.DataFrame,
    metrics_text: Dict[str, str],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    figure_paths: List[Dict[str, str]] = None,
    output_dir: str = '.',
    ppt_filename: str = None,
    upload_to_gdrive: bool = False,
    gdrive_folder_id: str = None,
    gdrive_credentials_path: str = None,
    use_oauth: bool = False,
    token_path: str = None
) -> Dict[str, Any]:
    """
    Create a PowerPoint presentation with metrics summary and figures,
    optionally upload to Google Drive.
    
    Args:
        df: DataFrame containing the data
        metrics_text: Dictionary mapping metric names to their pre-formatted explanation text
        metric_anomaly_map: Map of metrics to their anomaly information
        figure_paths: List of dictionaries, each with 'path' and optional 'title'
        output_dir: Directory to save the presentation
        ppt_filename: Filename for the presentation (if None, will generate based on date)
        region_col: Name of the region column
        upload_to_gdrive: Whether to upload to Google Drive
        gdrive_folder_id: Google Drive folder ID to upload to
        gdrive_credentials_path: Path to Google Drive credentials
        use_oauth: Whether to use OAuth2 flow instead of service account
        token_path: Path to OAuth token file
        
    Returns:
        Dictionary with information about the created presentation:
        - 'local_path': Path to the local file
        - 'gdrive_url': Google Drive URL (if uploaded)
        - 'gdrive_id': Google Drive file ID (if uploaded)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if ppt_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ppt_filename = f"Metrics_Summary_{timestamp}.pptx"
    
    # Create presentation
    prs = Presentation()
    
    # Add metrics summary slide
    create_metrics_summary_slide(
        prs=prs,
        df=df,
        metrics_text=metrics_text,
        metric_anomaly_map=metric_anomaly_map,
        title="Metrics Summary"
    )
    
    # Add figure slides if provided
    if figure_paths:
        for fig_info in figure_paths:
            figure_path = fig_info.get('path')
            title = fig_info.get('title')
            add_figure_slide(prs, figure_path, title)
    
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
                use_oauth=use_oauth,
                token_path=token_path
            )
            result.update(gdrive_results)
            logger.info(f"Uploaded to Google Drive: {gdrive_results.get('gdrive_url', 'Unknown URL')}")
        except Exception as e:
            logger.error(f"Failed to upload to Google Drive: {e}")
            result['gdrive_error'] = str(e)
    
    return result

def upload_to_google_drive(
    file_path: str, 
    folder_id: str = None,
    credentials_path: str = None,
    use_oauth: bool = False,
    token_path: str = None
) -> Dict[str, str]:
    """
    Upload a file to Google Drive using either service account or OAuth2 flow.
    
    Args:
        file_path: Path to the file to upload
        folder_id: Google Drive folder ID to upload to
        credentials_path: Path to Google Drive credentials
        use_oauth: Whether to use OAuth2 flow instead of service account
        token_path: Path to OAuth token file
        
    Returns:
        Dictionary with:
        - 'gdrive_id': Google Drive file ID
        - 'gdrive_url': Google Drive URL
    """
    try:
        # Import the necessary libraries only when needed
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        
        # Define credentials based on authentication method
        drive_service = None
        
        if use_oauth:
            # OAuth2 flow for user authentication
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            
            # At least one of credentials_path or token_path must be provided
            if credentials_path is None and token_path is None:
                raise ValueError("For OAuth2 flow, either credentials_path or token_path must be provided")
            
            # Handle OAuth token flow
            creds = None
            
            # The file token.json stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first time
            if token_path and os.path.exists(token_path):
                creds = Credentials.from_authorized_user_info(
                    json.load(open(token_path)), SCOPES)
                
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if credentials_path is None:
                        raise ValueError("Need credentials_path for OAuth2 flow")
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                if token_path:
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
            
            drive_service = build('drive', 'v3', credentials=creds)
            
        else:
            # Service account authentication
            from google.oauth2 import service_account
            
            # Credentials path must be provided for service account
            if credentials_path is None:
                raise ValueError("credentials_path must be provided for service account authentication")
                
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
            
            # Create credentials
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=SCOPES
            )
            
            # Build the Drive service
            drive_service = build('drive', 'v3', credentials=credentials)
        
        # Prepare metadata
        file_metadata = {
            'name': os.path.basename(file_path),
            'mimeType': 'application/vnd.google-apps.presentation'  # Convert to Google Slides
        }
        
        # Set parent folder if provided
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        # Prepare file upload
        media = MediaFileUpload(
            file_path,
            mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation',
            resumable=True
        )
        
        # Upload file
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()
        
        return {
            'gdrive_id': file.get('id'),
            'gdrive_url': file.get('webViewLink')
        }
        
    except ImportError:
        logger.warning(
            "Google Drive upload requires additional packages. "
            "Install with: pip install google-api-python-client google-auth google-auth-oauthlib"
        )
        raise
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {e}")
        raise

def main(output_dir='.', upload_to_gdrive=False, gdrive_folder_id=None, 
        gdrive_credentials_path=None, use_oauth=False, token_path=None):
    """
    Example function demonstrating usage with dummy data.
    
    Args:
        output_dir: Directory for output files
        upload_to_gdrive: Whether to upload to Google Drive
        gdrive_folder_id: Google Drive folder ID to upload to
        gdrive_credentials_path: Path to Google Drive credentials
        use_oauth: Whether to use OAuth2 flow instead of service account 
        token_path: Path to OAuth token file
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data
    data = {
        'region': ['Global', 'North America', 'Europe', 'Asia', 'Latin America'],
        'conversion_rate_pct': [0.12, 0.08, 0.11, 0.13, 0.10],
        'avg_order_value': [75.0, 65.0, 80.0, 85.0, 72.0],
        'bounce_rate_pct': [0.35, 0.45, 0.32, 0.28, 0.34],
        'session_duration': [180, 120, 190, 210, 175]
    }
    df = pd.DataFrame(data)
    
    # Pre-formatted explanation texts
    metrics_text = {
        'conversion_rate_pct': 'Conversion Rate is 33.3% lower in North America compared to global average. Root cause: North America has 10pp higher Bounce Rate than global average, explaining 75% of the difference.',
        'avg_order_value': 'Average Order Value is 6.7% higher in Europe. Root cause: Session Duration is 5.6% higher than global average.'
    }
    
    # Example metric anomaly map
    metric_anomaly_map = {
        'conversion_rate_pct': {
            'anomalous_region': 'North America',
            'metric_val': df.loc[df['region'] == 'North America', 'conversion_rate_pct'].values[0],
            'global_val': df.loc[df['region'] == 'Global', 'conversion_rate_pct'].values[0],
            'direction': 'lower',
            'magnitude': 33.3,
            'higher_is_better': True
        },
        'avg_order_value': {
            'anomalous_region': 'Europe',
            'metric_val': df.loc[df['region'] == 'Europe', 'avg_order_value'].values[0],
            'global_val': df.loc[df['region'] == 'Global', 'avg_order_value'].values[0],
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
        gdrive_credentials_path=gdrive_credentials_path,
        use_oauth=use_oauth,
        token_path=token_path
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
    parser.add_argument('--credentials', help='Path to Google credentials file')
    parser.add_argument('--oauth', action='store_true', help='Use OAuth2 flow instead of service account')
    parser.add_argument('--token', help='Path to OAuth token file')
    
    args = parser.parse_args()
    
    # Run with provided arguments
    main(
        output_dir=args.output_dir,
        upload_to_gdrive=args.upload,
        gdrive_folder_id=args.folder_id,
        gdrive_credentials_path=args.credentials,
        use_oauth=args.oauth,
        token_path=args.token
    ) 