import os
import logging
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from typing import Optional

# Setup logging
logger = logging.getLogger(__name__)
# Configure root logger if necessary, or rely on calling script's config
# logging.basicConfig(level=logging.INFO)

# Define scopes for Google APIs
SCOPES = [
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/drive.file'
]

def get_credentials():
    creds = None
    # Get the base directory (RCA_automation folder)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    token_path = os.path.join(base_dir, 'token.json')
    credentials_path = os.path.join(base_dir, 'credentials.json')
    
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return creds

def generate_ppt(analysis_results: dict, output_dir: str, ppt_filename: str = "RCA_Summary.pptx", report_format: str = "detailed"):
    """Generates a simplified PowerPoint presentation from RCA results and visualizations.
    Only embeds figures without titles or additional elements.

    Args:
        analysis_results: Dictionary containing processed analysis results per metric.
        output_dir: Directory where visualization PNG files are saved and
                    where the PowerPoint file will be saved.
        ppt_filename: The name for the output PowerPoint file.
        report_format: Type of report/visualization expected.
    """
    prs = Presentation()
    # Use a blank slide layout
    try:
        blank_slide_layout = prs.slide_layouts[6]
    except IndexError:
        logger.warning("Blank slide layout (index 6) not found. Using first layout.")
        blank_slide_layout = prs.slide_layouts[0]

    logger.info(f"Generating simplified PowerPoint presentation: {ppt_filename}")

    metrics_processed = 0
    
    # Step 1: First add all metric plots
    for metric, results in analysis_results.items():
        # Check for metric plot
        metric_filename = f"metric_{metric}.png"
        metric_path = os.path.join(output_dir, metric_filename)
        
        if os.path.exists(metric_path):
            slide = prs.slides.add_slide(blank_slide_layout)
            try:
                # Use full slide for image
                img_width_inches = 10.0
                left = Inches(0.0)
                top = Inches(0.0)
                
                slide.shapes.add_picture(metric_path, left, top, width=Inches(img_width_inches))
                metrics_processed += 1
                logger.info(f"Added metric plot for '{metric}': {metric_filename}")
            except Exception as e:
                logger.error(f"Error adding metric image {metric_path} to slide: {e}")
    
    # Step 2: Add all ranked views
    for metric, results in analysis_results.items():
        primary_region = results.get('primary_region', 'Unknown')
        if primary_region in ["NoAnomaly", "NoData", None]:
            continue
            
        # Check for ranked view
        summary_plot_paths = results.get('summary_plot_paths', {})
        if 'ranked' in summary_plot_paths and os.path.exists(summary_plot_paths['ranked']):
            slide = prs.slides.add_slide(blank_slide_layout)
            try:
                # Use full slide for image
                img_width_inches = 10.0
                left = Inches(0.0)
                top = Inches(0.0)
                
                slide.shapes.add_picture(summary_plot_paths['ranked'], left, top, width=Inches(img_width_inches))
                metrics_processed += 1
                logger.info(f"Added ranked view for '{metric}'")
            except Exception as e:
                logger.error(f"Error adding ranked view image to slide: {e}")
    
    # Step 3: Add all descriptive views
    for metric, results in analysis_results.items():
        primary_region = results.get('primary_region', 'Unknown')
        if primary_region in ["NoAnomaly", "NoData", None]:
            continue
            
        # Check for descriptive view
        summary_plot_paths = results.get('summary_plot_paths', {})
        if 'descriptive' in summary_plot_paths and os.path.exists(summary_plot_paths['descriptive']):
            slide = prs.slides.add_slide(blank_slide_layout)
            try:
                # Use full slide for image
                img_width_inches = 10.0
                left = Inches(0.0)
                top = Inches(0.0)
                
                slide.shapes.add_picture(summary_plot_paths['descriptive'], left, top, width=Inches(img_width_inches))
                metrics_processed += 1
                logger.info(f"Added descriptive view for '{metric}'")
            except Exception as e:
                logger.error(f"Error adding descriptive view image to slide: {e}")

    # Save the presentation
    if metrics_processed > 0:
        ppt_path = os.path.join(output_dir, ppt_filename)
        try:
            prs.save(ppt_path)
            logger.info(f"PowerPoint presentation saved successfully to {ppt_path}")
            return ppt_path
        except Exception as e:
            logger.error(f"Error saving PowerPoint presentation to {ppt_path}: {e}")
            return None
    else:
        logger.warning("No images were found or added to the PowerPoint presentation.")
        return None

def upload_to_drive(file_path: str, folder_id: Optional[str] = None, mime_type: str = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'):
    """Upload a file (e.g., the generated PPTX) to Google Drive.

    Args:
        file_path: Path to the file to upload (e.g., the PPTX path).
        folder_id: Optional Google Drive folder ID to upload into.
        mime_type: MIME type of the file being uploaded.

    Returns:
        File ID of the uploaded file if successful, None otherwise.
    """
    if not file_path or not os.path.exists(file_path):
        logger.error(f"File not found or path is invalid: {file_path}")
        return None

    try:
        creds = get_credentials()
        if not creds:
            logger.error("Failed to get Google credentials. Cannot upload.")
            return None

        service = build('drive', 'v3', credentials=creds)
        file_name = os.path.basename(file_path)

        # Metadata for the file on Google Drive
        file_metadata = {
            'name': file_name,
            # Optionally convert to Google Slides format on upload
            'mimeType': 'application/vnd.google-apps.presentation'
        }
        if folder_id:
            file_metadata['parents'] = [folder_id]

        # Prepare the media for upload
        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

        # Perform the upload
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink' # Request fields needed
        ).execute()

        file_id = file.get('id')
        web_link = file.get('webViewLink')

        logger.info(f"Successfully uploaded '{file_name}' to Google Drive.")
        logger.info(f"File ID: {file_id}")
        logger.info(f"Web Link: {web_link}")

        return file_id

    except Exception as e:
        logger.error(f"Error uploading file '{file_path}' to Google Drive: {e}", exc_info=True)
        return None 