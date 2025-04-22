"""Utility functions shared across the RCA automation pipeline."""
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

SCOPES = ['https://www.googleapis.com/auth/presentations',
          'https://www.googleapis.com/auth/drive.file']

def convert_to_numeric(value):
    """Convert a value to numeric, handling percentage strings.
    
    Args:
        value: The value to convert, can be string (with optional % suffix) or numeric
        
    Returns:
        float: The numeric value, or 0 if conversion fails
    """
    if isinstance(value, str):
        # Remove any whitespace and % symbol
        value = value.strip().rstrip('%')
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0

def map_territory_to_region(territory, mapping=None):
    """Maps Territory format to Region format using provided mapping or default rules.
    
    Args:
        territory: The territory name to map
        mapping: Optional dictionary mapping territories to regions
        
    Returns:
        str: The mapped region name
    """
    if mapping and territory in mapping:
        return mapping[territory]
        
    # Convert territory to string for matching
    territory = str(territory).strip()
        
      
    # Fallback to partial matching
    if "APAC" in territory:
        return "APAC"
    elif "EMEA" in territory:
        return "EMEA"
    elif "LATAM" in territory:
        return "LATAM"
    elif "NA" in territory:
        return "NA"
    elif territory == "Global":
        return "Global"
    else:
        return territory

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

def convert_bool(val):
    """Convert a value to boolean, handling string representations"""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == 'true'
    return False 