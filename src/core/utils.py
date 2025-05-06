"""Utility functions shared across the RCA automation pipeline."""
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

# Define scopes for Google APIs
SCOPES = [
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/drive.file'
]

def convert_to_numeric(value):
    """Convert a value to numeric, handling percentage strings."""
    if isinstance(value, str):
        value = value.strip().rstrip('%')
        try:
            # Handle potential commas in numbers
            return float(value.replace(',', ''))
        except (ValueError, TypeError):
            return 0.0 # Return float 0.0 for consistency
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def map_territory_to_region(territory, mapping=None):
    """Maps Territory format to Region format using provided mapping or default rules."""
    if mapping and territory in mapping:
        return mapping[territory]

    territory = str(territory).strip()

    # Prioritize specific mappings if provided or needed
    # Example: if territory == 'North America': return 'NA'

    # Fallback to partial matching
    if "APAC" in territory:
        return "APAC"
    elif "EMEA" in territory:
        return "EMEA"
    elif "LATAM" in territory:
        return "LATAM"
    elif "NA" in territory:
        return "NA"
    elif territory.lower() == "global": # Case-insensitive Global check
        return "Global"
    else:
        # Optional: Log unknown territories
        # logger.warning(f"Unknown territory mapping: {territory}")
        return territory # Return original if no match

def get_credentials():
    """Gets user credentials for Google APIs, handling token refresh/creation."""
    creds = None
    # Assume token.json and credentials.json are in the ROOT directory of the project
    # Adjust path relative to this utils.py file (src/core/utils.py)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    token_path = os.path.join(project_root, 'token.json')
    credentials_path = os.path.join(project_root, 'credentials.json')

    if not os.path.exists(credentials_path):
         print(f"ERROR: credentials.json not found at {credentials_path}")
         print("Please download credentials from Google Cloud Console and place it in the project root.")
         return None

    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception as e:
            print(f"Error loading token.json: {e}. Will attempt re-authentication.")
            creds = None # Force re-auth

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}. Please re-authenticate.")
                # Optionally delete token.json to force full re-auth
                # if os.path.exists(token_path): os.remove(token_path)
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            except FileNotFoundError:
                 print(f"ERROR: credentials.json not found at {credentials_path}")
                 return None
            except Exception as e:
                 print(f"Error during authentication flow: {e}")
                 return None
        try:
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            print(f"Error saving token to {token_path}: {e}")

    return creds

def convert_bool(val):
    """Convert a value to boolean, handling string representations."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ['true', 'yes', '1']
    if isinstance(val, (int, float)):
         return bool(val) # Treat non-zero numbers as True
    return False 