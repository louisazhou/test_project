#!/usr/bin/env python3
"""
Google Drive utilities for uploading presentations and files.

Required dependencies:
- google-auth google-auth-oauthlib google-api-python-client
- oauth2client httplib2 (for enterprise environments)
"""

import os
from typing import Optional, Dict, Any

# Google Auth imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Enterprise auth imports
from oauth2client.service_account import ServiceAccountCredentials
import httplib2

# Google API client imports
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def get_credentials_local(credentials_path: Optional[str] = None, token_path: Optional[str] = None):
    """Get credentials using local files (development/personal use)."""
    SCOPES = ['https://www.googleapis.com/auth/presentations',
              'https://www.googleapis.com/auth/drive.file',
              'https://www.googleapis.com/auth/drive']
    
    creds = None
    if token_path and os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path or not os.path.exists(credentials_path):
                raise FileNotFoundError("credentials.json file is required for OAuth flow")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
            if token_path:
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
    
    return creds


def get_credentials_enterprise(credentials_dict: dict, proxy_info=None, ca_certs: Optional[str] = None):
    """Get credentials for enterprise environments with proxy support."""
    SCOPES = ['https://www.googleapis.com/auth/presentations',
              'https://www.googleapis.com/auth/drive.file',
              'https://www.googleapis.com/auth/drive']
    
    creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scopes=SCOPES)
    http = httplib2.Http(proxy_info=proxy_info, ca_certs=ca_certs)
    authorized_http = creds.authorize(http)
    
    return creds, authorized_http


def upload_to_google_drive(file_path: str, user_email: Optional[str] = None, **auth_kwargs):
    """Upload a file to Google Drive with smart authentication detection."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine auth method and get service
    if 'credentials_dict' in auth_kwargs:
        # Enterprise auth
        creds, authorized_http = get_credentials_enterprise(
            auth_kwargs['credentials_dict'],
            auth_kwargs.get('proxy_info'),
            auth_kwargs.get('ca_certs')
        )
        drive_service = build('drive', 'v3', http=authorized_http, cache_discovery=False)
    else:
        # Local auth
        creds = get_credentials_local(
            auth_kwargs.get('credentials_path'),
            auth_kwargs.get('token_path')
        )
        drive_service = build('drive', 'v3', credentials=creds)
    
    # Create user folder if user_email provided
    folder_id = None
    if user_email:
        username = user_email.split('@')[0] if '@' in user_email else user_email
        folder_name = f"RCA_Analysis_{username}"
        
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query).execute()
        
        if results['files']:
            folder_id = results['files'][0]['id']
        else:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = drive_service.files().create(body=folder_metadata).execute()
            folder_id = folder['id']
            
            if 'credentials_dict' in auth_kwargs:
                permission = {
                    'type': 'user',
                    'role': 'writer',
                    'emailAddress': user_email
                }
                drive_service.permissions().create(
                    fileId=folder_id,
                    body=permission,
                    sendNotificationEmail=False
                ).execute()
    
    # Upload file
    file_metadata = {
        'name': os.path.basename(file_path),
        'mimeType': 'application/vnd.google-apps.presentation'
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(
        file_path,
        mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation',
        resumable=True
    )
    
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id,webViewLink'
    ).execute()
    
    result = {
        'gdrive_id': file['id'],
        'gdrive_url': file['webViewLink']
    }
    
    if folder_id:
        result['folder_url'] = f"https://drive.google.com/drive/folders/{folder_id}"
    
    return result 