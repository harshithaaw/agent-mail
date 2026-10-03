from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials/client_secret.json"


def _get_credentials():
    """
    Shared OAuth credential logic: reuse saved token, refresh if expired,
    otherwise run the login flow. Used by both main() (standalone test)
    and get_gmail_service() (real usage by other modules).
    """
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            CREDENTIALS_FILE,
            SCOPES
        )
        creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return creds


def get_gmail_service():
    """
    Returns an authenticated Gmail API service object.
    This is what other modules (e.g. screening.py) should import and call
    to get something they can actually make Gmail API requests with.
    """
    creds = _get_credentials()
    return build("gmail", "v1", credentials=creds)


def main():
    get_gmail_service()
    print("Gmail authentication successful!")


if __name__ == "__main__":
    main()