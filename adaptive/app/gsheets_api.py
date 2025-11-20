# gsheets_api.py
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# Path to your service account JSON
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _get_service():
    """Authenticate and return a Google Sheets API service."""
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("sheets", "v4", credentials=creds)



def append_row_to_sheet(sheet_id: str, user_id: str, row_data: dict):
    """
    Append a single log row to a Google Sheet.

    - sheet_id: The ID of the spreadsheet (not the full URL).
    - user_id: Used as the sheet/tab name (1 tab per user).
    - row_data: dict of column_name -> value.
    """
    service = _get_service()
    sheets_api = service.spreadsheets()

    # --- 1. Ensure user-specific sheet/tab exists ---
    metadata = sheets_api.get(spreadsheetId=sheet_id).execute()
    existing_titles = [s["properties"]["title"] for s in metadata.get("sheets", [])]

    if user_id not in existing_titles:
        # Create the sheet
        add_sheet_req = {
            "requests": [
                {
                    "addSheet": {
                        "properties": {"title": user_id}
                    }
                }
            ]
        }
        sheets_api.batchUpdate(
            spreadsheetId=sheet_id,
            body=add_sheet_req
        ).execute()

        # Write header row (keys of the dict)
        header_values = [list(row_data.keys())]
        sheets_api.values().update(
            spreadsheetId=sheet_id,
            range=f"{user_id}!A1",
            valueInputOption="RAW",
            body={"values": header_values},
        ).execute()

    # --- 2. Append the actual data row (values of the dict) ---
    data_values = [list(row_data.values())]

    sheets_api.values().append(
        spreadsheetId=sheet_id,
        range=f"{user_id}!A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": data_values},
    ).execute()


def get_valid_user_ids(sheet_id: str):
    """Reads all valid user IDs from a sheet named 'Users'."""
    service = _get_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range="Users!A2:A"    # assumes header in A1
    ).execute()

    values = result.get("values", [])
    # Flatten list-of-lists and remove blanks
    return [row[0].strip() for row in values if row]
