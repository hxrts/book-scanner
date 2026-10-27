"""Append book rows to a Google Sheet using a service account."""

from datetime import datetime, timezone
from pathlib import Path

import gspread

COLUMNS = [
    "scanned_at", "isbn", "title", "subtitle", "authors", "publishers", "publish_date",
    "pages", "subjects", "lccn", "oclc", "lc_classification", "dewey", "cover_url",
    "open_library_url",
]


class BookSheet:
    def __init__(self, sheet_id, tab, credentials):
        client = gspread.service_account(filename=str(Path(credentials).expanduser()))
        spreadsheet = client.open_by_key(sheet_id)
        try:
            self.ws = spreadsheet.worksheet(tab)
        except gspread.WorksheetNotFound:
            self.ws = spreadsheet.add_worksheet(tab, rows=1000, cols=len(COLUMNS))
        if self.ws.row_values(1) != COLUMNS:
            self.ws.update([COLUMNS], "A1")

    def append(self, book):
        row = dict(book, scanned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.ws.append_row([row.get(c, "") for c in COLUMNS], value_input_option="RAW")
