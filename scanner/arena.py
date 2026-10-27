"""Upload images as blocks to an Are.na channel (API v3)."""

import time

import requests

API = "https://api.are.na/v3"
TEMP_BUCKET = "https://s3.amazonaws.com/arena_images-temp"


class Arena:
    def __init__(self, token, channel, user_agent):
        self.channel = channel
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "User-Agent": user_agent})

    def _request(self, method, path, **kwargs):
        for attempt in range(4):
            resp = self.session.request(method, f"{API}{path}", timeout=30, **kwargs)
            if resp.status_code != 429:
                break
            time.sleep(int(resp.headers.get("Retry-After", 2 ** attempt * 5)))
        resp.raise_for_status()
        return resp.json()

    def check(self):
        """Fail fast on a bad token or channel."""
        self._request("GET", "/me")
        self._request("GET", f"/channels/{self.channel}")

    def upload_image(self, path, title, description, metadata, alt_text=""):
        presign = self._request(
            "POST", "/uploads/presign",
            json={"files": [{"filename": path.name, "content_type": "image/jpeg"}]},
        )["files"][0]
        put = requests.put(
            presign["upload_url"], data=path.read_bytes(),
            headers={"Content-Type": presign["content_type"]}, timeout=120,
        )
        put.raise_for_status()

        body = {
            "value": f"{TEMP_BUCKET}/{presign['key']}",
            "title": title,
            "description": description,
            "metadata": metadata,
            "channel_ids": [self.channel],
        }
        if alt_text:
            body["alt_text"] = alt_text
        return self._request("POST", "/blocks", json=body)
