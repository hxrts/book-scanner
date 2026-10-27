"""Runtime configuration, read from environment (see scanner.env.example)."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name, default=None, required=False):
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"missing required setting: {name}")
    return value


@dataclass(frozen=True)
class Config:
    # camera
    device: str = field(default_factory=lambda: _env("SCANNER_DEVICE", "/dev/video0"))
    width: int = field(default_factory=lambda: int(_env("SCANNER_WIDTH", "1920")))
    height: int = field(default_factory=lambda: int(_env("SCANNER_HEIGHT", "1080")))
    fps: int = field(default_factory=lambda: int(_env("SCANNER_FPS", "15")))

    # detection tuning
    page_min_area: float = field(default_factory=lambda: float(_env("SCANNER_PAGE_MIN_AREA", "0.15")))
    page_stable_seconds: float = field(default_factory=lambda: float(_env("SCANNER_PAGE_STABLE_S", "1.0")))
    # minimum Laplacian-variance focus score for a page capture (logged per capture; tune from logs)
    min_sharpness: float = field(default_factory=lambda: float(_env("SCANNER_MIN_SHARPNESS", "30")))
    isbn_cooldown_seconds: float = field(default_factory=lambda: float(_env("SCANNER_ISBN_COOLDOWN_S", "60")))

    # storage
    data_dir: Path = field(default_factory=lambda: Path(_env("SCANNER_DATA_DIR", "~/scanner-data")).expanduser())
    # cap on images kept from finished jobs; oldest are deleted first
    keep_mb: int = field(default_factory=lambda: int(_env("SCANNER_KEEP_MB", "2048")))
    ocr_lang: str = field(default_factory=lambda: _env("SCANNER_OCR_LANG", "eng"))

    # google sheets
    sheet_id: str = field(default_factory=lambda: _env("GOOGLE_SHEET_ID", ""))
    sheet_tab: str = field(default_factory=lambda: _env("GOOGLE_SHEET_TAB", "Books"))
    google_credentials: str = field(default_factory=lambda: _env("GOOGLE_CREDENTIALS", "~/.config/scanner/service-account.json"))

    # are.na
    arena_token: str = field(default_factory=lambda: _env("ARENA_TOKEN", ""))
    arena_channel: str = field(default_factory=lambda: _env("ARENA_CHANNEL", ""))

    # Are.na asks clients to identify themselves: "AppName/version (+contact)"
    contact: str = field(default_factory=lambda: _env("SCANNER_CONTACT", ""))

    @property
    def user_agent(self):
        return f"PageScanner/0.1 (+{self.contact})" if self.contact else "PageScanner/0.1"
