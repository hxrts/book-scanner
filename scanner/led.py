"""The Pi's green activity LED as a scanner status light.

    off       ready for a page
    on        page held still, capturing
    blinking  last page was too blurry; rescan it

Needs write access to /sys/class/leds/ACT/{trigger,brightness}; deploy/99-scanner-led.rules
grants it. Without it (or off a Pi) the LED is silently skipped. On exit the
LED goes back to flashing on SD card activity.
"""

import atexit
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class StatusLed:
    def __init__(self, name="ACT", blink_seconds=0.15):
        self.dir = Path("/sys/class/leds") / name
        self.blink_seconds = blink_seconds
        self.state = None
        self._stop = threading.Event()
        self._thread = None
        try:
            self._original = (self.dir / "trigger").read_text().split("[")[1].split("]")[0]
            (self.dir / "trigger").write_text("none")
            self.available = True
        except (OSError, IndexError) as e:
            log.info("status LED not available (%s)", e)
            self.available = False
            return
        atexit.register(self.restore)
        self.off()

    def _write(self, on):
        try:
            (self.dir / "brightness").write_text("1" if on else "0")
        except OSError:
            pass

    def _set(self, state):
        if not self.available or state == self.state:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.state = state
        if state == "blink":
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._blink, args=(self._stop,), daemon=True)
            self._thread.start()
        else:
            self._write(state == "on")

    def _blink(self, stop):
        lit = False
        while not stop.wait(self.blink_seconds):
            lit = not lit
            self._write(lit)

    def on(self):
        self._set("on")

    def off(self):
        self._set("off")

    def blink(self):
        self._set("blink")

    def restore(self):
        self._set("off")
        try:
            (self.dir / "trigger").write_text(self._original)
        except OSError:
            pass
