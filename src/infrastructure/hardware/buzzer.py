import logging
import os
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class Buzzer:
    """
    Simple GPIO buzzer wrapper (active buzzer: ON/OFF).
    Uses gpiozero.Buzzer if available; otherwise becomes a no-op.
    """

    def __init__(self, pin: int = 17):
        self._buzzer: Optional[object] = None

        if str(os.getenv("DS_BUZZER_DISABLED", "0")).strip().lower() in ("1", "true", "yes", "on"):
            log.info("Buzzer disabled via DS_BUZZER_DISABLED=1")
            return

        try:
            from gpiozero import Buzzer as _GPIOBuzzer  # type: ignore

            # gpiozero uses BCM numbering by default.
            self._buzzer = _GPIOBuzzer(pin)
            log.info("Buzzer initialized on BCM pin %s", pin)
        except Exception as e:
            self._buzzer = None
            log.warning("Buzzer unavailable (%s). Continuing without buzzer.", e)

    def available(self) -> bool:
        return self._buzzer is not None

    def __bool__(self) -> bool:
        return self.available()

    def beep(self, on_time: float = 0.1, off_time: float = 0.1, background: bool = True):
        """Start repeating beep pattern."""
        if not self._buzzer:
            return
        try:
            # gpiozero.Buzzer.beep(on_time=..., off_time=..., n=None, background=True)
            self._buzzer.beep(on_time=on_time, off_time=off_time, background=background)  # type: ignore[attr-defined]
        except Exception as e:
            log.debug("Buzzer.beep failed: %s", e)

    def off(self):
        """Stop buzzer (and stop any repeating beep pattern)."""
        if not self._buzzer:
            return
        try:
            self._buzzer.off()  # type: ignore[attr-defined]
        except Exception as e:
            log.debug("Buzzer.off failed: %s", e)

    def pulse(self, duration_sec: float = 0.2, background: bool = True):
        """Single beep: ON for duration_sec then OFF."""
        if not self._buzzer:
            return

        def _run():
            try:
                self._buzzer.on()  # type: ignore[attr-defined]
                time.sleep(max(0.0, float(duration_sec)))
            except Exception:
                pass
            finally:
                try:
                    self._buzzer.off()  # type: ignore[attr-defined]
                except Exception:
                    pass

        if background:
            threading.Thread(target=_run, daemon=True).start()
        else:
            _run()

    def beep_for(self, on_time: float, off_time: float, duration_sec: float):
        """Beep pattern for a fixed duration, then stop."""
        if not self._buzzer:
            return

        try:
            self.beep(on_time=on_time, off_time=off_time, background=True)
            threading.Timer(max(0.0, float(duration_sec)), self.off).start()
        except Exception as e:
            log.debug("Buzzer.beep_for failed: %s", e)


# Manual self-test (run from repo root):
#   python3 src/infrastructure/hardware/buzzer.py
#   BUZZER_PIN=17 python3 src/infrastructure/hardware/buzzer.py
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pin = int(os.getenv("BUZZER_PIN", "17"))
    b = Buzzer(pin=pin)
    print("Buzzer available:", b.available())
    b.pulse(duration_sec=0.2, background=False)
    time.sleep(0.2)
    b.beep_for(on_time=0.1, off_time=0.1, duration_sec=2.0)
    time.sleep(2.2)
    b.off()