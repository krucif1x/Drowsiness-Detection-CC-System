import os
import sys
import time
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `import src...` works when running directly:
#   python3 tests/buzzer.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.infrastructure.hardware.buzzer import Buzzer  # noqa: E402


def _hardware_tests_enabled() -> bool:
    return str(os.getenv("DS_RUN_HARDWARE_TESTS", "0")).strip().lower() in ("1", "true", "yes", "on")


@pytest.mark.hardware
def test_buzzer_pulse_and_pattern_no_exceptions():
    """
    Hardware/integration test for the GPIO buzzer.

    By default this test is skipped.
    Enable explicitly:
      DS_RUN_HARDWARE_TESTS=1 pytest -q

    Optional:
      BUZZER_PIN=17 DS_RUN_HARDWARE_TESTS=1 pytest -q -k buzzer
      DS_BUZZER_DISABLED=1 DS_RUN_HARDWARE_TESTS=1 pytest -q
    """
    if not _hardware_tests_enabled():
        pytest.skip("Hardware tests disabled. Set DS_RUN_HARDWARE_TESTS=1 to run.")

    pin = int(os.getenv("BUZZER_PIN", "17"))
    b = Buzzer(pin=pin)

    if not b.available():
        pytest.skip("Buzzer not available (gpiozero missing, no GPIO, init failed, or DS_BUZZER_DISABLED=1).")

    b.pulse(duration_sec=0.2, background=False)
    time.sleep(0.1)

    b.beep_for(on_time=0.1, off_time=0.1, duration_sec=1.0)
    time.sleep(1.2)

    b.off()


if __name__ == "__main__":
    # Run without pytest:
    #   DS_RUN_HARDWARE_TESTS=1 python3 tests/buzzer.py
    if not _hardware_tests_enabled():
        raise SystemExit("Hardware tests disabled. Set DS_RUN_HARDWARE_TESTS=1")

    pin = int(os.getenv("BUZZER_PIN", "17"))
    b = Buzzer(pin=pin)
    print(f"buzzer.available() = {b.available()}")
    if not b.available():
        raise SystemExit("Buzzer not available (gpiozero missing, no GPIO, init failed, or DS_BUZZER_DISABLED=1).")

    print("Pulse (0.2s)...")
    b.pulse(duration_sec=0.2, background=False)
    time.sleep(0.2)

    print("Beep pattern (1.0s)...")
    b.beep_for(on_time=0.1, off_time=0.1, duration_sec=1.0)
    time.sleep(1.2)

    print("Off")
    b.off()