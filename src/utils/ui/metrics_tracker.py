import time
from collections import deque

class FpsTracker:
    """
    Measures FPS over a sampling window (stable), optionally with EMA smoothing.

    Call update() exactly once per processed/rendered frame.
    """
    def __init__(self, sample_period_sec: float = 0.5, ema_alpha: float | None = None):
        self.sample_period_sec = float(sample_period_sec)
        self.ema_alpha = ema_alpha  # e.g. 0.2 for smoothing, or None to disable

        self._t0 = time.perf_counter()
        self._last = self._t0
        self._frames = 0

        self.current_fps = 0.0          # last reported windowed fps
        self.smoothed_fps = 0.0         # EMA of windowed fps (if enabled)

    def update(self) -> float:
        now = time.perf_counter()
        self._frames += 1

        elapsed = now - self._t0
        if elapsed >= self.sample_period_sec:
            fps = self._frames / elapsed
            self.current_fps = fps

            if self.ema_alpha is not None:
                a = float(self.ema_alpha)
                if self.smoothed_fps == 0.0:
                    self.smoothed_fps = fps
                else:
                    self.smoothed_fps = a * fps + (1.0 - a) * self.smoothed_fps

            # reset window
            self._t0 = now
            self._frames = 0

        self._last = now
        return self.smoothed_fps if self.ema_alpha is not None else self.current_fps


class RollingAverage:
    """
    An optimized rolling average calculator using O(1) updates.
    Replaces the manual deque management in DetectionLoop.
    """
    def __init__(self, duration_sec: float, target_fps: float = 30.0):
        buffer_size = max(1, int(target_fps * duration_sec))
        self.buffer = deque(maxlen=buffer_size)
        self.current_sum = 0.0

    def update(self, value: float) -> float:
        if value is None:
            return self.get_average()

        if len(self.buffer) == self.buffer.maxlen:
            self.current_sum -= self.buffer[0]

        self.buffer.append(value)
        self.current_sum += value

        return self.current_sum / len(self.buffer)

    def get_average(self) -> float:
        return self.current_sum / len(self.buffer) if self.buffer else 0.0