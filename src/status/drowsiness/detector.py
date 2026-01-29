import time
from collections import deque

from src.logging.system_logger import SystemLogger
from src.status.drowsiness.config import load_drowsiness_config
from src.status.drowsiness.rules import log_drowsy_episode, log_yawn


class DrowsinessDetector:
    """
    Detects drowsiness (low EAR over time) and yawning (MAR/expr/hands).
    """

    def __init__(self, logger: SystemLogger, fps: float = 30.0, config_path: str = "config/detector_config.yaml"):
        self.logger = logger
        self.fps = fps
        self.cfg = load_drowsiness_config(config_path, fps)
        self.user = None

        self._last_frame_rgb = None

        self.ear_history = deque(maxlen=int(self.cfg["ear"]["history_frames"]))
        self.ear_drop_detected = False

        # NEW: EAR smoothing + perclos history
        self._ear_ema = None
        self._eyes_closed_hist = deque(maxlen=int(self.cfg["perclos"]["window_frames"]))
        self._perclos = 0.0

        self.dynamic_ear_thresh = float(self.cfg["ear"]["low"])

        self.counters = {
            "DROWSINESS": 0,
            "RECOVERY": 0,
            "BLINK": 0,
            "EYES_CLOSED": 0,
            "SMILE_SUP": 0,
            "LAUGH_SUP": 0,
            "YAWN": 0,
            "YAWN_COOL": 0,
        }

        self.states = {"IS_DROWSY": False, "IS_YAWNING": False, "EYES_CLOSED": False}
        self.episode = {"active": False, "start_time": None, "min_ear": 1.0, "start_frame": None}

        self.yawn_timestamps = deque(maxlen=int(self.cfg["yawn"]["timestamps_max"]))
        self.yawn_count = 0

    def set_last_frame(self, frame):
        self._last_frame_rgb = frame.copy() if frame is not None else None

    def set_active_user(self, user_profile):
        self.user = user_profile
        base_ear = float(user_profile.ear_threshold) if user_profile else float(self.cfg["ear"]["low"])

        self.dynamic_ear_thresh = base_ear
        self.cfg["ear"]["low"] = base_ear

        ratio = float(self.cfg["ear"].get("high_ratio", 1.2))
        self.cfg["ear"]["high"] = base_ear * ratio

        self._reset_state()

    def _reset_state(self):
        for k in self.counters.keys():
            self.counters[k] = 0
        self.episode.update({"active": False, "start_time": None, "min_ear": 1.0, "start_frame": None})
        for k in self.states.keys():
            self.states[k] = False
        self.ear_history.clear()
        self.yawn_timestamps.clear()

        # NEW
        self._ear_ema = None
        self._eyes_closed_hist.clear()
        self._perclos = 0.0

    def detect(self, ear, mar, expression, hands_data=None, face_center=None, pitch=None):
        ear = float(ear)
        mar = float(mar)

        # NEW: smooth EAR (EMA)
        alpha = float(self.cfg["ear"]["ema_alpha"])
        if self._ear_ema is None:
            self._ear_ema = ear
        else:
            self._ear_ema = (1.0 - alpha) * float(self._ear_ema) + alpha * ear

        ear_used = float(self._ear_ema)

        self.ear_history.append(ear_used)
        self._detect_sudden_ear_drop()

        # closed decision uses low threshold on smoothed EAR
        self.states["EYES_CLOSED"] = ear_used < float(self.cfg["ear"]["low"])

        # NEW: perclos update
        self._update_perclos(self.states["EYES_CLOSED"])

        self._update_suppression(expression)
        self._update_blink(ear_used)
        self._update_drowsiness(ear_used)
        self._update_yawn(mar, expression, hands_data, face_center)

        if self.states["IS_DROWSY"]:
            return "DROWSY", (0, 0, 255)
        if self.states["IS_YAWNING"]:
            return "YAWNING", (0, 255, 255)
        return "NORMAL", (0, 255, 0)

    def _update_perclos(self, eyes_closed: bool) -> None:
        self._eyes_closed_hist.append(1 if eyes_closed else 0)
        if not self._eyes_closed_hist:
            self._perclos = 0.0
            return
        self._perclos = float(sum(self._eyes_closed_hist)) / float(len(self._eyes_closed_hist))

    def _detect_sudden_ear_drop(self):
        n = int(self.cfg["ear"]["drop_window_frames"])
        if len(self.ear_history) < n:
            return
        drop = self.ear_history[-n] - self.ear_history[-1]
        self.ear_drop_detected = drop > float(self.cfg["ear"]["drop"])

    def _update_suppression(self, expr):
        if expr == "SMILE":
            self.counters["SMILE_SUP"] = int(self.cfg["expression_suppression"]["smile_frames"])
        elif expr == "LAUGH":
            self.counters["LAUGH_SUP"] = int(self.cfg["expression_suppression"]["laugh_frames"])
        else:
            if self.counters["SMILE_SUP"] > 0:
                self.counters["SMILE_SUP"] -= 1
            if self.counters["LAUGH_SUP"] > 0:
                self.counters["LAUGH_SUP"] -= 1

    def _update_drowsiness(self, ear: float):
        is_suppressed = self.counters["SMILE_SUP"] > 0 or self.counters["LAUGH_SUP"] > 0

        start_frames = int(self.cfg["episode"]["start_frames"])
        end_frames = int(self.cfg["episode"]["end_frames"])
        ear_low = float(self.cfg["ear"]["low"])
        ear_high = float(self.cfg["ear"]["high"])

        # NEW: perclos trigger (window-based)
        perclos_thr = float(self.cfg["perclos"]["threshold"])
        perclos_active = (self._perclos >= perclos_thr)

        # NEW: fast onset when sudden drop detected
        if self.ear_drop_detected:
            start_frames = max(1, int(start_frames * float(self.cfg["episode"]["drop_start_multiplier"])))

        if self.episode["active"]:
            self.episode["min_ear"] = min(self.episode["min_ear"], ear)

            if ear >= ear_high or is_suppressed:
                self.counters["RECOVERY"] += 1
                if self.counters["RECOVERY"] >= end_frames:
                    dur = time.time() - float(self.episode["start_time"] or time.time())

                    if dur >= float(self.cfg["episode"]["min_episode_sec"]):
                        sev_cfg = self.cfg["severity"]["drowsiness"]
                        if dur >= float(sev_cfg["critical_sec"]):
                            severity = "Critical"
                        elif dur >= float(sev_cfg["high_sec"]):
                            severity = "High"
                        elif dur >= float(sev_cfg["medium_sec"]):
                            severity = "Medium"
                        else:
                            severity = "Low"

                        log_drowsy_episode(
                            self.logger,
                            self.user,
                            dur,
                            float(self.episode["min_ear"]),
                            self.episode["start_frame"],
                            severity,
                        )

                    self.episode["active"] = False
                    self.counters["DROWSINESS"] = 0
            else:
                self.counters["RECOVERY"] = 0

        elif (ear < ear_low or perclos_active) and not is_suppressed:
            self.counters["DROWSINESS"] += 1
            if self.counters["DROWSINESS"] >= start_frames:
                self.episode.update(
                    {"active": True, "start_time": time.time(), "start_frame": self._last_frame_rgb, "min_ear": 1.0}
                )
                self.counters["RECOVERY"] = 0
        else:
            self.counters["DROWSINESS"] = 0

        self.states["IS_DROWSY"] = bool(self.episode["active"])

    def _update_blink(self, ear: float):
        if ear < float(self.dynamic_ear_thresh):
            self.counters["EYES_CLOSED"] += 1
            return

        closed = int(self.counters["EYES_CLOSED"])
        min_f = int(self.cfg["blink"]["min_closed_frames"])
        max_f = int(self.cfg["blink"]["max_closed_frames"])
        if min_f <= closed <= max_f:
            self.counters["BLINK"] += 1
        self.counters["EYES_CLOSED"] = 0

    def _update_yawn(self, mar: float, expr, hands, face):
        if self.counters["YAWN_COOL"] > 0:
            self.counters["YAWN_COOL"] -= 1
            self.states["IS_YAWNING"] = False
            return

        covered = False
        if hands and face:
            thr = float(self.cfg["yawn"]["hand_cover_distance_norm"])
            thr2 = thr * thr
            fx, fy = float(face[0]), float(face[1])

            for h in hands:
                if not h or len(h) <= 12:
                    continue
                hx, hy = float(h[12][0]), float(h[12][1])
                dx = hx - fx
                dy = hy - fy
                if (dx * dx + dy * dy) < thr2:
                    covered = True
                    break

        # NEW: if "covered", require MAR to be above a minimum to count as yawn evidence
        covered_mar_min = float(self.cfg["yawn"]["covered_mar_min"])

        if (expr == "YAWN") or (covered and mar >= covered_mar_min and expr not in ["SMILE", "LAUGH"]):
            self.counters["YAWN"] += 1
        else:
            self.counters["YAWN"] = 0

        if self.counters["YAWN"] >= int(self.cfg["yawn"]["thresh_frames"]):
            if not self.states["IS_YAWNING"]:
                now = time.time()
                self.yawn_timestamps.append(now)
                self.yawn_count += 1

                win = float(self.cfg["yawn"]["frequency_window_sec"])
                hi_n = int(self.cfg["yawn"]["high_frequency_count"])
                recent_yawns = sum(1 for t in self.yawn_timestamps if now - t < win)

                if recent_yawns >= hi_n:
                    severity = "High"
                    alert_detail = f"Frequent Yawning ({recent_yawns} in {int(win)} sec)"
                    if covered:
                        alert_detail += " with Hand Near Face"
                elif covered:
                    severity = "Medium"
                    alert_detail = "Yawning with Hand Near Face"
                else:
                    severity = "Low"
                    alert_detail = "Yawning"

                log_yawn(self.logger, self.user, mar, self._last_frame_rgb, alert_detail, severity)

                self.counters["YAWN_COOL"] = int(self.cfg["yawn"]["cooldown_frames"])
                self.states["IS_YAWNING"] = True

    def get_detailed_state(self):
        # NEW: simple trend estimate from ear history
        ear_trend = "STABLE"
        if len(self.ear_history) >= 6:
            delta = float(self.ear_history[-1]) - float(self.ear_history[-6])
            if delta < -0.03:
                ear_trend = "FALLING"
            elif delta > 0.03:
                ear_trend = "RISING"

        return {
            "is_drowsy": self.states["IS_DROWSY"],
            "is_yawning": self.states["IS_YAWNING"],
            "eyes_closed": self.states["EYES_CLOSED"],
            "blink_count": self.counters["BLINK"],
            "ear_trend": ear_trend,
            "perclos": float(self._perclos),
        }