# file: src/infrastructure/data/drowsiness_events/repository.py
import sqlite3
import logging
from typing import Optional, Iterable, Any

class DrowsinessEventRepository:
    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection

    def add_event_row(
        self,
        vehicle_identification_number: str,
        user_id: int,
        time: str,
        status: str,
        img_drowsiness: Optional[bytes],
        img_path: Optional[str],  # Argument kept for compatibility, but ignored below
        duration: float = 0.0,
        value: float = 0.0
    ) -> int:
        cur = self.conn.cursor()
        try:
            # REMOVED: img_path from INSERT
            cur.execute(
                """
                INSERT INTO drowsiness_events
                (vehicle_identification_number, user_id, time, status, img_drowsiness, duration, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (vehicle_identification_number, user_id, time, status, img_drowsiness, duration, value),
            )
        except sqlite3.OperationalError as e:
            logging.error(f"Database Insert Failed: {e}")
            return -1
            
        self.conn.commit()
        event_id = int(cur.lastrowid)
        logging.info(f"Event saved: id={event_id} type={status} dur={duration:.1f}s val={value:.2f}")
        return event_id

    def _fetchall(self, query: str, params: Iterable[Any] = ()) -> list[tuple]:
        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()

    # REMOVED: img_path from all SELECT statements below
    def get_all_events(self) -> list[tuple]:
        return self._fetchall(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, duration, value
            FROM drowsiness_events
            ORDER BY time DESC
            """
        )

    def get_events_by_user(self, user_id: int) -> list[tuple]:
        return self._fetchall(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, duration, value
            FROM drowsiness_events
            WHERE user_id = ?
            ORDER BY time DESC
            """,
            (user_id,),
        )

    def get_events_by_vehicle(self, vin: str) -> list[tuple]:
        return self._fetchall(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, duration, value
            FROM drowsiness_events
            WHERE vehicle_identification_number = ?
            ORDER BY time DESC
            """,
            (vin,),
        )