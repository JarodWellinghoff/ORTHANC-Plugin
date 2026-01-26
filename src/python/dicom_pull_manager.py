#!/usr/bin/env python3

"""Scheduled DICOM pull manager for Orthanc plugin."""

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import orthanc  # type: ignore
import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore
from psycopg2 import pool  # type: ignore


class DicomPullManager:
    """Manage scheduled DICOM pulls from external modalities into Orthanc."""

    SCHEMA_STATEMENTS = [
        "CREATE SCHEMA IF NOT EXISTS workflow AUTHORIZATION postgres",
        """
        CREATE OR REPLACE FUNCTION workflow.touch_updated_at()
        RETURNS trigger AS $$
        BEGIN
          NEW.updated_at = now();
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
        """
        CREATE TABLE IF NOT EXISTS workflow.dicom_pull_batches (
            id BIGSERIAL PRIMARY KEY,
            requested_by TEXT NULL,
            display_name TEXT NULL,
            remote_modality TEXT NOT NULL,
            start_time TIMESTAMPTZ NOT NULL,
            end_time TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                'pending',
                'in_progress',
                'paused',
                'completed',
                'failed',
                'cancelled',
                'expired'
            )),
            estimated_total_seconds INTEGER NOT NULL DEFAULT 0,
            actual_total_seconds INTEGER NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL,
            failure_reason TEXT NULL,
            notes TEXT NULL,
            timezone TEXT DEFAULT 'UTC'
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS workflow.dicom_pull_items (
            id BIGSERIAL PRIMARY KEY,
            batch_id BIGINT NOT NULL REFERENCES workflow.dicom_pull_batches(id) ON DELETE CASCADE,
            external_patient_id TEXT NULL,
            patient_name TEXT NULL,
            study_instance_uid TEXT NOT NULL,
            series_instance_uid TEXT NULL,
            description TEXT NULL,
            modality TEXT NULL,
            body_part TEXT NULL,
            study_date TEXT NULL,
            series_date TEXT NULL,
            number_of_instances INTEGER NULL,
            estimated_seconds INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                'pending',
                'in_progress',
                'completed',
                'failed',
                'skipped',
                'expired',
                'cancelled'
            )),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL,
            failure_reason TEXT NULL,
            orthanc_series_id TEXT NULL,
            metadata JSONB NULL
        );
        """,
        "CREATE INDEX IF NOT EXISTS dicom_pull_items_batch_idx ON workflow.dicom_pull_items (batch_id)",
        "CREATE INDEX IF NOT EXISTS dicom_pull_batches_status_idx ON workflow.dicom_pull_batches (status, start_time)",
        "CREATE INDEX IF NOT EXISTS dicom_pull_items_status_idx ON workflow.dicom_pull_items (status)",
        "DROP TRIGGER IF EXISTS dicom_pull_batches_touch_updated_at ON workflow.dicom_pull_batches",
        """
        CREATE TRIGGER dicom_pull_batches_touch_updated_at
        BEFORE UPDATE ON workflow.dicom_pull_batches
        FOR EACH ROW EXECUTE FUNCTION workflow.touch_updated_at();
        """,
        "DROP TRIGGER IF EXISTS dicom_pull_items_touch_updated_at ON workflow.dicom_pull_items",
        """
        CREATE TRIGGER dicom_pull_items_touch_updated_at
        BEFORE UPDATE ON workflow.dicom_pull_items
        FOR EACH ROW EXECUTE FUNCTION workflow.touch_updated_at();
        """,
    ]

    ITEM_TERMINAL_STATUSES = {"completed", "failed", "skipped", "expired", "cancelled"}
    BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

    def __init__(self) -> None:
        self.db_config = {
            "host": os.getenv("DICOM_PULL_DB_HOST", "postgres"),
            "port": int(os.getenv("DICOM_PULL_DB_PORT", "5432")),
            "database": os.getenv("DICOM_PULL_DB_NAME", "orthanc"),
            "user": os.getenv("DICOM_PULL_DB_USER", "postgres"),
            "password": os.getenv("DICOM_PULL_DB_PASSWORD", "pgpassword"),
        }
        self.seconds_per_instance = float(os.getenv("DICOM_PULL_SECONDS_PER_INSTANCE", "3.0"))
        self.min_item_seconds = int(os.getenv("DICOM_PULL_MIN_ITEM_SECONDS", "20"))
        self.batch_overhead_seconds = int(os.getenv("DICOM_PULL_BATCH_OVERHEAD_SECONDS", "30"))
        self.poll_interval_seconds = int(os.getenv("DICOM_PULL_POLL_INTERVAL_SECONDS", "30"))
        self.job_poll_interval_seconds = float(os.getenv("DICOM_PULL_JOB_POLL_INTERVAL_SECONDS", "2.0"))
        self.job_timeout_seconds = int(os.getenv("DICOM_PULL_JOB_TIMEOUT_SECONDS", "1800"))
        self.max_consecutive_failures = int(os.getenv("DICOM_PULL_MAX_CONSECUTIVE_FAILURES", "3"))

        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop_event = threading.Event()

        self._init_connection_pool()
        self._ensure_schema()
        self._start_worker()

    def _init_connection_pool(self) -> None:
        if self._pool is None:
            self._pool = pool.ThreadedConnectionPool(1, 5, **self.db_config)

    @contextmanager
    def _get_conn(self):
        if self._pool is None:
            self._init_connection_pool()
        assert self._pool is not None
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def _ensure_schema(self) -> None:
        with self._get_conn() as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                for statement in self.SCHEMA_STATEMENTS:
                    cursor.execute(statement)
            conn.autocommit = False

    def shutdown(self) -> None:
        """Stop worker thread and close pool."""
        self._worker_stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        if self._pool:
            self._pool.closeall()

    # -------------------------------------------------------------------------
    # Public API used by REST callbacks
    # -------------------------------------------------------------------------

    def list_batches(self, limit: int = 50) -> List[Dict[str, Any]]:
        query = """
            SELECT
                b.*,
                COALESCE(SUM(i.estimated_seconds), 0) AS computed_estimated_total_seconds,
                COUNT(i.id) AS total_items,
                COUNT(i.id) FILTER (WHERE i.status = 'completed') AS completed_items,
                COUNT(i.id) FILTER (WHERE i.status = 'failed') AS failed_items,
                COUNT(i.id) FILTER (WHERE i.status = 'pending') AS pending_items,
                COUNT(i.id) FILTER (WHERE i.status = 'in_progress') AS in_progress_items,
                COUNT(i.id) FILTER (WHERE i.status = 'expired') AS expired_items
            FROM workflow.dicom_pull_batches b
            LEFT JOIN workflow.dicom_pull_items i ON i.batch_id = b.id
            GROUP BY b.id
            ORDER BY b.created_at DESC
            LIMIT %s
        """
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_batch(self, batch_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM workflow.dicom_pull_batches WHERE id = %s",
                    (batch_id,),
                )
                batch = cursor.fetchone()
                if not batch:
                    return None
                cursor.execute(
                    """
                    SELECT *
                    FROM workflow.dicom_pull_items
                    WHERE batch_id = %s
                    ORDER BY id ASC
                    """,
                    (batch_id,),
                )
                items = cursor.fetchall()
        batch_dict = dict(batch)
        batch_dict["items"] = [dict(row) for row in items]
        return batch_dict

    def cancel_batch(self, batch_id: int, reason: Optional[str] = None) -> bool:
        query_update_batch = """
            UPDATE workflow.dicom_pull_batches
            SET status = 'cancelled',
                failure_reason = COALESCE(%s, failure_reason),
                completed_at = now()
            WHERE id = %s AND status NOT IN ('completed', 'cancelled', 'failed', 'expired')
        """
        query_update_items = """
            UPDATE workflow.dicom_pull_items
            SET status = 'cancelled',
                failure_reason = COALESCE(%s, failure_reason)
            WHERE batch_id = %s
              AND status NOT IN ('completed', 'failed', 'expired', 'cancelled')
        """
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query_update_batch, (reason, batch_id))
                updated = cursor.rowcount
                if updated == 0:
                    conn.rollback()
                    return False
                cursor.execute(query_update_items, (reason, batch_id))
            conn.commit()
        return True

    def create_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        items_payload = payload.get("items", [])
        if not items_payload:
            raise ValueError("At least one series must be provided.")

        start_time = payload.get("startTime")
        end_time = payload.get("endTime")

        if start_time is None:
            # YYYY-MM-DDTHH:MM:SS.sssZ
            start_time = datetime.now()
            start_time = start_time.replace(tzinfo=timezone.utc).isoformat()

        if end_time is None:
            # YYYY-MM-DDTHH:MM:SS.sssZ
            end_time = datetime.now() + timedelta(days=3650)
            end_time = end_time.replace(tzinfo=timezone.utc).isoformat()

        start_time = self._parse_datetime(start_time)
        end_time = self._parse_datetime(end_time)

        if start_time is None or end_time is None:
            raise ValueError("Invalid or missing scheduling window.")
        if end_time <= start_time:
            raise ValueError("End time must be after start time.")

        remote_modality = payload.get("modality")
        if not remote_modality:
            raise ValueError("Remote modality is required.")

        timezone_name = payload.get("timezone") or "UTC"

        normalized_items: List[Dict[str, Any]] = []
        estimated_total_seconds = 0

        for raw in items_payload:
            item = self._normalize_item(raw)
            estimated_total_seconds += item["estimated_seconds"]
            normalized_items.append(item)

        estimated_total_seconds += self.batch_overhead_seconds

        available_window = (end_time - start_time).total_seconds()
        if estimated_total_seconds > available_window:
            raise ValueError(
                "Estimated transfer time exceeds the allowed scheduling window."
            )

        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO workflow.dicom_pull_batches (
                        requested_by,
                        display_name,
                        remote_modality,
                        start_time,
                        end_time,
                        status,
                        estimated_total_seconds,
                        notes,
                        timezone
                    )
                    VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        payload.get("requestedBy"),
                        payload.get("displayName"),
                        remote_modality,
                        start_time,
                        end_time,
                        estimated_total_seconds,
                        payload.get("notes"),
                        timezone_name,
                    ),
                )
                batch_row = cursor.fetchone()
                batch_id = batch_row["id"]

                insert_items = """
                    INSERT INTO workflow.dicom_pull_items (
                        batch_id,
                        external_patient_id,
                        patient_name,
                        study_instance_uid,
                        series_instance_uid,
                        description,
                        modality,
                        body_part,
                        study_date,
                        series_date,
                        number_of_instances,
                        estimated_seconds,
                        metadata
                    )
                    VALUES (
                        %(batch_id)s,
                        %(external_patient_id)s,
                        %(patient_name)s,
                        %(study_instance_uid)s,
                        %(series_instance_uid)s,
                        %(description)s,
                        %(modality)s,
                        %(body_part)s,
                        %(study_date)s,
                        %(series_date)s,
                        %(number_of_instances)s,
                        %(estimated_seconds)s,
                        %(metadata)s
                    )
                """
                psycopg2.extras.execute_batch(
                    cursor,
                    insert_items,
                    [
                        {
                            "batch_id": batch_id,
                            **item,
                            "metadata": psycopg2.extras.Json(item.get("metadata")),
                        }
                        for item in normalized_items
                    ],
                )
            conn.commit()

        batch_dict = dict(batch_row)
        batch_dict["items"] = normalized_items
        return batch_dict
    def query_remote(
        self,
        modality: str,
        level: str,
        query: Dict[str, Any],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        level_upper = level.title()
        payload = {
            "Level": level_upper,
            "Query": query,
            "Limit": limit,
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            response = orthanc.RestApiPost(
                f"/modalities/{modality}/query", body
            )
        except Exception as exc:
            raise RuntimeError(f"Query to modality {modality} failed: {exc}")

        query_result = json.loads(response.decode("utf-8"))
        print(query_result)
        query_id = query_result.get("ID") or query_result.get("Query")
        if not query_id:
            raise RuntimeError("Unexpected response from Orthanc query.")

        try:
            answers_blob = orthanc.RestApiGet(f"/queries/{query_id}/answers")
            answers_paths = json.loads(answers_blob.decode("utf-8"))
            print(query_id)
            print(answers_paths)
            results: List[Dict[str, Any]] = []
            for path in answers_paths:
                index = path.rstrip("/").split("/")[-1]
                detail_blob = orthanc.RestApiGet(
                    f"/queries/{query_id}/answers/{index}/content?simplify"
                )
                detail = json.loads(detail_blob.decode("utf-8"))
                print(detail)
                normalized = self._normalize_query_answer(detail, level_upper)
                if normalized:
                    results.append(normalized)
            return results
        finally:
            try:
                orthanc.RestApiDelete(f"/queries/{query_id}")
            except Exception:
                pass
    # -------------------------------------------------------------------------
    # Background processing
    # -------------------------------------------------------------------------

    def _start_worker(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="DicomPullWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        consecutive_failures = 0
        while not self._worker_stop_event.is_set():
            try:
                processed = self._process_due_batches()
                consecutive_failures = 0
                if not processed:
                    self._worker_stop_event.wait(self.poll_interval_seconds)
            except Exception as exc:
                consecutive_failures += 1
                print(f"[DicomPullWorker] Error: {exc}")
                if consecutive_failures >= self.max_consecutive_failures:
                    print(
                        "[DicomPullWorker] Too many consecutive failures, pausing worker."
                    )
                    return
                self._worker_stop_event.wait(self.poll_interval_seconds)

    def _process_due_batches(self) -> bool:
        now = datetime.now(timezone.utc)
        batches = self._fetch_candidate_batches(now)
        any_processed = False
        for batch in batches:
            processed = self._process_single_batch(batch, now)
            any_processed = any_processed or processed
        return any_processed

    def _fetch_candidate_batches(self, now: datetime) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM workflow.dicom_pull_batches
            WHERE status IN ('pending', 'in_progress', 'paused')
            ORDER BY start_time ASC, id ASC
        """
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query)
                return [dict(row) for row in cursor.fetchall()]

    def _process_single_batch(self, batch: Dict[str, Any], now: datetime) -> bool:
        batch_id = batch["id"]
        start_time = batch["start_time"]
        end_time = batch["end_time"]

        if now < start_time:
            return False

        if now >= end_time:
            self._expire_batch(batch_id, reason="Scheduling window elapsed.")
            return True

        remaining_seconds = (end_time - now).total_seconds()
        items = self._fetch_batch_items(batch_id)
        pending_items = [
            item for item in items if item["status"] in ("pending", "in_progress")
        ]

        if not pending_items:
            self._mark_batch_completed(batch_id)
            return True

        next_item = None
        for item in pending_items:
            if item["status"] == "in_progress":
                next_item = item
                break
        if not next_item:
            next_item = pending_items[0]

        if next_item["status"] == "pending":
            if next_item["estimated_seconds"] > remaining_seconds:
                self._pause_batch(batch_id, reason="Insufficient time remaining.")
                return True
            self._mark_item_in_progress(next_item["id"])

        self._mark_batch_in_progress(batch_id)
        success = self._retrieve_item(batch, next_item)
        if success:
            self._mark_item_completed(next_item["id"])
        else:
            self._mark_item_failed(next_item["id"], "Retrieval failed or timed out.")

        updated_items = self._fetch_batch_items(batch_id)
        if all(item["status"] in self.ITEM_TERMINAL_STATUSES for item in updated_items):
            if any(item["status"] == "failed" for item in updated_items):
                self._mark_batch_failed(batch_id, "One or more series failed to retrieve.")
            elif any(item["status"] == "expired" for item in updated_items):
                self._mark_batch_expired(batch_id, "One or more series expired.")
            else:
                self._mark_batch_completed(batch_id)
        return True

    def _fetch_batch_items(self, batch_id: int) -> List[Dict[str, Any]]:
        query = """
            SELECT *
            FROM workflow.dicom_pull_items
            WHERE batch_id = %s
            ORDER BY id ASC
        """
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, (batch_id,))
                return [dict(row) for row in cursor.fetchall()]

    def _retrieve_item(self, batch: Dict[str, Any], item: Dict[str, Any]) -> bool:
        modality = batch["remote_modality"]
        query = {
            "Asynchronous": True,
            "Level": "Series" if item.get("series_instance_uid") else "Study",
            "Resources": [{}],
        }
        if item.get("series_instance_uid"):
            query["Resources"][0]["SeriesInstanceUID"] = item["series_instance_uid"]
        if item.get("study_instance_uid"):
            query["Resources"][0]["StudyInstanceUID"] = item["study_instance_uid"]

        payload = json.dumps(query).encode("utf-8")
        print(payload)

        started = time.monotonic()
        try:
            response = orthanc.RestApiPost(
                f"/modalities/{modality}/get", payload
            )
        except Exception as exc:
            print(f"[DicomPullWorker] Failed to start retrieve for item {item['id']}: {exc}")
            return False

        job_info = json.loads(response.decode("utf-8"))
        job_id = job_info.get("ID") or job_info.get("Job")
        if not job_id:
            print(f"[DicomPullWorker] Unexpected job response for item {item['id']}: {job_info}")
            return False

        deadline = time.monotonic() + self.job_timeout_seconds
        while time.monotonic() < deadline:
            try:
                job_blob = orthanc.RestApiGet(f"/jobs/{job_id}")
                job_state = json.loads(job_blob.decode("utf-8"))
            except Exception as exc:
                print(f"[DicomPullWorker] Failed to poll job {job_id}: {exc}")
                time.sleep(self.job_poll_interval_seconds)
                continue

            state = job_state.get("State")
            if state == "Success":
                self._set_item_duration(item["id"], time.monotonic() - started)
                self._record_retrieved_series_id(item)
                return True
            if state in ("Failure", "Cancelled"):
                print(f"[DicomPullWorker] Job {job_id} ended in state {state}: {job_state}")
                return False

            time.sleep(self.job_poll_interval_seconds)

        try:
            orthanc.RestApiPost(f"/jobs/{job_id}/cancel", b"")
        except Exception:
            pass
        return False

    def _record_retrieved_series_id(self, item: Dict[str, Any]) -> None:
        if not item.get("series_instance_uid"):
            return
        lookup_payload = json.dumps(
            {
                "Level": "Series",
                "Query": {"SeriesInstanceUID": item["series_instance_uid"]},
            }
        ).encode("utf-8")
        try:
            response = orthanc.RestApiPost("/tools/lookup", lookup_payload)
            matches = json.loads(response.decode("utf-8"))
            if not matches:
                return
            match = matches[0]
            orthanc_id = match.get("ID") or match.get("ID_")
            if not orthanc_id:
                return
            with self._get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE workflow.dicom_pull_items
                        SET orthanc_series_id = %s
                        WHERE id = %s
                        """,
                        (orthanc_id, item["id"]),
                    )
                conn.commit()
        except Exception as exc:
            print(f"[DicomPullWorker] Failed to lookup retrieved series: {exc}")

    def _set_item_duration(self, item_id: int, duration_seconds: float) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_items
                    SET completed_at = now(),
                        estimated_seconds = GREATEST(estimated_seconds, %s)
                    WHERE id = %s
                    """,
                    (int(duration_seconds), item_id),
                )
            conn.commit()
    # -------------------------------------------------------------------------
    # State transitions helpers
    # -------------------------------------------------------------------------

    def _mark_batch_in_progress(self, batch_id: int) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_batches
                    SET status = 'in_progress',
                        started_at = COALESCE(started_at, now())
                    WHERE id = %s AND status <> 'in_progress'
                    """,
                    (batch_id,),
                )
            conn.commit()

    def _mark_batch_completed(self, batch_id: int) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_batches
                    SET status = 'completed',
                        completed_at = now()
                    WHERE id = %s
                    """,
                    (batch_id,),
                )
            conn.commit()

    def _mark_batch_failed(self, batch_id: int, reason: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_batches
                    SET status = 'failed',
                        failure_reason = %s,
                        completed_at = now()
                    WHERE id = %s
                    """,
                    (reason, batch_id),
                )
            conn.commit()

    def _mark_batch_expired(self, batch_id: int, reason: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_batches
                    SET status = 'expired',
                        failure_reason = %s,
                        completed_at = now()
                    WHERE id = %s
                    """,
                    (reason, batch_id),
                )
            conn.commit()

    def _pause_batch(self, batch_id: int, reason: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_batches
                    SET status = 'paused',
                        failure_reason = %s
                    WHERE id = %s
                    """,
                    (reason, batch_id),
                )
            conn.commit()

    def _expire_batch(self, batch_id: int, reason: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_batches
                    SET status = 'expired',
                        failure_reason = %s,
                        completed_at = now()
                    WHERE id = %s
                    """,
                    (reason, batch_id),
                )
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_items
                    SET status = 'expired',
                        failure_reason = %s
                    WHERE batch_id = %s AND status NOT IN ('completed', 'failed', 'cancelled', 'expired')
                    """,
                    (reason, batch_id),
                )
            conn.commit()

    def _mark_item_in_progress(self, item_id: int) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_items
                    SET status = 'in_progress',
                        started_at = COALESCE(started_at, now())
                    WHERE id = %s
                    """,
                    (item_id,),
                )
            conn.commit()

    def _mark_item_completed(self, item_id: int) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_items
                    SET status = 'completed',
                        completed_at = now()
                    WHERE id = %s
                    """,
                    (item_id,),
                )
            conn.commit()

    def _mark_item_failed(self, item_id: int, reason: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workflow.dicom_pull_items
                    SET status = 'failed',
                        failure_reason = %s,
                        completed_at = now()
                    WHERE id = %s
                    """,
                    (reason, item_id),
                )
            conn.commit()
    # -------------------------------------------------------------------------
    # Normalization helpers
    # -------------------------------------------------------------------------

    def _normalize_item(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if "studyInstanceUID" not in raw:
            raise ValueError("studyInstanceUID is required for each item.")

        number_of_instances = raw.get("numberOfInstances")
        estimated_seconds = raw.get("estimatedSeconds")
        if estimated_seconds is None:
            estimated_seconds = self._estimate_seconds(number_of_instances)

        return {
            "external_patient_id": raw.get("patientId"),
            "patient_name": raw.get("patientName"),
            "study_instance_uid": raw["studyInstanceUID"],
            "series_instance_uid": raw.get("seriesInstanceUID"),
            "description": raw.get("description"),
            "modality": raw.get("modality"),
            "body_part": raw.get("bodyPart"),
            "study_date": raw.get("studyDate"),
            "series_date": raw.get("seriesDate"),
            "number_of_instances": number_of_instances,
            "estimated_seconds": max(int(estimated_seconds), self.min_item_seconds),
            "metadata": raw,
        }

    def _normalize_query_answer(
        self, detail: Dict[str, Any], level: str
    ) -> Optional[Dict[str, Any]]:
        if level == "Series":
            series_instance_uid = detail.get("SeriesInstanceUID")
            study_instance_uid = detail.get("StudyInstanceUID")
            if not study_instance_uid:
                study_instance_uid = detail.get("ParentMainDicomTags", {}).get(
                    "StudyInstanceUID"
                )
            if not study_instance_uid:
                parents = detail.get("ParentResources") or []
                for parent in parents:
                    tags = parent.get("MainDicomTags", {})
                    if "StudyInstanceUID" in tags:
                        study_instance_uid = tags["StudyInstanceUID"]
                        break
            if not study_instance_uid:
                return None

            number_instances = None
            if "NumberOfSeriesRelatedInstances" in detail:
                try:
                    number_instances = int(detail["NumberOfSeriesRelatedInstances"])
                except Exception:
                    pass

            return {
                "studyInstanceUID": study_instance_uid,
                "seriesInstanceUID": series_instance_uid,
                "patientName": detail.get("PatientName"),
                "patientId": detail.get("PatientID"),
                "modality": detail.get("Modality"),
                "bodyPart": detail.get("BodyPartExamined"),
                "description": detail.get("SeriesDescription"),
                "studyDate": detail.get("StudyDate"),
                "seriesDate": detail.get("SeriesDate"),
                "numberOfInstances": number_instances,
                "estimatedSeconds": self._estimate_seconds(number_instances),
                "raw": detail,
            }

        if level == "Study":
            study_instance_uid = detail.get("StudyInstanceUID")
            if not study_instance_uid:
                return None
            return {
                "studyInstanceUID": study_instance_uid,
                "patientName": detail.get("PatientName"),
                "patientId": detail.get("PatientID"),
                "studyDate": detail.get("StudyDate"),
                "modality": detail.get("ModalitiesInStudy"),
                "raw": detail,
            }
        return None

    def _estimate_seconds(self, number_of_instances: Optional[Any]) -> int:
        try:
            count = int(number_of_instances)
            if count <= 0:
                raise ValueError()
        except Exception:
            count = 1
        estimate = count * self.seconds_per_instance
        return int(max(estimate, self.min_item_seconds))

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt


# Global singleton instance used by the plugin
dicom_pull_manager = DicomPullManager()
