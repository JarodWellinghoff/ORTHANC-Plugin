"""Stage reruns without changing accepted results.

Drafts survive browser refreshes, not plugin restarts. The repository is injected
so lifecycle tests do not require Orthanc, DICOM, PostgreSQL or MinIO.
"""
import copy
import hashlib
import json
import math
import threading
import time
import uuid
from datetime import date, datetime, time as datetime_time
from decimal import Decimal


class ReviewError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return json_safe(float(value))
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fingerprint(record):
    encoded = json.dumps(json_safe(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def has_analysis(record):
    # A metadata-only placeholder has processing_time=0 and no measurements.
    keys = ("ctdivol_avg", "ssde", "dw_avg", "dlp", "dlp_ssde",
            "average_index_of_detectability", "average_noise_level", "ctdivol",
            "dw", "cho_detectability", "noise_level", "nps")
    return bool(record) and any(record.get(key) is not None and record.get(key) != [] for key in keys)


class ReviewManager:
    def __init__(self, repository):
        self.repository = repository
        self._lock = threading.RLock()
        self._runs = {}
        self._series = {}

    def begin(self, series_uuid, series_uid):
        with self._lock:
            if series_uid in self._series:
                raise ReviewError("This series is running or has a rerun awaiting review.")
            token = uuid.uuid4().hex
            self._series[series_uid] = token
            self._runs[token] = {
                "id": token, "series_uuid": series_uuid, "series_uid": series_uid,
                "state": "running", "created_at": time.time(), "baseline": None,
            }
        try:
            baseline = self.repository.load(series_uid)
            needs_review = has_analysis(baseline)
            with self._lock:
                self._runs[token]["baseline"] = copy.deepcopy(baseline) if needs_review else None
            return token, needs_review
        except Exception:
            self.fail(token)
            raise

    def complete(self, token, payload):
        with self._lock:
            run = self._get(token)
            if run["baseline"] is None:
                self._remove(token)
                return False
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
                raise ReviewError("The calculation returned no reviewable results.", 500)
            if payload.get("series", {}).get("series_instance_uid") != run["series_uid"]:
                raise ReviewError("The calculated series does not match the review.", 500)
            if any(not isinstance(payload.get(key), dict) for key in ("patient", "study", "scanner", "ct")):
                raise ReviewError("The calculation returned incomplete metadata.", 500)
            run["candidate"] = copy.deepcopy(json_safe(payload))
            run["state"] = "ready"
            return True

    def fail(self, token):
        with self._lock:
            if token in self._runs:
                self._remove(token)

    def _get(self, token):
        if token not in self._runs:
            raise ReviewError("This review is no longer available. Reload the saved results.", 404)
        return self._runs[token]

    def _remove(self, token):
        run = self._runs.pop(token)
        self._series.pop(run["series_uid"], None)

    def pending(self):
        with self._lock:
            return [{"id": run["id"], "series_uuid": run["series_uuid"],
                     "series_uid": run["series_uid"], "created_at": run["created_at"],
                     "protocol": run["baseline"].get("protocol_name")}
                    for run in self._runs.values() if run["state"] == "ready"]

    def detail(self, token):
        with self._lock:
            run = self._get(token)
            if run["state"] != "ready":
                raise ReviewError("This calculation is not ready for review.")
            return copy.deepcopy(json_safe(run))

    def decide(self, token, decision):
        if decision not in ("overwrite", "discard"):
            raise ReviewError("Decision must be overwrite or discard.", 400)
        # Retain the reservation and draft until saving actually succeeds.
        with self._lock:
            run = self._get(token)
            if run["state"] != "ready":
                raise ReviewError("This calculation is not ready for review.")
            if decision == "overwrite":
                self.repository.overwrite(run["series_uid"], run["baseline"], run["candidate"])
            result = {"decision": decision, "series_uuid": run["series_uuid"]}
            self._remove(token)
            return result
