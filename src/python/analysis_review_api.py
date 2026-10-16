"""Orthanc and storage adapters for non-destructive rerun review."""
import json
import os
import orthanc
from results_storage import CHOResultsStorage, cho_storage
from analysis_review import ReviewError, ReviewManager, fingerprint, json_safe


class _ScopedStorage(CHOResultsStorage):
    """Retain one independent connection across the row lock, read and upsert."""
    def __init__(self):
        self.__dict__.update(vars(cho_storage))
        self.postgres_config = dict(cho_storage.postgres_config)
        self.postgres_connection = None

    def connect_postgres(self):
        if self.postgres_connection is not None and not self.postgres_connection.closed:
            return True
        return super().connect_postgres()

    def close(self):
        if self.postgres_connection is not None:
            self.postgres_connection.close()

    def record(self, series_uid):
        row, description = self.get_result(series_uid)
        return dict(zip((column[0] for column in description), row)) if row else None


class ResultRepository:
    def load(self, series_uid):
        storage = _ScopedStorage()
        try:
            return storage.record(series_uid)
        finally:
            storage.close()

    def overwrite(self, series_uid, baseline, candidate):
        storage = _ScopedStorage()
        try:
            if not storage.connect_postgres():
                raise ReviewError("Could not connect to result storage. The draft was retained.", 503)
            with storage.postgres_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT r.id FROM analysis.results r
                    JOIN dicom.series s ON s.id = r.series_id_fk
                    WHERE s.series_instance_uid = %s FOR UPDATE OF r
                """, (series_uid,))
                if cursor.fetchone() is None:
                    raise ReviewError("The saved result was deleted while this rerun was being reviewed.")
            if fingerprint(storage.record(series_uid)) != fingerprint(baseline):
                raise ReviewError("The saved result changed during this rerun. Discard this draft and rerun against the latest result.")
            if not storage.save_results(candidate["patient"], candidate["study"], candidate["scanner"],
                                        candidate["series"], candidate["ct"], candidate["results"]):
                raise ReviewError("Saving failed. The old result and new draft were retained.", 503)
        finally:
            # Uncommitted errors roll back on close. The existing save path
            # commits before writing the accepted coronal image to MinIO.
            storage.close()


review_manager = ReviewManager(ResultRepository())


def _answer(output, payload, status=200):
    content = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
    if status != 200:
        output.SetHttpHeader("Content-Type", "application/json")
        output.SendHttpStatus(status, content)
    else:
        output.AnswerBuffer(content, "application/json")


def handle_reviews(output, url, **request):
    output.SetHttpHeader("Access-Control-Allow-Origin", os.getenv("CORS_ALLOWED_ORIGIN", "*"))
    output.SetHttpHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    output.SetHttpHeader("Access-Control-Allow-Headers", "Content-Type, Authorization")
    output.SetHttpHeader("Cache-Control", "no-store")
    if os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true":
        output.SetHttpHeader("Access-Control-Allow-Credentials", "true")
    if request.get("method") == "OPTIONS":
        output.SendHttpStatusCode(204)
        return
    try:
        groups = request.get("groups") or []
        token = groups[0] if groups else None
        method = request.get("method")
        if method == "GET":
            _answer(output, review_manager.detail(token) if token else {"reviews": review_manager.pending()})
        elif method == "POST" and token:
            body = request.get("body", b"{}")
            payload = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
            if not isinstance(payload, dict):
                raise ReviewError("Expected a JSON object.", 400)
            _answer(output, review_manager.decide(token, payload.get("decision")))
        else:
            output.SendMethodNotAllowed("GET, POST, OPTIONS" if token else "GET, OPTIONS")
    except ReviewError as exc:
        _answer(output, {"error": str(exc)}, exc.status)
    except (ValueError, UnicodeDecodeError):
        _answer(output, {"error": "Invalid JSON request."}, 400)
    except Exception:
        _answer(output, {"error": "Review request failed. Reload the saved result before retrying."}, 500)


orthanc.RegisterRestCallback("/cho-result-reviews", handle_reviews)
orthanc.RegisterRestCallback("/cho-result-reviews/([a-f0-9]{32})", handle_reviews)
