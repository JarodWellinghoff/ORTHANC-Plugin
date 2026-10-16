"""Contract tests with synthetic records and Orthanc/storage service doubles."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
ROOT = Path(__file__).parents[1] / "src/python"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.row = {"id": 1, "ctdivol_avg": 5, "series_instance_uid": "1.2.3"}
        owner = self

        class Cursor:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, sql, values): owner.events.append(("lock", sql, values))
            def fetchone(self): return (1,) if owner.row else None

        class Connection:
            closed = False
            def cursor(self): return Cursor()
            def close(self):
                self.closed = True
                owner.events.append("close")

        class Storage:
            def connect_postgres(self):
                owner.events.append("connect")
                self.postgres_connection = Connection()
                return True
            def get_result(self, uid):
                self.connect_postgres()
                owner.events.append("read")
                return (tuple(owner.row.values()), [(key,) for key in owner.row]) if owner.row else (None, [])
            def save_results(self, patient, study, scanner, series, ct, results):
                self.connect_postgres()
                owner.events.append("save")
                owner.row.update(results)
                owner.events.append("commit")
                return True

        self.core = load("analysis_review")
        storage = types.SimpleNamespace(CHOResultsStorage=Storage, cho_storage=types.SimpleNamespace(postgres_config={}))
        self.modules = patch.dict(sys.modules, {"orthanc": types.SimpleNamespace(RegisterRestCallback=lambda *args: None),
                                               "results_storage": storage, "analysis_review": self.core})
        self.modules.start()
        self.api = load("analysis_review_api")
        self.payload = {"patient": {}, "study": {}, "scanner": {}, "ct": {},
                        "series": {"series_instance_uid": "1.2.3"}, "results": {"ctdivol_avg": 8}}

    def tearDown(self): self.modules.stop()

    def test_locked_overwrite_reuses_one_connection_until_commit(self):
        self.api.ResultRepository().overwrite("1.2.3", copy.deepcopy(self.row), self.payload)
        self.assertEqual(self.events.count("connect"), 1)
        self.assertIn("FOR UPDATE OF r", self.events[1][1])
        self.assertEqual(self.events[1][2], ("1.2.3",))
        self.assertEqual(self.events[2:], ["read", "save", "commit", "close"])
        self.assertEqual(self.row["ctdivol_avg"], 8)

    def test_conflict_does_not_save(self):
        baseline = copy.deepcopy(self.row)
        self.row["ctdivol_avg"] = 9
        with self.assertRaises(self.core.ReviewError):
            self.api.ResultRepository().overwrite("1.2.3", baseline, self.payload)
        self.assertNotIn("save", self.events)
        self.assertEqual(self.events[-1], "close")

    def test_deleted_result_not_silently_recreated(self):
        baseline, self.row = self.row, None
        with self.assertRaises(self.core.ReviewError):
            self.api.ResultRepository().overwrite("1.2.3", baseline, self.payload)
        self.assertNotIn("save", self.events)

    def test_api_list_and_preview_do_not_write(self):
        manager = self.core.ReviewManager(self.api.ResultRepository())
        token, _ = manager.begin("orthanc", "1.2.3")
        manager.complete(token, self.payload)
        self.api.review_manager = manager
        responses = []
        output = types.SimpleNamespace(SetHttpHeader=lambda *args: None,
            AnswerBuffer=lambda content, mime: responses.append(json.loads(content)),
            SendHttpStatus=lambda *args: self.fail("Unexpected error"))
        self.api.handle_reviews(output, "/cho-result-reviews", method="GET")
        self.api.handle_reviews(output, "/cho-result-reviews/" + token, method="GET", groups=[token])
        self.assertEqual(responses[0]["reviews"][0]["id"], token)
        self.assertEqual(responses[1]["candidate"]["results"]["ctdivol_avg"], 8)
        self.assertNotIn("save", self.events)

    def test_bad_json_produces_one_response_with_status_and_body(self):
        responses = []
        output = types.SimpleNamespace(SetHttpHeader=lambda *args: None,
            SendHttpStatus=lambda status, body: responses.append((status, json.loads(body))))
        self.api.handle_reviews(output, "/cho-result-reviews/x", method="POST", groups=["x"], body=b"not json")
        self.assertEqual(responses, [(400, {"error": "Invalid JSON request."})])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.threads, self.cleanup, self.failed = [], [], []
        self.needs_review = True
        owner = self

        class Manager:
            def begin(self, uuid, uid): return "run-token", owner.needs_review
            def complete(self, token, payload): owner.completed = (token, payload)
            def fail(self, token): owner.failed.append(token)
        self.current = {"metadata": {"run_id": "run-token"}}
        tracker = types.SimpleNamespace(
            start_calculation=lambda uid, metadata: setattr(self, "metadata", metadata),
            update_progress=lambda *args: None, fail_calculation=lambda *args: None,
            get_calculation_status=lambda uid: self.current, cleanup_calculation=lambda uid: self.cleanup.append(uid))

        def calculate(slices, config):
            self.params_used = config["custom_parameters"]
            return {"series": {"series_instance_uid": "1.2.3"}, "results": {"ctdivol_avg": 8}}
        orthanc = types.SimpleNamespace(
            RestApiGet=lambda path: json.dumps([{"ID": "slice"}] if path.endswith("instances") else {"MainDicomTags": {"SeriesInstanceUID": "1.2.3"}}),
            GetDicomForInstance=lambda uid: b"synthetic")
        self.modules = patch.dict(sys.modules, {"orthanc": orthanc,
            "pydicom": types.SimpleNamespace(dcmread=lambda buffer: types.SimpleNamespace(SliceLocation=0)),
            "progress_tracker": types.SimpleNamespace(progress_tracker=tracker),
            "analysis_review_api": types.SimpleNamespace(review_manager=Manager()),
            "patient_specific_calculation": types.SimpleNamespace(main=calculate)})
        self.modules.start()
        self.worker = load("calculation_wrapper")

        class Thread:
            def __init__(self, target, args=(), daemon=False):
                self.target, self.args = target, args
                owner.threads.append(self)
            def start(self): pass
        self.thread_patch = patch.object(self.worker.threading, "Thread", Thread)
        self.thread_patch.start()

    def tearDown(self):
        self.thread_patch.stop()
        self.modules.stop()

    def test_rerun_disables_autosave_and_deletion_without_mutating_request(self):
        params = {"saveResults": True, "deleteAfterCompletion": True, "report_progress": True}
        self.worker.run_cho_calculation_with_progress("orthanc", True, params)
        self.threads[0].target(*self.threads[0].args)
        self.assertFalse(self.params_used["saveResults"])
        self.assertFalse(self.params_used["deleteAfterCompletion"])
        self.assertTrue(params["saveResults"])
        self.assertTrue(params["deleteAfterCompletion"])
        self.assertTrue(self.metadata["review_required"])
        self.assertEqual(self.completed[0], "run-token")

    def test_first_run_preserves_autosave(self):
        self.needs_review = False
        self.worker.run_cho_calculation_with_progress("orthanc", True, {"saveResults": True})
        self.assertTrue(self.threads[0].args[2]["saveResults"])
        self.assertFalse(self.metadata["review_required"])

    def test_old_cleanup_timer_does_not_delete_new_run(self):
        self.worker.run_cho_calculation_with_progress("orthanc", True, {})
        self.threads[0].target(*self.threads[0].args)
        self.current = {"metadata": {"run_id": "new-run"}}
        with patch.object(self.worker.time, "sleep", lambda seconds: None):
            self.threads[1].target()
        self.assertEqual(self.cleanup, [])

    def test_worker_failure_releases_reservation(self):
        self.worker.run_cho_calculation_with_progress("orthanc", True, {})
        with patch.object(self.worker, "unified_cho_main", side_effect=ValueError("synthetic failure")), contextlib.redirect_stdout(io.StringIO()):
            self.threads[0].target(*self.threads[0].args)
        self.assertEqual(self.failed, ["run-token"])


if __name__ == "__main__":
    unittest.main()
