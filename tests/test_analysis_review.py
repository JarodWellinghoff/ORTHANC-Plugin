import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("analysis_review", Path(__file__).parents[1] / "src/python/analysis_review.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ReviewManager, ReviewError = module.ReviewManager, module.ReviewError


class Repository:
    def __init__(self, record):
        self.record, self.saved, self.error = copy.deepcopy(record), [], False

    def load(self, uid):
        if self.error:
            raise RuntimeError("database unavailable")
        return copy.deepcopy(self.record)

    def overwrite(self, uid, baseline, payload):
        if self.error:
            raise RuntimeError("save failed")
        if module.fingerprint(self.record) != module.fingerprint(baseline):
            raise ReviewError("Saved result changed")
        self.saved.append(copy.deepcopy(payload))
        self.record = copy.deepcopy(payload["results"])


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {"ctdivol_avg": 5, "patient_id": "synthetic", "protocol_name": "Test"}
        self.repo = Repository(self.baseline)
        self.manager = ReviewManager(self.repo)
        self.payload = {"patient": {}, "study": {}, "scanner": {}, "ct": {},
                        "series": {"series_instance_uid": "1.2.3"}, "results": {"ctdivol_avg": 8}}

    def ready(self):
        token, review = self.manager.begin("orthanc-id", "1.2.3")
        self.assertTrue(review)
        self.manager.complete(token, self.payload)
        return token

    def test_preview_never_saves(self):
        token = self.ready()
        self.assertEqual(self.repo.record, self.baseline)
        self.assertEqual(self.repo.saved, [])
        self.assertEqual(self.manager.detail(token)["candidate"]["results"]["ctdivol_avg"], 8)

    def test_explicit_overwrite(self):
        self.manager.decide(self.ready(), "overwrite")
        self.assertEqual(self.repo.record["ctdivol_avg"], 8)
        self.assertEqual(len(self.repo.saved), 1)
        self.assertEqual(self.manager.pending(), [])

    def test_discard_keeps_baseline(self):
        self.manager.decide(self.ready(), "discard")
        self.assertEqual(self.repo.record, self.baseline)
        self.assertEqual(self.repo.saved, [])

    def test_failed_save_retains_draft_for_retry(self):
        token = self.ready()
        self.repo.error = True
        with self.assertRaises(RuntimeError):
            self.manager.decide(token, "overwrite")
        self.assertEqual(self.manager.detail(token)["state"], "ready")
        self.assertEqual(self.repo.record, self.baseline)
        self.repo.error = False
        self.manager.decide(token, "overwrite")
        self.assertEqual(len(self.repo.saved), 1)

    def test_stale_overwrite_rejected_but_discard_allowed(self):
        token = self.ready()
        self.repo.record["ctdivol_avg"] = 10
        with self.assertRaises(ReviewError):
            self.manager.decide(token, "overwrite")
        self.manager.decide(token, "discard")
        self.assertEqual(self.repo.record["ctdivol_avg"], 10)

    def test_duplicate_runs_blocked_until_decision(self):
        token = self.ready()
        with self.assertRaises(ReviewError):
            self.manager.begin("other-orthanc-id", "1.2.3")
        self.manager.decide(token, "discard")
        self.assertTrue(self.manager.begin("orthanc-id", "1.2.3")[1])

    def test_duplicate_decisions_rejected(self):
        token = self.ready()
        self.manager.decide(token, "overwrite")
        with self.assertRaises(ReviewError):
            self.manager.decide(token, "overwrite")
        self.assertEqual(len(self.repo.saved), 1)

    def test_invalid_decision_retains_draft(self):
        token = self.ready()
        with self.assertRaises(ReviewError):
            self.manager.decide(token, "invalid")
        self.assertEqual(len(self.manager.pending()), 1)

    def test_first_run_and_metadata_placeholder_do_not_prompt(self):
        for record in (None, {"processing_time": 0, "ctdivol_avg": None}):
            manager = ReviewManager(Repository(record))
            token, needs_review = manager.begin("new", "1.2.3")
            self.assertFalse(needs_review)
            self.assertFalse(manager.complete(token, self.payload))
            self.assertEqual(manager.pending(), [])

    def test_database_failure_not_mistaken_for_first_run(self):
        self.repo.error = True
        with self.assertRaises(RuntimeError):
            self.manager.begin("orthanc-id", "1.2.3")
        self.repo.error = False
        self.assertTrue(self.manager.begin("orthanc-id", "1.2.3")[1])

    def test_calculation_failure_releases_reservation(self):
        token, _ = self.manager.begin("orthanc-id", "1.2.3")
        self.manager.fail(token)
        self.assertTrue(self.manager.begin("orthanc-id", "1.2.3")[1])
        self.assertEqual(self.repo.record, self.baseline)

    def test_json_safe_defensive_copy(self):
        self.payload["results"]["nps"] = [float("nan"), float("inf"), 0]
        token = self.ready()
        detail = self.manager.detail(token)
        self.assertEqual(detail["candidate"]["results"]["nps"], [None, None, 0])
        detail["candidate"]["results"]["ctdivol_avg"] = 999
        self.assertEqual(self.manager.detail(token)["candidate"]["results"]["ctdivol_avg"], 8)

    def test_preview_survives_refresh(self):
        token = self.ready()
        for _ in range(3):
            self.assertEqual(self.manager.pending()[0]["id"], token)
        self.assertEqual(self.repo.saved, [])

    def test_wrong_series_empty_results_and_incomplete_metadata_rejected(self):
        token, _ = self.manager.begin("orthanc-id", "1.2.3")
        with self.assertRaises(ReviewError):
            self.manager.complete(token, {})
        payload = copy.deepcopy(self.payload)
        del payload["patient"]
        with self.assertRaises(ReviewError):
            self.manager.complete(token, payload)
        self.payload["series"]["series_instance_uid"] = "wrong"
        with self.assertRaises(ReviewError):
            self.manager.complete(token, self.payload)
        self.assertEqual(self.repo.record, self.baseline)

    def test_zero_dose_is_an_existing_measurement(self):
        self.assertTrue(module.has_analysis({"ctdivol_avg": 0}))


if __name__ == "__main__":
    unittest.main()
