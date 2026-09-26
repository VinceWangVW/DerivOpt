import json

import pytest

from derivopt import acceptance


def test_suite_refuses_existing_nonempty_output(tmp_path):
    (tmp_path / "user-file").write_text("keep")
    with pytest.raises(FileExistsError):
        acceptance.run_suite(tmp_path)
    assert (tmp_path / "user-file").read_text() == "keep"


def test_suite_refuses_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        acceptance.run_suite(tmp_path, "not-a-suite")


def test_failed_execution_is_reported_not_treated_as_pass(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate execution failure")
    monkeypatch.setattr(acceptance, "_training_check", fail)
    monkeypatch.setattr(acceptance, "prepare", lambda *args, **kwargs: None)
    with pytest.raises(acceptance.AcceptanceFailure, match="13 acceptance checks failed"):
        acceptance.run_suite(tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["run_type"] == "smoke"
    assert report["status"] == "failed" and report["passed"] == 0 and report["failed"] == 13
    assert all("deliberate execution failure" in row["error"] for row in report["checks"])
