import json

import pytest

from src.rep_counting.evaluate_final import evaluate, metrics


def test_metrics_include_false_positive_and_missing_exercise():
    result = metrics([{"annotated": 0, "predicted": 2},
                      {"annotated": 4, "predicted": 0}])
    assert result["mae"] == 3
    assert result["wape"] == 1.5
    assert result["bias"] == -1
    assert result["exact_match"] == 0


def test_final_evaluation_covers_only_indexed_test(synthetic_bundle, tmp_path):
    a = synthetic_bundle
    target = tmp_path / "report"
    result = evaluate(target, model_path=a["model"], windows_path=a["windows"],
                      features_dir=a["features"], synth_dir=a["synth"])
    assert result["n_videos"] == 1
    assert result["end_to_end_video_exercise"]["overall"]["n"] == 3
    assert result["oracle_gt_segment"]["overall"]["n"] == 3
    saved = json.loads((target / "predictions.json").read_text())
    assert len(saved) == 1 and saved[0]["video_id"].startswith("synth_v2_test_")
    assert saved[0]["ground_truth"] is None
    with pytest.raises(FileExistsError):
        evaluate(target)
