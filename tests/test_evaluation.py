import pytest
from src.evaluation.segmentation_metrics import (
    frame_accuracy, edit_score, f1_at_k, boundary_error, evaluate_segmentation)
from src.evaluation.rep_metrics import rep_count_metrics

GT = [("transition", 0, 30), ("squat", 30, 130), ("transition", 130, 160),
      ("push_up", 160, 260), ("transition", 260, 300)]


def test_perfect():
    r = evaluate_segmentation(GT, GT, fps=30)
    assert r["frame_acc"] == 1 and r["edit"] == 1
    assert r["f1@10"] == r["f1@25"] == r["f1@50"] == 1
    assert r["boundary_mean_abs_s"] == 0


def test_shifted_one_second():
    pred = [("transition", 0, 60), ("squat", 60, 160), ("transition", 160, 190),
            ("push_up", 190, 290), ("transition", 290, 300)]
    assert boundary_error(GT, pred, fps=30)["mean_abs_s"] == pytest.approx(1.0)
    assert 0.5 < frame_accuracy(GT, pred) < 1
    assert edit_score(GT, pred) == 1                 # same label order
    # exercise segments still overlap (IoU 0.54) but two short transition segments drop below 0.5
    assert f1_at_k(GT, pred, 50) == pytest.approx(0.6)
    assert f1_at_k(GT, pred, 10) > f1_at_k(GT, pred, 50)


def test_wrong_label_and_oversegmentation():
    wrong = [("transition", 0, 30), ("squat", 30, 130), ("transition", 130, 160),
             ("squat", 160, 260), ("transition", 260, 300)]
    assert f1_at_k(GT, wrong, 50) < 1
    split = [("transition", 0, 30), ("squat", 30, 80), ("transition", 80, 85), ("squat", 85, 130),
             ("transition", 130, 160), ("push_up", 160, 260), ("transition", 260, 300)]
    assert edit_score(GT, split) < 1                 # extra segments cost edit score
    assert frame_accuracy(GT, split) > 0.9


def test_missing_segment():
    pred = [("transition", 0, 160), ("push_up", 160, 260), ("transition", 260, 300)]
    assert f1_at_k(GT, pred, 10) < 1
    assert edit_score(GT, pred) < 1
    b = boundary_error(GT, pred, fps=30)
    assert b["n_missed"] == 2 and b["n_gt"] == 4      # squat start and end boundaries not found


def test_rep_metrics():
    m = rep_count_metrics({"a": 10, "b": 9, "c": 14}, {"a": 10, "b": 10, "c": 12})
    assert m["exact_match"] == pytest.approx(1 / 3)
    assert m["off_by_one"] == pytest.approx(2 / 3)
    assert m["mae"] == pytest.approx(1.0)
