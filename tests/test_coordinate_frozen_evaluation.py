import numpy as np
import pandas as pd
import pytest

from src.rep_counting.evaluate_coordinate_frozen import paired_summary, sensitivity, evaluate
from src.rep_counting.coordinate_failure_table import missing_fraction


def test_paired_bootstrap_preserves_video_clusters_and_sign():
    rows=pd.DataFrame(dict(video_id=['a','a','b'],annotated=[5,5,5],old_predicted=[7,7,7],predicted=[6,6,6]))
    result=paired_summary(rows)
    assert result['n_video_clusters']==2
    assert result['mae_change']==-1
    np.testing.assert_allclose(result['mae_change_ci95'],[-1,-1])
    assert result['frozen']['within_one_fraction']==1
    assert result['old']['within_one_fraction']==0


def test_sensitivity_removes_same_source_reuses_from_both_methods():
    rows=pd.DataFrame(dict(video_id=['a','b','c','d'],clip_id=['bad1','bad1','bad2','keep'],annotated=[5]*4,old_predicted=[0,0,0,6],predicted=[4,4,4,5]))
    result=sensitivity(rows,['bad1','bad2'])
    assert result['all']['old']['n']==4
    assert result['without_top_two']['old']['n']==1
    assert result['without_top_two']['frozen']['n']==1
    assert result['without_top_two']['frozen']['mae']==0


def test_evaluation_refuses_overwriting_or_canonical_output(tmp_path):
    with pytest.raises(ValueError,match='new directory under outputs'):
        evaluate(tmp_path/'canonical')


def test_missing_observations_require_finite_signal_and_confidence():
    frame=pd.DataFrame(dict(signal=[1.,np.nan,2.,3.],signal_confidence=[.5,1.,.49,np.nan]))
    assert missing_fraction(frame,'signal')==.75
