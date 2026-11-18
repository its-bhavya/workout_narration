from src.segmentation.diagnose_oof_classes import duration_confusion, class_metrics, stages


def segment(label,start,end):
    return dict(exercise=label,start_time=start,end_time=end)


def test_temporal_confusion_uses_exact_overlap_and_gt_rows():
    truth=[segment('jump_jack',0,3),segment('transition',3,4)]
    predicted=[segment('jump_jack',0,1.5),segment('transition',1.5,4)]
    matrix=duration_confusion(truth,predicted)
    assert matrix.loc['jump_jack','transition']==1.5
    assert matrix.loc['transition','jump_jack']==0
    assert matrix.to_numpy().sum()==4
    assert class_metrics(matrix)['jump_jack']['recall']==.5


def test_stages_expose_transition_exemption_and_short_exercise_loss():
    values=['transition','jump_jack','jump_jack','transition']
    result=stages(values,[.1,.9,.9,.1],.5,3)
    assert result['confidence']==values
    assert result['minimum_run']==['transition']*4
    assert result['final']==['transition']*4


def test_final_merge_can_bridge_transition_with_a_different_exercise():
    labels=['push_up']*3+['transition']*6+['push_up']*3
    result=stages(labels,[.9]*len(labels),.5,3)
    assert result['minimum_run']==labels
    assert result['final']==['push_up']*len(labels)
