from src.rep_counting.classify_jumping_jack_windows import classify_gt_window, classify_predicted_window


def seg(label,start,end,index=0):
    return dict(label=label,start=start,end=end,index=index,clip_id=f'clip{index}')


def test_clipping_fragmentation_and_label_loss_are_distinct_flags():
    predicted=[seg('transition',0,2),seg('jump_jack',2,5),seg('squat',5,8),seg('jump_jack',8,9),seg('transition',9,10)]
    row=classify_gt_window(0,10,predicted)
    assert row['clipped_start'] and row['clipped_end'] and row['fragmented']
    assert row['jj_coverage_fraction']==.4
    assert not row['wholly_missed']


def test_wholly_mislabeled_gt_does_not_invent_a_matched_boundary():
    row=classify_gt_window(0,10,[seg('squat',0,10)])
    assert row['wholly_missed'] and row['dominant_predicted_label']=='squat'
    assert row['first_correct_label_s'] is None
    assert not row['clipped_start'] and not row['fragmented']


def test_merged_and_overlong_can_coexist_without_being_mostly_wrong():
    truth=[seg('jump_jack',1,5,1),seg('transition',5,6,2),seg('jump_jack',6,10,3)]
    row=classify_predicted_window(0,10,truth)
    assert row['merged_gt_windows'] and row['overlong_start'] and row['overlong_end']
    assert row['matched_gt_index']==1
    assert row['start_delta_s']==-1 and row['end_delta_s']==5
    assert row['jj_purity_fraction']==.8 and not row['mostly_wrong_label']


def test_false_predicted_class_keeps_its_actual_dominant_gt_label():
    row=classify_predicted_window(0,10,[seg('push_up',0,10)])
    assert row['dominant_gt_label']=='push_up' and row['mostly_wrong_label']
    assert row['gt_start_s'] is None
    assert row['off_class_duration_s']==10
