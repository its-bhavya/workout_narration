"""Exercise real MP4/JSON persistence, interrupted recovery and validation."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'/'data_pipeline'))
import concatenate_videos as concat
from generate_v2 import run_v2


def test_resume_archives_orphans_and_audits_manifests(tmp_path, monkeypatch):
    manifests = tmp_path/'manifests'
    manifests.mkdir()
    splits, folds, boundaries = [], [], {}
    for fold in range(5):
        for exercise in ['jump_jack','push_up','squat']:
            for i in range(3):
                clip = f'train_{fold}_{exercise}_{i}.mp4'
                splits.append(dict(clip_id=clip,exercise=exercise,split='train',original_split='train',status='pass',rep_count=1))
                folds.append(dict(clip_id=clip,fold=fold))
    for exercise in ['jump_jack','push_up','squat']:
        for i in range(10):
            splits.append(dict(clip_id=f'test_{exercise}_{i}.mp4',exercise=exercise,split='test',original_split='val',status='pass',rep_count=1))
    for row in splits:
        boundaries[row['clip_id']] = dict(fps=30,total_frames=6,pairs=[[1,4]],rep_count=1)
    pd.DataFrame(splits).to_csv(manifests/'splits_v2.csv',index=False)
    pd.DataFrame(folds).to_csv(manifests/'folds.csv',index=False)
    (manifests/'boundaries.json').write_text(json.dumps(boundaries))
    monkeypatch.setattr(concat,'find_video',lambda *args:tmp_path/'source.mp4')
    def render_fixture(clips,path,orig_map):
        writer = cv2.VideoWriter(str(path),concat.FOURCC,30,(640,360))
        assert writer.isOpened()
        count = sum(c.total_frames for c in clips)
        for i in range(count):
            writer.write(np.full((360,640,3),i,dtype=np.uint8))
        writer.release()
        return count
    monkeypatch.setattr(concat,'write_video',render_fixture)
    args = concat.parse_args([
        '--splits-file',str(manifests/'splits_v2.csv'), '--folds-file',str(manifests/'folds.csv'),
        '--boundaries-file',str(manifests/'boundaries.json'),'--exclusions-file',str(manifests/'absent.csv'),
        '--run-id','test_v2','--n-train-per-fold','5','--n-test','5','--test-cap','1',
        '--out',str(tmp_path/'synthetic'),'--max-videos','1'])
    run_v2(args)
    checkpoint_path = manifests/'batch_checkpoint.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    assert len(checkpoint['completed'])==1 and checkpoint['status']=='running'
    entry = checkpoint['entries'][1]
    orphan = tmp_path/'synthetic'/'train'/(entry['video_id']+'.mp4')
    orphan.write_bytes(b'interrupted video must be preserved')
    args.max_videos = None
    run_v2(args)
    assert any(p.read_bytes()==b'interrupted video must be preserved' for p in (tmp_path/'_interrupted_archive'/'synthetic').glob('*'))
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint['status']=='complete' and len(checkpoint['completed'])==30
    assert pd.read_csv(manifests/'batch_manifest.csv').shape[0]==30
    original_bytes = {p:p.read_bytes() for p in manifests.glob('*')}
    args.validate_only = True
    run_v2(args)
    assert {p:p.read_bytes() for p in original_bytes}==original_bytes
    usage = pd.read_csv(manifests/'clip_usage.csv')
    usage.loc[0,'times_used'] += 1
    usage.to_csv(manifests/'clip_usage.csv',index=False)
    with pytest.raises(AssertionError):
        run_v2(args)
    args.seed = 43
    with pytest.raises(AssertionError,match='Checkpoint input/parameter mismatch'):
        run_v2(args)


def test_decode_failure_does_not_insert_frames(tmp_path,monkeypatch):
    class Capture:
        released = False
        def isOpened(self):return True
        def read(self):return False,None
        def release(self):self.released=True
    capture = Capture()
    monkeypatch.setattr(concat,'find_video',lambda *args:tmp_path/'missing_frame.mp4')
    monkeypatch.setattr(concat.cv2,'VideoCapture',lambda *args:capture)
    clip = concat.ClipInfo('broken.mp4','squat','train',0,2,3,30,[[0,2]],1)
    with pytest.raises(RuntimeError,match='no footage substituted'):
        concat.write_video([clip],tmp_path/'partial.mp4',{'broken.mp4':'train'})
    assert capture.released
