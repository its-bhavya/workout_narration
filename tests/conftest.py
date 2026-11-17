"""Synthetic, tiny CV fixtures; never fit a model to repository benchmark data."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
for folder in [ROOT/'src'/'segmentation',ROOT/'src'/'data_pipeline']:
    sys.path.insert(0,str(folder))


@pytest.fixture
def synthetic_cv_data(tmp_path,monkeypatch):
    import workflow
    import src.pipeline.extractor as resolver
    from make_windows import make_windows_for_video
    from features import feature_names
    import json
    features=tmp_path/'features';features.mkdir()
    synth=tmp_path/'synthetic';synth.mkdir()
    videos={}; rows=[]
    classes=['squat','push_up','jump_jack','transition']
    for fold in range(6):
        split='train' if fold<5 else 'test'
        vid=f'synth_v2_{split}_fixture_{fold}'
        labels=np.repeat(classes,90)
        frames=np.arange(0,720,2);times=frames/30
        frame=pd.DataFrame(dict(frame=frames,time_s=times,seg_label=labels,run_id='v2',
                                 video_id=vid,split=split,fold=fold if fold<5 else np.nan))
        knee=np.repeat([60.,140.,110.,175.],90)+np.sin(times*4)
        elbow=np.repeat([150.,60.,130.,175.],90)+np.cos(times*4)
        for name,values in [('knee_best',knee),('elbow_best',elbow),('knee_L',knee),('knee_R',knee),
                            ('elbow_L',elbow),('elbow_R',elbow),('arm_raise_L',knee),('arm_raise_R',knee),
                            ('leg_spread',elbow)]:frame[name]=values
        for joint in ['knee','elbow']:
            for side in ['L','R']:frame[f'{joint}_{side}_vis']=1.
        frame.loc[4,'knee_best']=np.nan
        frame.to_csv(features/f'{vid}_angles.csv',index=False)
        meta=dict(run_id='v2',video_id=vid,split=split,fold=fold if fold<5 else None,
                  fps=30,n_frames=720,clip_ids_in_order=[f'{split}_{fold}_{e}.mp4' for e in classes[:3]],
                  segments=[dict(type='transition' if e=='transition' else 'exercise',label=e,
                                 start_frame=i*180,end_frame=(i+1)*180,rep_count_annotated=0)
                            for i,e in enumerate(classes)])
        videos[vid]=meta
        directory=synth/split;directory.mkdir(exist_ok=True)
        (directory/f'{vid}.json').write_text(json.dumps(meta))
        windows=make_windows_for_video(frame,vid,split)
        for row in windows:row.update(run_id='v2',fold=meta['fold'])
        rows.extend(windows)
    windows=tmp_path/'windows.csv';pd.DataFrame(rows).to_csv(windows,index=False)
    monkeypatch.setattr(workflow,'catalog',lambda *args:videos)
    # The resolver's guard uses the shared v2_protocol.catalog when handed files.
    import v2_protocol
    monkeypatch.setattr(v2_protocol,'catalog',lambda *args:videos)
    import src.data_pipeline.v2_protocol as package_protocol
    monkeypatch.setattr(package_protocol,'catalog',lambda *args:videos)
    return dict(features=features,synth=synth,videos=videos,windows=windows,out=tmp_path/'outputs')


@pytest.fixture
def synthetic_bundle(synthetic_cv_data):
    import workflow
    data=synthetic_cv_data
    selection=workflow.run_selection(data['windows'],data['features'],data['out'],['lr'],[.8],
                                     [.5],[1],42,'fixture',data['videos'])
    model=workflow.refit_selection(selection,data['windows'],data['features'],True,data['out'],data['videos'])
    return data | dict(selection=selection,model=model)
