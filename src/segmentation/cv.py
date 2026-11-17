"""Source-clip CV partitions only; this module never fits or scores a model."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'data_pipeline'))
from v2_protocol import catalog, require_v2
from config import MIN_PURITY_TRAIN


def validate_windows(windows, videos=None):
    videos = catalog() if videos is None else videos
    if 'run_id' not in windows or not windows.run_id.eq('v2').all():
        raise ValueError('Windows contain run_id != v2')
    for video_id, rows in windows.groupby('video_id'):
        if video_id not in videos:
            raise ValueError(f'Unknown v2 video: {video_id}')
        meta = require_v2(videos[video_id])
        assert rows.split.eq(meta['split']).all()
        assert rows.fold.eq(meta['fold']).all() if meta['fold'] is not None else rows.fold.isna().all()
    return videos


def get_cv_split(windows, fold, videos=None, min_purity_train=MIN_PURITY_TRAIN):
    if fold not in range(5):
        raise ValueError('fold must be 0..4')
    videos = validate_windows(windows, videos)
    train_pool = windows.loc[windows.split.eq('train')]
    training = train_pool.loc[train_pool.fold.ne(fold)]
    validation = train_pool.loc[train_pool.fold.eq(fold)].copy()
    def clips(rows):
        return {clip for vid in rows.video_id.unique() for clip in videos[vid]['clip_ids_in_order']}
    # Check complete video provenance BEFORE filtering low-purity windows.
    assert clips(training).isdisjoint(clips(validation)), 'Source-clip overlap across CV partitions'
    assert set(training.video_id).isdisjoint(validation.video_id)
    training = training.loc[training.purity.ge(min_purity_train)].copy()
    assert training.split.eq('train').all() and validation.split.eq('train').all()
    return training, validation


def count_windows(windows):
    """Only training eligibility is filtered; test counts remain unchanged."""
    validate_windows(windows)
    tables = []
    for stage, rows in [('before',windows), ('after_train_purity',windows.loc[windows.split.ne('train') | windows.purity.ge(MIN_PURITY_TRAIN)])]:
        counts = rows.groupby(['split','fold','label'],dropna=False).size().rename('windows').reset_index()
        counts.insert(0,'stage',stage)
        tables.append(counts)
        print(f'\n{stage}: min_purity_train={MIN_PURITY_TRAIN}; transition counts included explicitly')
        display=counts.copy()
        display['fold']=display['fold'].fillna(-1).astype(int)
        print(display.pivot_table(index=['split','fold'],columns='label',values='windows',fill_value=0)
              .rename(columns={'transition':'TRANSITION'}).to_string())
        print('Totals per split/class:')
        print(rows.groupby(['split','label']).size().unstack(fill_value=0).to_string())
    return pd.concat(tables,ignore_index=True)
