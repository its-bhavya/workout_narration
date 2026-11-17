"""Shared fail-closed provenance checks for the active source-clip protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'v2'
SYNTH_DIR = ROOT/'data'/'synthetic'
FEATURES_DIR = ROOT/'data'/'features'/'v2'
MANIFESTS = ROOT/'data'/'manifests'
INDEX = MANIFESTS/'synthetic_index.csv'
SPLITS = MANIFESTS/'splits_v2.csv'
FOLDS = MANIFESTS/'folds.csv'
OUTPUT_DIR = ROOT/'outputs'/'segmentation_v2'
SUMMARY = MANIFESTS/'pose_extraction_summary_v2.csv'


def active_path(path):
    path = Path(path).resolve()
    if '_archive_v1' in path.parts:
        raise ValueError(f'Archived v1 artifacts cannot be loaded: {path}')
    return path


def require_v2(meta):
    if meta.get('run_id') != RUN_ID:
        raise ValueError(f"Expected run_id=v2, found {meta.get('run_id')!r}: {meta.get('video_id', '')}")
    return meta


def load_json(path):
    return require_v2(json.loads(active_path(path).read_text(encoding='utf-8')))


def catalog(synth_dir=SYNTH_DIR, index_path=INDEX, splits_path=SPLITS, folds_path=FOLDS):
    synth_dir = active_path(synth_dir)
    index = pd.read_csv(active_path(index_path))
    splits = pd.read_csv(active_path(splits_path))
    folds = pd.read_csv(active_path(folds_path))
    if 'run_id' not in index or not index.run_id.eq(RUN_ID).all():
        raise ValueError('Index contains videos whose run_id != v2')
    assert index.video_id.is_unique and splits.clip_id.is_unique and folds.clip_id.is_unique
    assert set(splits.split) == {'train','test'} and set(folds.fold) == set(range(5))
    assert (splits.split == splits.original_split.map({'train':'train','val':'test','test':'test'})).all()
    assert set(folds.clip_id) == set(splits.loc[splits.split=='train','clip_id'])
    source = splits.set_index('clip_id').to_dict('index')
    assignment = folds.set_index('clip_id').fold.to_dict()
    videos = {}
    for row in index.to_dict('records'):
        path = synth_dir/row['split']/(row['video_id']+'.json')
        meta = load_json(path)
        assert meta['video_id']==row['video_id'] and meta['split']==row['split']
        ids = meta['clip_ids_in_order']
        assert ids == row['source_clips'].split('|')
        assert all(c in source and source[c]['split']==meta['split'] for c in ids)
        if meta['split']=='train':
            assert meta['fold']==row['fold'] and all(assignment[c]==meta['fold'] for c in ids)
        else:
            assert meta['fold'] is None and pd.isna(row['fold'])
        assert path.with_suffix('.mp4').is_file()
        videos[meta['video_id']] = meta
    found = {p.stem for p in synth_dir.rglob('*.json')}
    assert found == set(videos), 'Unindexed or missing synthetic JSONs'
    assert {p.stem for p in synth_dir.rglob('*.mp4')} == set(videos), 'Unindexed or missing synthetic MP4s'
    return videos


def feature_frame(path, meta=None):
    path = active_path(path)
    frame = pd.read_csv(path)
    if 'run_id' not in frame or not frame.run_id.eq(RUN_ID).all():
        raise ValueError(f'Feature run_id != v2: {path}')
    if meta is None:
        video_id = path.stem.removesuffix('_angles')
        meta = catalog()[video_id]
    require_v2(meta)
    assert frame.video_id.eq(meta['video_id']).all()
    assert frame.split.eq(meta['split']).all()
    assert frame.fold.eq(meta['fold']).all() if meta['fold'] is not None else frame.fold.isna().all()
    return frame


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retired_workflow():
    raise RuntimeError('The separate-val training/evaluation workflow is retired. Use cv.get_cv_split for source-clip CV; model training/evaluation is not authorized in Phase 4.')
