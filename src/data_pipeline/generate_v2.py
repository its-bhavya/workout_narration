"""Render a frozen v2 plan, resume safely, and audit the complete run.

The checkpoint contains the complete ordered clip plan and input fingerprints.
Completed MP4/JSON pairs are recovered into manifests after interruption.
Partial files are archived, never deleted. Legacy manifests are never imported.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd

from plan_folds import EXERCISES
from plan_targets import draw_lengths, plan_lengths

log = logging.getLogger(__name__)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def archive_partial(path, root):
    if path.exists():
        # Keep interrupted media outside the active synthetic tree so recursive
        # downstream feature discovery cannot mistake it for accepted videos.
        destination = root.parent / "_interrupted_archive" / root.name / f"{time.time_ns()}_{path.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        log.warning("Archived incomplete artifact to %s", destination)


def check_video(path, expected_frames, full=False):
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened(), f"Unreadable video: {path}"
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames, path
        assert abs(cap.get(cv2.CAP_PROP_FPS) - 30) < .001, path
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (640, 360), path
        if full:
            decoded = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                decoded += 1
            assert decoded == expected_frames, f"Decode count mismatch: {path}"
    finally:
        cap.release()


def expected_metadata(entry, clip_map):
    from concatenate_videos import build_raw_segments, merge_transitions, seg_to_dict
    clips = [clip_map[c] for c in entry['clips']]
    raw, nframes = build_raw_segments(clips, "transition")
    segments = [seg_to_dict(s) for s in merge_transitions(raw)]
    joins = []
    offset = 0
    for c in clips[:-1]:
        offset += round(c.total_frames*30/c.src_fps)
        joins.append(offset)
    return nframes, segments, joins


def verify_json(meta, entry, checkpoint, clip_map, fold_map):
    assert meta['video_id'] == entry['video_id']
    assert meta['run_id'] == checkpoint['config']['run_id']
    assert meta['plan_hash'] == checkpoint['plan_hash']
    assert meta['split'] == entry['split'] and meta['fold'] == entry['fold']
    assert meta['clip_ids_in_order'] == entry['clips']
    assert len(set(entry['clips'])) == len(entry['clips'])
    clips = [clip_map[c] for c in entry['clips']]
    assert 3 <= len(clips) <= 5 and len({c.exercise for c in clips}) >= 2
    assert all(a.exercise != b.exercise for a, b in zip(clips, clips[1:]))
    assert {c.split for c in clips} == {entry['split']}
    if entry['split'] == 'train':
        assert {fold_map[c.clip_id] for c in clips} == {entry['fold']}
    nframes, segments, joins = expected_metadata(entry, clip_map)
    assert (meta['fps'], meta['width'], meta['height']) == (30, 640, 360)
    assert meta['n_frames'] == nframes
    assert meta['segments'] == segments
    assert segments[0]['start_frame'] == 0 and segments[-1]['end_frame'] == nframes
    assert all(s['end_frame'] > s['start_frame'] for s in segments)
    assert all(a['end_frame'] == b['start_frame'] for a, b in zip(segments, segments[1:]))
    assert meta['clip_join_frames'] == joins
    assert len(joins) == len(clips)-1 and all(0 < j < nframes for j in joins)
    assert joins == sorted(set(joins))


def write_csv(path, rows, columns):
    temporary = path.with_suffix('.csv.tmp')
    pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
    temporary.replace(path)


def persist_manifests(manifests, records, checkpoint, clip_map, fold_map):
    indexes, batches = [], []
    usage = Counter()
    for meta in records:
        segments = meta['segments']
        joins = meta['clip_join_frames']
        direct = sum(any(s['type'] == 'exercise' and s['start_frame'] <= j < s['end_frame']
                         for s in segments) for j in joins)
        transitions = sum(s['end_frame']-s['start_frame'] for s in segments if s['type'] == 'transition')
        ids = meta['clip_ids_in_order']
        usage.update(ids)
        indexes.append(dict(video_id=meta['video_id'], split=meta['split'],
                            n_segments=len(segments), duration_s=round(meta['n_frames']/30, 3),
                            exercises='|'.join(sorted({clip_map[c].exercise for c in ids})),
                            source_clips='|'.join(ids), transition_frame_share=transitions/meta['n_frames'],
                            direct_cuts=direct, seed=meta['seed'], batch=1,
                            fold=meta['fold'], run_id=meta['run_id']))
        batches.append({k: meta[k] for k in ['video_id', 'split', 'seed', 'fold', 'run_id']} | {'batch': 1})
    usages = [dict(clip_id=c, split=clip_map[c].split, exercise=clip_map[c].exercise,
                   times_used=n, batch=1, fold=fold_map.get(c), run_id=checkpoint['config']['run_id'])
              for c, n in sorted(usage.items())]
    write_csv(manifests/'synthetic_index.csv', indexes,
              ['video_id','split','n_segments','duration_s','exercises','source_clips',
               'transition_frame_share','direct_cuts','seed','batch','fold','run_id'])
    write_csv(manifests/'clip_usage.csv', usages,
              ['clip_id','split','exercise','times_used','batch','fold','run_id'])
    write_csv(manifests/'batch_manifest.csv', batches,
              ['video_id','split','batch','seed','fold','run_id'])


def audit(records, checkpoint, clip_map, fold_map, manifests, complete):
    groups = defaultdict(list)
    split_sets = defaultdict(set)
    usage = Counter()
    for meta in records:
        groups[(meta['split'], meta['fold'])].append(meta)
        split_sets[meta['split']].update(meta['clip_ids_in_order'])
        usage.update(meta['clip_ids_in_order'])
    assert split_sets['train'].isdisjoint(split_sets['test'])
    for c, n in usage.items():
        assert n <= (5 if clip_map[c].split == 'train' else checkpoint['config']['test_cap'])
    saved = pd.read_csv(manifests/'clip_usage.csv')
    assert dict(zip(saved.clip_id, saved.times_used)) == dict(usage)
    assert saved.clip_id.is_unique
    for row in saved.to_dict('records'):
        assert row['run_id'] == checkpoint['config']['run_id']
        assert row['split'] == clip_map[row['clip_id']].split
        assert row['exercise'] == clip_map[row['clip_id']].exercise
        if row['split'] == 'train':
            assert row['fold'] == fold_map[row['clip_id']]
        else:
            assert pd.isna(row['fold'])
    index = pd.read_csv(manifests/'synthetic_index.csv')
    batch = pd.read_csv(manifests/'batch_manifest.csv')
    assert index.video_id.is_unique and batch.video_id.is_unique
    assert set(index.video_id) == set(batch.video_id) == {m['video_id'] for m in records}
    assert set(index.run_id) == set(batch.run_id) == {checkpoint['config']['run_id']}
    metas_by_id = {m['video_id']:m for m in records}
    for table in [index,batch]:
        for row in table.to_dict('records'):
            meta = metas_by_id[row['video_id']]
            assert row['split'] == meta['split']
            assert row['fold'] == meta['fold'] if meta['fold'] is not None else pd.isna(row['fold'])
            if 'source_clips' in row:
                assert row['source_clips'].split('|') == meta['clip_ids_in_order']
    print('AUDIT ' + ('COMPLETE' if complete else 'PARTIAL (remaining plan not rendered)'))
    for (split, fold), metas in sorted(groups.items(), key=lambda item: str(item[0])):
        counts = Counter(c for m in metas for c in m['clip_ids_in_order'])
        slots = Counter(clip_map[c].exercise for m in metas for c in m['clip_ids_in_order'])
        all_clips = [c for c in clip_map.values() if c.split == split and (split == 'test' or fold_map[c.clip_id] == fold)]
        distribution = Counter(counts[c.clip_id] for c in all_clips)
        frames = sum(m['n_frames'] for m in metas)
        transitions = sum(s['end_frame']-s['start_frame'] for m in metas for s in m['segments'] if s['type']=='transition')
        joins = sum(len(m['clip_join_frames']) for m in metas)
        cuts = sum(any(s['type']=='exercise' and s['start_frame'] <= j < s['end_frame'] for s in m['segments'])
                   for m in metas for j in m['clip_join_frames'])
        if complete:
            mean = sum(slots.values())/3
            assert all(.9*mean-1e-9 <= slots[e] <= 1.1*mean+1e-9 for e in EXERCISES)
            expected = checkpoint['config']['n_train_per_fold'] if split == 'train' else checkpoint['config']['n_test']
            assert len(metas) == expected
        for exercise in EXERCISES:
            values = [counts[c.clip_id] for c in all_clips if c.exercise==exercise]
            assert max(values)-min(values) <= 1
        print(f"{split} fold={fold}: videos={len(metas)}, minutes={frames/1800:.4f}, "
              f"transition_share={100*transitions/frames:.4f}%, direct_cuts={cuts}, joins={joins}, "
              f"exercise_slots={dict(slots)}, reuse_distribution={dict(sorted(distribution.items()))}, distinct_clips={len(counts)}")
    if complete:
        assert len(records) == len(checkpoint['entries'])
        for split in ['train','test']:
            expected_ids = {m['video_id'] for m in records if m['split']==split}
            directory = Path(checkpoint['config']['out'])/split
            assert {p.stem for p in directory.glob('*.mp4')} == expected_ids
            assert {p.stem for p in directory.glob('*.json')} == expected_ids
        print('PASS: caps; split isolation; train fold isolation; test/train disjointness; contiguous exclusive-end '
              'positive-length segments; exact clip joins; JSON/usage counts; balance; reuse spread; video frame counts/fps/dimensions.')


def run_v2(args):
    from concatenate_videos import load_exclusions, load_pool, find_video, make_video
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id), 'Invalid run ID'
    assert args.max_videos is None or args.max_videos > 0
    manifests = Path(args.splits_file).resolve().parent
    out = Path(args.out).resolve()
    checkpoint_path = manifests/'batch_checkpoint.json'
    inputs = [Path(args.splits_file), Path(args.folds_file), Path(args.boundaries_file)]
    exclusions = Path(args.exclusions_file)
    config = dict(run_id=args.run_id, seed=args.seed, n_train_per_fold=args.n_train_per_fold,
                  n_test=args.n_test, test_cap=args.test_cap, out=str(out),
                  input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  exclusions_hash=hashlib.sha256(exclusions.read_bytes()).hexdigest() if exclusions.exists() else None,
                  planner_version=1)
    splits = pd.read_csv(args.splits_file)
    folds = pd.read_csv(args.folds_file)
    assert splits.clip_id.is_unique and folds.clip_id.is_unique
    assert set(splits.split) == {'train','test'} and set(folds.fold) == set(range(5))
    assert set(folds.clip_id) == set(splits.loc[splits.split=='train','clip_id'])
    assert (splits.split == splits.original_split.map({'train':'train','val':'test','test':'test'})).all()
    fold_map = dict(zip(folds.clip_id, folds.fold))
    boundaries = json.loads(Path(args.boundaries_file).read_text())
    pool = load_pool(Path(args.splits_file), boundaries, load_exclusions(exclusions))
    clip_map = {c.clip_id:c for clips in pool.values() for c in clips}
    orig_map = dict(zip(splits.clip_id, splits.original_split))
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        assert checkpoint['config'] == config, 'Checkpoint input/parameter mismatch; archive before a new run'
        assert hashlib.sha256(json.dumps(checkpoint['entries'], sort_keys=True).encode()).hexdigest() == checkpoint['plan_hash']
    else:
        assert not args.validate_only, 'No v2 checkpoint to validate'
        assert not list(out.rglob('*.mp4')) and not list(out.rglob('*.json')), 'Archive old synthetic files first'
        assert not any((manifests/name).exists() for name in ['synthetic_index.csv','clip_usage.csv','batch_manifest.csv']), 'Archive old manifests first'
        entries = []
        for split, fold, clips, n, cap in [('train', f, [c for c in pool['train'] if fold_map[c.clip_id]==f], args.n_train_per_fold, 5) for f in range(5)] + [('test',None,pool['test'],args.n_test,args.test_cap)]:
            plan = plan_lengths(clips, cap, draw_lengths(args.seed,n), args.seed)
            for i, seq in enumerate(plan['sequences'],1):
                suffix = f'train_f{fold}' if split=='train' else 'test'
                entries.append(dict(video_id=f'synth_{args.run_id}_{suffix}_{i:04d}',split=split,fold=fold, clips=[c.clip_id for c in seq]))
        checkpoint = dict(config=config, entries=entries, plan_hash=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest(), completed=[], status='running')
        # All selected sources must exist before committing any render work.
        missing = [c for c in {c for e in entries for c in e['clips']} if find_video(c,orig_map[c]) is None]
        assert not missing, f'Missing source clips: {missing}'
        atomic_json(checkpoint_path, checkpoint)
    records = []
    rendered = 0
    start = time.perf_counter()
    for entry in checkpoint['entries']:
        destination = out/entry['split']
        video = destination/(entry['video_id']+'.mp4')
        metadata = destination/(entry['video_id']+'.json')
        if video.exists() and metadata.exists():
            meta = json.loads(metadata.read_text())
            verify_json(meta, entry, checkpoint, clip_map, fold_map)
            check_video(video, meta['n_frames'], full=args.validate_only)
        else:
            if args.validate_only:
                raise AssertionError(f'Missing rendered video/JSON: {entry["video_id"]}')
            if args.max_videos is not None and rendered >= args.max_videos:
                break
            archive_partial(video,out)
            archive_partial(metadata,out)
            staging = out/'_staging'/entry['video_id']
            archive_partial(staging,out)
            make_video(entry['video_id'], [clip_map[c] for c in entry['clips']], entry['split'], staging,
                       'transition',args.splits_file,args.seed,orig_map,run_id=args.run_id,fold=entry['fold'],plan_hash=checkpoint['plan_hash'])
            meta = json.loads((staging/metadata.name).read_text())
            verify_json(meta,entry,checkpoint,clip_map,fold_map)
            check_video(staging/video.name,meta['n_frames'],full=True)
            destination.mkdir(parents=True,exist_ok=True)
            (staging/video.name).rename(video)
            (staging/metadata.name).rename(metadata)
            rendered += 1
        records.append(meta)
        if not args.validate_only:
            persist_manifests(manifests,records,checkpoint,clip_map,fold_map)
            checkpoint['completed'] = [m['video_id'] for m in records]
            checkpoint['status'] = 'complete' if len(records)==len(checkpoint['entries']) else 'running'
            atomic_json(checkpoint_path,checkpoint)
        log.info('v2 progress %d/%d (%s), elapsed %.1fs',len(records),len(checkpoint['entries']),entry['video_id'],time.perf_counter()-start)
    audit(records,checkpoint,clip_map,fold_map,manifests,len(records)==len(checkpoint['entries']))
