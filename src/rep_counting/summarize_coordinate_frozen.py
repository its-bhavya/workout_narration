"""Derive transparent count-only tables/plots from the single frozen run."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import sha256
from .evaluate_coordinate_frozen import paired_summary


def summarize(output):
    output=Path(output)
    target=output/'supplement.json'
    if target.exists(): raise FileExistsError('Preserve existing derived analysis')
    report=json.loads((output/'report.json').read_text())
    train=pd.read_csv(output/'training_window_signed_errors.csv')
    gt=pd.read_csv(output/'gt_pushup_window_signed_errors.csv')
    videos=pd.read_csv(output/'video_exercise_comparison.csv')
    # The raw replay tables retain earlier-run auxiliary metadata; export
    # count-only tables so old timestamps/hits cannot be mistaken for new ones.
    for name in ('video_exercise_comparison','all_gt_window_comparison'):
        rows=pd.read_csv(output/(name+'.csv'))
        columns=['video_id','exercise','annotated','old_predicted','predicted','error']
        if 'segment_index' in rows: columns.insert(1,'segment_index')
        rows[columns].to_csv(output/(name+'_counts_only.csv'),index=False)
    def transitions(rows):
        old=rows.old_predicted-rows.annotated
        new=rows.predicted-rows.annotated
        def label(v): return np.where(v<0,'under',np.where(v>0,'over','exact'))
        return pd.crosstab(label(old),label(new)).to_dict(), dict(
            old_under=int((old<0).sum()),old_exact=int((old==0).sum()),old_over=int((old>0).sum()),
            new_under=int((new<0).sum()),new_exact=int((new==0).sum()),new_over=int((new>0).sum()),
            newly_overcounted=int(((old<=0)&(new>0)).sum()))
    train_cross,train_signed=transitions(train)
    gt_cross,gt_signed=transitions(gt)
    active=videos[videos.exercise.eq('push_up') & videos.annotated.gt(0)]
    result=dict(training_signed=train_signed,test_gt_signed=gt_signed,
                training_transition_matrix=train_cross,test_gt_transition_matrix=gt_cross,
                active_pushup_videos=paired_summary(active),
                provenance={'report_sha256':sha256(output/'report.json'),'source_sha256':sha256(__file__)},
                raw_replay_auxiliary_column_note='In raw video_exercise_comparison/all_gt_window_comparison, method/rep_times/hits/missed/extra/flag retain earlier-run metadata. Use counts_only exports for count comparisons and dedicated push-up sidecars/window files for frozen timelines. No frozen timing F1 is claimed here.')
    target.write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for ax,rows,title in [(axes[0],train,'Training windows (in-sample)'),(axes[1],gt,'Exploratory test GT windows')]:
        x=rows.old_predicted-rows.annotated; y=rows.predicted-rows.annotated
        ax.scatter(x,y,alpha=.65)
        lo=min(x.min(),y.min())-2; hi=max(x.max(),y.max())+2
        ax.plot([lo,hi],[lo,hi],color='grey',linestyle='--')
        ax.axhline(0,color='grey',linewidth=.5); ax.axvline(0,color='grey',linewidth=.5)
        ax.set(title=title,xlabel='Previous signed error (reps)',ylabel='Frozen signed error (reps)')
    for row in gt[gt.clip_id.isin(['stu9_52.mp4','stu4_46.mp4','stu10_44.mp4','val901.mp4'])].to_dict('records'):
        axes[1].annotate(row['clip_id'],(row['old_predicted']-row['annotated'],row['error']),xytext=(4,5),textcoords='offset points',fontsize=8)
    entries=[('All video/exercise',report['end_to_end']),('Push-ups, predicted',report['end_to_end_by_exercise']['push_up']),('Push-ups, GT',report['gt_by_exercise']['push_up'])]
    for i,(name,value) in enumerate(entries):
        low,high=value['mae_change_ci95']; center=value['mae_change']
        axes[2].errorbar(center,i,xerr=[[center-low],[high-center]],fmt='o',capsize=4)
    axes[2].axvline(0,color='grey',linestyle='--')
    axes[2].set(yticks=range(3),yticklabels=[n for n,_ in entries],xlabel='MAE change: frozen minus previous (95% CI)',title='Paired video-cluster bootstrap')
    fig.savefig(output/'signed_errors_and_confidence.png',dpi=160); plt.close(fig)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(summarize(args.run_dir),indent=2))
