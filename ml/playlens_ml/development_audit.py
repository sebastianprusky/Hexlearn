"""Offline, development-only model comparison. Never writes live model artifacts.

Run from playlens with PYTHONPATH=service:ml .venv/bin/python -m
playlens_ml.development_audit. Candidates and chronological folds are fixed below;
results are exploratory selection evidence and require a new prospective test.
"""
from __future__ import annotations
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from playlens_api.features import aggregate_feature_names
from playlens_ml.common import DATASET_DIR, ROOT, HORIZONS_SECONDS, monotonic_probabilities, remaining_from_probabilities
from playlens_ml.train_baseline import fit_calibrated, bootstrap_indices, average_duration_probabilities

OUT = ROOT / 'artifacts/experiments/development-audit-2026-09-30'
FOLDS = ((9, 15, 20), (14, 20, 25), (19, 25, 30))
CANDIDATES = ('average_duration', 'empirical_duration', 'elapsed_logistic', 'elapsed_tree',
              'existing_visual_ensemble', 'visual_logistic', 'visual_logistic_calibrated',
              'board_current_logistic', 'board_tree', 'hybrid_tree_diagnostic')


def development_data(path):
    with np.load(path) as source:
        mask = source['run_order'] <= 30
        data = {key: source[key][mask].copy() for key in
                ('X', 'sequences', 'y_failure', 'session_ids', 'run_order', 'elapsed_seconds', 'time_to_failure_seconds')}
    if not np.isfinite(data['X']).all() or not np.isfinite(data['sequences']).all():
        raise ValueError('Non-finite visual features')
    return data


def fit_probabilities(x, y, train, calibration, test, tree=False, calibrated=False):
    results = []
    for h in range(y.shape[1]):
        if len(np.unique(y[train,h])) < 2:
            results.append(np.full(test.sum(), y[train,h].mean())); continue
        if tree:
            model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7,
                       min_samples_leaf=40, l2_regularization=5, learning_rate=.05,
                       early_stopping=False, random_state=42)
        else:
            model = make_pipeline(VarianceThreshold(1e-10), StandardScaler(),
                        LogisticRegression(C=.1, max_iter=2000, solver='liblinear'))
        model.fit(x[train], y[train,h])
        p = model.predict_proba(x[test])[:,1]
        if calibrated and len(np.unique(y[calibration,h])) > 1:
            z = model.decision_function(x[calibration]).reshape(-1,1)
            calibrator = LogisticRegression(C=1, solver='liblinear').fit(z,y[calibration,h])
            p = calibrator.predict_proba(model.decision_function(x[test]).reshape(-1,1))[:,1]
        results.append(p)
    return monotonic_probabilities(np.column_stack(results))


def per_run_metrics(probabilities, data, mask):
    p = monotonic_probabilities(probabilities)
    y = data['y_failure'][mask]; remaining = data['time_to_failure_seconds'][mask]
    runs = data['run_order'][mask]; elapsed = data['elapsed_seconds'][mask]
    predicted = remaining_from_probabilities(p)
    out=[]
    for run in np.unique(runs):
        m=runs==run; last=m & (remaining<=60); final=m & (remaining<=10)
        early=m & (remaining>60); alarm=m & (p[:,-1]>=.5)
        indices=np.flatnonzero(m)[np.argsort(elapsed[m])]
        first=next((i for i in indices if alarm[i]),None)
        out.append(dict(run=int(run), brier=float(np.mean((p[m]-y[m])**2)),
          mae=float(np.mean(np.abs(predicted[last]-remaining[last]))),
          final10Mae=float(np.mean(np.abs(predicted[final]-remaining[final]))),
          falseWarningFraction=float(np.mean(alarm[early])) if early.any() else 0.,
          firstWarningLead=float(remaining[first]) if first is not None else None,
          firstWarningUseful=bool(first is not None and 15<=remaining[first]<=60)))
    return out


def summarize(rows):
    return {key:float(np.mean([r[key] for r in rows])) for key in
            ('brier','mae','final10Mae','falseWarningFraction','firstWarningUseful')}


def paired_intervals(rows, baseline):
    rng=np.random.default_rng(20260930); n=len(rows)
    indices=rng.integers(0,n,size=(5000,n)); out={}
    for metric in ('brier','mae','final10Mae'):
        a=np.array([r[metric] for r in rows]); b=np.array([r[metric] for r in baseline])
        delta=np.mean(b[indices]-a[indices],axis=1)
        out[metric]={'improvement':float(b.mean()-a.mean()),
          'interval95':np.quantile(delta,[.025,.975]).tolist(),
          'runsWon':int(np.sum(a<b)),'runs':n}
    return out


def main():
    path=DATASET_DIR/'personal_windows.npz'; data=development_data(path)
    OUT.mkdir(parents=True,exist_ok=True)
    order=data['run_order']; y=data['y_failure']; x=data['X'].astype(float)
    names=aggregate_feature_names()
    # Exclude global image/score/UI summaries; use board geometry and its recent change.
    board_columns=[i for i,name in enumerate(names) if i%27>=8]
    board=np.column_stack((x[:,board_columns], data['sequences'][:,-1,8:]))
    elapsed=data['elapsed_seconds'].astype(float)
    results={name:[] for name in CANDIDATES}; folds=[]; caught=[]
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        for train_end, cal_end, test_end in FOLDS:
            train=order<=train_end; cal=(order>train_end)&(order<=cal_end)
            test=(order>cal_end)&(order<=test_end)
            fold={'train': [1,train_end], 'calibration':[train_end+1,cal_end], 'evaluation':[cal_end+1,test_end], 'models':{}}
            duration=np.array([np.median((elapsed+data['time_to_failure_seconds'])[order==r]) for r in range(1,train_end+1)])
            for name in CANDIDATES:
                if name=='average_duration':
                    p=average_duration_probabilities(elapsed,data['time_to_failure_seconds'],train,data['session_ids'])[test]
                elif name=='empirical_duration':
                    t=elapsed[test,None]; d=duration[None,:]
                    # Conditional duration distribution given survival to current elapsed time; half-count smoothing.
                    denom=(d>t).sum(axis=1)+1.
                    p=np.column_stack([(((d>t)&(d<=t+h)).sum(axis=1)+.5)/denom for h in HORIZONS_SECONDS])
                elif name=='existing_visual_ensemble':
                    members=[]
                    for member in range(5):
                        idx=bootstrap_indices(data['session_ids'],train,41+member)
                        members.append(np.column_stack([fit_calibrated(x,y[:,h],idx,cal).predict_proba(x[test])[:,1] for h in range(7)]))
                    p=np.median([monotonic_probabilities(m) for m in members],axis=0)
                else:
                    inputs=elapsed[:,None] if name.startswith('elapsed_') else (board if name.startswith('board') else x)
                    if name=='hybrid_tree_diagnostic': inputs=np.column_stack((board,elapsed))
                    if name=='board_current_logistic': inputs=data['sequences'][:,-1,8:]
                    p=fit_probabilities(inputs,y,train,cal,test,tree='tree' in name,calibrated=name.endswith('_calibrated'))
                if not np.isfinite(p).all(): raise ValueError(f'{name}: non-finite predictions')
                rows=per_run_metrics(p,data,test); results[name].extend(rows)
                fold['models'][name]=summarize(rows)
                print(f'fold {cal_end+1}-{test_end} {name}: {fold["models"][name]}',flush=True)
            folds.append(fold)
        caught=[str(w.message) for w in seen]
    report={'protocol':'Exploratory development-only chronological evaluation. Runs 31-40 excluded from all candidate fitting and scoring. No live promotion.',
      'datasetSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'folds':folds,'warningCounts':{m:caught.count(m) for m in sorted(set(caught))},
      'models':{name:{'summary':summarize(rows),'perRun':rows,'vsAverageDuration':paired_intervals(rows,results['average_duration']), 'vsElapsedTree':paired_intervals(rows,results['elapsed_tree'])} for name,rows in results.items()},
      'limitations':['15 evaluation games, overlapping training folds; run-bootstrap intervals are descriptive, not independent confirmation.',
        'Candidate selection uses these development results; a frozen revision needs fresh future runs.',
        'Hybrid diagnostic includes elapsed time and is outside the current visual-only production contract.',
        'Existing PyTorch model cannot be evaluated fairly here because it trained on some evaluation games.']}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',OUT/'results.json',flush=True)

if __name__=='__main__': main()
