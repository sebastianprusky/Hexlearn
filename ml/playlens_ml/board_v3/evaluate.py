from __future__ import annotations
import hashlib
import json
import numpy as np
import joblib
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from playlens_api.model_types import ConstantProbabilityModel, stable_linear_scores
from playlens_ml.common import HORIZONS_SECONDS, monotonic_probabilities
from playlens_ml.development_audit import FOLDS, paired_intervals
from playlens_ml.train_baseline import average_duration_probabilities
from . import SCHEMA
from playlens_ml.board_experiment import ALERT_SCHEMA
from .features import fingerprint as extractor_hash
from .data import FEATURE_NAMES
from playlens_ml.board_experiment.features import GEOMETRY_NAMES, BASE_NAMES
from .data import OUT as DEFAULT_OUT
from playlens_ml.board_experiment.data import source_data
from .checkpoint import checkpoint
from playlens_ml.board_experiment.metrics import evaluate, screen

BASELINES=('average_duration','empirical_duration','elapsed_logistic','elapsed_tree')
GROUPS={'geometry':len(GEOMETRY_NAMES),'colors':len(BASE_NAMES),'temporal':len(FEATURE_NAMES)}
CANDIDATES=tuple(f'{g}_{family}' for g in GROUPS for family in ('logistic','tree'))


def load_data(out):
    with np.load(out/'windows.npz') as source: data={k:source[k].copy() for k in source.files}
    manifest=json.loads((out/'manifest.json').read_text())
    if str(data['schema'])!=SCHEMA or data['feature_names'].tolist()!=FEATURE_NAMES or manifest['extractorHash']!=extractor_hash():
        raise ValueError('Incompatible feature schema/extractor')
    if manifest['sourceDatasetSha256']!=source_data()[1]: raise ValueError('Source dataset changed')
    if np.any(data['run_order']>30) or not np.isfinite(data['X']).all(): raise ValueError('Future runs or nonfinite features')
    return data,manifest


def raw_scores(model,x):
    if isinstance(model,ConstantProbabilityModel):
        p=np.clip(model.probability,1e-6,1-1e-6); return np.full(len(x),np.log(p/(1-p)))
    if hasattr(model,'steps'): return stable_linear_scores(model,x)
    return model.decision_function(x)


def fit(x,y,train,cal,test,family):
    probabilities=[]; fitted=[]
    for h in range(7):
        if len(np.unique(y[train,h]))<2: model=ConstantProbabilityModel(float(y[train,h].mean()))
        elif family=='tree':
            model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=40,
                l2_regularization=5,learning_rate=.05,early_stopping=False,random_state=42).fit(x[train],y[train,h])
        else:
            model=make_pipeline(VarianceThreshold(1e-10),StandardScaler(),
                LogisticRegression(C=.1,max_iter=2000,solver='liblinear')).fit(x[train],y[train,h])
        z=raw_scores(model,x[cal]); target=y[cal,h]
        if len(np.unique(target))<2:
            calibrator=ConstantProbabilityModel(float(target.mean())); p=np.full(test.sum(),calibrator.probability)
        else:
            calibrator=LogisticRegression(C=1,solver='liblinear',max_iter=1000).fit(z[:,None],target)
            p=expit(raw_scores(model,x[test])*calibrator.coef_[0,0]+calibrator.intercept_[0])
        probabilities.append(p); fitted.append((model,calibrator))
    return monotonic_probabilities(np.column_stack(probabilities)),fitted


def references(data,train,cal,test):
    elapsed=data['elapsed_seconds']; rem=data['time_to_failure_seconds']; ids=data['session_ids']; y=data['y_failure']
    duration=np.array([np.median((elapsed+rem)[(ids==sid)&train]) for sid in np.unique(ids[train])])
    t=elapsed[test,None]; d=duration[None,:]; denom=(d>t).sum(axis=1)+1
    result={'average_duration':average_duration_probabilities(elapsed,rem,train,ids)[test],
        'empirical_duration':np.column_stack([(((d>t)&(d<=t+h)).sum(axis=1)+.5)/denom for h in HORIZONS_SECONDS])}
    for family in ('logistic','tree'):
        result['elapsed_'+family]=fit(elapsed[:,None],y,train,cal,test,family)[0]
    return result


def run(out=DEFAULT_OUT):
    gate=checkpoint(out)
    if not gate['passed']: raise ValueError('Extraction checkpoint failed: training is blocked')
    data,manifest=load_data(out); order=data['run_order']; valid=data['valid']
    coverage=[r['coverage'] for r in manifest['runs'] if r['run']>=16]
    rows=np.zeros(len(order),dtype=bool); all_predictions={name:np.zeros((len(order),7)) for name in (*BASELINES,*CANDIDATES,'always_warning','never_warning')}
    folds=[]; unsupported=[]
    with threadpool_limits(limits=1):
        for train_end,cal_end,test_end in FOLDS:
            train=(order<=train_end)&valid; cal=(order>train_end)&(order<=cal_end)&valid
            test=(order>cal_end)&(order<=test_end)&valid
            for mask,start,end in ((train,1,train_end),(cal,train_end+1,cal_end),(test,cal_end+1,test_end)):
                if set(order[mask])!=set(range(start,end+1)): raise ValueError('A game has no valid windows; refusing to shift folds')
            predictions=references(data,train,cal,test); models={}
            for group,width in GROUPS.items():
                for family in ('logistic','tree'):
                    key=f'{group}_{family}'; p,_=fit(data['X'][:,:width],data['y_failure'],train,cal,test,family)
                    predictions[key]=p
            predictions['always_warning']=np.ones((test.sum(),7)); predictions['never_warning']=np.zeros((test.sum(),7))
            for key,p in predictions.items():
                all_predictions[key][test]=p; models[key]=evaluate(p,data,test)['summary']
            rows|=test; folds.append({'fit':[1,train_end],'calibration':[train_end+1,cal_end],'evaluation':[cal_end+1,test_end],'models':models})
            # References on every eligible window, including those the extractor cannot support.
            all_test=(order>cal_end)&(order<=test_end)
            baseline_all=references(data,train,cal,all_test)
            unsupported.append({'evaluation':[cal_end+1,test_end],'allEligibleBaselines':{k:evaluate(p,data,all_test) for k,p in baseline_all.items()},
                                'unsupportedWindows':int(np.sum(all_test&~valid))})
    results={name:evaluate(p[rows],data,rows) for name,p in all_predictions.items()}
    for name in CANDIDATES:
        result=results[name]; baseline={b:results[b] for b in BASELINES}
        fold_results=[(f['models'][name],{b:f['models'][b] for b in BASELINES}) for f in folds]
        result['gates']=screen(result,baseline,fold_results,coverage)
        result['qualifies']=all(result['gates'].values())
        result['comparisons']={b:paired_intervals(result['perRun'],results[b]['perRun']) for b in BASELINES}
    qualifying=sorted([name for name in CANDIDATES if results[name]['qualifies']],key=lambda n:(results[n]['summary']['brier'],results[n]['summary']['mae']))
    report={'schema':SCHEMA,'alertSchema':ALERT_SCHEMA,'models':results,'folds':folds,'coverage':[{k:r[k] for k in ('run','eligibleWindows','validWindows','coverage','exclusions')} for r in manifest['runs']],
            'allEligibleEvaluation':unsupported,'selected':qualifying[0] if qualifying else None,
            'decision':'proceed_to_new_prospective_test' if qualifying else 'no_candidate_qualifies',
            'limitations':['15 development evaluation games; overlapping training histories. Bootstrap intervals are descriptive.',
                          'These development results are selection evidence, not independent validation.'],
            'datasetSha256':hashlib.sha256((out/'windows.npz').read_bytes()).hexdigest(),'extractorHash':extractor_hash()}
    if qualifying:
        name=qualifying[0]; group,family=name.rsplit('_',1); width=GROUPS[group]
        train=(order<=24)&valid; cal=(order>24)&(order<=30)&valid
        _,fitted=fit(data['X'][:,:width],data['y_failure'],train,cal,cal,family)
        artifact={'schema':SCHEMA,'feature_names':FEATURE_NAMES[:width],'extractorHash':extractor_hash(),'models':fitted,
                  'fitRuns':[1,24],'calibrationRuns':[25,30],'eligible_for_live':False,'purpose':'frozen prospective candidate',
                  'inputContract':{'kind':'canvas pixels','decodedWidth':960,'sampleHz':4,'historySeconds':8,'videoDelaySeconds':.5},
                  'temporalMethod':'sorted distributions; estimates, not tracked physical blocks'}
        joblib.dump(artifact,out/'frozen_candidate.joblib')
        report['frozenCandidateSha256']=hashlib.sha256((out/'frozen_candidate.joblib').read_bytes()).hexdigest()
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['decision'],flush=True)
    return report

if __name__=="__main__":run()
