"""Fixed-family ablations on common windows and on all supported windows."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from .data import OUT
from . import SCHEMA
from .checkpoint import checkpoint
from .windows import feature_hash
from playlens_ml.board_compact.data import OUT as PREVIOUS
from playlens_ml.board_compact.features import FEATURE_NAMES
from playlens_ml.board_compact.evaluate import fit,references,BASELINES
from playlens_ml.board_experiment.metrics import evaluate,screen
from playlens_ml.board_experiment.data import source_data
from playlens_ml.development_audit import FOLDS,paired_intervals
from playlens_ml.board_v2.data import digest


def check_data(data,manifest,previous):
    if str(data['schema'])!=SCHEMA or data['feature_names'].tolist()!=FEATURE_NAMES or manifest['featureHash']!=feature_hash():raise ValueError('Schema mismatch')
    if manifest['sourceDatasetSha256']!=source_data()[1]:raise ValueError('Source changed')
    if np.any((data['run_order']<1)|(data['run_order']>30)) or not np.isfinite(data['X']).all():raise ValueError('Invalid development data')
    for key in ('run_order','session_ids','elapsed_seconds','time_to_failure_seconds','y_failure'):
        if not np.array_equal(data[key],previous[key]):raise ValueError('Comparison anchors changed')


def comparison(data,valid,representations,label):
    order=data['run_order'];rows=np.zeros(len(order),bool)
    names=[f'{name}_{family}' for name in representations for family in ('logistic','tree')]
    predictions={name:np.zeros((len(order),7)) for name in [*BASELINES,*names,'always_warning','never_warning']};folds=[];all_eligible=[]
    coverage=[float(valid[order==n].mean()) for n in range(16,31)]
    with threadpool_limits(limits=1):
        for fit_end,cal_end,test_end in FOLDS:
            train=(order<=fit_end)&valid;cal=(order>fit_end)&(order<=cal_end)&valid;test=(order>cal_end)&(order<=test_end)&valid
            for mask,start,end in ((train,1,fit_end),(cal,fit_end+1,cal_end),(test,cal_end+1,test_end)):
                if set(order[mask])!=set(range(start,end+1)):raise ValueError('Missing fold game')
            p=references(data,train,cal,test)
            for name,x in representations.items():
                for family in ('logistic','tree'):p[f'{name}_{family}']=fit(x,data['y_failure'],train,cal,test,family)[0]
            p['always_warning']=np.ones((test.sum(),7));p['never_warning']=np.zeros((test.sum(),7))
            for name,values in p.items():
                if not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid probabilities')
                predictions[name][test]=values
            folds.append({'fit':[1,fit_end],'calibration':[fit_end+1,cal_end],'evaluation':[cal_end+1,test_end],
                          'models':{name:evaluate(values,data,test)['summary'] for name,values in p.items()}})
            rows|=test
            all_test=(order>cal_end)&(order<=test_end)
            all_eligible.append({'evaluation':[cal_end+1,test_end],'unsupportedWindows':int((all_test&~valid).sum()),
                                'baselines':{name:evaluate(values,data,all_test) for name,values in references(data,train,cal,all_test).items()}})
            print(label,'fold',test_end,'done',flush=True)
    results={name:evaluate(p[rows],data,rows) for name,p in predictions.items()}
    for name in names:
        b={k:results[k] for k in BASELINES};f=[(fold['models'][name],{k:fold['models'][k] for k in BASELINES}) for fold in folds]
        results[name]['gates']=screen(results[name],b,f,coverage)
        results[name]['qualifies']=all(results[name]['gates'].values())
        results[name]['comparisons']={k:paired_intervals(results[name]['perRun'],results[k]['perRun']) for k in BASELINES}
    np.savez_compressed(OUT/f'{label}_predictions.npz',run_order=order[rows],elapsed_seconds=data['elapsed_seconds'][rows],
                        remaining_seconds=data['time_to_failure_seconds'][rows],segments=data['segments'][rows],
                        **{name:p[rows] for name,p in predictions.items()})
    return {'models':results,'folds':folds,'allEligibleEvaluation':all_eligible,'scoredWindows':int(rows.sum()),'minimumCoverage':min(coverage)}


def run():
    if not checkpoint()['passed']:raise ValueError('Extraction checkpoint failed')
    with np.load(OUT/'compact_windows.npz') as s:data={k:s[k].copy() for k in s.files}
    with np.load(PREVIOUS/'windows.npz') as s:previous={k:s[k].copy() for k in s.files}
    manifest=json.loads((OUT/'manifest.json').read_text());check_data(data,manifest,previous)
    common=data['valid']&previous['valid']
    # Identical segmentation for every method in the common-window comparison;
    # pair native segment ids so either extractor reset remains an alert boundary.
    common_data={**data,'segments':data['segments']*10000000+previous['segments']}
    a=comparison(common_data,common,{'previous':previous['X'],'pixel_fix_old_motion':data['geometry_motion'],'corrected':data['X']},'common')
    b=comparison(data,data['valid'],{'pixel_fix_old_motion':data['geometry_motion'],'corrected':data['X']},'supported')
    qualifying=sorted([n for n,r in b['models'].items() if r.get('qualifies')],key=lambda n:(b['models'][n]['summary']['brier'],b['models'][n]['summary']['mae']))
    result={'schema':SCHEMA,'featureHash':feature_hash(),'datasetSha256':digest(OUT/'compact_windows.npz'),
            'common':a,'supported':b,'featureComparisons':{name:paired_intervals(v['perRun'],a['models']['previous_logistic']['perRun']) for name,v in a['models'].items() if name.startswith(('pixel_fix','corrected'))},'coverage':manifest['runs'],'selected':qualifying[0] if qualifying else None,
            'decision':'candidate_requires_pipeline_freeze' if qualifying else 'no_candidate_qualifies',
            'limitations':['Development selection only; four new feature/family combinations plus prior model references.',
                           'Motion/clear labels are assistant annotations, not independent ground truth.']}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result['decision'],flush=True)
    return result

if __name__=='__main__':run()
