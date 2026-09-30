"""Postmortem of the frozen original candidate; does not train or promote models.

This intentionally inspects the original exposed test games to explain failures.
It is separate from development_audit, whose experiments exclude runs 31-40.
"""
import hashlib
import json
import warnings
import joblib
import numpy as np
from scipy.special import expit
from playlens_api.features import aggregate_feature_names
from playlens_api.model_types import stable_linear_scores
from playlens_ml.common import DATASET_DIR, MODEL_DIR, monotonic_probabilities
from playlens_ml.development_audit import OUT


def main():
    model_path = MODEL_DIR / 'baseline_v3_candidate.joblib'
    artifact = joblib.load(model_path)
    data = np.load(DATASET_DIR / 'personal_windows.npz')
    x = data['X'].astype(float); runs = data['run_order']; labels = data['y_failure']
    members = []; discrepancy = 0.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for models in artifact['ensemble_models']:
            values = []
            for model in models:
                p = model.predict_proba(x)[:, 1]
                if hasattr(model, 'calibrator'):
                    z = stable_linear_scores(model.base_model, x)
                    c = model.calibrator
                    scalar = expit(z * c.coef_[0,0] + c.intercept_[0])
                    discrepancy = max(discrepancy, float(np.max(np.abs(p-scalar))))
                values.append(p)
            members.append(monotonic_probabilities(np.column_stack(values)))
    probabilities = monotonic_probabilities(np.median(members, axis=0))
    train = x[runs<=24]; mu = train.mean(0); sd = train.std(0)
    names = aggregate_feature_names(); rows = []
    for run in np.unique(runs):
        mask = runs==run
        # Only nonconstant features get a standardized shift.
        z = np.abs((x[mask].mean(0)-mu)/np.maximum(sd,1e-12))
        z[sd<1e-10]=0
        indices = np.argsort(z)[-5:][::-1]
        rows.append({'run':int(run),'brier':float(np.mean((probabilities[mask]-labels[mask])**2)),
            'meanProbabilities':probabilities[mask].mean(0).tolist(),
            'largestStandardizedMeanShifts':{names[i]:float(z[i]) for i in indices}})
    result = {'purpose':'Original candidate postmortem, not new test evidence or model selection',
        'modelSha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'finitePredictions':bool(np.isfinite(probabilities).all()),
        'maximumDifferenceFromScalarSigmoid':discrepancy,
        'warningCounts':{msg:sum(str(w.message)==msg for w in caught) for msg in sorted({str(w.message) for w in caught})},
        'perRun':rows}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'original_candidate_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',OUT/'original_candidate_audit.json')

if __name__ == '__main__': main()
