"""Separately authorized exploratory subset; original imminent-loss-v1 is closed."""
from playlens_ml.common import ROOT
SCHEMA = 'imminent-loss-27-v1'
OUT = ROOT / 'artifacts/experiments' / SCHEMA
EXCLUDED = (5, 14, 17)
RUNS = tuple(n for n in range(1, 31) if n not in EXCLUDED)
FOLDS = ((9, 15, 20), (14, 20, 25), (19, 25, 30))
