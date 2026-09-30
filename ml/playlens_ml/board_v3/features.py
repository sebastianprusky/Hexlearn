"""Reviewed native extractor with an occlusion-resistant geometry fallback."""
import hashlib
from pathlib import Path
from playlens_ml.board_v2 import features as primary, fallback


def needs_repair(result):
    return not result['valid'] or max(result['heights'])>.84


def extract(image):
    result=primary.extract(image)
    return fallback.extract(image) if needs_repair(result) else result


def fingerprint():
    return hashlib.sha256(b''.join(Path(p).read_bytes() for p in (__file__,primary.__file__,fallback.__file__))).hexdigest()
