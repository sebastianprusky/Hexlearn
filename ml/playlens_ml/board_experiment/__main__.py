import argparse
import json
from pathlib import Path
from .data import DEFAULT_OUT, build
from .review import prepare, checkpoint
from .robustness import check
from .evaluate import run
from .report import generate


def main():
    parser=argparse.ArgumentParser(description='Offline pixel-only board experiment; never promotes a live model.')
    parser.add_argument('command',choices=('review','extract','robustness','checkpoint','evaluate','report'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    # CLI output is confined to experimental artifacts, never production dataset/models.
    if not args.out.resolve().is_relative_to(DEFAULT_OUT.parent.resolve()):
        parser.error('--out must be inside artifacts/experiments')
    action={'review':prepare,'extract':build,'robustness':check,'checkpoint':checkpoint,'evaluate':run,'report':generate}[args.command]
    try:
        result=action(args.out)
    except (ValueError,FileNotFoundError) as exc:
        parser.exit(2,f'{exc}\n')
    if args.command=='checkpoint': print(json.dumps(result,indent=2))
    elif args.command=='report': print(result)

if __name__=='__main__': main()
