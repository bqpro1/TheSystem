"""Run a GPU evaluation or reconstruct the saved report offline."""
import argparse
from . import evaluate, reconstruct_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default=None)
parser.add_argument("--output", required=True)
parser.add_argument("--cache", default=None)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--saved", action="store_true", help="Reconstruct bundled evidence without GPU inference")
args = parser.parse_args()
if args.saved:
    if args.smoke:
        parser.error("--saved and --smoke are separate workflows")
    reconstruct_report(args.root, args.output)
else:
    evaluate(args.root, args.output, cache=args.cache, smoke=args.smoke)
