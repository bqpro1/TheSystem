"""Command-line entry point for exact training recipes."""
import argparse
from . import train_lora, train_probe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=("train-probe", "train-lora"))
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--root", default=None, help="Verified release resources; default downloads pinned Hub snapshots")
parser.add_argument("--output", required=True)
parser.add_argument("--cache", default=None)
args = parser.parse_args()
(train_probe if args.action == "train-probe" else train_lora)(args.root, args.output, cache=args.cache, smoke=args.smoke)
