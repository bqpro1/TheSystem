"""Text progress bars visible in VS Code and Colab without widget support."""
from functools import partial
import sys
from tqdm import tqdm as _tqdm

tqdm = partial(_tqdm, file=sys.stdout, mininterval=2.0, dynamic_ncols=True)
