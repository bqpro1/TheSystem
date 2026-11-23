# Reproducing TheSystem

## Exact selections

| Component | Selection |
|---|---|
| Backbone | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; frozen, BF16 |
| Probe-256 | One hidden layer, 4096 → 256 → GELU → 18, dropout 0; seed 42, validation-selected epoch 6; no LoRA |
| LoRA | Selected epoch 2; rank 16, alpha 32; 124 projections in layers 17–32 |
| Shared `<FAST>` | Diverse-42, seed 42, selected epoch 9; RMS-normalized last prompt state after layer 16; threshold 0.5 |

Model files and datasets are downloaded from immutable Hugging Face revisions and
checked against `RELEASE-MANIFEST.json`. No private research checkout or cached
features are required. The tokenizer and chat wrapper are pinned; thinking is off.

## Setup and access

Use Python **3.12 or 3.13**. In Colab, select the latest runtime and a GPU
with native BF16 support. Check the GPU and free memory reported by the
notebook. The BF16 backbone weights alone use about 18 GB, and working memory
depends on the request. The demos report available memory and impose no fixed
minimum. Their setup checks do not establish that a full model will fit.
They run in BF16 without automatic quantization, CPU offload or input truncation.
Notebook 03 can run on an NVIDIA L4 with 22 GiB of visible GPU memory using its
pinned CUDA 13 environment.

Open a notebook from the README's Colab links. The setup helpers, wheels, weights
and datasets are public downloads. You can also download a notebook from GitHub
and open it using **File → Upload notebook** in Colab.

### Demo setup (01–02)

Start with a fresh runtime and run **Prepare runtime**. The cell reuses Colab’s
PyTorch 2.14.x and NumPy, installs the binary inference dependencies and verifies
the pinned helper and `0.1.0rc3` wheel by SHA-256. Progress appears every 20
seconds and installer output is saved to `thesystem-prepare-runtime.log`. If
prompted, restart the session and rerun the cell. Load the model only after
**READY** appears.

The demos use `load_model(..., execution="inference")` with the released
weights and Transformers 5.17.0. This path uses built-in PyTorch operations.
Kernel differences can affect speed and introduce small numerical changes.
Use notebook 03 for the reference evaluation.

### Evaluation setup (03)

Run its five steps: **Prepare runtime → Choose run and save location → Download
resources → Run evaluation → Download results and finish**. The first cell downloads a hash-verified helper and
`0.1.0rc4` evaluation wheel, then runs the reference
`python -m thesystem._vendor.lora.setup_runtime` installer. A source checkout
first needs `python -m pip install -e .`. The installer sets up the pinned CUDA
build and accelerated kernels; the GPU extra alone does not build the extension.
If setup asks for a restart, choose **Runtime → Restart session**. After Colab reconnects, click **Run all**.
The helper reports four setup stages and elapsed-time heartbeats.
Keep this reference environment separate from the lighter demo environment.

## Run order

Start with either demo notebook. The evaluation notebook loads original Qwen,
LoRA and Probe-256 in separate processes, using one backbone at a time. Its small
smoke check verifies mechanics; the full run produces the reportable measurements.
New outputs are stored outside downloaded inputs, with logs, resumable journals,
identity checks and heartbeat files. Reuse an output directory only for the same
configuration. A changed identity stops instead of silently combining runs. The
updated evaluation defaults to `release-evaluation-rc4`; do not reuse an older
wheel's run folder. `SMOKE=True` uses a separate `-smoke` folder and is not a full
benchmark result. Memory failures stop and preserve completed examples; they do
not shorten inputs, quantize weights, or lower generation limits.

## Training data and recipe

Training notebooks are not included in this release. Exact training and validation
data remain available in [TheSystem-Data](https://huggingface.co/datasets/Lbuk/TheSystem-Data),
with their original order, splits and hashes. Training code and configurations are retained.

Both answer models use the same 18,000 training and 2,000 validation rows, in the
same order. Source proportions are 4,000 each MASSIVE, BoolQ, MultiNLI and WinoGrande,
plus 2,000 HelpSteer2 in training. Test scores do not select checkpoints.

Probe training uses seed 42, batch 256, AdamW learning rate 0.001, weight decay 0.01,
at most 30 epochs and patience 5. Frozen train/validation features are recomputed;
the original private feature cache is unnecessary. The released gate remains fixed.

LoRA training uses seed 42, micro-batch 2, accumulation 8, AdamW learning rate 0.0002,
weight decay 0.01, 5% warm-up and three epochs. The gate recipe trains the original
gate, then continues it on the exact diverse routing mixture. The starting gate
checkpoint is included for reproducing that continuation independently; it is not
an additional released inference model.

The recorded reference stack uses PyTorch 2.14.0+cu130, Transformers 5.17.0,
FLA 0.5.2 and causal-conv1d 1.7.0. The reference setup and runtime checks apply to training.
Original GPU runs used an RTX PRO 6000 Blackwell, 96 GB. Platform and kernel
variations can change retrained tensors or scores. New training outputs remain
separate from the selected release checkpoints.

## Evaluation results and storage

A full GPU run makes new predictions. A smoke run checks the evaluation path on
a small sample. Reconstructing a report from saved predictions recomputes the
tables without running the models. See the [evaluation guide](EVALUATION.md).

With `USE_DRIVE=False`, notebook 03 saves to local Colab storage. Set it to
`True` to retain progress in Drive. The inline table summarizes the five
benchmarks. The downloaded ZIP contains the detailed results and supports
inspection of partial progress after a failed run. Check that the ZIP is saved
before confirming **Shut down runtime**. Deleting the runtime erases local files.

If Colab displays a blank output instead of the download button, run this in a new cell:

```python
import shutil
from google.colab import files

files.download(shutil.make_archive(str(OUTPUT), "zip", OUTPUT))
```

Open the downloaded ZIP and check for `SUMMARY.json` and `REPORT.html` after a
completed run. Then choose **Runtime → Disconnect and delete runtime**.
