# Run the evaluation

Open [notebook 03](../notebooks/03-Reproduce-Evaluation.ipynb) in a GPU runtime and run its cells in order. Leave `SMOKE = False` to evaluate all five benchmarks, or set it to `True` for a quick setup check. See the [setup and access instructions](REPRODUCIBILITY.md#setup-and-private-access).

## Decision accuracy

The full run scores 1,648 multiple-choice decisions per model across VitaminC, PAWS, SQuAD2, Aegis2 and PubMedQA. Automatic accuracy counts a decision as correct only when the gate selects decision mode and the model selects the correct answer. The notebook labels the published Nimble and Jev comparison scores separately.

## Internal held-out-context test

The routing test uses 4,000 prompts from 500 held-out contexts. Each context has four requests to choose an answer and four to generate text. In the confusion matrix, rows give the requested mode and columns give the mode the gate selected. The generation checks compare each model’s output tokens with those from the original Qwen model on the same writing prompt.

## Save your results

Notebook 03 generates `REPORT.html` for inspecting individual cases, `SUMMARY.json` for programmatic use, and `comparison.csv` for the benchmark table. Download the results ZIP before releasing a local Colab runtime, or select persistent Drive storage when setting up the run.

To reconstruct the reference report from the saved Hugging Face predictions on a CPU, install the package with its `audit` extra and run:

```python
from thesystem.artifacts import resolve_release_resources
from thesystem.evaluation import reconstruct_report

reconstruct_report(resolve_release_resources(), "outputs/reference-audit")
```
