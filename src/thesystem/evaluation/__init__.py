"""Evaluate the selected checkpoints, original Qwen and fixed published references."""
from thesystem.reproduction import _paths, _run


def evaluate(root, output, cache=None, smoke=False):
    """Run five benchmarks, internal/external routing and generation preservation.

    A full run uses 1,648 decisions, 1,648 paired writing requests, the 4,000-case
    internal routing set, and 45 unique generation checks for each release model.
    Original Qwen shares the frozen probe forward pass. The two release model
    arms load sequentially, so only one backbone occupies GPU memory at a time.
    ``smoke=True`` is a small mechanics check and cannot reproduce benchmark scores.
    Existing, matching output directories resume saved shards after interruption.
    """
    root, output, cache = _paths(root, output, cache)
    for arm in ("probe", "lora"):
        _run("thesystem.evaluation.evaluation_worker",
             ["--arm", arm, "--output", output / arm, "--cache", cache,
              *(["--smoke"] if smoke else [])], root, output / f"{arm}.log")
    _run("thesystem.evaluation.build_report", [output], root, output / "report.log")
    return output


def reconstruct_report(root, output):
    """Rebuild the bundled measured results from per-case evidence, without a GPU."""
    root, output, _ = _paths(root, output, None)
    _run("thesystem.evaluation.saved_results", [output], root, output / "reconstruction.log")
    return output
