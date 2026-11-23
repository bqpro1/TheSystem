"""Export a standalone report; published reference results stay explicitly separate."""
import argparse
import csv
import html
import json
from pathlib import Path

from .evaluation_data import ROOT, SUBSETS, COUNTS, load_suites
from .evaluation_io import digest, read_json, read_stage, sha, write_json
from .evaluation_metrics import decision_metrics, group_interval, routing_metrics, writing_metrics


def collect(output):
    decisions, routing, writing, all_rows = {}, {}, {}, {}
    identities = []
    for arm in ("probe", "lora"):
        folder = output / arm
        identity = read_json(folder / "identity.json")
        identities.append(identity)
        if read_json(folder / "COMPLETE.json") != dict(complete=True, identity=digest(identity)):
            raise ValueError(f"Incomplete model evaluation: {arm}")
        if identity["package_sha256"] != sha(ROOT / "RELEASE-MANIFEST.json"):
            raise ValueError("Report package differs from evaluated package")
        for name, expected in zip(SUBSETS, ([2] * len(SUBSETS) if identity.get("smoke") else COUNTS)):
            rows = read_stage(folder / "decisions" / name)
            if len(rows) != expected:
                raise ValueError("Wrong benchmark size")
            all_rows[arm, name] = rows
            decisions.setdefault(name, {})
            for model in rows[0]["scores"]:
                decisions[name][model] = decision_metrics(rows, model)
                if model != "qwen":
                    decisions[name][model + "_automatic"] = decision_metrics(rows, model, automatic=True)
        expected_suites = load_suites(smoke=identity.get("smoke", False))
        for stage in ("paired_writing", "routing_historical", "writing"):
            if [row["id"] for row in read_stage(folder / stage)] != [row["id"] for row in expected_suites[stage]]:
                raise ValueError(f"Report inputs differ from the frozen suite: {arm}/{stage}")
        routing[arm] = {name: routing_metrics(read_stage(folder / name))
                        for name in ("paired_writing", "routing_historical")}
        decision_rows = [dict(r, target=1) for name in SUBSETS for r in all_rows[arm, name]]
        routing[arm]["external_decisions"] = routing_metrics(decision_rows)
        writing[arm] = writing_metrics(read_stage(folder / "writing"))
    if any(identities[0][key] != identities[1][key] for key in ("batch_size", "packages", "generation_caps", "smoke", "source_sha256")):
        raise ValueError("The two arms used different evaluation settings")
    paired = {}
    for name in SUBSETS:
        left, right = all_rows["lora", name], all_rows["probe", name]
        signature = lambda rows: [(r["id"], r["target"], r["input_sha256"], r["option_keys"]) for r in rows]
        if signature(left) != signature(right):
            raise ValueError("Model comparison has different inputs, labels or row order")
        def correct(row, model):
            scores = row["scores"][model]
            return int(max(range(len(scores)), key=scores.__getitem__) == row["target"])
        deltas = [correct(a, "lora") - correct(b, "qwen") for a, b in zip(left, right)]
        probe_deltas = [correct(r, "probe256") - correct(r, "qwen") for r in right]
        paired[name] = {"lora_vs_qwen": dict(difference=sum(deltas)/len(deltas), ci95=group_interval(left, deltas)),
                        "probe_vs_qwen": dict(difference=sum(probe_deltas)/len(probe_deltas), ci95=group_interval(right, probe_deltas))}
    return dict(complete=True, smoke=identities[0].get("smoke", False), decisions=decisions, routing=routing, writing=writing, paired=paired,
                origin=identities[0].get("origin", "gpu_evaluation"), inference_performed=identities[0].get("inference_performed", True),
                published=read_json(ROOT / "provenance/published-results.json"),
                limitations=["Published Nimble/Jev scores were not rerun; native model prompt wrappers differ.",
                  "No paired significance test against published aggregates; no latency comparison.",
                  "Intervals resample source families, retaining related questions from the same context together.",
                  "Exact token equality is generation-path preservation, not independent writing-quality assessment.",
                  "These sources were absent from our answer fine-tuning; base-model pretraining overlap is unknown.",

                  "PubMedQA uses the publisher's 250-case expert-labeled subset, not the official leaderboard split.",
                  "Internal routing/writing checks were used during development; external paired writing uses two explicit instruction endings."]), all_rows


def table(headers, rows):
    escape = lambda value: html.escape(str(value))
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{escape(x)}</th>" for x in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(x)}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def render(report, all_rows, output):
    rows = []
    for name in SUBSETS:
        result = report["decisions"][name]
        reference = report["published"]["subsets"][name]
        row = [name, result["qwen"]["rows"]]
        row += [f"{100*result[key]['accuracy']:.2f}%" for key in
                ("qwen", "lora", "lora_automatic", "probe256", "probe256_automatic")]
        row += [f"{reference[key]['accuracy_percent']:.1f}%" for key in ("nimble", "jev")]
        rows.append(row)
    headers = ["Dataset", "n", "Qwen", "LoRA forced", "LoRA auto", "Probe forced", "Probe auto", "Nimble published", "Jev published"]
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(headers); writer.writerows(rows)
    body = "<h1>TheSystem: five-benchmark evaluation</h1><p>Frozen release checkpoints · Evaluation only · No retraining</p>"
    if not report.get("inference_performed", True):
        body += "<p>Reconstructed from frozen per-case evidence. No new GPU inference was run.</p>"
    if report.get("smoke"):
        body += "<p><strong>SMOKE CHECK ONLY. Small subsets test execution; these are not benchmark results.</strong></p>"
    body += "<h2>Decision accuracy</h2><p>Same frozen published examples and reference labels (1,648 in a full run). Automatic accuracy counts a wrong writing route as failure.</p>" + table(headers, rows)
    body += "<p>Nimble: Bespoke-Nimble-9B, revision 594dfdcfb6f94e3d0c0db7535180d3c71689169a. Jev: 1.13.0. Published values are rounded aggregates, not new runs. Input semantics and candidate order are preserved; model-specific prompt wrappers differ.</p>"
    body += "<h2>&lt;FAST&gt; routing</h2><p>Rows are requested modes; columns are selected modes. Each cell is a request count. Decision means select a supplied answer; writing means generate text. These measure routing, separately from answer accuracy. Original Qwen has no learned gate.</p>"
    for arm, suites in report["routing"].items():
        internal = suites["routing_historical"]["matrix"]
        decision = suites["external_decisions"]["matrix"][0]
        writing = suites["paired_writing"]["matrix"][1]
        for label, matrix in ((f"Internal: {sum(map(sum, internal)):,} fixed routing requests", internal),
                              ("External: paired decision and writing instructions", [decision, writing])):
            body += f"<h3>{html.escape(arm)} — {html.escape(label)}</h3>"
            body += table(["Requested mode", "Selected decision", "Selected writing"],
                          [["Decision", *matrix[0]], ["Writing", *matrix[1]]])
    writing_rows = []
    for arm, suites in report["writing"].items():
        for suite, values in suites.items():
            writing_rows.append([arm, {"historical": "Internal writing prompts", "external-paired": "Benchmark writing requests"}.get(suite, suite), values['rows'], values['forced_matches'], values['automatic_matches'],
                                 values['wrong_routes'], values['capped'], values['passed']])
    body += "<h2>Generation preservation</h2><p>Greedy generation, seed 42; 512-token cap, retry at 1,024 when capped. A capped match does not pass. Forced writing checks the generation path; automatic writing also tests routing.</p>"
    body += table(["Model", "Suite", "n", "Forced exact match", "Auto exact match", "Wrong routes", "Still capped", "Full pass"], writing_rows)
    ci_rows = [[name, arm, f"{100*v['difference']:+.2f}", f"[{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]"]
               for name, comparisons in report["paired"].items() for arm, v in comparisons.items()]
    body += "<details><summary>Differences from original Qwen and uncertainty</summary><p>Percentage points; paired 95% intervals from 2,000 source-family bootstrap draws. These intervals describe these checkpoints and cases.</p>" + table(["Dataset", "Comparison", "Difference", "95% interval"], ci_rows) + "</details>"
    sources = load_suites(smoke=report.get("smoke", False))["decisions"]
    body += "<details><summary>Decision disagreements and wrong routes</summary>"
    for name in SUBSETS:
        for source, left, right in zip(sources[name], all_rows["lora", name], all_rows["probe", name]):
            scores = dict(left["scores"], **right["scores"])
            predictions = {k: max(range(len(v)), key=v.__getitem__) for k, v in scores.items()}
            if all(p == source["target_index"] for p in predictions.values()) and left["route"] and right["route"]:
                continue
            lines = {"correct": source["options"][source["target_index"]],
                     "predictions": {k: source["options"][v] for k, v in predictions.items()},
                     "LoRA route": left["route"], "Probe route": right["route"]}
            body += f"<details><summary>{html.escape(source['id'])}</summary><pre>{html.escape(source['prompt'])}</pre><pre>{html.escape(json.dumps(lines,indent=2))}</pre></details>"
    body += "</details><h2>Interpretation limits</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in report["limitations"]) + "</ul>"
    link = html.escape(report["published"]["source_url"], quote=True)
    body += f"<p><a href='{link}'>Publisher's frozen results and protocol</a> · Full scores, intervals, writing tokens and progress journals accompany this report.</p>"
    css = "body{font:16px/1.5 system-ui;margin:40px auto;max-width:1200px;padding:0 24px;color:#1e293b;background:#f8fafc}h1,h2{color:#0f172a}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#e2e8f0}.scroll{overflow:auto}details{margin:18px 0;padding:12px;background:white;border:1px solid #ddd;border-radius:8px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}p{max-width:100ch}"
    return "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>TheSystem release evaluation</title><style>" + css + "</style><body>" + body + "</body></html>"


def build(output):
    output = Path(output)
    report, rows = collect(output)
    (output / "REPORT.html").parent.mkdir(parents=True, exist_ok=True)
    write_json(output / "SUMMARY.json", report)
    (output / "REPORT.html").write_text(render(report, rows, output))
    print("Report saved:", output / "REPORT.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
