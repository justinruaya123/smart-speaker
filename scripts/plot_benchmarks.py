#!/usr/bin/env python3
"""Regenerate Iris release figures from committed receipts; no audio/cache needed.

Run from a fresh clone with ``python scripts/plot_benchmarks.py``. Matplotlib is
already in the project environment. Original history-cache paths in the numeric
snapshot are provenance only; this script never reads them or runs inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("experiments/iris-release-v1")
HISTORIES = Path("docs/assets/iris-training-histories.json")
BASELINE = "#0072B2"
CANDIDATE = "#D55E00"


def receipt(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def integer_score(rate: float, n: int) -> int:
    value = rate * n
    if not math.isclose(value, round(value), abs_tol=1e-8):
        raise ValueError("Score does not represent an integer numerator")
    return round(value)


def build_plot_data(root: Path) -> dict:
    names = ("recognition-results", "release-choice", "training-execution",
             "development-selection", "pipeline-resources")
    files = {name: root / EXPERIMENT / f"{name}.json" for name in names}
    inputs = {name: json.loads(path.read_text()) for name, path in files.items()}
    recognition = inputs["recognition-results"]
    choice = inputs["release-choice"]
    if choice["promoted"] or choice["guards"]["human_unknown_not_above_baseline"]:
        raise ValueError("These release figures require the recorded failed human UNKNOWN guard")
    history_path = root / HISTORIES
    histories = json.loads(history_path.read_text())
    for source in histories["source_receipts"]:
        if receipt(root / source["path"], root) != source:
            raise ValueError(f"Training history source receipt changed: {source['path']}")

    benchmarks = []
    official = {"id": "official196", "title": "Official scorer · 196 rows",
                "population": "186 supported commands + 10 out-of-scope rows",
                "denominator": 196, "negative_denominator": 10,
                "metric": "Exact scorer decisions, including correct rejections",
                "source_pointers": ["/release_official", "/official"]}
    for model, field in (("baseline", "release_official"), ("candidate", "official")):
        scores = recognition[field]
        raw, policy = (scores[k]["command_level"] for k in ("raw_diagnostic", "accepted"))
        if raw["n"] != 196 or policy["n"] != 196:
            raise ValueError("The unchanged official 196-row population is required")
        if raw["n_out_of_scope"] != 10 or policy["n_out_of_scope"] != 10:
            raise ValueError("Official negative population changed")
        rejects = policy["per_class"]["REJECT"]
        correct_rejects = integer_score(rejects["recall"], rejects["support"])
        official[model] = {
            "raw_correct": integer_score(raw["accuracy"], 196),
            "policy_exact": integer_score(policy["accuracy"], 196),
            "policy_correct_rejects": correct_rejects,
            "correct_accepts": integer_score(policy["accuracy"], 196) - correct_rejects,
            "unknown_accepts": policy["false_accepts"],
        }
    benchmarks.append(official)
    cohorts = (
        ("personal36", "personal-test", "Personal commands · 36 positives", 36, 52),
        ("human636", "test", "Original human · 636 positives", 636, 229),
        ("freshsynthetic444", "voice-test", "Fresh designed voices · 444 positives", 444, 120),
    )
    for key, split, title, positive_n, negative_n in cohorts:
        row = {"id": key, "title": title, "denominator": positive_n,
               "negative_denominator": negative_n,
               "metric": "Whole-command correct positives / positive recordings",
               "population": "original human" if split == "test" else "synthetic voices",
               "original_human_positive_n": positive_n if split == "test" else 0,
               "source_pointers": []}
        for model, field in (("baseline", "release_results"), ("candidate", "results")):
            suffix = "strata/human" if split == "test" else "overall"
            value = recognition[field][split]
            stats = value["strata"]["human"] if split == "test" else value["overall"]
            if stats["positive_n"] != positive_n or stats["negative_n"] != negative_n:
                raise ValueError(f"Changed population: {key}")
            if stats["n"] != positive_n + negative_n + stats["ambiguous_n"]:
                raise ValueError(f"Unaccounted recordings: {key}")
            row["ambiguous_excluded"] = stats["ambiguous_n"]
            row[model] = {name: stats[name] for name in (
                "raw_correct", "correct_accepts", "wrong_accepted_positives", "unknown_accepts")}
            row[model]["policy_exact"] = stats["correct_accepts"]
            row["source_pointers"].append(f"/{field}/{split}/{suffix}")
        benchmarks.append(row)

    runs = []
    execution = inputs["training-execution"]
    selections = inputs["development-selection"]
    for saved in histories["runs"]:
        key = tuple(saved[k] for k in ("model", "arm", "seed"))
        run = next(r for r in execution["runs"] if tuple(r[k] for k in ("model", "arm", "seed")) == key)
        selected = next(r for r in selections["all_runs"] if tuple(r[k] for k in ("model", "arm", "seed")) == key)
        points = saved["history"]
        original_bytes = (json.dumps(points, indent=2) + "\n").encode()
        if (hashlib.sha256(original_bytes).hexdigest() != saved["original_history"]["sha256"] or
                len(original_bytes) != saved["original_history"]["bytes"]):
            raise ValueError(f"Training history content changed: {saved['run_id']}")
        if (len(points) != run["epochs"] or
                [p["epoch"] for p in points] != list(range(1, run["epochs"] + 1)) or
                saved["checkpoint_sha256"] != run["checkpoint_sha256"] or
                saved["selected_epoch"] != selected["epoch"] or
                points[selected["epoch"] - 1]["accepted_selection"] != selected["key"]):
            raise ValueError(f"Training history differs from locked selection: {saved['run_id']}")
        runs.append({**saved, "parameters": selected["parameters"],
                     "training_seconds": run["seconds"]})
    if len(runs) != execution["completed_training_runs"] or len(runs) != 6:
        raise ValueError("All six completed training histories are required")
    if histories["test_used_for_selection"] or execution["test_used_for_selection"]:
        raise ValueError("Training histories must retain development-only selection")

    resource = inputs["pipeline-resources"]
    measured = resource["measurements"]
    desktop = {"physical_pi_measured": resource["physical_pi_measured"],
               "hardware": resource["hardware"], "engine": resource["bindings"]["engine"],
               "threads": resource["bindings"]["threads"], "sample_rate": resource["bindings"]["sample_rate"],
               "command_n": measured["command_n"], "wake_n": measured["wake_n"],
               "component_latency_ms": measured["component_latency_ms"],
               "complete_compute_budget_ms": measured["complete_compute_budget_ms"],
               "total_session_load": resource["total_session_load"],
               "memory": resource["memory"], "sizes": resource["sizes"],
               "timing_excludes": resource["timing_excludes"],
               "timing_scope": resource["timing_scope"], "input_policy": resource["input_policy"]}
    return {"schema_version": 1, "experiment": "iris-release-v1",
            "regenerate": "python scripts/plot_benchmarks.py",
            "sources": [receipt(path, root) for path in files.values()] + [receipt(history_path, root)],
            "promotion": choice, "command_benchmarks": benchmarks,
            "training": {"runs": runs, "test_used_for_selection": False,
                         "initialization": histories["initialization"],
                         "selection_rule": selections["rule"]},
            "desktop_resources": desktop,
            "limits": ["Candidate was not promoted: human UNKNOWN accepts increased from 12/229 to 14/229.",
                       "Official196 exact score includes correct negative rejections; other panels score positives only.",
                       "Personal and fresh designed-voice positives are synthetic, with no original-human personal positives.",
                       "Human positive accuracy excludes 185 ambiguous references; negative accepts are separate.",
                       "Desktop component profiling is not physical Pi or live microphone latency."]}


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E6EA", linewidth=0.7)
    ax.set_axisbelow(True)


def save_figure(fig, directory, name):
    for extension in ("png", "svg"):
        path = directory / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else {"Software": "scripts/plot_benchmarks.py"}
        fig.savefig(path, dpi=180, facecolor="white", metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        print(path)


def command_figure(plt, data, directory):
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.17, hspace=0.52, wspace=0.23)
    fig.suptitle("Whole-command recognition: raw prediction and locked acceptance", fontsize=16, y=0.965)
    fig.legend(handles=[Patch(facecolor="white", edgecolor="#3D4B58", hatch="///", label="Raw argmax (diagnostic)"),
                        Patch(facecolor="#617483", label="After locked acceptance policy")],
               loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    for ax, row in zip(axes.flat, data["command_benchmarks"]):
        n = row["denominator"]
        for x, (model, color) in enumerate((("baseline", BASELINE), ("candidate", CANDIDATE))):
            for offset, field in ((-0.18, "raw_correct"), (0.18, "policy_exact")):
                count = row[model][field]
                value = 100 * count / n
                ax.bar(x + offset, value, width=0.31, color="white" if offset < 0 else color,
                       edgecolor=color, hatch="///" if offset < 0 else None, linewidth=1.2)
                ax.text(x + offset, value + 2, f"{count}/{n}\n{value:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(row["title"], fontsize=12, pad=13)
        ax.set_xticks([0, 1], ["Shipped V3 baseline", "Candidate (not promoted)"], fontsize=9)
        ax.set_ylim(0, 119)
        ax.set_yticks(range(0, 101, 25))
        ax.set_ylabel("Exact decisions (%)" if row["id"] == "official196" else "Correct positives (%)")
        style_axes(ax)
    fig.text(0.08, 0.09, "Official196 includes correct rejections; its accepted score is not a count of accepted positives.\n"
             "Personal and fresh-voice cohorts are synthetic. Human accuracy excludes 185 ambiguous references.\n"
             "Candidate not promoted: human UNKNOWN accepts increased from 12/229 to 14/229. Command-only replay; no Pi/live-mic claim.",
             fontsize=9, va="center", linespacing=1.6)
    save_figure(fig, directory, "iris-command-benchmarks")
    plt.close(fig)


def negatives_figure(plt, data, directory):
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.25)
    fig.suptitle("UNKNOWN / out-of-scope accepts after the locked policy", fontsize=16, y=0.96)
    labels = ["Official scorer\n10 negatives", "Personal synthetic\n52 negatives",
              "Original human\n229 UNKNOWN", "Fresh synthetic\n120 negatives"]
    for offset, model, color, label in ((-0.19, "baseline", BASELINE, "Shipped V3 baseline"),
                                        (0.19, "candidate", CANDIDATE, "Candidate (not promoted)")):
        for i, row in enumerate(data["command_benchmarks"]):
            count, n = row[model]["unknown_accepts"], row["negative_denominator"]
            value = 100 * count / n
            ax.bar(i + offset, value, width=0.34, color=color, label=label if i == 0 else None)
            ax.text(i + offset, value + 0.3, f"{count}/{n}\n{value:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(range(4), labels, fontsize=10)
    ax.set_ylim(0, 13.5)
    ax.set_ylabel("Negative recordings falsely accepted (%) · lower is better")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
    fig.text(0.08, 0.08, "The human nonregression guard failed: 14/229 exceeded the predeclared baseline's 12/229.\n"
             "No thresholds were retuned after test. These are clip-level errors, not false wakes per hour.", fontsize=10, linespacing=1.5)
    save_figure(fig, directory, "iris-negative-accepts")
    plt.close(fig)


def training_figure(plt, data, directory):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), sharex="col")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.81, bottom=0.19, hspace=0.2, wspace=0.24)
    fig.suptitle("Six completed continuation fits · actual epoch histories", fontsize=16, y=0.965)
    colors = {("ce", 53): "#0072B2", ("supcon", 53): "#D55E00",
              ("ce", 67): "#009E73", ("supcon", 67): "#A45A9E"}
    for run in data["training"]["runs"]:
        col = 0 if run["model"] == "compact" else 1
        history = run["history"]
        epochs = [p["epoch"] for p in history]
        color = colors[run["arm"], run["seed"]]
        label = f"{'CE' if run['arm'] == 'ce' else 'CE + contrastive'} · seed {run['seed']}"
        linestyle = "-" if run["arm"] == "ce" else "--"
        axes[0, col].plot(epochs, [p["loss"] for p in history], color=color, linestyle=linestyle, label=label)
        values = [100 * p["accepted_selection"][0] for p in history]
        axes[1, col].plot(epochs, values, color=color, linestyle=linestyle)
        epoch = run["selected_epoch"]
        axes[1, col].scatter([epoch], [values[epoch - 1]], color=color, s=60, marker="o", edgecolor="white", zorder=4)
        if run["run_id"] == "compact-supcon-s67":
            axes[1, col].annotate("Selected candidate\nepoch 3 · 35.61%", xy=(epoch, values[epoch - 1]),
                                  xytext=(4.2, 37.7), fontsize=9,
                                  arrowprops={"arrowstyle": "->", "color": color}, color=color)
    axes[0, 0].set_title("Compact · 969,928 parameters")
    axes[0, 1].set_title("Full · 2,118,280 parameters")
    for col in range(2):
        axes[0, col].set_ylabel("Mean training objective")
        axes[1, col].set_ylabel("Dev human macro-speaker\ncorrect acceptance (%)")
        axes[1, col].set_xlabel("Completed epoch")
        axes[1, col].set_ylim(15, 43)
        axes[1, col].set_xticks(range(1, 10, 2) if col == 0 else range(2, 17, 2))
        for row in range(2):
            style_axes(axes[row, col])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", color="#617483", linestyle="None", label="Development-selected epoch"))
    fig.legend(handles=handles, labels=labels + ["Development-selected epoch"],
               loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False, fontsize=9)
    fig.text(0.09, 0.075, "Continuation from project V2 initializers; 62 completed epochs across six fits. CE + contrastive adds 0.1 × contrastive loss.\n"
             "Objective values therefore have different definitions. Circles mark each run's development-selected epoch; test did not select weights.\n"
             "The development-selected candidate later failed the human UNKNOWN test guard and was not promoted.", fontsize=9, linespacing=1.6)
    save_figure(fig, directory, "iris-training-curves")
    plt.close(fig)


def resources_figure(plt, data, directory):
    resource = data["desktop_resources"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.9), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.28, wspace=0.30)
    fig.suptitle("Recorded desktop CPU compute and packaged model assets", fontsize=16, y=0.96)
    fig.text(0.09, 0.865, f"x86_64 Linux · {resource['engine']} · one CPU thread per model · mono {resource['sample_rate']:,} Hz · eight synthetic wake/command pairs",
             fontsize=10)
    latency = resource["component_latency_ms"]
    values = [latency["vcm_final"]["wall_ms"], latency["vcm_preview"]["wall_ms"],
              resource["complete_compute_budget_ms"]["wall_ms"]]
    labels = ["Final VCM call\nn = 8", "One VCM preview\nn = 35", "Complete serial budget\nn = 8 pairs"]
    for offset, metric, color in ((-0.18, "p50", BASELINE), (0.18, "p95", CANDIDATE)):
        for i, value in enumerate(values):
            axes[0].bar(i + offset, value[metric], width=0.31, color=color, label=metric if i == 0 else None)
            axes[0].text(i + offset, value[metric] + 2, f"{value[metric]:.1f}", ha="center", fontsize=10)
    axes[0].set_xticks(range(3), labels, fontsize=9)
    axes[0].set_ylabel("Wall-clock compute (ms)")
    axes[0].set_ylim(0, 150)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    sizes = resource["sizes"]["by_component"]
    for i, (component, color) in enumerate(zip(("vcm", "wake", "owner", "silero"), (BASELINE, "#009E73", "#A45A9E", "#617483"))):
        value = sizes[component] / 2**20
        axes[1].bar(i, value, color=color, width=0.65)
        axes[1].text(i, value + 0.15, f"{value:.2f}", ha="center", fontsize=10)
    axes[1].set_xticks(range(4), ["VCM", "Wake", "Speaker", "VAD"], fontsize=9)
    axes[1].set_ylabel("Packaged graph + metadata size (MiB)")
    axes[1].set_ylim(0, 9.4)
    axes[1].set_title("ONNX + ncnn variants included", fontsize=10)
    for ax in axes:
        style_axes(ax)
    total_mib = resource["sizes"]["all_packaged_model_asset_bytes"] / 2**20
    fig.text(0.09, 0.12, f"All model assets: {total_mib:.2f} MiB. Five-session load: {resource['total_session_load']['wall_ms']:.1f} ms. "
             f"Whole Python process peak RSS: {resource['memory']['peak_process_rss_mib']:.1f} MiB.\n"
             "Complete budget includes wake, all scheduled previews, final VCM, VAD and speaker checks; it excludes elapsed audio, endpoint waiting,\n"
             "file decoding, worker IPC, UI and actions. It measures serial compute, not live user latency. Physical Pi performance remains unmeasured.",
             fontsize=9, linespacing=1.6)
    save_figure(fig, directory, "iris-desktop-resources")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    data = build_plot_data(ROOT)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "iris-release-v1",
                         "axes.titleweight": "normal", "axes.labelcolor": "#283744",
                         "text.color": "#283744", "axes.edgecolor": "#88949D"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for draw in (command_figure, negatives_figure, training_figure, resources_figure):
        draw(plt, data, args.output_dir)
    path = args.output_dir / "iris-benchmark-data.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
