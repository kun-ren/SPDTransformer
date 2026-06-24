from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


RUN_START_RE = re.compile(r"(?m)^\[Run (?P<index>\d+)\]\s+(?P<run_id>\S+)\s*$")
DONE_RE = re.compile(
    r"\[Run (?P<index>\d+)\] done \| "
    r"best_epoch=(?P<best_epoch>\d+) "
    r"best_val_mf1=(?P<best_val_mf1>[0-9.]+) "
    r"test_acc=(?P<test_acc>[0-9.]+) "
    r"test_mf1=(?P<test_mf1>[0-9.]+)"
)
EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)/(?:\d+)\s+\|\s+"
    r"train loss=(?P<train_loss>[0-9.]+)\s+"
    r"acc=(?P<train_acc>[0-9.]+)\s+"
    r"mf1=(?P<train_mf1>[0-9.]+)\s+\|\s+"
    r"val loss=(?P<val_loss>[0-9.]+)\s+"
    r"acc=(?P<val_acc>[0-9.]+)\s+"
    r"mf1=(?P<val_mf1>[0-9.]+)"
)

GROUP_KEYS = [
    "model.learnable_metric_mode",
    "model.dropout",
    "model.attention_dropout",
    "model.tau",
    "model.depth",
    "model.classifier_type",
    "model.pooling",
    "training.learning_rate",
    "training.gradient_clip_norm",
    "training.apply_weight_decay_to_special_parameters",
]


def extract_balanced_dict(text: str, marker: str) -> dict[str, Any]:
    marker_index = text.find(marker)
    if marker_index < 0:
        return {}

    start = text.find("{", marker_index)
    if start < 0:
        return {}

    depth = 0
    quote: str | None = None
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                raw = text[start:index + 1]
                # Some logs are copied from a terminal that hard-wraps long
                # lines, even in the middle of quoted keys such as
                # 'debug_tensor\n_stats'.  Removing newlines inside the
                # balanced dict restores the original Python literal while
                # leaving commas and spaces intact.
                raw = raw.replace("\r", "").replace("\n", "")
                try:
                    parsed = ast.literal_eval(raw)
                except (SyntaxError, ValueError):
                    return {"_parse_error": raw}
                if isinstance(parsed, dict):
                    return parsed
                return {}

    return {}


def parse_float_fields(match: re.Match[str], fields: list[str]) -> dict[str, float]:
    return {field: float(match.group(field)) for field in fields}


def flatten(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{key}": value for key, value in payload.items()}


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def parse_runs(log_text: str) -> list[dict[str, Any]]:
    starts = list(RUN_START_RE.finditer(log_text))
    runs: list[dict[str, Any]] = []

    for position, match in enumerate(starts):
        section_start = match.start()
        section_end = starts[position + 1].start() if position + 1 < len(starts) else len(log_text)
        section = log_text[section_start:section_end]

        run: dict[str, Any] = {
            "run_index": int(match.group("index")),
            "run_id": match.group("run_id"),
            "model": extract_balanced_dict(section, "model="),
            "training": extract_balanced_dict(section, "training="),
            "epochs": [],
        }

        for epoch_match in EPOCH_RE.finditer(section):
            epoch = {"epoch": int(epoch_match.group("epoch"))}
            epoch.update(
                parse_float_fields(
                    epoch_match,
                    [
                        "train_loss",
                        "train_acc",
                        "train_mf1",
                        "val_loss",
                        "val_acc",
                        "val_mf1",
                    ],
                )
            )
            run["epochs"].append(epoch)

        done_match = DONE_RE.search(section)
        if done_match:
            run.update(
                {
                    "best_epoch": int(done_match.group("best_epoch")),
                    "best_val_mf1": float(done_match.group("best_val_mf1")),
                    "test_acc": float(done_match.group("test_acc")),
                    "test_mf1": float(done_match.group("test_mf1")),
                }
            )

        add_derived_metrics(run)
        runs.append(run)

    return runs


def add_derived_metrics(run: dict[str, Any]) -> None:
    epochs = run.get("epochs", [])
    if not epochs:
        return

    best_epoch = run.get("best_epoch")
    best_row = next((row for row in epochs if row["epoch"] == best_epoch), None)
    final_row = epochs[-1]
    best_val_loss_row = min(epochs, key=lambda row: row["val_loss"])
    peak_val_mf1_row = max(epochs, key=lambda row: row["val_mf1"])

    run["final_epoch"] = final_row["epoch"]
    run["final_train_mf1"] = final_row["train_mf1"]
    run["final_val_mf1"] = final_row["val_mf1"]
    run["final_val_loss"] = final_row["val_loss"]
    run["final_train_val_mf1_gap"] = final_row["train_mf1"] - final_row["val_mf1"]
    run["min_val_loss_epoch"] = best_val_loss_row["epoch"]
    run["min_val_loss"] = best_val_loss_row["val_loss"]
    run["peak_val_mf1_epoch"] = peak_val_mf1_row["epoch"]
    run["peak_val_mf1"] = peak_val_mf1_row["val_mf1"]

    if best_row is not None:
        run["best_train_mf1"] = best_row["train_mf1"]
        run["best_val_loss"] = best_row["val_loss"]
        run["best_train_val_mf1_gap"] = best_row["train_mf1"] - best_row["val_mf1"]
        run["val_mf1_drop_after_best"] = best_row["val_mf1"] - final_row["val_mf1"]
        run["val_loss_increase_after_best"] = final_row["val_loss"] - best_row["val_loss"]

    if "best_val_mf1" in run and "test_mf1" in run:
        run["val_test_mf1_gap"] = run["best_val_mf1"] - run["test_mf1"]


def row_for_run(run: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        key: value
        for key, value in run.items()
        if key not in {"model", "training", "epochs"}
    }
    row.update(flatten("model", run.get("model", {})))
    row.update(flatten("training", run.get("training", {})))
    return {key: csv_value(value) for key, value in row.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_epoch_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for epoch in run.get("epochs", []):
            row = {
                "run_index": run["run_index"],
                "run_id": run["run_id"],
                **epoch,
            }
            rows.append(row)
    return rows


def value_at_key(run: dict[str, Any], dotted_key: str) -> Any:
    section, key = dotted_key.split(".", 1)
    return run.get(section, {}).get(key)


def group_stats(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete_runs = [run for run in runs if "test_mf1" in run]
    rows = []

    for key in GROUP_KEYS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in complete_runs:
            value = value_at_key(run, key)
            groups[str(value)].append(run)

        for value, group in sorted(groups.items()):
            rows.append(
                {
                    "key": key,
                    "value": value,
                    "n": len(group),
                    "avg_test_mf1": mean(run["test_mf1"] for run in group),
                    "max_test_mf1": max(run["test_mf1"] for run in group),
                    "avg_best_val_mf1": mean(run["best_val_mf1"] for run in group),
                    "avg_final_gap": mean(
                        run.get("final_train_val_mf1_gap", 0.0)
                        for run in group
                    ),
                    "avg_val_test_gap": mean(
                        run.get("val_test_mf1_gap", 0.0)
                        for run in group
                    ),
                }
            )

    return rows


def format_run(run: dict[str, Any]) -> str:
    model = run.get("model", {})
    training = run.get("training", {})
    return (
        f"Run {run['run_index']} {run['run_id']}: "
        f"test_mf1={run.get('test_mf1', 0.0):.4f}, "
        f"best_val_mf1={run.get('best_val_mf1', 0.0):.4f}, "
        f"best_epoch={run.get('best_epoch')}, "
        f"mode={model.get('learnable_metric_mode')}, "
        f"lr={training.get('learning_rate')}, "
        f"clip={training.get('gradient_clip_norm')}, "
        f"wd_special={training.get('apply_weight_decay_to_special_parameters')}, "
        f"attn_drop={model.get('attention_dropout')}, "
        f"dropout={model.get('dropout')}"
    )


def make_report(runs: list[dict[str, Any]], stats: list[dict[str, Any]]) -> str:
    complete_runs = [run for run in runs if "test_mf1" in run]
    top_test = sorted(complete_runs, key=lambda run: run["test_mf1"], reverse=True)[:10]
    top_val = sorted(complete_runs, key=lambda run: run["best_val_mf1"], reverse=True)[:10]
    top_gap = sorted(
        complete_runs,
        key=lambda run: run.get("final_train_val_mf1_gap", 0.0),
        reverse=True,
    )[:10]

    lines = [
        "# Training Log Analysis",
        "",
        f"- parsed_runs: {len(runs)}",
        f"- completed_runs: {len(complete_runs)}",
        "",
        "## Top Runs By Test Macro-F1",
        "",
    ]
    lines.extend(f"- {format_run(run)}" for run in top_test)

    lines.extend(["", "## Top Runs By Validation Macro-F1", ""])
    lines.extend(f"- {format_run(run)}" for run in top_val)

    lines.extend(["", "## Largest Final Train-Val Macro-F1 Gaps", ""])
    for run in top_gap:
        lines.append(
            "- "
            f"Run {run['run_index']}: "
            f"final_gap={run.get('final_train_val_mf1_gap', 0.0):.4f}, "
            f"final_train_mf1={run.get('final_train_mf1', 0.0):.4f}, "
            f"final_val_mf1={run.get('final_val_mf1', 0.0):.4f}, "
            f"final_val_loss={run.get('final_val_loss', 0.0):.4f}"
        )

    lines.extend(["", "## Parameter Group Averages", ""])
    for key in GROUP_KEYS:
        lines.append(f"### {key}")
        key_rows = sorted(
            [row for row in stats if row["key"] == key],
            key=lambda row: row["avg_test_mf1"],
            reverse=True,
        )
        for row in key_rows:
            lines.append(
                "- "
                f"{row['value']}: n={row['n']}, "
                f"avg_test_mf1={row['avg_test_mf1']:.4f}, "
                f"max_test_mf1={row['max_test_mf1']:.4f}, "
                f"avg_val_test_gap={row['avg_val_test_gap']:.4f}, "
                f"avg_final_gap={row['avg_final_gap']:.4f}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(log_path: Path, out_dir: Path) -> dict[str, Path]:
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    runs = parse_runs(log_text)
    stats = group_stats(runs)
    report = make_report(runs, stats)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = log_path.stem
    paths = {
        "runs_csv": out_dir / f"{stem}_runs.csv",
        "epochs_csv": out_dir / f"{stem}_epochs.csv",
        "groups_csv": out_dir / f"{stem}_group_stats.csv",
        "json": out_dir / f"{stem}_analysis.json",
        "report": out_dir / f"{stem}_report.md",
    }

    write_csv(paths["runs_csv"], [row_for_run(run) for run in runs])
    write_csv(paths["epochs_csv"], make_epoch_rows(runs))
    write_csv(paths["groups_csv"], [{key: csv_value(value) for key, value in row.items()} for row in stats])
    paths["json"].write_text(
        json.dumps(
            {
                "log_path": str(log_path),
                "runs": runs,
                "group_stats": stats,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths["report"].write_text(report, encoding="utf-8")
    print(report)
    print("Wrote:")
    for path in paths.values():
        print(f"  {path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SPDTransformer grid-search configs and metrics from a training log."
    )
    parser.add_argument("log_path", type=Path, help="Path to output.log.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON/report outputs. Defaults to the log directory.",
    )
    args = parser.parse_args()

    log_path = args.log_path.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else log_path.parent
    write_outputs(log_path=log_path, out_dir=out_dir)


if __name__ == "__main__":
    main()
