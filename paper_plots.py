"""Create paper-ready plots from interval-fairness experiment CSV files.

This module only reads experiment output.  It never reruns or changes an
algorithm.  Run, for example:

    python paper_plots.py experiment_results.csv --output-dir plots
    python paper_plots.py results_directory --output-dir plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba


# Keep this section small and explicit: it is the only place normally needed
# when a future experiment changes an algorithm identifier or assignment name.
ALGORITHM_LABELS = {
    "offline_greedy": "Global Greedy",
    "offline_deterministic": "Offline Deterministic",
    "offline_randomized": "Offline Randomized",
    "simple_online_greedy": "Simple Online Greedy",
    "crs": "CRS",
    "online_randomized_fair": "Online Randomized Fair",
    # Historical names accepted by older result files.
    "online_randomized": "Online Randomized Fair",
    "online_randomized_level_greedy": "CRS",
}
ALGORITHM_ALIASES = {
    "online_randomized": "online_randomized_fair",
    "online_randomized_level_greedy": "crs",
}
OFFLINE_ALGORITHM_ORDER = [
    "offline_greedy",
    "offline_deterministic",
    "offline_randomized",
]
# Offline Greedy remains in cleaned/aggregated CSVs, but is not part of the
# paper comparison figures.
OFFLINE_PLOT_ALGORITHM_ORDER = [
    "offline_deterministic",
    "offline_randomized",
]
ONLINE_ALGORITHM_ORDER = [
    "simple_online_greedy",
    "crs",
    "online_randomized_fair",
]

# Longer patterns must precede shorter patterns.  Add project-specific naming
# variants here rather than encoding assumptions in the filename parser.
ASSIGNMENT_PATTERNS = {
    "containment_quantile": [r"containment[_-]quantile"],
    "length_quantile": [r"length[_-]quantile"],
    "length_delta": [r"length[_-]delta"],
    "exponential": [r"exponential"],
    "uniform": [r"uniform"],
    "containment": [r"containment"],
    "length": [r"length"],
}
ASSIGNMENT_ORDER = [
    "containment_quantile",
    "length_quantile",
    "length_delta",
    "exponential",
    "uniform",
]
ASSIGNMENT_LABELS = {
    "containment_quantile": "Containment Quantile",
    "length_quantile": "Length Quantile",
    "length_delta": "Length Delta",
    "exponential": "Exponential",
    "uniform": "Uniform",
}

REQUIRED_COLUMNS = {
    "input_file", "k", "algorithm", "algorithm_type", "runs", "selected",
    "fairness", "fraction_opt", "inverse_ratio", "opt_by_group",
    "selected_by_group",
}
RUN_RANGE_COLUMNS = {
    "selected": ("min_selected", "max_selected"),
    "inverse_ratio": ("min_inverse_ratio", "max_inverse_ratio"),
    "fairness": ("min_fairness", "max_fairness"),
    "minimum_group_coverage": ("minimum_group_coverage", "max_group_coverage"),
}
PLOT_STYLE: dict[str, dict[str, Any]] = {}


def parse_number(value: Any) -> float:
    """Parse blank, ordinary, and infinity CSV values without throwing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return math.nan
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan


def parse_json_dict(value: Any) -> dict[str, float]:
    """Parse a JSON mapping with numeric values; malformed input becomes {}."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw in parsed.items():
        number = parse_number(raw)
        if not math.isnan(number):
            result[str(key)] = number
    return result


def parse_metadata(input_file: Any) -> dict[str, Any]:
    """Extract workload, assignment and optional seed from a flexible path."""
    raw_path = "" if input_file is None else str(input_file)
    pieces = [piece for piece in re.split(r"[\\/]+", raw_path) if piece]
    filename = pieces[-1] if pieces else raw_path
    stem = Path(filename).stem
    workload = pieces[-2] if len(pieces) >= 2 else "unknown"

    assignment = "unknown"
    for name in ASSIGNMENT_ORDER + sorted(ASSIGNMENT_PATTERNS):
        for pattern in ASSIGNMENT_PATTERNS[name]:
            if re.search(pattern, stem, flags=re.IGNORECASE):
                assignment = name
                break
        if assignment != "unknown":
            break

    seed_match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:[_-]|$)", stem, re.I)
    return {
        "workload": workload,
        "assignment": assignment,
        "instance_seed": seed_match.group(1) if seed_match else np.nan,
    }


def group_diagnostics(opt_by_group: dict[str, float], selected_by_group: dict[str, float]) -> dict[str, Any]:
    """Calculate unserved and worst-group diagnostics from the two JSON maps."""
    ratios: list[tuple[str, float, float, float]] = []
    coverages: list[float] = []
    unserved = 0
    for group, opt in opt_by_group.items():
        if opt <= 0:
            continue
        selected = selected_by_group.get(group, 0.0)
        ratio = math.inf if selected <= 0 else opt / selected
        coverage = 0.0 if selected <= 0 else selected / opt
        if selected <= 0:
            unserved += 1
        ratios.append((group, ratio, opt, selected))
        coverages.append(coverage)
    if not ratios:
        no_eligible_group = bool(opt_by_group)
        return {
            "num_unserved_groups": 0,
            "worst_fairness_group": np.nan,
            "worst_group_opt": np.nan,
            "worst_group_selected": np.nan,
            "recomputed_fairness": 1.0 if no_eligible_group else math.nan,
            "minimum_group_coverage": 1.0 if no_eligible_group else math.nan,
            "max_group_coverage": 1.0 if no_eligible_group else math.nan,
        }
    # Deterministic tie-breaking is useful when several groups are unserved.
    worst = max(ratios, key=lambda item: (item[1], -int(item[0]) if item[0].isdigit() else 0))
    return {
        "num_unserved_groups": unserved,
        "worst_fairness_group": worst[0],
        "worst_group_opt": worst[2],
        "worst_group_selected": worst[3],
        "recomputed_fairness": worst[1],
        "minimum_group_coverage": min(coverages),
        "max_group_coverage": max(coverages),
    }


def csv_files(input_path: Path) -> list[Path]:
    """Return one CSV or all immediate CSV files in a result directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.csv"))
        if files:
            return files
    raise FileNotFoundError(f"Expected a CSV file or directory containing CSV files: {input_path}")


def read_results(input_path: Path) -> pd.DataFrame:
    """Read and validate compatible result CSVs, preserving infinity values."""
    files = csv_files(input_path)
    frames: list[pd.DataFrame] = []
    columns: set[str] | None = None
    for path in files:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        current = set(frame.columns)
        if columns is None:
            columns = current
        elif current != columns:
            missing, extra = columns - current, current - columns
            raise ValueError(f"CSV columns differ in {path}: missing={sorted(missing)}, extra={sorted(extra)}")
        absent = REQUIRED_COLUMNS - current
        if absent:
            raise ValueError(f"{path} is missing required columns: {sorted(absent)}")
        frame["source_csv"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean_results(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add metadata and diagnostics while retaining original experiment values."""
    frame = raw.copy()
    warnings: list[str] = []
    metadata = frame["input_file"].apply(parse_metadata).apply(pd.Series)
    frame = pd.concat([frame, metadata], axis=1)
    frame["algorithm"] = frame["algorithm"].map(lambda value: ALGORITHM_ALIASES.get(value, value))
    for column in [
        "k", "runs", "selected", "min_selected", "max_selected", "fairness",
        "min_fairness", "max_fairness", "fraction_opt", "min_fraction_opt",
        "max_fraction_opt", "inverse_ratio", "min_inverse_ratio", "max_inverse_ratio",
        "delta", "num_levels",
    ]:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = frame[column].map(parse_number)
    frame["k"] = frame["k"].round().astype("Int64")

    diagnostics = []
    for _, row in frame.iterrows():
        opt = parse_json_dict(row.get("opt_by_group"))
        selected = parse_json_dict(row.get("selected_by_group"))
        diagnostic = group_diagnostics(opt, selected)
        diagnostics.append(diagnostic)
        observed = row["fairness"]
        recomputed = diagnostic["recomputed_fairness"]
        compatible = bool(opt) and bool(selected)
        if compatible and not (
            (math.isinf(observed) and math.isinf(recomputed))
            or math.isclose(observed, recomputed, rel_tol=1e-7, abs_tol=1e-8)
        ):
            warnings.append(
                f"fairness mismatch: {row['input_file']} [{row['algorithm']}], "
                f"CSV={observed}, group JSON={recomputed}"
            )
    frame = pd.concat([frame, pd.DataFrame(diagnostics)], axis=1)
    return frame, warnings


def scalar_summary(values: Iterable[float], fairness: bool = False) -> dict[str, float]:
    """Summarize values without allowing infinity to contaminate finite means."""
    values_array = np.asarray(list(values), dtype=float)
    finite = values_array[np.isfinite(values_array)]
    inf_count = int(np.isposinf(values_array).sum())
    result = {
        "mean": float(np.mean(finite)) if len(finite) else (math.inf if inf_count else math.nan),
        "min_workload": float(np.min(finite)) if len(finite) else math.nan,
        "max_workload": float(np.max(finite)) if len(finite) else math.nan,
        "std_workload": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if len(finite) == 1 else math.nan,
        "num_finite_workloads": int(len(finite)),
        "num_infinite_workloads": inf_count,
    }
    if fairness:
        result.update({
            "mean_finite_fairness": result["mean"],
            "min_finite_fairness": result["min_workload"],
            "max_finite_fairness": result["max_workload"],
        })
    return result


def mean_preserving_infinity(values: Iterable[float]) -> float:
    """Average values, returning +inf when at least one instance seed is +inf."""
    array = np.asarray(list(values), dtype=float)
    if np.isposinf(array).any():
        return math.inf
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else math.nan


def aggregate_results(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Aggregate instance seeds within workload, then give workloads equal weight."""
    metrics = ["inverse_ratio", "fairness", "minimum_group_coverage", "fraction_opt"]
    seed_keys = ["workload", "assignment", "k", "algorithm", "algorithm_type", "instance_seed"]
    workload_keys = ["workload", "assignment", "k", "algorithm", "algorithm_type"]
    seed_rows: list[dict[str, Any]] = []
    for key, group in cleaned.groupby(seed_keys, dropna=False):
        row = dict(zip(seed_keys, key))
        row["duplicate_rows"] = len(group)
        for metric in metrics:
            row[metric] = mean_preserving_infinity(group[metric])
        seed_rows.append(row)
    by_seed = pd.DataFrame(seed_rows)

    workload_rows: list[dict[str, Any]] = []
    for key, group in by_seed.groupby(workload_keys, dropna=False):
        row = dict(zip(workload_keys, key))
        row["num_instance_seeds"] = len(group)
        for metric in metrics:
            row[metric] = mean_preserving_infinity(group[metric])
        workload_rows.append(row)
    by_workload = pd.DataFrame(workload_rows)

    rows: list[dict[str, Any]] = []
    aggregate_keys = ["assignment", "k", "algorithm", "algorithm_type"]
    for key, group in by_workload.groupby(aggregate_keys, dropna=False):
        common = dict(zip(aggregate_keys, key))
        for metric in metrics:
            summary = scalar_summary(group[metric], fairness=(metric == "fairness"))
            rows.append({
                **common,
                "metric": metric,
                **summary,
                "num_workloads": int(len(group)),
            })
    return pd.DataFrame(rows), by_workload, by_seed


def algorithm_style(algorithms: Iterable[str]) -> None:
    """Assign stable default-cycle colours and varied markers once per run."""
    markers = ["o", "s", "^", "D", "P", "X"]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, algorithm in enumerate(sorted(set(algorithms), key=algorithm_rank)):
        PLOT_STYLE[algorithm] = {
            "color": colours[index % len(colours)],
            "marker": markers[index % len(markers)],
        }


def algorithm_rank(algorithm: str) -> tuple[int, str]:
    order = OFFLINE_ALGORITHM_ORDER + ONLINE_ALGORITHM_ORDER
    return (order.index(algorithm), algorithm) if algorithm in order else (len(order), algorithm)


def ordered_assignments(values: Iterable[str]) -> list[str]:
    present = set(values)
    configured = [name for name in ASSIGNMENT_ORDER if name in present]
    return configured + sorted(present - set(configured))


def finite_plot_limit(data: pd.DataFrame, metric: str, log_scale: bool) -> tuple[float, float]:
    finite = data.loc[np.isfinite(data["mean"]), ["mean", "min_workload", "max_workload"]].to_numpy(dtype=float)
    values = finite[np.isfinite(finite)]
    if log_scale:
        values = values[values > 0]
    if not len(values):
        return (0.8, 2.0)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if log_scale:
        return max(lower / 1.35, 1e-6), upper * 1.35
    padding = max((upper - lower) * 0.10, 0.05)
    return max(0.0, lower - padding), upper + padding


def plot_grid(
    aggregated: pd.DataFrame,
    cleaned: pd.DataFrame,
    algorithms: list[str],
    metric: str,
    title: str,
    y_label: str,
    output_dir: Path,
    filename: str,
    error_mode: str,
    log_fairness: bool,
    assignments_override: list[str] | None = None,
    layout: tuple[int, int] | None = None,
    force_log_scale: bool | None = None,
    offset_algorithms: bool = True,
    subtle_error_bars: bool = False,
    show_infinite_markers: bool = True,
    show_error_bars: bool = True,
    reference: str = "one",
    fixed_y_limits: tuple[float, float] | None = None,
) -> None:
    """Create one five-assignment paper figure and matching PNG/PDF files."""
    data = aggregated[(aggregated["metric"] == metric) & aggregated["algorithm"].isin(algorithms)].copy()
    if assignments_override is not None:
        data = data[data["assignment"].isin(assignments_override)]
    assignments = assignments_override or ordered_assignments(data["assignment"])
    assignments = [assignment for assignment in assignments if assignment in set(data["assignment"])]
    if data.empty or not assignments:
        print(f"Skipping {filename}: no matching data.")
        return
    shown_algorithms = sorted(data["algorithm"].unique(), key=algorithm_rank)
    offsets = {
        # With three algorithms this is exactly -0.12, 0, +0.12.  Two
        # algorithms use a smaller symmetric dodge, while tick locations and
        # all aggregation remain at the original integer k values.
        algorithm: (index - (len(shown_algorithms) - 1) / 2) * 0.12
        for index, algorithm in enumerate(shown_algorithms)
    }
    log_scale = metric == "fairness" and log_fairness if force_log_scale is None else force_log_scale
    y_min, y_max = fixed_y_limits or finite_plot_limit(data, metric, log_scale)
    nrows, ncols = layout or (2, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.35 * nrows), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(np.asarray(axes).reshape(-1))
    for axis in flat_axes[len(assignments):]:
        axis.remove()

    for axis, assignment in zip(flat_axes, assignments):
        panel = data[data["assignment"] == assignment]
        for algorithm in shown_algorithms:
            points = panel[panel["algorithm"] == algorithm].sort_values("k")
            finite_points = points[np.isfinite(points["mean"])]
            if not finite_points.empty:
                x = finite_points["k"].to_numpy(dtype=float)
                if offset_algorithms:
                    x = x + offsets[algorithm]
                y = finite_points["mean"].to_numpy(dtype=float)
                if error_mode == "std":
                    lower = upper = finite_points["std_workload"].fillna(0.0).to_numpy(dtype=float)
                else:
                    lower = np.maximum(y - finite_points["min_workload"].to_numpy(dtype=float), 0.0)
                    upper = np.maximum(finite_points["max_workload"].to_numpy(dtype=float) - y, 0.0)
                style = PLOT_STYLE[algorithm]
                error_style = {}
                if subtle_error_bars:
                    error_style = {
                        "ecolor": to_rgba(style["color"], alpha=0.42),
                        "elinewidth": 0.75,
                        "capthick": 0.75,
                    }
                else:
                    error_style = {
                        "elinewidth": 1.25,
                        "capthick": 1.15,
                    }
                axis.plot(
                    x,
                    y,
                    label=ALGORITHM_LABELS.get(algorithm, algorithm),
                    linewidth=1.8,
                    markersize=5,
                    **style,
                )
                if show_error_bars:
                    finite_errors = np.isfinite(lower) & np.isfinite(upper)
                    if finite_errors.any():
                        axis.errorbar(
                            x[finite_errors],
                            y[finite_errors],
                            yerr=np.vstack([lower[finite_errors], upper[finite_errors]]),
                            fmt="none",
                            capsize=1.8 if subtle_error_bars else 3.5,
                            **error_style,
                        )
            if metric == "fairness" and show_infinite_markers:
                infinite = points[points["num_infinite_workloads"] > 0]
                previous_count: int | None = None
                previous_k: int | None = None
                for _, point in infinite.iterrows():
                    count = int(point["num_infinite_workloads"])
                    total = int(point["num_workloads"])
                    x = float(point["k"]) + (offsets[algorithm] if offset_algorithms else 0.0)
                    style = PLOT_STYLE[algorithm]
                    axis.plot(x, 0.94, marker="^", linestyle="None", color=style["color"],
                              transform=axis.get_xaxis_transform(), clip_on=False, markersize=6)
                    is_new_segment = previous_k is None or int(point["k"]) != previous_k + 1
                    if is_new_segment or count != previous_count:
                        axis.annotate(f"∞ {count}/{total}", (x, 0.94), xycoords=axis.get_xaxis_transform(),
                                      xytext=(0, -13), textcoords="offset points", ha="center", va="top", fontsize=7,
                                      color=style["color"])
                    previous_count = count
                    previous_k = int(point["k"])
            if metric == "inverse_ratio" and show_error_bars:
                infinite_ranges = points[
                    np.isfinite(points["mean"]) & np.isposinf(points["max_workload"])
                ]
                previous_k = None
                for _, point in infinite_ranges.iterrows():
                    x = float(point["k"]) + (offsets[algorithm] if offset_algorithms else 0.0)
                    style = PLOT_STYLE[algorithm]
                    axis.plot(
                        x,
                        0.94,
                        marker="^",
                        linestyle="None",
                        color=style["color"],
                        transform=axis.get_xaxis_transform(),
                        clip_on=False,
                        markersize=5.5,
                    )
                    is_new_segment = previous_k is None or int(point["k"]) != previous_k + 1
                    if is_new_segment:
                        axis.annotate(
                            "max inf",
                            (x, 0.94),
                            xycoords=axis.get_xaxis_transform(),
                            xytext=(0, -13),
                            textcoords="offset points",
                            ha="center",
                            va="top",
                            fontsize=7,
                            color=style["color"],
                        )
                    previous_k = int(point["k"])
        if reference == "one":
            axis.axhline(1.0, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
        elif reference == "coverage_threshold":
            k_values = np.asarray(sorted(panel["k"].dropna().unique()), dtype=float)
            axis.plot(
                k_values,
                1.0 / k_values,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
                label="Fairness threshold (1/k)" if axis is flat_axes[0] else None,
                zorder=0,
            )
        axis.set_title(ASSIGNMENT_LABELS.get(assignment, assignment.replace("_", " ").title()), fontsize=10)
        axis.set_xticks(sorted(panel["k"].dropna().unique()))
        axis.grid(True, alpha=0.22, linewidth=0.6)
        axis.set_ylim(y_min, y_max)
        if log_scale:
            axis.set_yscale("log")

    for axis in flat_axes[:len(assignments)]:
        axis.set_xlabel("k")
    for row_index in range(nrows):
        flat_axes[row_index * ncols].set_ylabel(y_label)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    # "outside" reserves layout space above every subplot, so the shared
    # legend cannot collide with assignment titles in the 1x2/1x3 figures.
    legend_columns = len(shown_algorithms) + (1 if reference == "coverage_threshold" else 0)
    fig.legend(handles, labels, loc="outside upper center", ncol=legend_columns, frameon=False)
    has_infinite_fairness = (
        metric == "fairness"
        and show_infinite_markers
        and bool((data["num_infinite_workloads"] > 0).any())
    )
    if has_infinite_fairness:
        fig.text(0.5, -0.015, "Upward triangles indicate infinite fairness values; annotations give infinite workloads / total workloads.",
                 ha="center", fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_workload_figures(
    cleaned: pd.DataFrame,
    output_dir: Path,
    show_run_range: bool | None,
    log_fairness: bool,
) -> None:
    """Write appendix figures, one four-figure set per workload.

    Selected-count plots show their run-level min/max range. Fairness and
    coverage show their within-workload group ranges. Competitive ratio
    remains a point estimate unless ``--show-run-range`` is supplied.
    """
    for workload, workload_rows in cleaned.groupby("workload", dropna=False):
        workload_dir = output_dir / "by_workload" / str(workload)
        for base_name in ("offline_fairness", "online_fairness", "online_inverse_ratio"):
            for suffix in ("png", "pdf"):
                stale_path = workload_dir / f"{base_name}.{suffix}"
                if stale_path.exists():
                    stale_path.unlink()
        rows: list[dict[str, Any]] = []
        for keys, group in workload_rows.groupby(["assignment", "k", "algorithm", "algorithm_type"], dropna=False):
            assignment, k, algorithm, algorithm_type = keys
            for metric in ("selected", "inverse_ratio", "fairness", "minimum_group_coverage"):
                values = group[metric].map(parse_number)
                finite = values[np.isfinite(values)]
                record = {
                    "assignment": assignment, "k": k, "algorithm": algorithm, "algorithm_type": algorithm_type,
                    "metric": metric, "mean": mean_preserving_infinity(values),
                    "min_workload": float(finite.min()) if len(finite) else math.nan,
                    "max_workload": float(finite.max()) if len(finite) else math.nan,
                    "std_workload": 0.0, "num_workloads": len(group),
                    "num_infinite_workloads": int(np.isposinf(values).sum()),
                }
                use_run_range = metric == "inverse_ratio" and show_run_range is True
                use_selected_run_range = metric == "selected" and show_run_range is not False
                use_group_range = metric in {"fairness", "minimum_group_coverage"}
                if (use_run_range or use_selected_run_range or use_group_range) and len(group) == 1 and metric in RUN_RANGE_COLUMNS:
                    low, high = RUN_RANGE_COLUMNS[metric]
                    record["min_workload"] = parse_number(group.iloc[0].get(low))
                    record["max_workload"] = parse_number(group.iloc[0].get(high))
                rows.append(record)
        aggregate = pd.DataFrame(rows)
        for kind, algorithms in (("offline", OFFLINE_PLOT_ALGORITHM_ORDER), ("online", ONLINE_ALGORITHM_ORDER)):
            for metric, label, plot_name in (
                ("selected", "Selected intervals", "selected"),
                ("inverse_ratio", "Approximation ratio", "approximation_ratio"),
                ("fairness", "Fairness ratio", "fairness_ratio"),
                ("minimum_group_coverage", "Minimum group coverage", "minimum_group_coverage"),
            ):
                if kind == "online" and metric == "inverse_ratio":
                    label = "Competitive ratio"
                    plot_name = "competitive_ratio"
                plot_algorithms = OFFLINE_ALGORITHM_ORDER if kind == "offline" and metric in {"selected", "fairness", "minimum_group_coverage"} else algorithms
                plot_grid(aggregate, workload_rows, plot_algorithms, metric, "", label, workload_dir,
                          f"{kind}_{plot_name}", "range", log_fairness,
                          show_error_bars=metric in {"selected", "fairness", "minimum_group_coverage"} or (metric == "inverse_ratio" and show_run_range is True),
                          reference="coverage_threshold" if metric == "minimum_group_coverage" else "none" if metric == "selected" else "one",
                          fixed_y_limits=(0.0, 1.0) if metric == "minimum_group_coverage" else None,
                          force_log_scale=metric == "inverse_ratio" and show_run_range is True)


def print_validation(cleaned: pd.DataFrame, warnings: list[str]) -> None:
    """Print a compact data-quality report before plotting."""
    combination = ["workload", "assignment", "k", "algorithm", "instance_seed"]
    duplicates = cleaned.duplicated(combination, keep=False).sum()
    expected = set()
    workloads = cleaned["workload"].unique()
    assignments = cleaned["assignment"].unique()
    ks = cleaned["k"].dropna().unique()
    algorithms = cleaned["algorithm"].unique()
    for workload in workloads:
        for assignment in assignments:
            for k in ks:
                for algorithm in algorithms:
                    expected.add((workload, assignment, int(k), algorithm))
    observed = set(tuple(value) for value in cleaned[["workload", "assignment", "k", "algorithm"]].dropna().drop_duplicates().itertuples(index=False, name=None))
    missing = expected - observed
    print(f"Number of rows: {len(cleaned)}")
    print(f"Workloads found: {', '.join(map(str, sorted(workloads)))}")
    print(f"Assignments found: {', '.join(map(str, ordered_assignments(assignments)))}")
    print(f"k values found: {', '.join(map(str, sorted(int(k) for k in ks)))}")
    print(f"Algorithms found: {', '.join(sorted(map(str, algorithms), key=algorithm_rank))}")
    print(f"Missing combinations: {len(missing)}")
    print(f"Number of infinite fairness rows: {int(np.isposinf(cleaned['fairness']).sum())}")
    print(f"Number of rows with unserved groups: {int((cleaned['num_unserved_groups'] > 0).sum())}")
    if duplicates:
        print(f"WARNING: duplicate workload/assignment/k/algorithm/seed rows: {duplicates}")
    for warning in warnings[:10]:
        print(f"WARNING: {warning}")
    if len(warnings) > 10:
        print(f"WARNING: {len(warnings) - 10} additional fairness mismatches suppressed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate interval-fairness results and create paper plots.")
    parser.add_argument("input", type=Path, help="Experiment CSV, or directory containing CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument(
        "--show-run-range",
        dest="show_run_range",
        action="store_true",
        help="Show per-run min/max error bars for all appendix metrics, including approximation ratio.",
    )
    range_group.add_argument(
        "--hide-run-range",
        dest="show_run_range",
        action="store_false",
        help="Hide all min/max error bars in appendix workload plots.",
    )
    parser.set_defaults(show_run_range=None)
    parser.add_argument("--no-log-fairness", action="store_true")
    parser.add_argument("--aggregate-error", choices=["range", "std"], default="range")
    args = parser.parse_args()

    raw = read_results(args.input)
    cleaned, warnings = clean_results(raw)
    aggregated, _, _ = aggregate_results(cleaned)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output_dir / "cleaned_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    aggregated.to_csv(args.output_dir / "aggregated_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print_validation(cleaned, warnings)

    algorithm_style(cleaned["algorithm"])
    log_fairness = not args.no_log_fairness
    plot_grid(aggregated, cleaned, OFFLINE_PLOT_ALGORITHM_ORDER, "inverse_ratio", "", "Approximation ratio",
              args.output_dir, "offline_approximation_ratio", args.aggregate_error, log_fairness)
    # Keep the bounded coverage and the reciprocal fairness-ratio views side
    # by side.  They answer complementary questions about the same schedules.
    for base_name in (
        "offline_fairness",
        "online_fairness",
        "offline_fairness_structure_based",
        "offline_fairness_distribution_based",
        "online_inverse_ratio",
        "crs_approximation_ratio",
    ):
        for suffix in ("png", "pdf"):
            old_path = args.output_dir / f"{base_name}.{suffix}"
            if old_path.exists():
                old_path.unlink()
    plot_grid(
        aggregated, cleaned, OFFLINE_ALGORITHM_ORDER, "minimum_group_coverage", "", "Minimum group coverage",
        args.output_dir, "offline_minimum_group_coverage_structure_based", args.aggregate_error, log_fairness,
        assignments_override=["containment_quantile", "length_quantile", "length_delta"],
        layout=(1, 3), force_log_scale=False, show_infinite_markers=False,
        reference="coverage_threshold", fixed_y_limits=(0.0, 1.0),
    )
    plot_grid(
        aggregated, cleaned, OFFLINE_ALGORITHM_ORDER, "fairness", "", "Fairness ratio",
        args.output_dir, "offline_fairness_ratio_structure_based", args.aggregate_error, log_fairness,
        assignments_override=["containment_quantile", "length_quantile", "length_delta"],
        layout=(1, 3), force_log_scale=True, show_infinite_markers=False,
        reference="one",
    )
    plot_grid(
        aggregated, cleaned, OFFLINE_ALGORITHM_ORDER, "fairness", "", "Fairness ratio",
        args.output_dir, "offline_fairness_ratio_distribution_based", args.aggregate_error, log_fairness,
        assignments_override=["exponential", "uniform"], layout=(1, 2), force_log_scale=False,
        subtle_error_bars=True, reference="one",
    )
    plot_grid(
        aggregated, cleaned, OFFLINE_ALGORITHM_ORDER, "minimum_group_coverage", "", "Minimum group coverage",
        args.output_dir, "offline_minimum_group_coverage_distribution_based", args.aggregate_error, log_fairness,
        assignments_override=["exponential", "uniform"], layout=(1, 2), force_log_scale=False,
        offset_algorithms=True, subtle_error_bars=True, show_infinite_markers=False,
        reference="coverage_threshold", fixed_y_limits=(0.0, 1.0),
    )
    plot_grid(aggregated, cleaned, ONLINE_ALGORITHM_ORDER, "inverse_ratio", "", "Competitive ratio",
              args.output_dir, "online_competitive_ratio", args.aggregate_error, log_fairness)
    plot_grid(
        aggregated, cleaned, ONLINE_ALGORITHM_ORDER, "minimum_group_coverage", "", "Minimum group coverage",
        args.output_dir, "online_minimum_group_coverage", args.aggregate_error, log_fairness,
        show_infinite_markers=False, reference="coverage_threshold", fixed_y_limits=(0.0, 1.0),
    )
    plot_grid(
        aggregated, cleaned, ONLINE_ALGORITHM_ORDER, "fairness", "", "Fairness ratio",
        args.output_dir, "online_fairness_ratio", args.aggregate_error, log_fairness,
        reference="one",
    )
    # CRS-only views make its behaviour readable without overlap from the
    # fair online algorithm.  They use the same workload aggregation as the
    # comparison plots above.
    plot_grid(
        aggregated, cleaned, ["crs"], "inverse_ratio", "", "Competitive ratio",
        args.output_dir, "crs_competitive_ratio", args.aggregate_error, log_fairness,
        reference="one",
    )
    plot_grid(
        aggregated, cleaned, ["crs"], "fairness", "", "Fairness ratio",
        args.output_dir, "crs_fairness_ratio", args.aggregate_error, log_fairness,
        reference="one",
    )
    plot_grid(
        aggregated, cleaned, ["crs"], "minimum_group_coverage", "", "Minimum group coverage",
        args.output_dir, "crs_minimum_group_coverage", args.aggregate_error, log_fairness,
        show_infinite_markers=False, reference="coverage_threshold", fixed_y_limits=(0.0, 1.0),
    )
    plot_workload_figures(cleaned, args.output_dir, args.show_run_range, log_fairness)
    print(f"Wrote cleaned and aggregated CSV files plus plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
