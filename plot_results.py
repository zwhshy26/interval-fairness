import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is not installed. Install it with:\n"
        "  python3 -m pip install matplotlib"
    ) from exc

try:
    from scipy.stats import t
except ModuleNotFoundError as exc:
    raise SystemExit(
        "scipy is not installed. Install it with:\n"
        "  python3 -m pip install scipy"
    ) from exc


FILENAME_RE = re.compile(
    r"_interval_(?P<assignment_method>.+)_(?P<k>\d+)groups(?:_seed(?P<assignment_seed>\d+))?\.csv$"
)

ALGORITHM_LABELS = {
    "offline_greedy": "Global OPT",
    "offline_deterministic": "Offline Deterministic",
    "offline_randomized": "Offline Randomized",
    "simple_online_greedy": "Simple Online Greedy",
    "online_randomized": "Online Randomized",
    "online_randomized_level_greedy": "CRS",
}

ALGORITHM_COLORS = {
    "offline_greedy": "#2563eb",
    "offline_deterministic": "#16a34a",
    "offline_randomized": "#f97316",
    "simple_online_greedy": "#7c3aed",
    "online_randomized": "#dc2626",
    "online_randomized_level_greedy": "#0891b2",
}

ALGORITHM_X_OFFSETS = {
    "offline_greedy": -0.08,
    "offline_deterministic": 0.0,
    "offline_randomized": 0.08,
}

ALGORITHM_ORDER = {
    "offline_greedy": 0,
    "offline_deterministic": 1,
    "offline_randomized": 2,
    "simple_online_greedy": 3,
    "online_randomized": 4,
    "online_randomized_level_greedy": 5,
}

ALGORITHM_MARKERS = {
    "offline_deterministic": "o",
    "offline_greedy": "s",
    "offline_randomized": "^",
    "simple_online_greedy": "D",
    "online_randomized": "o",
    "online_randomized_level_greedy": "^",
}

ALGORITHM_SETTINGS = {
    "offline_greedy": "offline",
    "offline_deterministic": "offline",
    "offline_randomized": "offline",
    "simple_online_greedy": "online",
    "online_randomized": "online",
    "online_randomized_level_greedy": "online",
}

ALGORITHMS_BY_SETTING = {
    "offline": [
        ("offline_greedy", "Greedy"),
        ("offline_deterministic", "Deterministic"),
        ("offline_randomized", "Randomized"),
    ],
    "online": [
        ("online_randomized", "Randomized"),
        ("online_randomized_level_greedy", "CRS"),
    ],
}

OFFLINE_FAIRNESS_ALGORITHMS = [
    ("offline_greedy", "Global OPT"),
    ("offline_deterministic", "Deterministic"),
    ("offline_randomized", "Randomized"),
]

RATIO_ALGORITHMS_BY_SETTING = {
    "offline": [
        ("offline_deterministic", "Deterministic"),
        ("offline_randomized", "Randomized"),
    ],
    "online": [
        ("online_randomized", "Randomized"),
        ("online_randomized_level_greedy", "CRS"),
    ],
}

EXCLUDED_PLOT_ALGORITHMS: set[str] = {"simple_online_greedy"}

METRICS = {
    "fairness": "Fairness Ratio",
    "fraction_opt": "Fraction OPT",
    "inverse_ratio": "Ratio",
    "selected": "Selected",
}

CI_BAND_ALPHA = 0.22


DELTA_CACHE = {}


def parse_float(value: str) -> float:
    if value == "" or value is None:
        return math.nan
    return float(value)


def parse_int(value: str) -> int | None:
    if value == "" or value is None:
        return None
    return int(float(value))


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def offline_fairness_ratio(k: int) -> float:
    return float(k)


def offline_approximation_ratio(delta: float, k: int) -> float:
    denominator = delta + k - 1
    return safe_ratio(delta * k, denominator)


def online_fairness_ratio(delta: float, k: int) -> float:
    return k * math.log2(delta) / 2 if delta > 0 else math.nan


def online_approximation_ratio(delta: float, k: int) -> float:
    return math.log2(delta) / 2 if delta > 0 else math.nan


def compute_delta_from_input(input_file: str) -> float:
    if input_file in DELTA_CACHE:
        return DELTA_CACHE[input_file]

    path = Path(input_file)
    lengths = []
    if path.exists() and path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            length_col = next(
                (name for name in ["length", "Length", "run_time", "Run Time"] if name in fieldnames),
                None,
            )
            start_col = next(
                (name for name in ["start", "Start Time", "submit_time"] if name in fieldnames),
                None,
            )
            finish_col = next(
                (name for name in ["finish", "Finish Time"] if name in fieldnames),
                None,
            )
            for row in reader:
                try:
                    if length_col is not None:
                        length = float(row[length_col])
                    elif start_col is not None and finish_col is not None:
                        length = float(row[finish_col]) - float(row[start_col])
                    else:
                        continue
                except (TypeError, ValueError):
                    continue
                if length > 0:
                    lengths.append(length)

    delta = safe_ratio(max(lengths), min(lengths)) if lengths else math.nan
    DELTA_CACHE[input_file] = delta
    return delta


def finite_values(values) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def mean_and_ci95(values) -> tuple[float, float]:
    cleaned = finite_values(values)
    if not cleaned:
        return math.nan, math.nan
    if len(cleaned) == 1:
        return cleaned[0], 0.0
    sample_mean = mean(cleaned)
    sample_std = stdev(cleaned)
    n = len(cleaned)
    standard_error = sample_std / math.sqrt(n)
    critical_value = t.ppf(0.975, df=n - 1)
    return sample_mean, critical_value * standard_error


def parse_input_metadata(input_file: str) -> dict:
    parts = [part for part in re.split(r"[\\/]+", input_file) if part]
    filename = parts[-1] if parts else Path(input_file).name
    workload = parts[-2] if len(parts) >= 2 else "unknown"
    match = FILENAME_RE.search(filename)
    if not match:
        return {
            "workload": workload,
            "assignment_method": "unknown",
            "assignment_k": None,
            "assignment_seed": None,
        }

    return {
        "workload": workload,
        "assignment_method": match.group("assignment_method"),
        "assignment_k": int(match.group("k")),
        "assignment_seed": (
            int(match.group("assignment_seed"))
            if match.group("assignment_seed") is not None
            else None
        ),
    }


def load_results(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata = parse_input_metadata(row["input_file"])
            delta = parse_float(row.get("delta"))
            if not math.isfinite(delta):
                delta = compute_delta_from_input(row["input_file"])
            enriched = {
                **row,
                **metadata,
                "k": parse_int(row["k"]),
                "alpha": parse_float(row["alpha"]),
                "r": parse_int(row["r"]),
                "runs": parse_int(row["runs"]),
                "selected": parse_float(row["selected"]),
                "fairness": parse_float(row["fairness"]),
                "fraction_opt": parse_float(row["fraction_opt"]),
                "inverse_ratio": parse_float(row["inverse_ratio"]),
                "num_levels": parse_int(row["num_levels"]),
                "delta": delta,
            }
            rows.append(enriched)
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["workload"],
            row["assignment_method"],
            row["k"],
            row["algorithm"],
        )
        grouped[key].append(row)

    summary = []
    for (workload, assignment_method, k, algorithm), items in sorted(grouped.items()):
        # First-stage aggregation: within one workload, average across
        # group-assignment seeds. The ci95_* values here describe seed-level
        # variation for that workload and are kept for the summary CSV, but
        # they are not the uncertainty bands used in the main aggregated plots.
        mean_selected, ci95_selected = mean_and_ci95(row["selected"] for row in items)
        mean_fairness, ci95_fairness = mean_and_ci95(row["fairness"] for row in items)
        mean_fraction_opt, ci95_fraction_opt = mean_and_ci95(
            row["fraction_opt"] for row in items
        )
        mean_inverse_ratio, ci95_inverse_ratio = mean_and_ci95(
            row["inverse_ratio"] for row in items
        )
        mean_delta, _ = mean_and_ci95(row["delta"] for row in items)
        summary.append(
            {
                "workload": workload,
                "assignment_method": assignment_method,
                "k": k,
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                "setting": ALGORITHM_SETTINGS.get(algorithm, "unknown"),
                "num_instances": len(items),
                "mean_selected": mean_selected,
                "ci95_selected": ci95_selected,
                "mean_fairness": mean_fairness,
                "ci95_fairness": ci95_fairness,
                "mean_fraction_opt": mean_fraction_opt,
                "ci95_fraction_opt": ci95_fraction_opt,
                "mean_inverse_ratio": mean_inverse_ratio,
                "ci95_inverse_ratio": ci95_inverse_ratio,
                "mean_delta": mean_delta,
                "offline_fairness_ratio": offline_fairness_ratio(k),
                "offline_approximation_ratio": offline_approximation_ratio(mean_delta, k),
                "online_fairness_ratio": online_fairness_ratio(mean_delta, k),
                "online_approximation_ratio": online_approximation_ratio(mean_delta, k),
            }
        )
    return summary


def save_summary_csv(summary: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "assignment_method",
        "k",
        "algorithm",
        "algorithm_label",
        "setting",
        "num_instances",
        "mean_selected",
        "ci95_selected",
        "mean_fairness",
        "ci95_fairness",
        "mean_fraction_opt",
        "ci95_fraction_opt",
        "mean_inverse_ratio",
        "ci95_inverse_ratio",
        "mean_delta",
        "offline_fairness_ratio",
        "offline_approximation_ratio",
        "online_fairness_ratio",
        "online_approximation_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def points_for_chart(
    summary: list[dict],
    assignment_method: str,
    setting: str,
    metric: str,
) -> dict:
    values_by_algorithm_k = defaultdict(list)
    metric_key = f"mean_{metric}"

    for row in summary:
        if row["algorithm"] in EXCLUDED_PLOT_ALGORITHMS:
            continue
        if row["assignment_method"] != assignment_method:
            continue
        if row["setting"] != setting:
            continue
        values_by_algorithm_k[(row["algorithm"], row["k"])].append(row[metric_key])

    series = defaultdict(list)
    for (algorithm, k), values in values_by_algorithm_k.items():
        # Main-figure aggregation: compute the plotted mean and CI across
        # workload-level means from aggregate_rows(). This preserves workloads
        # as the independent units instead of flattening workload x seed rows.
        point_mean, point_ci95 = mean_and_ci95(values)
        series[algorithm].append((k, point_mean, point_ci95))

    return {
        algorithm: sorted(points)
        for algorithm, points in sorted(
            series.items(),
            key=lambda item: algorithm_sort_key(item[0]),
        )
    }


def points_for_workload_chart(
    summary: list[dict],
    workload: str,
    assignment_method: str,
    setting: str,
    metric: str,
) -> dict:
    metric_key = f"mean_{metric}"
    ci_key = f"ci95_{metric}"
    series = defaultdict(list)

    for row in summary:
        if row["algorithm"] in EXCLUDED_PLOT_ALGORITHMS:
            continue
        if row["workload"] != workload:
            continue
        if row["assignment_method"] != assignment_method:
            continue
        if row["setting"] != setting:
            continue
        series[row["algorithm"]].append(
            (row["k"], row[metric_key], row[ci_key])
        )

    return {
        algorithm: sorted(points)
        for algorithm, points in sorted(
            series.items(),
            key=lambda item: algorithm_sort_key(item[0]),
        )
    }


def theory_points_for_chart(
    summary: list[dict],
    assignment_method: str,
    setting: str,
    metric: str,
) -> list[tuple[int, float]]:
    ratio_key = theory_ratio_key(setting, metric)
    if ratio_key is None:
        return []

    values_by_k = defaultdict(list)
    for row in summary:
        if row["algorithm"] in EXCLUDED_PLOT_ALGORITHMS:
            continue
        if row["assignment_method"] != assignment_method:
            continue
        if ALGORITHM_SETTINGS.get(row["algorithm"], "unknown") != setting:
            continue
        values_by_k[row["k"]].append(row[ratio_key])

    points = []
    for k, values in values_by_k.items():
        point_mean, _ = mean_and_ci95(values)
        points.append((k, point_mean))
    return sorted(points)


def theory_ratio_key(setting: str, metric: str) -> str | None:
    if metric == "fairness" and setting == "online":
        return "online_fairness_ratio"
    if metric == "inverse_ratio" and setting == "offline":
        return "offline_approximation_ratio"
    if metric == "inverse_ratio" and setting == "online":
        return "online_approximation_ratio"
    return None


def theory_ratio_label(setting: str, metric: str) -> str | None:
    if metric == "fairness" and setting == "online":
        return "k*log2(delta)/2"
    if metric == "inverse_ratio" and setting == "offline":
        return "delta*k/(delta+k-1)"
    if metric == "inverse_ratio" and setting == "online":
        return "log2(delta)/2"
    return None


def format_assignment_method(value: str) -> str:
    return value.replace("_", " ").title()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def format_workload_label(value: str) -> str:
    return re.sub(r"-\d{4}$", "", value)


def algorithm_sort_key(algorithm: str) -> tuple[int, str]:
    return (ALGORITHM_ORDER.get(algorithm, len(ALGORITHM_ORDER)), algorithm)


def chart_title(metric_label: str, assignment_method: str, setting: str, metric: str) -> str:
    method_label = format_assignment_method(assignment_method)
    if setting == "offline" and metric == "fairness":
        return f"Fairness Ratio under {method_label} Group Assignment"
    if setting == "offline" and metric == "inverse_ratio":
        return f"Approximation Ratio under {method_label} Group Assignment"
    if setting == "online" and metric == "inverse_ratio":
        return f"Competitive Ratio under {method_label} Group Assignment"
    return f"{metric_label} by k ({assignment_method}, {setting}, mean +/- 95% CI)"


def chart_y_label(metric_label: str, setting: str, metric: str) -> str:
    if setting == "offline" and metric == "fairness":
        return r"Fairness Ratio ($\max_g\ \mathrm{OPT}_g / \mathrm{ALG}_g$)"
    if setting == "offline" and metric == "inverse_ratio":
        return "Approximation Ratio"
    if setting == "online" and metric == "inverse_ratio":
        return "Competitive Ratio"
    return metric_label


def save_offline_fairness_small_multiples(
    series: dict[str, list[tuple[int, float, float]]],
    assignment_method: str,
    output_path: Path,
    title_prefix: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if assignment_method == "length_quantile":
        save_length_quantile_offline_fairness_figure(
            series=series,
            assignment_method=assignment_method,
            output_path=output_path,
            title_prefix=title_prefix,
        )
        return

    algorithms = OFFLINE_FAIRNESS_ALGORITHMS
    ci_values = [
        value
        for algorithm, _ in algorithms
        if algorithm in series
        for _, mean_value, ci95 in series[algorithm]
        for value in (mean_value - ci95, mean_value + ci95)
        if math.isfinite(value)
    ]
    if not ci_values:
        return

    log_scale = assignment_method == "containment_quantile"
    if log_scale:
        positive_ci_values = [value for value in ci_values if value > 0]
        if not positive_ci_values:
            return
        y_min = max(min(positive_ci_values) * 0.82, 1e-12)
        y_max = max(positive_ci_values) * 1.22
    else:
        y_min = max(1.0, min(ci_values) - max((max(ci_values) - min(ci_values)) * 0.10, 0.05))
        y_max = max(ci_values) + max((max(ci_values) - min(ci_values)) * 0.10, 0.05)
    if y_max <= y_min:
        y_max = y_min * 1.1 if log_scale else y_min + 0.1

    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(max(3.2 * len(algorithms), 6.4), 3.1),
        dpi=300,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(algorithms) == 1:
        axes = [axes]

    for panel_index, (ax, (algorithm, panel_title)) in enumerate(zip(axes, algorithms)):
        ax.set_axisbelow(True)
        color = ALGORITHM_COLORS.get(algorithm, "#2563eb")
        marker = ALGORITHM_MARKERS.get(algorithm, "o")

        points = [
            (k, mean_value, ci95)
            for k, mean_value, ci95 in series[algorithm]
            if math.isfinite(mean_value) and math.isfinite(ci95)
        ]
        if points:
            x_values = [k for k, _, _ in points]
            mean_values = [mean_value for _, mean_value, _ in points]
            lower_values = [mean_value - ci95 for _, mean_value, ci95 in points]
            upper_values = [mean_value + ci95 for _, mean_value, ci95 in points]
            ax.fill_between(
                x_values,
                lower_values,
                upper_values,
                color=color,
                alpha=CI_BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x_values,
                mean_values,
                f"-{marker}",
                color=color,
                linewidth=2.5,
                markersize=5.8,
                zorder=2,
            )

        ax.set_xticks(list(range(2, 11)))
        ax.set_xlim(1.7, 10.3)
        ax.set_ylim(y_min, y_max)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, which="major", alpha=0.22, linewidth=0.7)
        if log_scale:
            ax.grid(True, which="minor", axis="y", alpha=0.10, linewidth=0.55)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlabel("k", fontsize=10)
        if panel_index == 0:
            ax.set_ylabel(
                r"Fairness Ratio ($\max_g\ \mathrm{OPT}_g / \mathrm{ALG}_g$)",
                fontsize=10,
            )

    fig.savefig(output_path)
    plt.close(fig)


def save_workload_fairness_grid(
    series_by_workload: dict[str, dict[str, list[tuple[int, float, float]]]],
    assignment_method: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workloads = sorted(series_by_workload)[:4]
    algorithms = OFFLINE_FAIRNESS_ALGORITHMS
    if not workloads:
        return

    log_scale = assignment_method == "length_quantile"
    y_values = []
    for workload in workloads:
        series = series_by_workload[workload]
        for algorithm, _ in algorithms:
            for _, mean_value, ci95 in series.get(algorithm, []):
                for value in (mean_value - ci95, mean_value, mean_value + ci95):
                    if math.isfinite(value) and (not log_scale or value > 0):
                        y_values.append(value)

    if not y_values:
        return

    if log_scale:
        y_min = max(min(y_values) * 0.82, 1e-12)
        y_max = max(y_values) * 1.22
    else:
        y_min = max(1.0, min(y_values) - max((max(y_values) - min(y_values)) * 0.10, 0.05))
        y_max = max(y_values) + max((max(y_values) - min(y_values)) * 0.10, 0.05)
    if y_max <= y_min:
        y_max = y_min * 1.1 if log_scale else y_min + 0.1

    fig, axes = plt.subplots(
        len(workloads),
        len(algorithms),
        figsize=(9.8, 8.6),
        dpi=300,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(workloads) == 1:
        axes = [axes]

    for row_index, workload in enumerate(workloads):
        series = series_by_workload[workload]
        for column_index, (algorithm, label) in enumerate(algorithms):
            ax = axes[row_index][column_index]
            ax.set_axisbelow(True)
            points = [
                (k, mean_value, ci95)
                for k, mean_value, ci95 in series.get(algorithm, [])
                if math.isfinite(mean_value) and math.isfinite(ci95)
            ]
            _plot_mean_with_ci_band(
                ax,
                points,
                algorithm,
                label,
                log_scale=log_scale,
                show_ci=True,
            )
            if log_scale:
                ax.set_yscale("log")
            if row_index == 0:
                ax.set_title(label, fontsize=11)
            if column_index == 0:
                ax.set_ylabel(format_workload_label(workload), fontsize=10)
            if row_index == len(workloads) - 1:
                ax.set_xlabel("k", fontsize=10)
            ax.set_xticks(list(range(2, 11)))
            ax.set_xlim(1.7, 10.3)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, which="major", alpha=0.22, linewidth=0.7)
            if log_scale:
                ax.grid(True, which="minor", axis="y", alpha=0.10, linewidth=0.55)
            ax.tick_params(axis="both", labelsize=8)

    fig.savefig(output_path)
    plt.close(fig)


def _finite_fairness_points(
    series: dict[str, list[tuple[int, float, float]]],
    algorithm: str,
    require_positive_mean: bool = False,
) -> list[tuple[int, float, float]]:
    points = []
    for k, mean_value, ci95 in series.get(algorithm, []):
        if not (math.isfinite(mean_value) and math.isfinite(ci95)):
            continue
        if require_positive_mean and mean_value <= 0:
            continue
        points.append((k, mean_value, ci95))
    return points


def _plot_mean_with_ci_band(
    ax,
    points: list[tuple[int, float, float]],
    algorithm: str,
    label: str,
    *,
    log_scale: bool = False,
    show_ci: bool = True,
) -> list[float]:
    if not points:
        return []

    color = ALGORITHM_COLORS.get(algorithm, "#2563eb")
    marker = ALGORITHM_MARKERS.get(algorithm, "o")
    x_values = [k for k, _, _ in points]
    mean_values = [mean_value for _, mean_value, _ in points]
    plotted_y_values = [mean_value for mean_value in mean_values if math.isfinite(mean_value)]

    band_points = []
    if show_ci:
        band_points = [
            (k, mean_value - ci95, mean_value + ci95)
            for k, mean_value, ci95 in points
            if math.isfinite(mean_value - ci95)
            and math.isfinite(mean_value + ci95)
            and (not log_scale or mean_value - ci95 > 0)
        ]
    if band_points:
        band_x = [k for k, _, _ in band_points]
        lower_values = [lower for _, lower, _ in band_points]
        upper_values = [upper for _, _, upper in band_points]
        ax.fill_between(
            band_x,
            lower_values,
            upper_values,
            color=color,
            alpha=CI_BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )
        plotted_y_values.extend(lower_values)
        plotted_y_values.extend(upper_values)

    ax.plot(
        x_values,
        mean_values,
        f"-{marker}",
        color=color,
        linewidth=2.3,
        markersize=5.2,
        label=label,
        zorder=2,
    )
    return [value for value in plotted_y_values if math.isfinite(value)]


def save_metric_small_multiples(
    series: dict[str, list[tuple[int, float, float]]],
    assignment_method: str,
    setting: str,
    metric: str,
    metric_label: str,
    output_path: Path,
    title_prefix: str | None = None,
    lower_bound: float | None = None,
    algorithms: list[tuple[str, str]] | None = None,
    theory_points: list[tuple[int, float]] | None = None,
    theory_label: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = algorithms or ALGORITHMS_BY_SETTING.get(setting, [])
    if not algorithms:
        return

    ci_values = [
        value
        for algorithm, _ in algorithms
        if algorithm in series
        for _, mean_value, ci95 in series[algorithm]
        for value in (mean_value - ci95, mean_value + ci95)
        if math.isfinite(value)
    ]
    if not ci_values:
        return
    filtered_theory_points = [
        (k, value)
        for k, value in (theory_points or [])
        if math.isfinite(value)
    ]
    theory_values = [value for _, value in filtered_theory_points]
    y_values = ci_values + theory_values

    y_min = min(y_values) - max((max(y_values) - min(y_values)) * 0.10, 0.05)
    if lower_bound is not None:
        y_min = max(lower_bound, y_min)
    y_max = max(y_values) + max((max(y_values) - min(y_values)) * 0.10, 0.05)
    if y_max <= y_min:
        y_max = y_min + 0.1

    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(max(3.2 * len(algorithms), 6.4), 3.1),
        dpi=300,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(algorithms) == 1:
        axes = [axes]

    for panel_index, (ax, (algorithm, panel_title)) in enumerate(zip(axes, algorithms)):
        ax.set_axisbelow(True)
        points = [
            (k, mean_value, ci95)
            for k, mean_value, ci95 in series.get(algorithm, [])
            if math.isfinite(mean_value) and math.isfinite(ci95)
        ]
        _plot_mean_with_ci_band(
            ax,
            points,
            algorithm,
            panel_title,
            show_ci=True,
        )
        if filtered_theory_points:
            ax.plot(
                [k for k, _ in filtered_theory_points],
                [value for _, value in filtered_theory_points],
                "--",
                linewidth=1.8,
                color="#111827",
                label=theory_label or "Theory ratio",
                zorder=1,
            )
        ax.set_xticks(list(range(2, 11)))
        ax.set_xlim(1.7, 10.3)
        ax.set_ylim(y_min, y_max)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlabel("k", fontsize=10)
        if panel_index == 0:
            ax.set_ylabel(chart_y_label(metric_label, setting, metric), fontsize=10)
        ax.legend(loc="best", frameon=False, fontsize=8)

    fig.savefig(output_path)
    plt.close(fig)


def save_length_quantile_offline_fairness_figure(
    series: dict[str, list[tuple[int, float, float]]],
    assignment_method: str,
    output_path: Path,
    title_prefix: str | None = None,
) -> None:
    algorithms = OFFLINE_FAIRNESS_ALGORITHMS

    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(9.2, 3.1),
        dpi=300,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    log_y_values = []
    for panel_index, (ax, (algorithm, panel_title)) in enumerate(zip(axes, algorithms)):
        ax.set_axisbelow(True)
        points = _finite_fairness_points(series, algorithm, require_positive_mean=True)
        log_y_values.extend(
            _plot_mean_with_ci_band(
                ax,
                points,
                algorithm,
                panel_title,
                log_scale=True,
                show_ci=True,
            )
        )
        ax.set_yscale("log")
        ax.set_xticks(list(range(2, 11)))
        ax.set_xlim(1.7, 10.3)
        ax.set_xlabel("k", fontsize=10)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, which="major", alpha=0.23, linewidth=0.75)
        ax.grid(True, which="minor", axis="y", alpha=0.10, linewidth=0.55)
        ax.tick_params(axis="both", labelsize=9)
        if panel_index == 0:
            ax.set_ylabel(
                r"Fairness Ratio ($\max_g\ \mathrm{OPT}_g / \mathrm{ALG}_g$)",
                fontsize=10,
            )

    positive_log_values = [value for value in log_y_values if value > 0]
    if positive_log_values:
        axes[0].set_ylim(
            min(positive_log_values) * 0.82,
            max(positive_log_values) * 1.22,
        )

    fig.savefig(output_path)
    plt.close(fig)


def save_offline_fairness_deterministic_randomized_detail(
    series: dict[str, list[tuple[int, float, float]]],
    assignment_method: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    algorithms = [
        ("offline_deterministic", "Deterministic"),
        ("offline_randomized", "Randomized"),
    ]
    log_scale = False
    y_values = []

    for algorithm, _ in algorithms:
        for _, mean_value, ci95 in series.get(algorithm, []):
            for value in (mean_value - ci95, mean_value, mean_value + ci95):
                if math.isfinite(value) and (not log_scale or value > 0):
                    y_values.append(value)

    if not y_values:
        return

    if log_scale:
        y_min = max(min(y_values) * 0.82, 1e-12)
        y_max = max(y_values) * 1.22
    else:
        y_min = max(
            1.0,
            min(y_values) - max((max(y_values) - min(y_values)) * 0.12, 0.05),
        )
        y_max = max(y_values) + max((max(y_values) - min(y_values)) * 0.12, 0.05)
    if y_max <= y_min:
        y_max = y_min * 1.1 if log_scale else y_min + 0.1

    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(max(3.2 * len(algorithms), 6.4), 3.1),
        dpi=300,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(algorithms) == 1:
        axes = [axes]

    for panel_index, (ax, (algorithm, panel_title)) in enumerate(zip(axes, algorithms)):
        ax.set_axisbelow(True)
        points = _finite_fairness_points(series, algorithm)
        _plot_mean_with_ci_band(
            ax,
            points,
            algorithm,
            panel_title,
            log_scale=log_scale,
            show_ci=True,
        )
        ax.set_ylim(y_min, y_max)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(list(range(2, 11)))
        ax.set_xlim(1.7, 10.3)
        ax.set_xlabel("k", fontsize=10)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, which="major", alpha=0.23, linewidth=0.75)
        if log_scale:
            ax.grid(True, which="minor", axis="y", alpha=0.10, linewidth=0.55)
        ax.tick_params(axis="both", labelsize=9)
        if panel_index == 0:
            ax.set_ylabel(
                r"Fairness Ratio ($\max_g\ \mathrm{OPT}_g / \mathrm{ALG}_g$)",
                fontsize=10,
            )

    fig.savefig(output_path)
    plt.close(fig)


def save_assignment_method_fairness_comparison(
    series_by_assignment_method: dict[str, dict[str, list[tuple[int, float, float]]]],
    algorithm: str,
    algorithm_label: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assignment_methods = [
        ("length_quantile", "Length Quantile", "#16a34a", "o", -0.04),
        ("containment_quantile", "Containment Quantile", "#f97316", "^", 0.04),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=300, constrained_layout=True)
    empirical_y_values = []

    for assignment_method, label, color, marker, offset in assignment_methods:
        points = [
            (k, mean_value, ci95)
            for k, mean_value, ci95 in series_by_assignment_method
            .get(assignment_method, {})
            .get(algorithm, [])
            if math.isfinite(mean_value) and math.isfinite(ci95)
        ]
        if not points:
            continue

        x_values = [k + offset for k, _, _ in points]
        mean_values = [mean_value for _, mean_value, _ in points]
        ci_values = [ci95 for _, _, ci95 in points]
        empirical_y_values.extend(
            value
            for _, mean_value, ci95 in points
            for value in (mean_value - ci95, mean_value + ci95)
            if math.isfinite(value)
        )
        errorbar = ax.errorbar(
            x_values,
            mean_values,
            yerr=ci_values,
            fmt=f"-{marker}",
            linewidth=2.2,
            elinewidth=0.9,
            markersize=5.0,
            capsize=2.8,
            capthick=0.9,
            color=color,
            ecolor=color,
            label=label,
            zorder=2,
        )
        for capline in errorbar[1]:
            capline.set_alpha(0.55)
        for barline_collection in errorbar[2]:
            barline_collection.set_alpha(0.45)

    if empirical_y_values:
        y_min = max(
            1.0,
            min(empirical_y_values)
            - max((max(empirical_y_values) - min(empirical_y_values)) * 0.12, 0.05),
        )
        y_max = max(empirical_y_values) + max(
            (max(empirical_y_values) - min(empirical_y_values)) * 0.12,
            0.05,
        )
        if y_max <= y_min:
            y_max = y_min + 0.1
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel(
        r"Fairness Ratio ($\max_g\ \mathrm{OPT}_g / \mathrm{ALG}_g$)",
        fontsize=10,
    )
    ax.set_title(algorithm_label, fontsize=11)
    ax.set_xticks(list(range(2, 11)))
    ax.set_xlim(1.7, 10.3)
    ax.grid(True, alpha=0.24, linewidth=0.75)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.savefig(output_path)
    plt.close(fig)


def save_line_chart(
    series: dict[str, list[tuple[int, float, float]]],
    title: str,
    y_label: str,
    output_path: Path,
    theory_points: list[tuple[int, float]] | None = None,
    theory_label: str | None = None,
    zoom_to_empirical: bool = False,
    lower_bound: float | None = None,
    optimal_reference: bool = False,
    x_ticks: list[int] | None = None,
    log_scale: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    all_k_values = sorted({x for points in series.values() for x, _, _ in points})
    empirical_y_values = []
    positive_floor = lower_bound if lower_bound and lower_bound > 0 else 1e-12

    for algorithm, points in series.items():
        filtered_points = [
            (x, y, ci95)
            for x, y, ci95 in points
            if math.isfinite(y)
            and math.isfinite(ci95)
            and (not log_scale or y > 0)
        ]
        if not filtered_points:
            continue
        offset = ALGORITHM_X_OFFSETS.get(algorithm, 0.0)
        x_values = [x + offset for x, _, _ in filtered_points]
        y_values = [y for _, y, _ in filtered_points]
        if log_scale:
            lower_values = [max(y - ci95, positive_floor) for _, y, ci95 in filtered_points]
            upper_values = [y + ci95 for _, y, ci95 in filtered_points]
            yerr = [
                [y - lower for y, lower in zip(y_values, lower_values)],
                [upper - y for y, upper in zip(y_values, upper_values)],
            ]
            empirical_y_values.extend(
                value
                for lower, y, upper in zip(lower_values, y_values, upper_values)
                for value in (lower, y, upper)
                if math.isfinite(value) and value > 0
            )
        else:
            ci_values = [ci95 for _, _, ci95 in filtered_points]
            yerr = ci_values
            empirical_y_values.extend(
                value
                for _, y, ci95 in filtered_points
                for value in (y - ci95, y + ci95)
                if math.isfinite(value)
            )
        color = ALGORITHM_COLORS.get(algorithm)
        marker = ALGORITHM_MARKERS.get(algorithm, "o")
        errorbar = ax.errorbar(
            x_values,
            y_values,
            yerr=yerr,
            fmt="-",
            marker=marker,
            linewidth=2.4,
            elinewidth=1.2,
            markersize=4,
            capsize=4,
            capthick=1.2,
            color=color,
            ecolor=color,
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
        )
        for capline in errorbar[1]:
            capline.set_alpha(0.55)
        for barline_collection in errorbar[2]:
            barline_collection.set_alpha(0.45)

    if theory_points:
        filtered_theory_points = [
            (x, y)
            for x, y in theory_points
            if math.isfinite(y) and (not log_scale or y > 0)
        ]
        if filtered_theory_points:
            x_values = [x for x, _ in filtered_theory_points]
            y_values = [y for _, y in filtered_theory_points]
            empirical_y_values.extend(y_values)
            ax.plot(
                x_values,
                y_values,
                "--",
                linewidth=2.0,
                color="#111827",
                label=theory_label or "Theory ratio",
            )

    if optimal_reference:
        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.2,
            color="#6b7280",
            alpha=0.75,
            label="Optimal (=1)",
        )

    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xticks(x_ticks if x_ticks is not None else all_k_values)
    ax.tick_params(axis="both", labelsize=10)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    if log_scale:
        ax.grid(True, which="minor", axis="y", alpha=0.10, linewidth=0.55)

    if log_scale and empirical_y_values:
        positive_values = [value for value in empirical_y_values if value > 0]
        if positive_values:
            y_min = max(min(positive_values) * 0.82, positive_floor)
            y_max = max(positive_values) * 1.22
            if y_max <= y_min:
                y_max = y_min * 1.1
            ax.set_ylim(y_min, y_max)
    elif zoom_to_empirical and empirical_y_values:
        empirical_min = min(empirical_y_values)
        empirical_max = max(empirical_y_values)
        data_range = empirical_max - empirical_min
        margin = max(data_range * 0.12, 0.05)
        y_min = empirical_min - margin
        if lower_bound is not None:
            y_min = max(lower_bound, y_min)
        y_max = empirical_max + margin
        if y_max <= y_min:
            y_max = y_min + 0.1
        ax.set_ylim(y_min, y_max)

    if optimal_reference:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
    else:
        ax.legend(loc="best", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_charts(summary: list[dict], output_dir: Path) -> list[Path]:
    chart_paths = []
    assignment_methods = sorted({row["assignment_method"] for row in summary})
    workloads = sorted({row["workload"] for row in summary})
    for assignment_method in assignment_methods:
        for setting in ["offline", "online"]:
            for metric, label in METRICS.items():
                series = points_for_chart(summary, assignment_method, setting, metric)
                if not series:
                    continue
                output_metric = metric
                if metric == "inverse_ratio" and setting == "offline":
                    output_metric = "approximation_ratio"
                elif metric == "inverse_ratio" and setting == "online":
                    output_metric = "competitive_ratio"
                output_path = output_dir / f"{assignment_method}_{setting}_{output_metric}_by_k.png"
                if setting == "offline" and metric == "fairness":
                    save_offline_fairness_small_multiples(
                        series=series,
                        assignment_method=assignment_method,
                        output_path=output_path,
                    )
                    chart_paths.append(output_path)
                    if assignment_method in {"containment_quantile", "length_quantile"}:
                        detail_output_path = (
                            output_dir
                            / f"{assignment_method}_offline_fairness_deterministic_randomized_by_k.png"
                        )
                        save_offline_fairness_deterministic_randomized_detail(
                            series=series,
                            assignment_method=assignment_method,
                            output_path=detail_output_path,
                        )
                        chart_paths.append(detail_output_path)
                    continue

                if metric == "inverse_ratio":
                    theory_points = (
                        theory_points_for_chart(
                            summary,
                            assignment_method,
                            setting,
                            metric,
                        )
                        if setting == "online"
                        else None
                    )
                    theory_label = theory_ratio_label(setting, metric) if setting == "online" else None
                    save_metric_small_multiples(
                        series=series,
                        assignment_method=assignment_method,
                        setting=setting,
                        metric=metric,
                        metric_label=label,
                        output_path=output_path,
                        lower_bound=1.0,
                        algorithms=RATIO_ALGORITHMS_BY_SETTING.get(setting),
                        theory_points=theory_points,
                        theory_label=theory_label,
                    )
                    chart_paths.append(output_path)
                    continue

                theory_points = theory_points_for_chart(
                    summary,
                    assignment_method,
                    setting,
                    metric,
                )
                theory_label = theory_ratio_label(setting, metric)
                is_offline_fairness = setting == "offline" and metric == "fairness"
                use_log_scale = (
                    assignment_method == "containment_quantile"
                    and setting == "online"
                    and metric == "fairness"
                )
                save_line_chart(
                    series=series,
                    title=chart_title(label, assignment_method, setting, metric),
                    y_label=chart_y_label(label, setting, metric),
                    output_path=output_path,
                    theory_points=theory_points,
                    theory_label=theory_label,
                    zoom_to_empirical=is_offline_fairness,
                    lower_bound=1.0 if is_offline_fairness or use_log_scale else None,
                    optimal_reference=False,
                    x_ticks=list(range(2, 11)) if is_offline_fairness else None,
                    log_scale=use_log_scale,
                )
                chart_paths.append(output_path)

        workload_fairness_series = {}
        for workload in workloads:
            series = points_for_workload_chart(
                summary,
                workload,
                assignment_method,
                setting="offline",
                metric="fairness",
            )
            if not series:
                continue
            workload_fairness_series[workload] = series
            output_path = (
                output_dir
                / "by_workload"
                / f"{safe_filename(workload)}_{assignment_method}_offline_fairness_by_k.png"
            )
            save_offline_fairness_small_multiples(
                series=series,
                assignment_method=assignment_method,
                output_path=output_path,
                title_prefix=workload,
            )
            chart_paths.append(output_path)
        if workload_fairness_series:
            output_path = (
                output_dir
                / "by_workload"
                / f"{assignment_method}_offline_fairness_grid_by_workload.png"
            )
            save_workload_fairness_grid(
                series_by_workload=workload_fairness_series,
                assignment_method=assignment_method,
                output_path=output_path,
            )
            chart_paths.append(output_path)
    comparison_series = {
        assignment_method: points_for_chart(
            summary,
            assignment_method,
            setting="offline",
            metric="fairness",
        )
        for assignment_method in ["length_quantile", "containment_quantile"]
    }
    for algorithm, algorithm_label, output_name in [
        (
            "offline_deterministic",
            "Offline Deterministic",
            "length_containment_offline_deterministic_fairness_by_k.png",
        ),
        (
            "offline_randomized",
            "Offline Randomized",
            "length_containment_offline_randomized_fairness_by_k.png",
        ),
    ]:
        output_path = output_dir / output_name
        save_assignment_method_fairness_comparison(
            series_by_assignment_method=comparison_series,
            algorithm=algorithm,
            algorithm_label=algorithm_label,
            output_path=output_path,
        )
        chart_paths.append(output_path)
    return chart_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts from experiment result CSV")
    parser.add_argument("--input", type=str, default="experiment_results.csv")
    parser.add_argument("--output-dir", type=str, default="figures_matplotlib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_results(args.input)
    summary = aggregate_rows(rows)
    summary_path = output_dir / "summary_by_algorithm.csv"
    save_summary_csv(summary, summary_path)
    chart_paths = make_charts(summary, output_dir)

    print(f"Loaded rows: {len(rows)}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved charts: {len(chart_paths)}")
    for path in chart_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
