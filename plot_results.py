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


FILENAME_RE = re.compile(
    r"_interval_(?P<assignment_method>.+)_(?P<k>\d+)groups_seed(?P<assignment_seed>\d+)\.csv$"
)

ALGORITHM_LABELS = {
    "offline_greedy": "Offline Greedy",
    "offline_deterministic": "Offline Deterministic",
    "offline_randomized": "Offline Randomized",
    "simple_online_greedy": "Simple Online Greedy",
    "online_randomized": "Online Randomized",
    "online_randomized_level_greedy": "Online Randomized Level",
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
    "offline_deterministic": -0.08,
    "offline_greedy": 0.0,
    "offline_randomized": 0.08,
}

ALGORITHM_SETTINGS = {
    "offline_greedy": "offline",
    "offline_deterministic": "offline",
    "offline_randomized": "offline",
    "simple_online_greedy": "online",
    "online_randomized": "online",
    "online_randomized_level_greedy": "online",
}

METRICS = {
    "fairness": "Fairness",
    "fraction_opt": "Fraction OPT",
    "inverse_ratio": "OPT / ALG",
    "selected": "Selected",
}


def parse_float(value: str) -> float:
    if value == "" or value is None:
        return math.nan
    return float(value)


def parse_int(value: str) -> int | None:
    if value == "" or value is None:
        return None
    return int(float(value))


def finite_values(values) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def mean_and_ci95(values) -> tuple[float, float]:
    cleaned = finite_values(values)
    if not cleaned:
        return math.nan, math.nan
    if len(cleaned) == 1:
        return cleaned[0], 0.0
    return mean(cleaned), 1.96 * stdev(cleaned) / math.sqrt(len(cleaned))


def parse_input_metadata(input_file: str) -> dict:
    path = Path(input_file)
    workload = path.parent.name or "unknown"
    match = FILENAME_RE.search(path.name)
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
        "assignment_seed": int(match.group("assignment_seed")),
    }


def load_results(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata = parse_input_metadata(row["input_file"])
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
        mean_selected, ci95_selected = mean_and_ci95(row["selected"] for row in items)
        mean_fairness, ci95_fairness = mean_and_ci95(row["fairness"] for row in items)
        mean_fraction_opt, ci95_fraction_opt = mean_and_ci95(
            row["fraction_opt"] for row in items
        )
        mean_inverse_ratio, ci95_inverse_ratio = mean_and_ci95(
            row["inverse_ratio"] for row in items
        )
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
        if row["assignment_method"] != assignment_method:
            continue
        if row["setting"] != setting:
            continue
        values_by_algorithm_k[(row["algorithm"], row["k"])].append(row[metric_key])

    series = defaultdict(list)
    for (algorithm, k), values in values_by_algorithm_k.items():
        point_mean, point_ci95 = mean_and_ci95(values)
        series[algorithm].append((k, point_mean, point_ci95))

    return {
        algorithm: sorted(points)
        for algorithm, points in sorted(series.items())
    }


def save_line_chart(
    series: dict[str, list[tuple[int, float, float]]],
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    all_k_values = sorted({x for points in series.values() for x, _, _ in points})

    for algorithm, points in series.items():
        filtered_points = [
            (x, y, ci95)
            for x, y, ci95 in points
            if math.isfinite(y) and math.isfinite(ci95)
        ]
        if not filtered_points:
            continue
        offset = ALGORITHM_X_OFFSETS.get(algorithm, 0.0)
        x_values = [x + offset for x, _, _ in filtered_points]
        y_values = [y for _, y, _ in filtered_points]
        ci_values = [ci95 for _, _, ci95 in filtered_points]
        color = ALGORITHM_COLORS.get(algorithm)
        ax.errorbar(
            x_values,
            y_values,
            yerr=ci_values,
            fmt="-o",
            marker="o",
            linewidth=2.4,
            elinewidth=1.1,
            markersize=4,
            capsize=4,
            capthick=1.1,
            color=color,
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
        )

    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel(y_label)
    ax.set_xticks(all_k_values)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_charts(summary: list[dict], output_dir: Path) -> list[Path]:
    chart_paths = []
    assignment_methods = sorted({row["assignment_method"] for row in summary})
    for assignment_method in assignment_methods:
        for setting in ["offline", "online"]:
            for metric, label in METRICS.items():
                series = points_for_chart(summary, assignment_method, setting, metric)
                if not series:
                    continue
                output_path = output_dir / f"{assignment_method}_{setting}_{metric}_by_k.png"
                save_line_chart(
                    series=series,
                    title=f"{label} by k ({assignment_method}, {setting}, mean +/- 95% CI)",
                    y_label=label,
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
