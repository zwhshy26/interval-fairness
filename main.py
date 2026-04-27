import argparse
import csv
import json
import math
import random
from bisect import bisect_right
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean


@dataclass
class Interval:
    start: int
    length: int
    group: int | None = None
    section: int | None = None
    accepted: bool = False

    @property
    def finish(self) -> int:
        return self.start + self.length

    def copy(self) -> "Interval":
        return Interval(
            start=self.start,
            length=self.length,
            group=self.group,
            section=self.section,
            accepted=self.accepted,
        )


@dataclass
class FairDPResult:
    feasible: bool
    max_cardinality: int
    quotas: list[int]
    opt_per_group: list[int]
    selected_intervals: list[tuple[int, int, int]] | None = None


@dataclass
class BestFairnessResult:
    best_beta: float
    best_quota_vector: list[int]
    max_cardinality_at_best_beta: int
    opt_per_group: list[int]
    selected_intervals: list[tuple[int, int, int]] | None = None


def _as_interval_tuple(interval: Interval | tuple[int, int, int]) -> tuple[int, int, int]:
    """Return an interval as (start, end, group)."""
    if isinstance(interval, Interval):
        if interval.group is None:
            raise ValueError("All intervals must have a group id")
        return (interval.start, interval.finish, interval.group)

    if len(interval) != 3:
        raise ValueError("Tuple intervals must have shape (start, end, group)")
    start, end, group = interval
    return (int(start), int(end), int(group))


def _normalize_intervals(
    intervals: list[Interval | tuple[int, int, int]], k: int
) -> list[tuple[int, int, int]]:
    """Validate and normalize intervals to the tuple representation used by the DP."""
    if k < 0:
        raise ValueError("k must be non-negative")

    normalized = [_as_interval_tuple(interval) for interval in intervals]
    for start, end, group in normalized:
        if end < start:
            raise ValueError(f"Interval has end before start: {(start, end, group)}")
        if not 0 <= group < k:
            raise ValueError(
                f"Group id {group} is outside the expected range [0, {k - 1}]"
            )
    return normalized


def compute_group_opts(
    intervals: list[Interval | tuple[int, int, int]], k: int
) -> list[int]:
    """
    Compute OPT_g for every group using classical greedy interval scheduling.

    For cardinality interval scheduling, sorting by end time and taking the next
    compatible interval is optimal. Compatibility uses left-closed, right-open
    intervals, so [a, b) and [b, c) are compatible.
    """
    normalized = _normalize_intervals(intervals, k)
    opt_per_group = [0] * k

    for group in range(k):
        group_intervals = sorted(
            (interval for interval in normalized if interval[2] == group),
            key=lambda interval: (interval[1], interval[0]),
        )
        current_end = None
        for start, end, _ in group_intervals:
            if current_end is None or current_end <= start:
                opt_per_group[group] += 1
                current_end = end

    return opt_per_group


def sort_intervals_and_compute_p(
    intervals: list[Interval | tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], list[int]]:
    """
    Sort intervals by end time and compute p for each sorted interval.

    The returned p values are DP row numbers, not interval indices: p[i] is the
    number of intervals ending at or before sorted_intervals[i].start among the
    intervals before i. This lets a transition for row i + 1 jump directly to
    DP[p[i]].
    """
    sorted_intervals = sorted(
        [_as_interval_tuple(interval) for interval in intervals],
        key=lambda interval: (interval[1], interval[0]),
    )
    ends = [end for _, end, _ in sorted_intervals]
    p = [
        bisect_right(ends, start, hi=i)
        for i, (start, _, _) in enumerate(sorted_intervals)
    ]
    return sorted_intervals, p


def generate_quota_vectors(opt_per_group: list[int]):
    """Yield every integer quota vector a with 0 <= a_g <= OPT_g."""
    ranges = [range(opt + 1) for opt in opt_per_group]
    yield from product(*ranges)


def _quota_after_taking(quota: tuple[int, ...], group: int) -> tuple[int, ...]:
    """Reduce the remaining quota for group by one, truncated at zero."""
    if quota[group] == 0:
        return quota
    updated = list(quota)
    updated[group] -= 1
    return tuple(updated)


def _run_quota_dp(
    sorted_intervals: list[tuple[int, int, int]],
    p: list[int],
    quota_vectors: list[tuple[int, ...]],
    reconstruct_target: tuple[int, ...] | None = None,
) -> tuple[list[dict[tuple[int, ...], int]], list[dict[tuple[int, ...], tuple]] | None]:
    """
    Dynamic program for max-cardinality schedules under every requested quota.

    DP[i][a] is the maximum number of intervals selectable from the first i
    end-time-sorted intervals while still satisfying quota vector a. Missing
    entries represent invalid states, i.e. negative infinity.

    Recurrence for interval i - 1 with group h:
      skip: DP[i][a] = DP[i - 1][a]
      take: DP[i][a] = max(DP[i][a], 1 + DP[p[i - 1]][a with h reduced])
    """
    n = len(sorted_intervals)
    if not quota_vectors:
        raise ValueError("quota_vectors must contain at least one quota vector")

    k = len(quota_vectors[0])
    upper_bounds = tuple(
        max(quota[group] for quota in quota_vectors) for group in range(k)
    )
    zero_quota = tuple(0 for _ in range(k))
    dp: list[dict[tuple[int, ...], int]] = [{zero_quota: 0}]
    parents: list[dict[tuple[int, ...], tuple]] | None = (
        [{} for _ in range(n + 1)] if reconstruct_target is not None else None
    )

    for row in range(1, n + 1):
        interval_index = row - 1
        _, _, group = sorted_intervals[interval_index]
        compatible_row = p[interval_index]
        previous_layer = dp[row - 1]
        compatible_layer = dp[compatible_row]
        # Start with the skip transition for every reachable state:
        # DP[i][a] >= DP[i - 1][a].
        current_layer: dict[tuple[int, ...], int] = dict(previous_layer)
        current_parents = parents[row] if parents is not None else None

        if current_parents is not None:
            for quota in previous_layer:
                current_parents[quota] = ("skip", row - 1, quota)

        # Take transition, written in forward form from DP[p(i)]:
        # taking an interval from group h can either leave an already-zero
        # remaining quota unchanged, or increase the satisfied target quota by
        # one in group h, up to the requested upper bound.
        for reduced_quota, take_base in compatible_layer.items():
            candidate_quotas = [reduced_quota]
            if reduced_quota[group] < upper_bounds[group]:
                increased_quota = list(reduced_quota)
                increased_quota[group] += 1
                candidate_quotas.append(tuple(increased_quota))

            take_value = take_base + 1
            for quota in candidate_quotas:
                best_value = current_layer.get(quota)
                if best_value is None or take_value > best_value:
                    current_layer[quota] = take_value
                    if current_parents is not None:
                        current_parents[quota] = (
                            "take",
                            compatible_row,
                            reduced_quota,
                            interval_index,
                        )

        dp.append(current_layer)

    return dp, parents


def _reconstruct_selected_intervals(
    sorted_intervals: list[tuple[int, int, int]],
    parents: list[dict[tuple[int, ...], tuple]],
    target_quota: tuple[int, ...],
) -> list[tuple[int, int, int]]:
    selected: list[tuple[int, int, int]] = []
    row = len(sorted_intervals)
    quota = target_quota

    while row > 0:
        parent = parents[row].get(quota)
        if parent is None:
            break
        if parent[0] == "skip":
            _, row, quota = parent
        else:
            _, row, quota, interval_index = parent
            selected.append(sorted_intervals[interval_index])

    selected.reverse()
    return selected


def fair_dp_with_beta(
    intervals: list[Interval | tuple[int, int, int]], k: int, beta: float
) -> FairDPResult:
    """
    Solve offline interval scheduling with group-fairness quotas induced by beta.

    The returned schedule has maximum cardinality among all feasible schedules
    that select at least ceil(beta * OPT_g) intervals from each group g.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    normalized = _normalize_intervals(intervals, k)
    opt_per_group = compute_group_opts(normalized, k)
    quotas = [math.ceil(beta * opt) for opt in opt_per_group]
    quota_tuple = tuple(quotas)

    sorted_intervals, p = sort_intervals_and_compute_p(normalized)
    quota_vectors = list(generate_quota_vectors(quotas))
    dp, parents = _run_quota_dp(
        sorted_intervals,
        p,
        quota_vectors,
        reconstruct_target=quota_tuple,
    )

    max_cardinality = dp[len(sorted_intervals)].get(quota_tuple)
    feasible = max_cardinality is not None
    selected_intervals = (
        _reconstruct_selected_intervals(sorted_intervals, parents, quota_tuple)
        if feasible and parents is not None
        else None
    )

    return FairDPResult(
        feasible=feasible,
        max_cardinality=max_cardinality if max_cardinality is not None else 0,
        quotas=quotas,
        opt_per_group=opt_per_group,
        selected_intervals=selected_intervals,
    )


def _fairness_ratio(quota: tuple[int, ...], opt_per_group: list[int]) -> float:
    ratios = [
        quota[group] / opt
        for group, opt in enumerate(opt_per_group)
        if opt > 0
    ]
    return min(ratios) if ratios else 1.0


def find_best_fairness_by_quota_enumeration(
    intervals: list[Interval | tuple[int, int, int]], k: int
) -> BestFairnessResult:
    """
    Enumerate all quota vectors and return the feasible one with best fairness.

    The fairness score of quota vector a is min_g a_g / OPT_g, ignoring groups
    with OPT_g = 0. Ties are broken by larger selected cardinality and then by
    the lexicographically larger quota vector.
    """
    normalized = _normalize_intervals(intervals, k)
    opt_per_group = compute_group_opts(normalized, k)
    sorted_intervals, p = sort_intervals_and_compute_p(normalized)
    quota_vectors = list(generate_quota_vectors(opt_per_group))

    dp, _ = _run_quota_dp(sorted_intervals, p, quota_vectors)
    final_layer = dp[len(sorted_intervals)]

    best_quota = tuple(0 for _ in range(k))
    best_beta = -1.0
    best_cardinality = -1

    for quota in quota_vectors:
        cardinality = final_layer.get(quota)
        if cardinality is None:
            continue

        beta = _fairness_ratio(quota, opt_per_group)
        candidate_key = (beta, cardinality, quota)
        best_key = (best_beta, best_cardinality, best_quota)
        if candidate_key > best_key:
            best_beta = beta
            best_cardinality = cardinality
            best_quota = quota

    _, parents = _run_quota_dp(
        sorted_intervals,
        p,
        quota_vectors,
        reconstruct_target=best_quota,
    )
    selected_intervals = (
        _reconstruct_selected_intervals(sorted_intervals, parents, best_quota)
        if parents is not None and best_cardinality >= 0
        else None
    )

    return BestFairnessResult(
        best_beta=best_beta if best_beta >= 0 else 0.0,
        best_quota_vector=list(best_quota),
        max_cardinality_at_best_beta=max(best_cardinality, 0),
        opt_per_group=opt_per_group,
        selected_intervals=selected_intervals,
    )


def do_intervals_intersect(interval1: Interval, interval2: Interval) -> bool:
    # Left-closed right-open: [start, finish)
    return interval1.start < interval2.finish and interval2.start < interval1.finish


def can_add_interval(new_interval: Interval, accepted_list: list[Interval]) -> bool:
    for existing_interval in accepted_list:
        if do_intervals_intersect(new_interval, existing_interval):
            return False
    return True


def run_online_algorithm(intervals_to_process: list[Interval]) -> list[Interval]:
    accepted_intervals = []
    for interval in intervals_to_process:
        if can_add_interval(interval, accepted_intervals):
            interval.accepted = True
            accepted_intervals.append(interval)
    return accepted_intervals


def run_greedy_algorithm(intervals_to_process: list[Interval]) -> list[Interval]:
    sorted_intervals = sorted(intervals_to_process, key=lambda iv: iv.finish)
    accepted_intervals = []
    for interval in sorted_intervals:
        if can_add_interval(interval, accepted_intervals):
            interval.accepted = True
            accepted_intervals.append(interval)
    return accepted_intervals


def generate_random_intervals(
    num_intervals: int = 50,
    start_low: int = 0,
    start_high: int = 100,
    length_low: int = 1,
    length_high: int = 20,
    num_groups: int = 5,
    seed: int | None = None,
) -> list[Interval]:
    rng = random.Random(seed)
    intervals = []
    for _ in range(num_intervals):
        start = rng.randint(start_low, start_high)
        length = rng.randint(length_low, length_high)
        group = rng.randint(1, num_groups)
        intervals.append(Interval(start=start, length=length, group=group))
    return intervals


def _normalize_column_name(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def _get_value_from_row(row: dict, candidates: list[str]):
    normalized = {_normalize_column_name(k): v for k, v in row.items()}
    for name in candidates:
        key = _normalize_column_name(name)
        if key in normalized:
            return normalized[key]
    return None


def _to_float(value, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field_name}: {value}") from exc


def load_intervals_from_csv(
    path: str,
    start_col: str = "Start Time",
    finish_col: str = "Finish Time",
    length_col: str | None = None,
    group_col: str = "Group",
) -> list[Interval]:
    intervals = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row")

        for row in reader:
            start_raw = _get_value_from_row(row, [start_col, "start"])
            finish_raw = _get_value_from_row(row, [finish_col, "finish", "end"])
            length_raw = _get_value_from_row(row, [length_col or "", "length", "run time"])
            group_raw = _get_value_from_row(row, [group_col, "group id", "group"])

            if start_raw is None:
                continue
            start = _to_float(start_raw, "start")

            if finish_raw is not None:
                finish = _to_float(finish_raw, "finish")
                length = finish - start
            elif length_raw is not None:
                length = _to_float(length_raw, "length")
            else:
                raise ValueError(
                    "Each row must provide either finish column or length column"
                )

            if length <= 0:
                continue

            group = int(group_raw) if group_raw is not None and str(group_raw) != "" else 1
            intervals.append(Interval(start=int(start), length=int(length), group=group))
    return intervals


def load_intervals_from_json(
    path: str,
    start_col: str = "Start Time",
    finish_col: str = "Finish Time",
    length_col: str | None = None,
    group_col: str = "Group",
) -> list[Interval]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of objects")

    intervals = []
    for row in data:
        if not isinstance(row, dict):
            continue
        start_raw = _get_value_from_row(row, [start_col, "start"])
        finish_raw = _get_value_from_row(row, [finish_col, "finish", "end"])
        length_raw = _get_value_from_row(row, [length_col or "", "length", "run time"])
        group_raw = _get_value_from_row(row, [group_col, "group id", "group"])

        if start_raw is None:
            continue
        start = _to_float(start_raw, "start")

        if finish_raw is not None:
            finish = _to_float(finish_raw, "finish")
            length = finish - start
        elif length_raw is not None:
            length = _to_float(length_raw, "length")
        else:
            raise ValueError(
                "Each JSON object must provide either finish column or length column"
            )

        if length <= 0:
            continue

        group = int(group_raw) if group_raw is not None and str(group_raw) != "" else 1
        intervals.append(Interval(start=int(start), length=int(length), group=group))
    return intervals


def load_intervals_from_swf(path: str) -> list[Interval]:
    # Standard Workload Format columns:
    # Submit Time (1), Wait Time (2), Run Time (3), Group ID (12), 0-based indexing in code.
    intervals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue

            submit_time = _to_float(parts[1], "Submit Time")
            wait_time = _to_float(parts[2], "Wait Time")
            run_time = _to_float(parts[3], "Run Time")
            group_id = int(float(parts[12]))

            if run_time <= 0:
                continue

            start = submit_time + wait_time
            intervals.append(
                Interval(start=int(start), length=int(run_time), group=group_id)
            )
    return intervals


def load_intervals_from_file(
    path: str,
    file_format: str | None = None,
    start_col: str = "Start Time",
    finish_col: str = "Finish Time",
    length_col: str | None = None,
    group_col: str = "Group",
) -> list[Interval]:
    resolved_format = (file_format or Path(path).suffix.lstrip(".")).lower()

    if resolved_format == "csv":
        return load_intervals_from_csv(path, start_col, finish_col, length_col, group_col)
    if resolved_format == "json":
        return load_intervals_from_json(path, start_col, finish_col, length_col, group_col)
    if resolved_format == "swf":
        return load_intervals_from_swf(path)

    raise ValueError(
        f"Unsupported format: {resolved_format}. Use csv/json/swf or provide --format."
    )


def _supported_input_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".csv", ".json", ".swf"}


def collect_input_files_from_manifest(manifest_path: str) -> list[Path]:
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise ValueError(f"Manifest file does not exist: {manifest}")

    collected: list[Path] = []
    seen: set[Path] = set()

    with manifest.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            entry = Path(line)
            if not entry.is_absolute():
                entry = (manifest.parent / entry).resolve()

            if entry.is_dir():
                for candidate in sorted(entry.rglob("*")):
                    resolved_candidate = candidate.resolve()
                    if _supported_input_file(resolved_candidate) and resolved_candidate not in seen:
                        collected.append(resolved_candidate)
                        seen.add(resolved_candidate)
                continue

            if _supported_input_file(entry.resolve()) and entry.resolve() not in seen:
                collected.append(entry.resolve())
                seen.add(entry.resolve())
                continue

            raise ValueError(
                f"Manifest entry must be an existing folder or a csv/json/swf file: {entry}"
            )

    if not collected:
        raise ValueError(f"No supported input files were found in manifest: {manifest}")

    return collected


def load_intervals_from_manifest(
    manifest_path: str,
    start_col: str = "Start Time",
    finish_col: str = "Finish Time",
    length_col: str | None = None,
    group_col: str = "Group",
) -> list[Interval]:
    intervals: list[Interval] = []
    input_files = collect_input_files_from_manifest(manifest_path)

    print("\nINPUT FILES")
    for input_file in input_files:
        file_intervals = load_intervals_from_file(
            path=str(input_file),
            file_format=None,
            start_col=start_col,
            finish_col=finish_col,
            length_col=length_col,
            group_col=group_col,
        )
        intervals.extend(file_intervals)
        print(f"{input_file}: {len(file_intervals)} intervals")

    return intervals


def assign_sections(intervals: list[Interval]) -> list[Interval]:
    # Old code's new_assign_sections idea: scan by start, overlap -> same section.
    sorted_with_idx = sorted(enumerate(intervals), key=lambda x: x[1].start)
    current_section_end = None
    current_section_number = 0
    section_by_index = {}

    for original_index, iv in sorted_with_idx:
        if current_section_end is not None and iv.start < current_section_end:
            current_section_end = max(current_section_end, iv.finish)
        else:
            current_section_number += 1
            current_section_end = iv.finish
        section_by_index[original_index] = current_section_number

    result = []
    for i, iv in enumerate(intervals):
        cloned = iv.copy()
        cloned.section = section_by_index[i]
        result.append(cloned)
    return result


def greedy_for_group(intervals: list[Interval]) -> list[Interval]:
    return run_greedy_algorithm([iv.copy() for iv in intervals])


def build_group_permutation(
    intervals: list[Interval],
    permutation: list[int] | None = None,
    seed: int | None = None,
) -> list[int]:
    groups = sorted({iv.group for iv in intervals if iv.group is not None})
    if not groups:
        return []

    if permutation is None:
        rng = random.Random(seed)
        ordered_groups = groups[:]
        rng.shuffle(ordered_groups)
        return ordered_groups

    available_groups = set(groups)
    seen: set[int] = set()
    ordered_groups: list[int] = []

    for group in permutation:
        if group not in available_groups or group in seen:
            continue
        ordered_groups.append(group)
        seen.add(group)

    for group in groups:
        if group not in seen:
            ordered_groups.append(group)

    return ordered_groups


def merge_without_conflict(
    selected_intervals: list[Interval], new_intervals: list[Interval]
) -> list[Interval]:
    if not selected_intervals:
        return greedy_for_group(new_intervals)
    if not new_intervals:
        return [iv.copy() for iv in selected_intervals]

    result = sorted([iv.copy() for iv in selected_intervals], key=lambda x: x.finish)
    for candidate in sorted(new_intervals, key=lambda x: x.finish):
        if can_add_interval(candidate, result):
            candidate.accepted = True
            result.append(candidate.copy())
            result.sort(key=lambda x: x.finish)
    return result


def select_intervals_all_sections(
    intervals_with_section: list[Interval],
    seed: int | None = None,
    permutation: list[int] | None = None,
) -> list[Interval]:
    ordered_groups = build_group_permutation(
        intervals_with_section,
        permutation=permutation,
        seed=seed,
    )
    selected_intervals: list[Interval] = []

    for group in ordered_groups:
        chosen_group_data = [
            iv.copy() for iv in intervals_with_section if iv.group == group
        ]
        selected_intervals = merge_without_conflict(
            selected_intervals,
            greedy_for_group(chosen_group_data),
        )

    return selected_intervals


def select_intervals_mixed(
    intervals: list[Interval],
    alpha: float,
    seed: int | None = None,
    permutation: list[int] | None = None,
) -> list[Interval]:
    rng = random.Random(seed)
    if rng.random() < alpha:
        return run_greedy_algorithm([iv.copy() for iv in intervals])
    return select_intervals_all_sections(intervals, seed=seed, permutation=permutation)


def compute_optimal_per_group(intervals: list[Interval]) -> tuple[int, dict[int, int]]:
    group_optimal = {}
    all_groups = sorted({iv.group for iv in intervals if iv.group is not None})

    for group in all_groups:
        group_intervals = [iv.copy() for iv in intervals if iv.group == group]
        group_optimal[group] = len(run_greedy_algorithm(group_intervals))

    return sum(group_optimal.values()), group_optimal


def count_by_group(intervals: list[Interval]) -> dict[int, int]:
    counts = {}
    for iv in intervals:
        counts[iv.group] = counts.get(iv.group, 0) + 1
    return dict(sorted(counts.items()))


def run_fair_algorithm_multiple_times(
    intervals_with_section: list[Interval],
    num_runs: int = 50,
    seed: int = 42,
    alpha: float = 0.0,
    permutation: list[int] | None = None,
) -> dict:
    run_counts = []
    run_totals = []

    for i in range(num_runs):
        selected = select_intervals_mixed(
            intervals_with_section,
            alpha=alpha,
            seed=seed + i,
            permutation=permutation,
        )
        group_counts = count_by_group(selected)
        run_counts.append(group_counts)
        run_totals.append(sum(group_counts.values()))

    all_groups = sorted({g for run in run_counts for g in run})
    mean_per_group = {}
    min_per_group = {}
    max_per_group = {}

    for g in all_groups:
        values = [run.get(g, 0) for run in run_counts]
        mean_per_group[g] = mean(values)
        min_per_group[g] = min(values)
        max_per_group[g] = max(values)

    return {
        "num_runs": num_runs,
        "run_counts": run_counts,
        "mean_per_group": mean_per_group,
        "min_per_group": min_per_group,
        "max_per_group": max_per_group,
        "mean_total_selected": mean(run_totals),
    }


def print_alg_opt_summary(alg_counts: dict[int, int], opt_by_group: dict[int, int]) -> None:
    ratios = {}
    for g in sorted(opt_by_group):
        opt = opt_by_group[g]
        sel = alg_counts.get(g, 0)
        ratios[g] = (sel / opt) if opt > 0 else 0.0

    ratio_values = list(ratios.values())
    min_ratio = min(ratio_values) if ratio_values else 0.0

    print("\nSUMMARY")
    print(f"Ratio: {min_ratio:.3f}")

    print("\nPER-GROUP")
    print(f"{'Group':<8}{'Alg':<8}{'Opt':<8}{'Ratio':<8}")
    for g in sorted(opt_by_group):
        sel = alg_counts.get(g, 0)
        opt = opt_by_group[g]
        print(f"{g:<8}{sel:<8}{opt:<8}{ratios[g]:<8.3f}")


def compare_online_vs_greedy(num_intervals_to_generate: int = 50) -> None:
    base = generate_random_intervals(num_intervals=num_intervals_to_generate, seed=7)
    online_accepted = run_online_algorithm([iv.copy() for iv in base])
    greedy_accepted = run_greedy_algorithm([iv.copy() for iv in base])

    online_total_length = sum(iv.length for iv in online_accepted)
    greedy_total_length = sum(iv.length for iv in greedy_accepted)

    print("\nONLINE vs GREEDY")
    print(
        f"Online: {len(online_accepted)} intervals, total length {online_total_length}"
    )
    print(
        f"Greedy: {len(greedy_accepted)} intervals, total length {greedy_total_length}"
    )


def demo_fairness() -> None:
    intervals = generate_random_intervals(num_intervals=120, num_groups=6, seed=42)
    with_sections = assign_sections(intervals)

    total_opt, opt_by_group = compute_optimal_per_group(with_sections)
    selected_once = select_intervals_mixed(with_sections, alpha=0.0, seed=42)
    alg_counts = count_by_group(selected_once)

    print("\nFAIRNESS DEMO")
    print(f"Total per-group optimal sum: {total_opt}")
    print(f"Single-run selected total:   {sum(alg_counts.values())}")
    print_alg_opt_summary(alg_counts, opt_by_group)

    multi = run_fair_algorithm_multiple_times(
        with_sections,
        num_runs=100,
        seed=42,
        alpha=0.0,
    )
    print("\nMULTI-RUN")
    print(f"Runs: {multi['num_runs']}")
    print(f"Average selected per run: {multi['mean_total_selected']:.2f}")
    print(f"Mean selected by group: {multi['mean_per_group']}")


def demo_offline_fair_dp() -> None:
    """Small example showing how to call the offline fairness DP functions."""
    intervals = [
        (0, 3, 0),
        (3, 5, 0),
        (1, 4, 1),
        (4, 7, 1),
        (5, 8, 0),
        (7, 9, 1),
    ]
    k = 2

    beta_result = fair_dp_with_beta(intervals, k, beta=0.3)
    best_result = find_best_fairness_by_quota_enumeration(intervals, k)

    print("\nOFFLINE FAIR DP DEMO")
    print(f"Beta result: {beta_result}")
    print(f"Best fairness by quota enumeration: {best_result}")


def run_fairness_with_intervals(
    intervals: list[Interval],
    num_runs: int,
    seed: int,
    alpha: float = 0.0,
    permutation: list[int] | None = None,
) -> None:
    if not intervals:
        raise ValueError("No valid intervals were loaded from input file")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    total_opt, opt_by_group = compute_optimal_per_group(intervals)
    selected_once = select_intervals_mixed(
        intervals,
        alpha=alpha,
        seed=seed,
        permutation=permutation,
    )
    alg_counts = count_by_group(selected_once)

    print("\nFAIRNESS (EXTERNAL DATA)")
    print(f"Input intervals:             {len(intervals)}")
    print(f"Total per-group optimal sum: {total_opt}")
    print(f"Alpha:                       {alpha:.3f}")
    print(f"Single-run selected total:   {sum(alg_counts.values())}")
    print_alg_opt_summary(alg_counts, opt_by_group)

    multi = run_fair_algorithm_multiple_times(
        intervals,
        num_runs=num_runs,
        seed=seed,
        alpha=alpha,
        permutation=permutation,
    )
    competitive_ratio = (
        multi["mean_total_selected"] / total_opt if total_opt > 0 else 0.0
    )
    mean_ratio_by_group = {}
    for group, opt in opt_by_group.items():
        mean_selected = multi["mean_per_group"].get(group, 0.0)
        mean_ratio_by_group[group] = (mean_selected / opt) if opt > 0 else 0.0

    print("\nMULTI-RUN")
    print(f"Runs: {multi['num_runs']}")
    print(f"Average selected per run: {multi['mean_total_selected']:.2f}")
    print(f"Competitive ratio: {competitive_ratio:.3f}")
    print(f"Mean selected by group: {multi['mean_per_group']}")
    print(f"Mean ratio by group: {mean_ratio_by_group}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interval fairness experiments")
    parser.add_argument("--input", type=str, help="Path to external data file")
    parser.add_argument(
        "--input-list",
        type=str,
        help="Path to a text file listing input files and/or folders to load",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "swf"],
        help="Input format. If omitted, infer from file extension.",
    )
    parser.add_argument("--start-col", type=str, default="Start Time")
    parser.add_argument("--finish-col", type=str, default="Finish Time")
    parser.add_argument("--length-col", type=str, default=None)
    parser.add_argument("--group-col", type=str, default="Group")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Use global greedy with probability alpha, otherwise use the permutation-based group greedy algorithm",
    )
    parser.add_argument(
        "--group-permutation",
        type=str,
        help="Optional comma-separated group order, for example: 3,1,2,4",
    )
    parser.add_argument(
        "--skip-random-demo",
        action="store_true",
        help="Skip built-in random demo when external input is provided.",
    )
    return parser.parse_args()


def parse_group_permutation(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None

    values = [part.strip() for part in raw_value.split(",")]
    values = [part for part in values if part]
    if not values:
        return []
    return [int(part) for part in values]


if __name__ == "__main__":
    args = parse_args()
    group_permutation = parse_group_permutation(args.group_permutation)

    if args.input and args.input_list:
        raise ValueError("Use either --input or --input-list, not both")

    if args.input or args.input_list:
        if args.input_list:
            external_intervals = load_intervals_from_manifest(
                manifest_path=args.input_list,
                start_col=args.start_col,
                finish_col=args.finish_col,
                length_col=args.length_col,
                group_col=args.group_col,
            )
        else:
            external_intervals = load_intervals_from_file(
                path=args.input,
                file_format=args.format,
                start_col=args.start_col,
                finish_col=args.finish_col,
                length_col=args.length_col,
                group_col=args.group_col,
            )
        run_fairness_with_intervals(
            intervals=external_intervals,
            num_runs=args.runs,
            seed=args.seed,
            alpha=args.alpha,
            permutation=group_permutation,
        )
        if not args.skip_random_demo:
            compare_online_vs_greedy(num_intervals_to_generate=50)
            demo_fairness()
    else:
        compare_online_vs_greedy(num_intervals_to_generate=50)
        demo_fairness()
        demo_offline_fair_dp()
