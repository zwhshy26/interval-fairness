import argparse
import csv
import gzip
import json
import math
import random
import sys
from bisect import bisect_right
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean


# DATA STRUCTURES


@dataclass
class Interval:
    start: int
    length: int
    group: int | None = None
    accepted: bool = False

    @property
    def finish(self) -> int:
        return self.start + self.length

    def copy(self) -> "Interval":
        return Interval(
            start=self.start,
            length=self.length,
            group=self.group,
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


@dataclass
class Block:
    group: int
    index: int
    intervals: list[Interval]

    @property
    def start(self) -> int:
        return min(interval.start for interval in self.intervals)

    @property
    def finish(self) -> int:
        return max(interval.finish for interval in self.intervals)

    @property
    def size(self) -> int:
        return len(self.intervals)


# PROGRESS


class ProgressBar:
    """Small terminal progress bar without external dependencies."""

    def __init__(self, total: int, label: str, enabled: bool = True, width: int = 30):
        self.total = total
        self.label = label
        self.enabled = enabled and total > 0
        self.width = width
        self.current = 0

    def update(self, current: int) -> None:
        if not self.enabled:
            return

        self.current = min(current, self.total)
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(ratio * 100)
        sys.stderr.write(
            f"\r{self.label}: [{bar}] {self.current}/{self.total} ({percent:3d}%)"
        )
        sys.stderr.flush()

    def advance(self) -> None:
        self.update(self.current + 1)

    def finish(self) -> None:
        if not self.enabled:
            return
        if self.current < self.total:
            self.update(self.total)
        sys.stderr.write("\n")
        sys.stderr.flush()


# COMMON INTERVAL HELPERS


def _as_interval_tuple(interval: Interval | tuple[int, int, int]) -> tuple[int, int, int]:
    """Return an interval as (start, end, normalized_group). Used by the DP only."""
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
    """Validate DP intervals using normalized group ids 0, ..., k - 1."""
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


def do_intervals_intersect(interval1: Interval, interval2: Interval) -> bool:
    # Left-closed, right-open intervals: [a, b) and [b, c) are compatible.
    return interval1.start < interval2.finish and interval2.start < interval1.finish


def can_add_interval(new_interval: Interval, accepted_list: list[Interval]) -> bool:
    return all(
        not do_intervals_intersect(new_interval, existing_interval)
        for existing_interval in accepted_list
    )


def intersects_any_interval(
    interval: Interval,
    sorted_nonoverlapping_intervals: list[Interval],
    starts: list[int],
) -> bool:
    if not sorted_nonoverlapping_intervals:
        return False
    candidate_index = bisect_right(starts, interval.start) - 1
    if (
        candidate_index >= 0
        and do_intervals_intersect(interval, sorted_nonoverlapping_intervals[candidate_index])
    ):
        return True
    next_index = candidate_index + 1
    return (
        next_index < len(sorted_nonoverlapping_intervals)
        and do_intervals_intersect(interval, sorted_nonoverlapping_intervals[next_index])
    )


def add_if_compatible_with_sorted_schedule(
    interval: Interval,
    schedule_by_start: list[Interval],
    starts: list[int],
) -> bool:
    insert_index = bisect_right(starts, interval.start)
    previous_index = insert_index - 1
    if (
        previous_index >= 0
        and do_intervals_intersect(interval, schedule_by_start[previous_index])
    ):
        return False
    if (
        insert_index < len(schedule_by_start)
        and do_intervals_intersect(interval, schedule_by_start[insert_index])
    ):
        return False

    schedule_by_start.insert(insert_index, interval)
    starts.insert(insert_index, interval.start)
    return True


def interval_sort_key(interval: Interval) -> tuple[int, int, int, int]:
    group = interval.group if interval.group is not None else -1
    return (interval.finish, interval.start, interval.length, group)


def count_by_group(intervals: list[Interval]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for interval in intervals:
        if interval.group is not None:
            counts[interval.group] = counts.get(interval.group, 0) + 1
    return dict(sorted(counts.items()))


def get_groups(intervals: list[Interval]) -> list[int]:
    return sorted({interval.group for interval in intervals if interval.group is not None})


def compatible_residual_intervals(
    intervals: list[Interval],
    selected_solution: list[Interval],
) -> list[Interval]:
    selected_by_start = sorted(
        [interval.copy() for interval in selected_solution],
        key=lambda interval: (interval.start, interval.finish, interval.group or -1),
    )
    starts = [interval.start for interval in selected_by_start]
    return [
        interval.copy()
        for interval in intervals
        if not intersects_any_interval(interval, selected_by_start, starts)
    ]


# OFFLINE GREEDY AND OPTIMA


def run_offline_greedy(intervals: list[Interval]) -> list[Interval]:
    """Earliest-finish-time greedy; exact global OPT for unweighted cardinality."""
    accepted: list[Interval] = []
    for interval in sorted([iv.copy() for iv in intervals], key=interval_sort_key):
        if not accepted or accepted[-1].finish <= interval.start:
            interval.accepted = True
            accepted.append(interval)
    return accepted


def compute_optimal_per_group(intervals: list[Interval]) -> tuple[int, dict[int, int]]:
    opt_by_group: dict[int, int] = {}
    for group in get_groups(intervals):
        group_intervals = [interval.copy() for interval in intervals if interval.group == group]
        opt_by_group[group] = len(run_offline_greedy(group_intervals))
    return sum(opt_by_group.values()), opt_by_group


# EXACT FAIRNESS DP, AUXILIARY ONLY


def _compute_dp_group_opts(
    intervals: list[Interval | tuple[int, int, int]], k: int
) -> list[int]:
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
    ranges = [range(opt + 1) for opt in opt_per_group]
    yield from product(*ranges)


def _run_quota_dp(
    sorted_intervals: list[tuple[int, int, int]],
    p: list[int],
    quota_vectors: list[tuple[int, ...]],
    reconstruct_target: tuple[int, ...] | None = None,
) -> tuple[list[dict[tuple[int, ...], int]], list[dict[tuple[int, ...], tuple]] | None]:
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
        current_layer: dict[tuple[int, ...], int] = dict(previous_layer)
        current_parents = parents[row] if parents is not None else None

        if current_parents is not None:
            for quota in previous_layer:
                current_parents[quota] = ("skip", row - 1, quota)

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
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    normalized = _normalize_intervals(intervals, k)
    opt_per_group = _compute_dp_group_opts(normalized, k)
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
        opt / quota[group] if quota[group] > 0 else math.inf
        for group, opt in enumerate(opt_per_group)
        if opt > 0
    ]
    return max(ratios) if ratios else 1.0


def find_best_fairness_by_quota_enumeration(
    intervals: list[Interval | tuple[int, int, int]], k: int
) -> BestFairnessResult:
    normalized = _normalize_intervals(intervals, k)
    opt_per_group = _compute_dp_group_opts(normalized, k)
    sorted_intervals, p = sort_intervals_and_compute_p(normalized)
    quota_vectors = list(generate_quota_vectors(opt_per_group))

    dp, _ = _run_quota_dp(sorted_intervals, p, quota_vectors)
    final_layer = dp[len(sorted_intervals)]

    best_quota = tuple(0 for _ in range(k))
    best_beta = math.inf
    best_cardinality = -1

    for quota in quota_vectors:
        cardinality = final_layer.get(quota)
        if cardinality is None:
            continue

        beta = _fairness_ratio(quota, opt_per_group)
        if (
            beta < best_beta
            or (beta == best_beta and (cardinality, quota) > (best_cardinality, best_quota))
        ):
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
        best_beta=best_beta if best_cardinality >= 0 else math.inf,
        best_quota_vector=list(best_quota),
        max_cardinality_at_best_beta=max(best_cardinality, 0),
        opt_per_group=opt_per_group,
        selected_intervals=selected_intervals,
    )


# OFFLINE DETERMINISTIC


def partition_schedule_into_blocks(
    schedule: list[Interval], group: int, r: int
) -> list[Block]:
    """
    Partition a group optimum into r consecutive near-equal blocks.

    Finite workloads may not divide evenly; the first remainder blocks receive
    one extra interval. Empty blocks are not materialized.
    """
    if r <= 0:
        raise ValueError("r must be positive")

    ordered_schedule = sorted([iv.copy() for iv in schedule], key=interval_sort_key)
    if not ordered_schedule:
        return []

    base_size, remainder = divmod(len(ordered_schedule), r)
    blocks: list[Block] = []
    cursor = 0
    for block_index in range(r):
        block_size = base_size + (1 if block_index < remainder else 0)
        if block_size == 0:
            continue
        block_intervals = ordered_schedule[cursor : cursor + block_size]
        blocks.append(Block(group=group, index=block_index + 1, intervals=block_intervals))
        cursor += block_size
    return blocks


def do_blocks_intersect(block1: Block, block2: Block) -> bool:
    return block1.start < block2.finish and block2.start < block1.finish


def select_blocks_by_earliest_finish(
    intervals: list[Interval],
    r: int,
) -> list[Block]:
    groups = get_groups(intervals)
    k = len(groups)
    if r < k:
        raise ValueError(f"r must be at least k. Got r={r}, k={k}")

    blocks_by_group: dict[int, list[Block]] = {}
    for group in groups:
        group_intervals = [iv.copy() for iv in intervals if iv.group == group]
        group_optimum = run_offline_greedy(group_intervals)
        blocks_by_group[group] = partition_schedule_into_blocks(
            group_optimum,
            group=group,
            r=r,
        )

    active_groups = groups[:]
    selected_blocks: list[Block] = []

    while active_groups and len(selected_blocks) < k:
        candidate_blocks = [
            block
            for group in active_groups
            for block in blocks_by_group.get(group, [])
        ]
        if not candidate_blocks:
            break

        chosen_block = min(
            candidate_blocks,
            key=lambda block: (
                block.finish,
                -block.start,
                block.group,
                block.index,
            ),
        )
        selected_blocks.append(chosen_block)
        active_groups.remove(chosen_block.group)

        for group in active_groups:
            blocks_by_group[group] = [
                block
                for block in blocks_by_group.get(group, [])
                if not do_blocks_intersect(block, chosen_block)
            ]

    return selected_blocks


def intervals_from_blocks(blocks: list[Block]) -> list[Interval]:
    result = sorted(
        [interval.copy() for block in blocks for interval in block.intervals],
        key=interval_sort_key,
    )
    for interval in result:
        interval.accepted = True
    return result


def run_offline_deterministic_r_block(
    intervals: list[Interval],
    r: int,
) -> tuple[list[Interval], list[Block]]:
    groups = get_groups(intervals)
    k = len(groups)
    if r < k:
        raise ValueError(f"r must be at least k. Got r={r}, k={k}")

    selected_blocks = select_blocks_by_earliest_finish(intervals, r=r)
    block_solution = intervals_from_blocks(selected_blocks)
    residual_intervals = compatible_residual_intervals(intervals, block_solution)
    residual_solution = run_offline_greedy(residual_intervals)
    result = sorted(block_solution + residual_solution, key=interval_sort_key)
    return result, selected_blocks


# OFFLINE RANDOMIZED


def run_offline_randomized_group_greedy(
    intervals: list[Interval],
    seed: int | None = None,
) -> tuple[list[Interval], int]:
    groups = get_groups(intervals)
    if not groups:
        return [], -1

    rng = random.Random(seed)
    chosen_group = rng.choice(groups)
    group_intervals = [
        interval.copy()
        for interval in intervals
        if interval.group == chosen_group
    ]
    group_solution = run_offline_greedy(group_intervals)
    residual_intervals = compatible_residual_intervals(intervals, group_solution)
    residual_solution = run_offline_greedy(residual_intervals)
    result = sorted(group_solution + residual_solution, key=interval_sort_key)
    return result, chosen_group


def run_offline_randomized_multiple_times(
    intervals: list[Interval],
    opt_by_group: dict[int, int],
    global_opt: int,
    runs: int,
    seed: int,
    show_progress: bool = True,
    debug_runs: bool = False,
) -> dict:
    run_counts: list[dict[int, int]] = []
    chosen_groups: list[int] = []
    progress = ProgressBar(runs, "Offline randomized", enabled=show_progress)

    for run_index in range(runs):
        solution, chosen_group = run_offline_randomized_group_greedy(
            intervals,
            seed=seed + run_index,
        )
        counts = count_by_group(solution)
        chosen_groups.append(chosen_group)
        run_counts.append(counts)
        if debug_runs:
            print(
                f"Offline randomized run {run_index + 1}: "
                f"group={chosen_group}, selected={len(solution)}"
            )
        progress.advance()

    progress.finish()
    mean_by_group = mean_counts_by_group(run_counts, opt_by_group.keys())
    expected_total = sum(mean_by_group.values())
    return {
        "runs": runs,
        "expected_total": expected_total,
        "mean_by_group": mean_by_group,
        "fairness": ex_ante_fairness(mean_by_group, opt_by_group),
        "fraction_opt": safe_fraction(expected_total, global_opt),
        "inverse_ratio": safe_inverse_ratio(global_opt, expected_total),
        "chosen_groups": chosen_groups,
    }


# ONLINE GREEDY BASELINE


def run_simple_online_greedy(intervals: list[Interval]) -> list[Interval]:
    accepted: list[Interval] = []
    accepted_by_start: list[Interval] = []
    starts: list[int] = []
    for interval in intervals:
        candidate = interval.copy()
        if add_if_compatible_with_sorted_schedule(candidate, accepted_by_start, starts):
            candidate.accepted = True
            accepted.append(candidate)
    return accepted


# ONLINE RANDOMIZED


def compute_length_level(length: int, min_length: int) -> int:
    """
    Return zero-based dyadic length level.

    Level j contains lengths satisfying 2^j * L_min <= length, up to the next
    dyadic boundary. Exact powers of two move into the higher level, e.g.
    L_min -> 0, 2 L_min -> 1, and 4 L_min -> 2.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if min_length <= 0:
        raise ValueError(f"min_length must be positive, got {min_length}")
    if length < min_length:
        raise ValueError(f"length {length} is below min_length {min_length}")

    level = 0
    boundary = min_length * 2
    while boundary <= length:
        level += 1
        boundary *= 2
    return level


def compute_num_length_levels(intervals: list[Interval]) -> int:
    lengths = [interval.length for interval in intervals if interval.length > 0]
    if not lengths:
        return 0
    min_length = min(lengths)
    max_length = max(lengths)
    return compute_length_level(max_length, min_length) + 1


def run_online_random_group_level(
    intervals: list[Interval],
    seed: int | None = None,
) -> tuple[list[Interval], int, int]:
    groups = get_groups(intervals)
    lengths = [interval.length for interval in intervals if interval.length > 0]
    if not groups or not lengths:
        return [], -1, -1

    min_length = min(lengths)
    num_levels = compute_num_length_levels(intervals)
    rng = random.Random(seed)
    chosen_group = rng.choice(groups)
    chosen_level = rng.randrange(num_levels)

    accepted: list[Interval] = []
    accepted_by_start: list[Interval] = []
    starts: list[int] = []
    for interval in intervals:
        if interval.group != chosen_group:
            continue
        if compute_length_level(interval.length, min_length) != chosen_level:
            continue
        candidate = interval.copy()
        if add_if_compatible_with_sorted_schedule(candidate, accepted_by_start, starts):
            candidate.accepted = True
            accepted.append(candidate)

    return accepted, chosen_group, chosen_level


def run_online_random_group_level_multiple_times(
    intervals: list[Interval],
    opt_by_group: dict[int, int],
    global_opt: int,
    runs: int,
    seed: int,
    show_progress: bool = True,
    debug_runs: bool = False,
) -> dict:
    run_counts: list[dict[int, int]] = []
    chosen_groups: list[int] = []
    chosen_levels: list[int] = []
    num_levels = compute_num_length_levels(intervals)
    progress = ProgressBar(runs, "Online randomized", enabled=show_progress)

    for run_index in range(runs):
        solution, chosen_group, chosen_level = run_online_random_group_level(
            intervals,
            seed=seed + run_index,
        )
        counts = count_by_group(solution)
        chosen_groups.append(chosen_group)
        chosen_levels.append(chosen_level)
        run_counts.append(counts)
        if debug_runs:
            print(
                f"Online randomized run {run_index + 1}: "
                f"group={chosen_group}, level={chosen_level}, selected={len(solution)}"
            )
        progress.advance()

    progress.finish()
    mean_by_group = mean_counts_by_group(run_counts, opt_by_group.keys())
    expected_total = sum(mean_by_group.values())
    return {
        "runs": runs,
        "num_levels": num_levels,
        "expected_total": expected_total,
        "mean_by_group": mean_by_group,
        "fairness": ex_ante_fairness(mean_by_group, opt_by_group),
        "fraction_opt": safe_fraction(expected_total, global_opt),
        "inverse_ratio": safe_inverse_ratio(global_opt, expected_total),
        "chosen_groups": chosen_groups,
        "chosen_levels": chosen_levels,
    }


def run_online_random_level_greedy(
    intervals: list[Interval],
    seed: int | None = None,
) -> tuple[list[Interval], int]:
    lengths = [interval.length for interval in intervals if interval.length > 0]
    if not lengths:
        return [], -1

    min_length = min(lengths)
    num_levels = compute_num_length_levels(intervals)
    rng = random.Random(seed)
    chosen_level = rng.randrange(num_levels)

    accepted: list[Interval] = []
    accepted_by_start: list[Interval] = []
    starts: list[int] = []
    for interval in intervals:
        if compute_length_level(interval.length, min_length) != chosen_level:
            continue
        candidate = interval.copy()
        if add_if_compatible_with_sorted_schedule(candidate, accepted_by_start, starts):
            candidate.accepted = True
            accepted.append(candidate)

    return accepted, chosen_level


def run_online_random_level_greedy_multiple_times(
    intervals: list[Interval],
    opt_by_group: dict[int, int],
    global_opt: int,
    runs: int,
    seed: int,
    show_progress: bool = True,
    debug_runs: bool = False,
) -> dict:
    run_counts: list[dict[int, int]] = []
    chosen_levels: list[int] = []
    num_levels = compute_num_length_levels(intervals)
    progress = ProgressBar(runs, "Online randomized level", enabled=show_progress)

    for run_index in range(runs):
        solution, chosen_level = run_online_random_level_greedy(
            intervals,
            seed=seed + run_index,
        )
        counts = count_by_group(solution)
        chosen_levels.append(chosen_level)
        run_counts.append(counts)
        if debug_runs:
            print(
                f"Online randomized level run {run_index + 1}: "
                f"level={chosen_level}, selected={len(solution)}"
            )
        progress.advance()

    progress.finish()
    mean_by_group = mean_counts_by_group(run_counts, opt_by_group.keys())
    expected_total = sum(mean_by_group.values())
    return {
        "runs": runs,
        "num_levels": num_levels,
        "expected_total": expected_total,
        "mean_by_group": mean_by_group,
        "fairness": ex_ante_fairness(mean_by_group, opt_by_group),
        "fraction_opt": safe_fraction(expected_total, global_opt),
        "inverse_ratio": safe_inverse_ratio(global_opt, expected_total),
        "chosen_levels": chosen_levels,
    }


# METRICS


def deterministic_fairness(
    counts: dict[int, int],
    opt_by_group: dict[int, int],
) -> float:
    ratios = [
        opt / counts.get(group, 0) if counts.get(group, 0) > 0 else math.inf
        for group, opt in opt_by_group.items()
        if opt > 0
    ]
    return max(ratios) if ratios else 1.0


def ex_ante_fairness(
    mean_by_group: dict[int, float],
    opt_by_group: dict[int, int],
) -> float:
    ratios = [
        opt / mean_by_group.get(group, 0.0)
        if mean_by_group.get(group, 0.0) > 0
        else math.inf
        for group, opt in opt_by_group.items()
        if opt > 0
    ]
    return max(ratios) if ratios else 1.0


def mean_counts_by_group(
    run_counts: list[dict[int, int]],
    groups: object,
) -> dict[int, float]:
    ordered_groups = sorted(groups)
    if not run_counts:
        return {group: 0.0 for group in ordered_groups}
    return {
        group: mean(counts.get(group, 0) for counts in run_counts)
        for group in ordered_groups
    }


def safe_fraction(value: float, denominator: float) -> float:
    return value / denominator if denominator > 0 else 0.0


def safe_inverse_ratio(reference: float, value: float) -> float:
    return reference / value if value > 0 else math.inf


def offline_approximation_bound(delta: float, k: int) -> float:
    denominator = delta + k - 1
    return (delta * k / denominator) if denominator > 0 else 0.0


def format_float(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.3f}"


def build_result_row(
    input_file: Path,
    k: int,
    alpha: float | None,
    r: int | None,
    algorithm: str,
    algorithm_type: str,
    runs: int,
    selected: float,
    fairness: float,
    fraction_opt: float,
    inverse_ratio: float,
    delta: float,
    num_levels: int | None = None,
) -> dict:
    return {
        "input_file": str(input_file),
        "k": k,
        "alpha": alpha,
        "r": r,
        "algorithm": algorithm,
        "algorithm_type": algorithm_type,
        "runs": runs,
        "selected": selected,
        "fairness": fairness,
        "fraction_opt": fraction_opt,
        "inverse_ratio": inverse_ratio,
        "delta": delta,
        "num_levels": num_levels,
    }


def save_results_to_csv(results: list[dict], output_path: str) -> None:
    if not results:
        raise ValueError("No experiment results to save")

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# OUTPUT


def print_instance_summary(
    input_file: Path,
    intervals: list[Interval],
    opt_by_group: dict[int, int],
    global_opt: int,
) -> None:
    groups = get_groups(intervals)
    lengths = [interval.length for interval in intervals if interval.length > 0]
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    delta = (max_length / min_length) if min_length > 0 else 0.0
    num_levels = compute_num_length_levels(intervals)

    print("\nINSTANCE INFORMATION")
    print(f"Input file:            {input_file}")
    print(f"Intervals:             {len(intervals)}")
    print(f"Groups (k):            {len(groups)}")
    print(f"Minimum length:        {min_length}")
    print(f"Maximum length:        {max_length}")
    print(f"Delta:                 {delta:.3f}")
    print(f"Offline fairness bound:{len(groups): .3f}")
    print(f"Offline approx bound:  {offline_approximation_bound(delta, len(groups)):.3f}")
    print(f"Length levels:         {num_levels}")
    print(f"Global offline OPT:    {global_opt}")
    print(f"OPT by group:          {opt_by_group}")


def print_algorithm_detail(
    name: str,
    selected: float,
    by_group,
    fairness: float,
    fraction_label: str,
    fraction_value: float,
    ratio_label: str,
    ratio_value: float,
) -> None:
    print(f"\n{name}")
    print(f"Selected:              {selected:.2f}" if isinstance(selected, float) and not selected.is_integer() else f"Selected:              {int(selected)}")
    print(f"Selected by group:     {by_group}")
    print(f"Fairness ratio:        {format_float(fairness)}")
    print(f"{fraction_label}:      {fraction_value:.3f}")
    print(f"{ratio_label}:         {format_float(ratio_value)}")


def print_comparison_table(
    title: str,
    rows: list[tuple[str, float, float, float, float]],
    online: bool = False,
) -> None:
    print(f"\n{title}")
    if online:
        print(
            f"{'Algorithm':<24}{'Selected':<12}{'Fairness Ratio':<16}"
            f"{'Fraction Offline OPT':<22}{'Offline OPT / ALG':<18}"
        )
    else:
        print(
            f"{'Algorithm':<24}{'Selected':<12}{'Fairness Ratio':<16}"
            f"{'Fraction OPT':<16}{'OPT / ALG':<12}"
        )

    for name, selected, fairness, fraction_opt, inverse_ratio in rows:
        selected_text = (
            str(int(selected))
            if float(selected).is_integer()
            else f"{selected:.2f}"
        )
        print(
            f"{name:<24}{selected_text:<12}{format_float(fairness):<16}"
            f"{fraction_opt:<16.3f}{format_float(inverse_ratio):<12}"
        )


def print_selected_blocks(blocks: list[Block]) -> None:
    print("\nSELECTED BLOCKS")
    print(f"{'Step':<8}{'Group':<8}{'Block':<8}{'Size':<8}{'Span':<20}")
    for step, block in enumerate(blocks, start=1):
        span = f"[{block.start}, {block.finish})"
        print(f"{step:<8}{block.group:<8}{block.index:<8}{block.size:<8}{span:<20}")


# INPUT LOADING


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
    intervals: list[Interval] = []
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

    intervals: list[Interval] = []
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
    intervals: list[Interval] = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
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
    suffixes = "".join(Path(path).suffixes).lower()
    resolved_format = (file_format or Path(path).suffix.lstrip(".")).lower()
    if suffixes.endswith(".swf.gz"):
        resolved_format = "swf"

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
    suffixes = "".join(path.suffixes).lower()
    return path.is_file() and (
        path.suffix.lower() in {".csv", ".json", ".swf"}
        or suffixes.endswith(".swf.gz")
    )


def collect_input_files_from_path(path: str) -> list[Path]:
    input_path = Path(path)
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        input_files = [
            candidate
            for candidate in sorted(input_path.rglob("*"))
            if _supported_input_file(candidate)
        ]
        if not input_files:
            raise ValueError(f"No supported input files found under: {input_path}")
        return input_files

    if _supported_input_file(input_path):
        return [input_path]

    raise ValueError(f"Input must be a folder or a csv/json/swf file: {input_path}")


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

            for input_file in collect_input_files_from_path(str(entry)):
                resolved = input_file.resolve()
                if resolved not in seen:
                    collected.append(resolved)
                    seen.add(resolved)

    if not collected:
        raise ValueError(f"No supported input files were found in manifest: {manifest}")
    return collected


# EXPERIMENT PIPELINE


def evaluate_workload(
    input_file: Path,
    intervals: list[Interval],
    alpha: float,
    runs: int,
    seed: int,
    show_progress: bool = True,
    debug_runs: bool = False,
    debug_blocks: bool = False,
) -> list[dict]:
    results: list[dict] = []

    if not intervals:
        raise ValueError(f"No valid intervals were loaded from input file: {input_file}")

    groups = get_groups(intervals)
    k = len(groups)
    if alpha <= 0:
        raise ValueError(f"alpha must be positive. Got alpha={alpha}")
    resolved_r = math.ceil(alpha * k)
    if resolved_r < k:
        raise ValueError(
            f"Input file {input_file} has k={k}, alpha={alpha}, "
            f"and r=ceil(alpha*k)={resolved_r}. Require r >= k, so use alpha >= 1."
        )

    global_opt_solution = run_offline_greedy(intervals)
    global_opt = len(global_opt_solution)
    _, opt_by_group = compute_optimal_per_group(intervals)
    lengths = [interval.length for interval in intervals if interval.length > 0]
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    delta = (max_length / min_length) if min_length > 0 else 0.0

    print("\n" + "=" * 80)
    print_instance_summary(input_file, intervals, opt_by_group, global_opt)

    print("\n==================================================")
    print("OFFLINE ALGORITHMS")
    print("==================================================")

    offline_greedy_counts = count_by_group(global_opt_solution)
    offline_greedy_fairness = deterministic_fairness(offline_greedy_counts, opt_by_group)
    offline_greedy_fraction = 1.0 if global_opt > 0 else 0.0
    offline_greedy_ratio = 1.0 if global_opt > 0 else math.inf
    print_algorithm_detail(
        "Offline Greedy / Global OPT",
        len(global_opt_solution),
        offline_greedy_counts,
        offline_greedy_fairness,
        "Fraction of global OPT",
        offline_greedy_fraction,
        "OPT / ALG",
        offline_greedy_ratio,
    )
    results.append(
        build_result_row(
            input_file=input_file,
            k=k,
            alpha=None,
            r=None,
            algorithm="offline_greedy",
            algorithm_type="deterministic",
            runs=1,
            selected=len(global_opt_solution),
            fairness=offline_greedy_fairness,
            fraction_opt=offline_greedy_fraction,
            inverse_ratio=offline_greedy_ratio,
            delta=delta,
            num_levels=None,
        )
    )

    deterministic_solution, selected_blocks = run_offline_deterministic_r_block(
        intervals,
        r=resolved_r,
    )
    deterministic_counts = count_by_group(deterministic_solution)
    deterministic_fair = deterministic_fairness(deterministic_counts, opt_by_group)
    deterministic_fraction = safe_fraction(len(deterministic_solution), global_opt)
    deterministic_ratio = safe_inverse_ratio(global_opt, len(deterministic_solution))
    approximation_bound = offline_approximation_bound(delta, k)
    print("\nOffline Deterministic")
    print(f"k:                     {k}")
    print(f"alpha:                 {alpha:.3f}")
    print(f"r:                     {resolved_r}")
    print(f"Offline fairness bound:{k: .3f}")
    print(f"Offline approx bound:  {approximation_bound:.3f}")
    print(f"Selected:              {len(deterministic_solution)}")
    print(f"Selected by group:     {deterministic_counts}")
    print(f"Fairness ratio:        {format_float(deterministic_fair)}")
    print(f"Fraction of global OPT:{deterministic_fraction: .3f}")
    print(f"Observed OPT / ALG ratio: {format_float(deterministic_ratio)}")
    if debug_blocks:
        print_selected_blocks(selected_blocks)
    results.append(
        build_result_row(
            input_file=input_file,
            k=k,
            alpha=alpha,
            r=resolved_r,
            algorithm="offline_deterministic",
            algorithm_type="deterministic",
            runs=1,
            selected=len(deterministic_solution),
            fairness=deterministic_fair,
            fraction_opt=deterministic_fraction,
            inverse_ratio=deterministic_ratio,
            delta=delta,
            num_levels=None,
        )
    )

    offline_randomized = run_offline_randomized_multiple_times(
        intervals,
        opt_by_group=opt_by_group,
        global_opt=global_opt,
        runs=runs,
        seed=seed,
        show_progress=show_progress,
        debug_runs=debug_runs,
    )
    print("\nOffline Randomized")
    print(f"Runs:                  {offline_randomized['runs']}")
    print(f"Expected selected:     {offline_randomized['expected_total']:.2f}")
    print(f"Mean selected by group:{offline_randomized['mean_by_group']}")
    print(f"Estimated ex-ante fairness ratio: {format_float(offline_randomized['fairness'])}")
    print(f"Fraction of global OPT:{offline_randomized['fraction_opt']: .3f}")
    print(f"Observed OPT / ALG ratio: {format_float(offline_randomized['inverse_ratio'])}")
    results.append(
        build_result_row(
            input_file=input_file,
            k=k,
            alpha=None,
            r=None,
            algorithm="offline_randomized",
            algorithm_type="randomized",
            runs=offline_randomized["runs"],
            selected=offline_randomized["expected_total"],
            fairness=offline_randomized["fairness"],
            fraction_opt=offline_randomized["fraction_opt"],
            inverse_ratio=offline_randomized["inverse_ratio"],
            delta=delta,
            num_levels=None,
        )
    )

    print_comparison_table(
        "OFFLINE COMPARISON",
        [
            ("Offline Greedy", len(global_opt_solution), offline_greedy_fairness, 1.0 if global_opt > 0 else 0.0, 1.0 if global_opt > 0 else math.inf),
            ("Offline Deterministic", len(deterministic_solution), deterministic_fair, deterministic_fraction, deterministic_ratio),
            ("Offline Randomized", offline_randomized["expected_total"], offline_randomized["fairness"], offline_randomized["fraction_opt"], offline_randomized["inverse_ratio"]),
        ],
    )

    print("\n==================================================")
    print("ONLINE ALGORITHMS")
    print("==================================================")

    simple_online_solution = run_simple_online_greedy(intervals)
    simple_online_counts = count_by_group(simple_online_solution)
    simple_online_fair = deterministic_fairness(simple_online_counts, opt_by_group)
    simple_online_fraction = safe_fraction(len(simple_online_solution), global_opt)
    simple_online_ratio = safe_inverse_ratio(global_opt, len(simple_online_solution))
    print_algorithm_detail(
        "Simple Online Greedy",
        len(simple_online_solution),
        simple_online_counts,
        simple_online_fair,
        "Fraction of global offline OPT",
        simple_online_fraction,
        "Observed online OPT / ALG ratio",
        simple_online_ratio,
    )
    results.append(
        build_result_row(
            input_file=input_file,
            k=k,
            alpha=None,
            r=None,
            algorithm="simple_online_greedy",
            algorithm_type="deterministic",
            runs=1,
            selected=len(simple_online_solution),
            fairness=simple_online_fair,
            fraction_opt=simple_online_fraction,
            inverse_ratio=simple_online_ratio,
            delta=delta,
            num_levels=None,
        )
    )

    online_randomized = run_online_random_group_level_multiple_times(
        intervals,
        opt_by_group=opt_by_group,
        global_opt=global_opt,
        runs=runs,
        seed=seed,
        show_progress=show_progress,
        debug_runs=debug_runs,
    )
    print("\nOnline Randomized")
    print(f"Runs:                  {online_randomized['runs']}")
    print(f"Length levels:         {online_randomized['num_levels']}")
    print(f"Expected selected:     {online_randomized['expected_total']:.2f}")
    print(f"Mean selected by group:{online_randomized['mean_by_group']}")
    print(f"Estimated ex-ante fairness ratio: {format_float(online_randomized['fairness'])}")
    print(f"Fraction of global offline OPT:{online_randomized['fraction_opt']: .3f}")
    print(f"Observed offline OPT / ALG ratio: {format_float(online_randomized['inverse_ratio'])}")
    results.append(
        build_result_row(
            input_file=input_file,
            k=k,
            alpha=None,
            r=None,
            algorithm="online_randomized",
            algorithm_type="randomized",
            runs=online_randomized["runs"],
            selected=online_randomized["expected_total"],
            fairness=online_randomized["fairness"],
            fraction_opt=online_randomized["fraction_opt"],
            inverse_ratio=online_randomized["inverse_ratio"],
            delta=delta,
            num_levels=online_randomized["num_levels"],
        )
    )

    online_randomized_level = run_online_random_level_greedy_multiple_times(
        intervals,
        opt_by_group=opt_by_group,
        global_opt=global_opt,
        runs=runs,
        seed=seed,
        show_progress=show_progress,
        debug_runs=debug_runs,
    )
    print("\nOnline Randomized Level Greedy")
    print(f"Runs:                  {online_randomized_level['runs']}")
    print(f"Length levels:         {online_randomized_level['num_levels']}")
    print(f"Expected selected:     {online_randomized_level['expected_total']:.2f}")
    print(f"Mean selected by group:{online_randomized_level['mean_by_group']}")
    print(f"Estimated ex-ante fairness ratio: {format_float(online_randomized_level['fairness'])}")
    print(f"Fraction of global offline OPT:{online_randomized_level['fraction_opt']: .3f}")
    print(f"Observed offline OPT / ALG ratio: {format_float(online_randomized_level['inverse_ratio'])}")
    results.append(
        build_result_row(
            input_file=input_file,
            k=k,
            alpha=None,
            r=None,
            algorithm="online_randomized_level_greedy",
            algorithm_type="randomized",
            runs=online_randomized_level["runs"],
            selected=online_randomized_level["expected_total"],
            fairness=online_randomized_level["fairness"],
            fraction_opt=online_randomized_level["fraction_opt"],
            inverse_ratio=online_randomized_level["inverse_ratio"],
            delta=delta,
            num_levels=online_randomized_level["num_levels"],
        )
    )

    print_comparison_table(
        "ONLINE COMPARISON",
        [
            ("Simple Online Greedy", len(simple_online_solution), simple_online_fair, simple_online_fraction, simple_online_ratio),
            ("Online Randomized", online_randomized["expected_total"], online_randomized["fairness"], online_randomized["fraction_opt"], online_randomized["inverse_ratio"]),
            ("Online Randomized Level", online_randomized_level["expected_total"], online_randomized_level["fairness"], online_randomized_level["fraction_opt"], online_randomized_level["inverse_ratio"]),
        ],
        online=True,
    )

    return results


# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interval scheduling experiments")
    parser.add_argument("--input", type=str, help="Path to one input file or folder")
    parser.add_argument(
        "--input-list",
        type=str,
        help="Path to a text file listing input files and/or folders",
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Offline Deterministic block multiplier. Uses r = ceil(alpha * k).",
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_results.csv",
        help="Path to the CSV file used to store all experiment result rows.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress bars for repeated randomized experiments.",
    )
    parser.add_argument(
        "--debug-runs",
        action="store_true",
        help="Print every randomized run.",
    )
    parser.add_argument(
        "--debug-blocks",
        action="store_true",
        help="Print selected blocks for Offline Deterministic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input and args.input_list:
        raise ValueError("Use either --input or --input-list, not both")
    if not args.input and not args.input_list:
        raise ValueError("Provide --input or --input-list")

    input_files = (
        collect_input_files_from_manifest(args.input_list)
        if args.input_list
        else collect_input_files_from_path(args.input)
    )

    all_results: list[dict] = []
    for input_file in input_files:
        intervals = load_intervals_from_file(
            path=str(input_file),
            file_format=args.format,
            start_col=args.start_col,
            finish_col=args.finish_col,
            length_col=args.length_col,
            group_col=args.group_col,
        )
        workload_results = evaluate_workload(
            input_file=input_file,
            intervals=intervals,
            alpha=args.alpha,
            runs=args.runs,
            seed=args.seed,
            show_progress=not args.no_progress,
            debug_runs=args.debug_runs,
            debug_blocks=args.debug_blocks,
        )
        all_results.extend(workload_results)

    save_results_to_csv(all_results, args.output)
    print(f"\nSaved experiment results to: {args.output}")


if __name__ == "__main__":
    main()
