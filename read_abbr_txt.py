import argparse
import math
import random
from pathlib import Path

from main import (
    Interval,
    compute_group_opts,
    fair_dp_with_beta,
    find_best_fairness_by_quota_enumeration,
    run_fairness_with_intervals,
)


def load_intervals_from_abbr_txt(
    path: str,
    group: int = 1,
    random_groups: bool = False,
    num_groups: int = 1,
    seed: int = 42,
) -> list[Interval]:
    intervals: list[Interval] = []
    rng = random.Random(seed)

    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Line {line_number} in {path} does not have two numeric columns"
                )

            length = int(float(parts[0]))
            start = int(float(parts[1]))

            if length <= 0:
                continue

            assigned_group = rng.randint(1, num_groups) if random_groups else group
            intervals.append(Interval(start=start, length=length, group=assigned_group))

    if not intervals:
        raise ValueError(f"No valid intervals found in {path}")

    return intervals


def intervals_to_zero_based_tuples(
    intervals: list[Interval], k: int
) -> list[tuple[int, int, int]]:
    """
    Convert loaded Interval objects to (start, end, group) tuples for the DP.

    The existing txt loader historically used groups 1..k. The offline DP follows
    the problem statement and expects groups 0..k-1, so this helper accepts either
    convention and normalizes to 0-based groups.
    """
    groups = {interval.group for interval in intervals}

    if groups.issubset(set(range(k))):
        group_offset = 0
    elif groups.issubset(set(range(1, k + 1))):
        group_offset = 1
    else:
        raise ValueError(
            f"Loaded group ids {sorted(groups)} do not match either 0..{k - 1} or 1..{k}"
        )

    return [
        (interval.start, interval.finish, int(interval.group) - group_offset)
        for interval in intervals
    ]


def quota_state_count(opt_per_group: list[int]) -> int:
    total = 1
    for opt in opt_per_group:
        total *= opt + 1
    return total


def check_dp_size_or_raise(
    dp_intervals: list[tuple[int, int, int]],
    k: int,
    max_quota_states: int,
    allow_large_dp: bool,
) -> list[int]:
    opt_per_group = compute_group_opts(dp_intervals, k)
    state_count = quota_state_count(opt_per_group)
    if not allow_large_dp and state_count > max_quota_states:
        raise ValueError(
            "Offline DP would be large for this input: "
            f"quota states={state_count}, opt_per_group={opt_per_group}. "
            "Try fewer groups/intervals, lower --max-quota-states only if appropriate, "
            "or pass --allow-large-dp to run anyway."
        )
    return opt_per_group


def check_beta_dp_size_or_raise(
    dp_intervals: list[tuple[int, int, int]],
    k: int,
    beta: float,
    max_quota_states: int,
    allow_large_dp: bool,
) -> list[int]:
    opt_per_group = compute_group_opts(dp_intervals, k)
    quotas = [math.ceil(beta * opt) for opt in opt_per_group]
    state_count = quota_state_count(quotas)
    if not allow_large_dp and state_count > max_quota_states:
        raise ValueError(
            "Offline beta DP would be large for this input: "
            f"quota states={state_count}, quotas={quotas}, "
            f"opt_per_group={opt_per_group}. "
            "Try fewer groups/intervals, increase --max-quota-states if appropriate, "
            "or pass --allow-large-dp to run anyway."
        )
    return opt_per_group


def greedy_select_tuples(
    intervals: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """Classical global greedy interval scheduling over all groups."""
    selected: list[tuple[int, int, int]] = []
    current_end = None

    for interval in sorted(intervals, key=lambda item: (item[1], item[0])):
        start, end, _ = interval
        if current_end is None or current_end <= start:
            selected.append(interval)
            current_end = end

    return selected


def count_tuple_groups(intervals: list[tuple[int, int, int]], k: int) -> list[int]:
    counts = [0] * k
    for _, _, group in intervals:
        counts[group] += 1
    return counts


def print_group_comparison(
    title: str,
    opt_per_group: list[int],
    quotas: list[int],
    dp_selected: list[tuple[int, int, int]] | None,
    greedy_selected: list[tuple[int, int, int]],
    k: int,
) -> None:
    dp_counts = count_tuple_groups(dp_selected or [], k)
    greedy_counts = count_tuple_groups(greedy_selected, k)

    print(f"\n{title}")
    print(f"{'Group':<8}{'Quota':<8}{'DP':<8}{'Greedy':<8}{'OPT':<8}{'DP/OPT':<10}{'Greedy/OPT':<12}")
    for group in range(k):
        opt = opt_per_group[group]
        dp_ratio = dp_counts[group] / opt if opt > 0 else 1.0
        greedy_ratio = greedy_counts[group] / opt if opt > 0 else 1.0
        print(
            f"{group:<8}"
            f"{quotas[group]:<8}"
            f"{dp_counts[group]:<8}"
            f"{greedy_counts[group]:<8}"
            f"{opt:<8}"
            f"{dp_ratio:<10.3f}"
            f"{greedy_ratio:<12.3f}"
        )

    print(f"Total DP selected: {sum(dp_counts)}")
    print(f"Total greedy selected: {len(greedy_selected)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read abbreviated interval txt files and run fairness analysis"
    )
    parser.add_argument("--input", required=True, help="Path to the *_abbr.txt file")
    parser.add_argument("--group", type=int, default=1, help="Group id for all intervals")
    parser.add_argument(
        "--random-groups",
        action="store_true",
        help="Assign each interval to a random group instead of using one fixed group",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=2,
        help="Number of groups used when --random-groups is enabled",
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of fairness runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Use global greedy with probability alpha, otherwise use the permutation-based group greedy algorithm",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="How many intervals to print before running fairness",
    )
    parser.add_argument(
        "--offline-beta",
        type=float,
        help="Run the offline fair DP for this beta value after loading the txt file",
    )
    parser.add_argument(
        "--best-fairness-dp",
        action="store_true",
        help="Run quota enumeration DP to find the best achievable fairness ratio",
    )
    parser.add_argument(
        "--skip-original-fairness",
        action="store_true",
        help="Only load the txt file and run requested offline DP commands",
    )
    parser.add_argument(
        "--max-intervals",
        type=int,
        help="Only load the first N valid intervals from the txt file, useful for testing DP",
    )
    parser.add_argument(
        "--max-quota-states",
        type=int,
        default=200_000,
        help="Safety limit for product_g (OPT_g + 1) before running offline DP",
    )
    parser.add_argument(
        "--allow-large-dp",
        action="store_true",
        help="Run offline DP even if the estimated quota state count is large",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input)
    if args.num_groups < 1:
        raise ValueError("--num-groups must be at least 1")

    intervals = load_intervals_from_abbr_txt(
        str(input_path),
        group=args.group,
        random_groups=args.random_groups,
        num_groups=args.num_groups,
        seed=args.seed,
    )
    if args.max_intervals is not None:
        intervals = intervals[: args.max_intervals]

    print(f"Loaded {len(intervals)} intervals from {input_path}")
    if args.random_groups:
        print(f"Assigned groups randomly in range [1, {args.num_groups}] with seed={args.seed}")
    else:
        print(f"Assigned one fixed group: {args.group}")
    print("Preview:")
    for interval in intervals[: args.preview]:
        print(
            f"start={interval.start}, length={interval.length}, finish={interval.finish}, group={interval.group}"
        )

    if args.offline_beta is not None:
        dp_intervals = intervals_to_zero_based_tuples(intervals, args.num_groups)
        greedy_selected = greedy_select_tuples(dp_intervals)
        check_beta_dp_size_or_raise(
            dp_intervals,
            args.num_groups,
            args.offline_beta,
            args.max_quota_states,
            args.allow_large_dp,
        )
        beta_result = fair_dp_with_beta(
            intervals=dp_intervals,
            k=args.num_groups,
            beta=args.offline_beta,
        )
        print("\nOFFLINE FAIR DP")
        print(f"Beta: {args.offline_beta:.3f}")
        print(f"Feasible: {beta_result.feasible}")
        print(f"Max cardinality: {beta_result.max_cardinality}")
        print(f"OPT per group: {beta_result.opt_per_group}")
        print(f"Quotas: {beta_result.quotas}")
        print_group_comparison(
            "PER-GROUP: OFFLINE DP vs GLOBAL GREEDY",
            beta_result.opt_per_group,
            beta_result.quotas,
            beta_result.selected_intervals,
            greedy_selected,
            args.num_groups,
        )
        print(f"Selected intervals: {beta_result.selected_intervals}")

    if args.best_fairness_dp:
        dp_intervals = intervals_to_zero_based_tuples(intervals, args.num_groups)
        greedy_selected = greedy_select_tuples(dp_intervals)
        check_dp_size_or_raise(
            dp_intervals,
            args.num_groups,
            args.max_quota_states,
            args.allow_large_dp,
        )
        best_result = find_best_fairness_by_quota_enumeration(
            intervals=dp_intervals,
            k=args.num_groups,
        )
        print("\nBEST FAIRNESS BY QUOTA ENUMERATION")
        print(f"Best beta: {best_result.best_beta:.3f}")
        print(f"Best quota vector: {best_result.best_quota_vector}")
        print(f"Max cardinality: {best_result.max_cardinality_at_best_beta}")
        print(f"OPT per group: {best_result.opt_per_group}")
        print_group_comparison(
            "PER-GROUP: BEST FAIRNESS DP vs GLOBAL GREEDY",
            best_result.opt_per_group,
            best_result.best_quota_vector,
            best_result.selected_intervals,
            greedy_selected,
            args.num_groups,
        )
        print(f"Selected intervals: {best_result.selected_intervals}")

    if not args.skip_original_fairness:
        run_fairness_with_intervals(
            intervals=intervals,
            num_runs=args.runs,
            seed=args.seed,
            alpha=args.alpha,
        )
