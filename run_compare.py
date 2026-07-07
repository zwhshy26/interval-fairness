import argparse
from pathlib import Path

from main import load_intervals_from_file, run_fairness_with_intervals


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "temp" / "CTC-SP2-1996-3.1-cln_interval_10groups.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "temp" / "CTC_full_compare.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interval fairness algorithms and export comparison CSV."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input CSV/JSON/SWF file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output comparison CSV path.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--bpa-variant-alpha", type=float, default=2.0)
    parser.add_argument(
        "--limit-intervals",
        type=int,
        default=None,
        help="Use only the first N intervals. Useful for quick tests.",
    )
    parser.add_argument(
        "--run-dp",
        action="store_true",
        help="Also run DP algorithms. Use with --limit-intervals for large data.",
    )
    parser.add_argument("--dp-beta", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intervals = load_intervals_from_file(args.input)
    loaded_count = len(intervals)

    if args.limit_intervals is not None:
        if args.limit_intervals <= 0:
            raise ValueError("--limit-intervals must be positive")
        intervals = intervals[: args.limit_intervals]

    print(f"Loaded intervals: {loaded_count}")
    print(f"Using intervals:  {len(intervals)}")
    if args.run_dp and args.limit_intervals is None:
        print("WARNING: --run-dp without --limit-intervals will run DP on all intervals.")

    dataset_name = Path(args.input).stem
    if args.limit_intervals is not None:
        dataset_name = f"{dataset_name}_first_{args.limit_intervals}"

    run_fairness_with_intervals(
        intervals=intervals,
        num_runs=args.runs,
        seed=args.seed,
        alpha=args.alpha,
        show_progress=False,
        bpa_variant_alpha=args.bpa_variant_alpha,
        dataset_name=dataset_name,
        results_csv=args.output,
        run_dp=args.run_dp,
        dp_beta=args.dp_beta,
    )


if __name__ == "__main__":
    main()
