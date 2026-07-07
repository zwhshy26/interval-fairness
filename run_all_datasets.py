import argparse
from pathlib import Path

from main import load_intervals_from_file, run_fairness_with_intervals


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "Input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "temp" / "all_runs"
SUPPORTED_SUFFIXES = {".csv", ".json", ".swf"}


def iter_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interval fairness experiments for every dataset in Input."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing dataset files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where per-dataset comparison CSVs are written.",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--bpa-blocks", type=int, default=None)
    parser.add_argument("--bpa-variant-alpha", type=float, default=None)
    parser.add_argument(
        "--limit-intervals",
        type=int,
        default=None,
        help="Use only the first N intervals from each dataset.",
    )
    parser.add_argument(
        "--run-dp",
        action="store_true",
        help="Also run DP algorithms. Use with --limit-intervals for large data.",
    )
    parser.add_argument("--dp-beta", type=float, default=0.5)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next dataset if one dataset fails.",
    )
    return parser.parse_args()


def run_one_dataset(path: Path, args: argparse.Namespace, output_dir: Path) -> None:
    intervals = load_intervals_from_file(str(path))
    loaded_count = len(intervals)

    if args.limit_intervals is not None:
        if args.limit_intervals <= 0:
            raise ValueError("--limit-intervals must be positive")
        intervals = intervals[: args.limit_intervals]

    dataset_name = path.stem
    if args.limit_intervals is not None:
        dataset_name = f"{dataset_name}_first_{args.limit_intervals}"

    output_csv = output_dir / f"{dataset_name}_compare.csv"

    print("\n" + "=" * 80)
    print(f"Dataset:          {path.name}")
    print(f"Loaded intervals: {loaded_count}")
    print(f"Using intervals:  {len(intervals)}")
    print(f"Output CSV:       {output_csv}")

    run_fairness_with_intervals(
        intervals=intervals,
        num_runs=args.runs,
        seed=args.seed,
        alpha=args.alpha,
        show_progress=False,
        bpa_blocks=args.bpa_blocks,
        bpa_variant_alpha=args.bpa_variant_alpha,
        dataset_name=dataset_name,
        results_csv=str(output_csv),
        run_dp=args.run_dp,
        dp_beta=args.dp_beta,
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = iter_input_files(input_dir)
    if not input_files:
        raise ValueError(f"No supported input files found in {input_dir}")

    print(f"Found {len(input_files)} dataset(s) in {input_dir}")
    if args.run_dp and args.limit_intervals is None:
        print("WARNING: --run-dp without --limit-intervals may be very slow.")

    failures = []
    for path in input_files:
        try:
            run_one_dataset(path, args, output_dir)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((path.name, str(exc)))
            print(f"FAILED: {path.name}: {exc}")

    print("\n" + "=" * 80)
    print(f"Finished {len(input_files) - len(failures)}/{len(input_files)} dataset(s).")
    if failures:
        print("Failures:")
        for name, message in failures:
            print(f"- {name}: {message}")


if __name__ == "__main__":
    main()
