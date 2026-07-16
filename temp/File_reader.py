import argparse
import math
import random
from pathlib import Path
import pandas as pd


SWF_COLUMNS = [
    "job_id",
    "submit_time",
    "wait_time",
    "run_time",
    "allocated_processors",
    "avg_cpu_time",
    "used_memory",
    "requested_processors",
    "requested_time",
    "requested_memory",
    "status",
    "user_id",
    "group_id",
    "executable_number",
    "queue_number",
    "partition_number",
    "preceding_job_number",
    "think_time",
]


ASSIGNMENT_METHODS = [
    "uniform",
    "exponential",
    "length_quantile",
    "length_delta",
    "containment_quantile",
]

RANDOM_ASSIGNMENT_METHODS = [
    "uniform",
    "exponential",
]

DEFAULT_START_SEED = 42
DEFAULT_NUM_SEEDS = 1
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 10
OUTPUT_FOLDER_NAME = "generated_intervals"


def workload_folder_name(file_path: Path) -> str:
    base_name = file_path.name.replace(".swf.gz", "")
    parts = base_name.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else base_name


def discrete_exponential_weights(num_groups: int, decay: float = 0.6) -> list[float]:
    return [
        decay ** group_index
        for group_index in range(num_groups)
    ]


def build_group_ids(
    df: pd.DataFrame,
    num_groups: int,
    seed: int | None,
    assignment_method: str,
) -> pd.Series:
    num_jobs = len(df)

    if assignment_method == "uniform":
        rng = random.Random(seed)
        group_ids = [
            rng.randint(1, num_groups)
            for _ in range(num_jobs)
        ]
        return pd.Series(group_ids, index=df.index)

    if assignment_method == "exponential":
        rng = random.Random(seed)
        groups = list(range(1, num_groups + 1))
        weights = discrete_exponential_weights(num_groups)
        group_ids = [
            rng.choices(groups, weights=weights, k=1)[0]
            for _ in range(num_jobs)
        ]
        return pd.Series(group_ids, index=df.index)

    if assignment_method == "length_quantile":
        sorted_index = df.sort_values(["run_time", "job_id"]).index
        group_ids = [
            rank * num_groups // num_jobs + 1
            for rank in range(num_jobs)
        ]
        return pd.Series(group_ids, index=sorted_index)

    if assignment_method == "length_delta":
        min_length = df["run_time"].min()
        max_length = df["run_time"].max()
        delta = max_length / min_length

        if delta == 1:
            return pd.Series([1] * num_jobs, index=df.index)

        log_delta = math.log(delta)
        group_ids = df["run_time"].apply(
            lambda length: min(
                int(num_groups * math.log(length / min_length) / log_delta) + 1,
                num_groups,
            )
        )
        return pd.Series(group_ids, index=df.index)

    if assignment_method == "containment_quantile":
        interval_df = pd.DataFrame(
            {
                "start": df["submit_time"],
                "finish": df["submit_time"] + df["run_time"],
            },
            index=df.index,
        )

        containment_counts = []
        for _, interval_i in interval_df.iterrows():
            count = (
                (interval_df["start"] >= interval_i["start"])
                & (interval_df["finish"] <= interval_i["finish"])
            ).sum() - 1
            containment_counts.append(count)

        interval_df["_containment_count"] = containment_counts
        group_ids = pd.qcut(
            interval_df["_containment_count"].rank(method="first"),
            q=num_groups,
            labels=False,
        ) + 1
        return pd.Series(group_ids, index=df.index)

    raise ValueError(
        f"Unknown assignment_method: {assignment_method}. "
        f"Use one of: {', '.join(ASSIGNMENT_METHODS)}"
    )


def convert_one_file(
    file_path: Path,
    output_folder: Path,
    num_groups: int = 10,
    seed: int | None = 42,
    assignment_method: str = "uniform",
) -> None:
    if seed is None:
        output_suffix = f"_interval_{assignment_method}_{num_groups}groups.csv"
    else:
        output_suffix = f"_interval_{assignment_method}_{num_groups}groups_seed{seed}.csv"
    output_name = file_path.name.replace(".swf.gz", output_suffix)
    output_path = output_folder / output_name

    print("\n" + "=" * 80)
    print(f"Reading: {file_path.name}")
    print(f"Assignment method: {assignment_method}")

    df = pd.read_csv(
        file_path,
        compression="gzip",
        comment=";",
        sep=r"\s+",
        names=SWF_COLUMNS,
        engine="python",
    )

    df = df[
        (df["submit_time"] >= 0)
        & (df["run_time"] > 0)
    ].copy()

    num_jobs = len(df)

    print(f"Rows after filtering: {len(df)}")
    print(f"Number of groups: {num_groups}")
    if seed is not None:
        print(f"Random seed: {seed}")

    if num_jobs == 0:
        interval_df = pd.DataFrame(columns=["start", "length", "group_id"])
        interval_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print("No valid jobs found.")
        return

    df["new_group_id"] = build_group_ids(
        df=df,
        num_groups=num_groups,
        seed=seed,
        assignment_method=assignment_method,
    )
    df["new_group_id"] = df["new_group_id"].astype(int)

    interval_df = pd.DataFrame({
        "start": df["submit_time"],
        "length": df["run_time"],
        "group_id": df["new_group_id"],
    })

    interval_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("Group sizes by number of jobs:")
    print(interval_df["group_id"].value_counts().sort_index())


def parse_seed_list(raw_value: str) -> list[int]:
    seeds = [
        int(part.strip())
        for part in raw_value.split(",")
        if part.strip()
    ]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def parse_args() -> argparse.Namespace:
    project_folder = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Convert SWF workloads into interval CSV files."
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=project_folder,
        help="Folder containing .swf.gz workload files.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=project_folder / OUTPUT_FOLDER_NAME,
        help="Folder where generated interval CSV files are written.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=DEFAULT_START_SEED,
        help="First seed to use when generating a consecutive seed range.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of consecutive seeds to use, starting from --start-seed.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="Optional comma-separated seed list. Overrides --start-seed/--num-seeds.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=DEFAULT_K_MIN,
        help="Smallest number of groups to generate.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=DEFAULT_K_MAX,
        help="Largest number of groups to generate.",
    )
    return parser.parse_args()


def build_seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        return parse_seed_list(args.seeds)

    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")

    return list(range(args.start_seed, args.start_seed + args.num_seeds))


def main():
    args = parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.k_min <= 0 or args.k_max < args.k_min:
        raise ValueError("--k-min must be positive and --k-max must be >= --k-min")

    seeds = build_seed_list(args)
    k_values = range(args.k_min, args.k_max + 1)
    files = sorted(input_folder.glob("*.swf.gz"))

    print("Input folder:", input_folder)
    print("Output folder:", output_folder)
    print("Number of .swf.gz files found:", len(files))
    print("K values:", list(k_values))
    print("Random seeds:", seeds)
    print("Random assignment methods:", RANDOM_ASSIGNMENT_METHODS)

    if len(files) == 0:
        print("No .swf.gz files found.")
        return

    for file_path in files:
        workload_output_folder = output_folder / workload_folder_name(file_path)
        workload_output_folder.mkdir(parents=True, exist_ok=True)

        for num_groups in k_values:
            for assignment_method in ASSIGNMENT_METHODS:
                seeds_to_use = (
                    seeds
                    if assignment_method in RANDOM_ASSIGNMENT_METHODS
                    else [None]
                )
                for seed in seeds_to_use:
                    convert_one_file(
                        file_path,
                        workload_output_folder,
                        num_groups=num_groups,
                        seed=seed,
                        assignment_method=assignment_method,
                    )

    print("\nAll files converted.")


if __name__ == "__main__":
    main()
