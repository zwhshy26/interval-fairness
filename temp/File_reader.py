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
]

SEEDS = [42, 43, 44, 45, 46]
K_VALUES = range(2, 11)
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
    seed: int,
    assignment_method: str,
) -> pd.Series:
    num_jobs = len(df)
    rng = random.Random(seed)

    if assignment_method == "uniform":
        group_ids = [
            rng.randint(1, num_groups)
            for _ in range(num_jobs)
        ]
        return pd.Series(group_ids, index=df.index)

    if assignment_method == "exponential":
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

    raise ValueError(
        f"Unknown assignment_method: {assignment_method}. "
        f"Use one of: {', '.join(ASSIGNMENT_METHODS)}"
    )


def convert_one_file(
    file_path: Path,
    output_folder: Path,
    num_groups: int = 10,
    seed: int = 42,
    assignment_method: str = "uniform",
) -> None:
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


def main():
    project_folder = Path(__file__).resolve().parent

    input_folder = project_folder
    output_folder = project_folder / OUTPUT_FOLDER_NAME
    output_folder.mkdir(parents=True, exist_ok=True)

    files = list(input_folder.glob("*.swf.gz"))

    print("Input folder:", input_folder)
    print("Output folder:", output_folder)
    print("Number of .swf.gz files found:", len(files))
    print("K values:", list(K_VALUES))
    print("Seeds:", SEEDS)

    if len(files) == 0:
        print("No .swf.gz files found.")
        return

    for file_path in files:
        workload_output_folder = output_folder / workload_folder_name(file_path)
        workload_output_folder.mkdir(parents=True, exist_ok=True)

        for num_groups in K_VALUES:
            for assignment_method in ASSIGNMENT_METHODS:
                for seed in SEEDS:
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
