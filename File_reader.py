import argparse
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


def convert_one_file(file_path: Path, output_folder: Path, num_groups: int = 5) -> None:
    output_name = file_path.name.replace(".swf.gz", f"_interval_{num_groups}groups.csv")
    output_path = output_folder / output_name

    print("\n" + "=" * 80)
    print(f"Reading: {file_path.name}")

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
        & (df["user_id"] >= 0)
    ].copy()

    unique_users = sorted(df["user_id"].unique())
    num_users = len(unique_users)

    print(f"Rows after filtering: {len(df)}")
    print(f"Number of unique user_id: {num_users}")

    if num_users == 0:
        print("No valid users found; skipping.")
        return

    user_to_group = {}
    for rank, user_id in enumerate(unique_users):
        group_id = rank * num_groups // num_users + 1
        user_to_group[user_id] = min(group_id, num_groups)

    df["new_group_id"] = df["user_id"].map(user_to_group)

    interval_df = pd.DataFrame(
        {
            "start": df["submit_time"],
            "length": df["run_time"],
            "group_id": df["new_group_id"],
        }
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    interval_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("Group sizes by number of jobs:")
    print(interval_df["group_id"].value_counts().sort_index())


def parse_args() -> argparse.Namespace:
    project_folder = Path(__file__).resolve().parent
    default_data_folder = project_folder.parent / "temp"
    default_output_folder = project_folder / "Input"

    parser = argparse.ArgumentParser(
        description="Convert SWF workload files into interval CSV files."
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=default_data_folder,
        help="Folder containing .swf.gz files.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=default_output_folder,
        help="Folder where interval CSV files will be written.",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=5,
        help="Number of user-based groups to create.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_groups <= 0:
        raise ValueError("--num-groups must be positive")

    files = sorted(args.input_folder.glob("*.swf.gz"))

    print("Input folder:", args.input_folder)
    print("Output folder:", args.output_folder)
    print("Number of .swf.gz files found:", len(files))
    print("Number of groups:", args.num_groups)

    if not files:
        print("No .swf.gz files found.")
        return

    for file_path in files:
        convert_one_file(file_path, args.output_folder, num_groups=args.num_groups)

    print("\nAll files converted.")


if __name__ == "__main__":
    main()
