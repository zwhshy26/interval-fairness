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


def convert_one_file(file_path: Path, output_folder: Path, num_groups: int = 10) -> None:
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

    user_to_group = {}

    for rank, user_id in enumerate(unique_users):
        group_id = rank * num_groups // num_users + 1

        if group_id > num_groups:
            group_id = num_groups

        user_to_group[user_id] = group_id

    df["new_group_id"] = df["user_id"].map(user_to_group)

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
    output_folder = project_folder

    files = list(input_folder.glob("*.swf.gz"))

    print("Input folder:", input_folder)
    print("Output folder:", output_folder)
    print("Number of .swf.gz files found:", len(files))

    if len(files) == 0:
        print("No .swf.gz files found.")
        return

    for file_path in files:
        convert_one_file(file_path, output_folder, num_groups=10)

    print("\nAll files converted.")


if __name__ == "__main__":
    main()