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


def convert_one_file(file_path: Path, output_folder: Path) -> None:
    output_name = file_path.name.replace(".swf.gz", "_interval.csv")
    output_path = output_folder / output_name

    print(f"Reading: {file_path.name}")

    df = pd.read_csv(
        file_path,
        compression="gzip",
        comment=";",
        sep=r"\s+",
        names=SWF_COLUMNS,
        engine="python",
    )

    interval_df = df[["submit_time", "run_time", "group_id"]].copy()

    interval_df.columns = ["start", "length", "group_id"]

    interval_df = interval_df[
        (interval_df["start"] >= 0)
        & (interval_df["length"] > 0)
        & (interval_df["group_id"] >= 0)
    ]

    interval_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(interval_df)}")


def main():
    project_folder = Path(__file__).resolve().parent

    input_folder = project_folder
    output_folder = project_folder

    files = list(input_folder.glob("*.swf.gz"))

    print("Number of files:", len(files))

    for file_path in files:
        convert_one_file(file_path, output_folder)


if __name__ == "__main__":
    main()