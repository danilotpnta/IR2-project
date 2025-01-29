import json
import random
import argparse
from pathlib import Path


def sample_jsonl(file_path, sample_size):
    """
    Randomly sample entries from a JSONL file and save them with a new filename.

    Parameters:
        file_path (str): Path to the input JSONL file.
        sample_size (int): Number of entries to sample (e.g., 50k or 25k).
    """
    # Load all entries from the JSONL file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Validate sample size
    original_sample_size = sample_size  # Keep track of the requested sample size
    if sample_size > len(lines):
        print(
            f"Sample size {sample_size} exceeds the number of entries in the file ({len(lines)})."
        )
        if sample_size == 50000:
            sample_size = int(len(lines) * 0.5)  # 50% of the data
        elif sample_size == 25000:
            sample_size = int(len(lines) * 0.25)  # 25% of the data
        else:
            sample_size = int(len(lines) * 0.1)  # 10% of the data

        print(f"Adjusting sample size to {sample_size} ({sample_size/len(lines)*100:.2f}%).")

    # Randomly sample the specified number of entries
    sampled_lines = random.sample(lines, sample_size)

    # Generate the output file name
    original_name = Path(file_path).stem
    if original_sample_size > len(lines):
        if original_sample_size == 50000:
            suffix = "50k"
        elif original_sample_size == 25000:
            suffix = "25k"
        else:
            suffix = "10k"
    else:
        suffix = f"{sample_size}k"

    new_file_name = f"{original_name}_{suffix}.jsonl"
    new_file_path = Path(file_path).with_name(new_file_name)

    # Save the sampled entries to the new file
    with open(new_file_path, "w", encoding="utf-8") as out_file:
        out_file.writelines(sampled_lines)

    print(f"Sampled {sample_size} entries and saved to {new_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly sample entries from a JSONL file."
    )
    parser.add_argument("file_path", type=str, help="Path to the input JSONL file.")
    parser.add_argument(
        "--sample_size",
        type=int,
        choices=[25000, 50000],
        required=True,
        help="Number of entries to sample (25k or 50k).",
    )

    # args = parser.parse_args()
    # sample_jsonl(args.file_path, args.sample_size)

    file_path = (
        "/home/scur2880/IR2-project/results/scifact/queries_Llama-3.1-8B_Agent.jsonl"
    )
    for i in [10000, 25000, 50000]:
        sample_jsonl(file_path, i)
