import os
import argparse
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import AutoTokenizer, AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset statistics")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-j-6B",
        choices=[
            "EleutherAI/gpt-j-6B",
            "meta-llama/Llama-3.1-8B",
            "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
        ],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "arguana",
            "bioasq",
            "climate-fever",
            "cqadupstack",
            "dbpedia-entity",
            "fever",
            "fiqa",
            "hotpotqa",
            "nfcorpus",
            "nq",
            "quora",
            "robust04",
            "scidocs",
            "scifact",
            "signal",
            "trec-covid",
            "trec-news",
            "webis-touche2020",
        ],
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate violin plots for datasets",
    )
    return parser.parse_args()


def get_model_max_length(model_name: str) -> int:
    config = AutoConfig.from_pretrained(model_name)
    max_length = getattr(config, "max_position_embeddings", None)
    if max_length is None:
        max_length = getattr(config, "n_positions", None)
    if max_length is None:
        max_length = getattr(config, "max_sequence_length", 2048)
    return max_length


def plot_violin(df, dataset_name: str, output_path: str):
    stats = df["word_count"].describe()
    plt.figure(figsize=(8, 7))
    sns.violinplot(data=df, y="word_count", inner="quartile", color="skyblue")
    plt.title(f"Word Count Distribution for '{dataset_name.capitalize()}'")
    plt.xlabel("Dataset")
    plt.ylabel("Word Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.ylim(None, stats["75%"] * 3)
    legend_elements = [
        Patch(color="none", label=f"Mean: {df['word_count'].mean():.2f}"),
        Patch(color="none", label=f"Std Dev: {df['word_count'].std():.2f}"),
        Patch(color="none", label=f"Min: {df['word_count'].min()}"),
        Patch(color="none", label=f"Max: {df['word_count'].max()}"),
    ]
    plt.legend(
        handles=legend_elements, title="Statistics", loc="upper right", frameon=False
    )
    plt.savefig(os.path.join(output_path, f"{dataset_name}_violin_plot.png"))
    plt.close()


def batch_count_tokens(texts: List[str], tokenizer, batch_size: int) -> List[int]:
    token_counts = [None] * len(texts)
    skipped_indices = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        test_encodings = tokenizer(
            batch, padding=False, truncation=False, return_tensors=None
        )
        input_ids = test_encodings["input_ids"]

        for idx, ids in enumerate(input_ids):
            global_idx = i + idx
            if len(ids) <= tokenizer.model_max_length:
                token_counts[global_idx] = len(ids)
            else:
                skipped_indices.append(global_idx)

    return token_counts, skipped_indices


def generate_dataset_stats(
    dataset_path: str,
    dataset_name: str,
    tokenizer,
    batch_size: int,
    context_limit: int,
    generate_plots: bool,
    output_dir: str,
) -> Dict:

    df = pd.read_json(dataset_path, lines=True)
    text_column = "text" if "text" in df.columns else "doc_text"
    df["doc_id"] = df["doc_id"].astype(str)
    df["word_count"] = df[text_column].str.split().str.len()

    token_counts, skipped_indices = batch_count_tokens(
        df[text_column].tolist(), tokenizer, batch_size
    )
    df["token_count"] = token_counts

    valid_df = df[df["token_count"].notna()]
    stats = valid_df["word_count"].describe()

    if generate_plots:
        plot_violin(valid_df, dataset_name, output_dir)

    return {
        "Dataset": dataset_name,
        "Model": tokenizer.name_or_path,
        "Model Context Length": context_limit,
        "Total Documents": len(df),
        "Valid Documents": len(valid_df),  # whithin the model's context length
        "Docs Exceeding Context (%)": round(100 * (1 - len(valid_df) / len(df)), 2),
        "Avg Documents Words": round(stats["mean"], 2),
        "Std Dev Documents Words": round(stats["std"], 2),
        "Avg Documents Tokens": round(valid_df["token_count"].mean(), 2),
        "Min Doc Words": f"{int(stats['min'])} (ID: {valid_df.loc[valid_df['word_count'].idxmin(), 'doc_id']})",
        "Min Doc Tokens": valid_df["token_count"].min(),
        "Max Doc Words": f"{int(stats['max'])} (ID: {valid_df.loc[valid_df['word_count'].idxmax(), 'doc_id']})",
        "Max Doc Tokens": valid_df["token_count"].max(),
        "Context Window Outliers Count": len(skipped_indices),
        "Context Window Outliers IDs": (
            ",".join(df.iloc[skipped_indices]["doc_id"]) if skipped_indices else "None"
        ),
    }


def save_progress(stats_list: List[Dict], output_path: str):
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_path, index=False)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    context_limit = get_model_max_length(args.model_name)
    print(f"Model: {args.model_name}")
    print(f"Context window size: {context_limit}")

    output_path = os.path.join(args.data_dir, "stats")
    csv_stats_path = os.path.join(output_path, "dataset_stats.csv")
    os.makedirs(output_path, exist_ok=True)

    all_stats = []
    for dataset in tqdm(args.datasets, desc="Progress"):
        dataset_path = os.path.join(args.data_dir, dataset, "queries.jsonl")
        try:
            print(f"\nProcessing {dataset}...")
            stats = generate_dataset_stats(
                dataset_path,
                dataset,
                tokenizer,
                args.batch_size,
                context_limit,
                args.generate_plots,
                output_path,
            )
            all_stats.append(stats)
            save_progress(all_stats, csv_stats_path)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    print(f"Done! Final statistics saved to {csv_stats_path}")


if __name__ == "__main__":
    main()
