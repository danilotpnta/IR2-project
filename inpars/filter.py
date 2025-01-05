import os
import json
import torch
import getpass
import argparse
import numpy as np
import requests
from tqdm import tqdm
from .rerank import Reranker
from .dataset import load_corpus
import random

from transformers import set_seed
from torch.cuda import empty_cache


def read_synthetic_data(args):
    rows = []
    with open(args.input, "r") as fin:
        for line in tqdm(fin, desc="Reading synthetic queries"):
            row = json.loads(line.strip())

            if args.keep_only_question:
                if "?" in row["query"]:
                    query, _, _ = row["query"].partition("?")
                    row["query"] = query.strip() + "?"

            if "log_probs" in row:
                if len(row["log_probs"]) < args.min_tokens:
                    continue
                if len(row["log_probs"]) > args.max_tokens:
                    continue
            if args.skip_questions_copied_from_context:
                if row["query"].lower() in row["doc_text"].lower():
                    continue
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input jsonl file with the synthetic queries to be filtered.",
    )
    parser.add_argument(
        "--dataset", default=None, type=str, help="Dataset name from BEIR collection."
    )
    parser.add_argument(
        "--dataset_source",
        default="ir_datasets",
        help="The dataset source: ir_datasets or pyserini",
    )
    parser.add_argument(
        "--filter_strategy",
        type=str,
        required=True,
        help="Filtering strategy: scores or reranker.",
    )
    parser.add_argument(
        "--keep_top_k",
        type=int,
        default=10_000,
        help="Write only top_k best scored query-doc pairs.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the filtered queries."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="castorini/monot5-3b-msmarco-10k",
        required=False,
        help="Reranker model to be used in case of filtering_strategy=reranker.",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=3,
        help="Skip question that have fewer than this number of words.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Skip question that have more than this number of words.",
    )
    parser.add_argument(
        "--skip_questions_copied_from_context",
        action="store_true",
        help="If passed, skip questions that were copied from the passage.",
    )
    parser.add_argument("--device", default=None, type=str, help="CPU or CUDA device.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use FP16 weights during inference.",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for inference."
    )
    parser.add_argument(
        "--use_scratch_shared_cache",
        action="store_true",
        help="Use scratch-shared directory for Hugging Face cache.",
    )
    parser.add_argument(
        "--keep_only_question",
        action="store_true",
        help="Keep only the question part of the query.",
    )
    parser.add_argument("--max_generations_considered", default=None, type=int)
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    assert args.filter_strategy in ["scores", "reranker"]

    if args.use_scratch_shared_cache:
        hf_cache_dir = f"/scratch-shared/{getpass.getuser()}/.cache/huggingface"
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir

    # Derive prompting strategy from generated input data path
    zeroshot = True if 'Zero-shot' in args.input else False
    CoT = True if 'CoT' in args.input else False
    if zeroshot and CoT:
        assert "Both zeroshot and CoT in input path."
    
    # If path does not exist in current environment, try to download it from huggingface. Otherwise throw exception. 
    if not os.path.exists(args.input) and (CoT or zeroshot):
        os.makedirs(os.path.dirname(args.input), exist_ok=True)
        try:
            prompt_strategy = 'Zero-shot' if zeroshot else 'CoT'
            response = requests.get(
            f"https://huggingface.co/datasets/inpars-plus/generated-data/resolve/main/{args.dataset}/queries_Llama-3.1-8B_{prompt_strategy}.jsonl"
            )
            with open(args.input, "wb") as f:
                f.write(response.content)

            print("downloaded data!")
        except Exception as e:
            assert "jsonl file with synthetic data cannot be found."    

    dataset = read_synthetic_data(args)
    
    # Shuffle dataset to be able to test different subsets if desired
    set_seed(args.seed)
    random.shuffle(dataset)
    
    # If max_generations_considered is smaller than dataset size, take first N. 
    if args.max_generations_considered:
        if len(dataset) > args.max_generations_considered:
            dataset = dataset[:args.max_generations_considered]

    model = None
    if len(dataset) <= args.keep_top_k:
        # Give all an arbitrary score, no filtering is needed.  
        for line in tqdm(dataset):
            line["score"] = 1.0
    elif args.filter_strategy == "scores":
        for line in tqdm(dataset):
            line["score"] = np.mean(line["log_probs"])
    else:
        corpus = load_corpus(args.dataset, source=args.dataset_source)
        corpus = dict(zip(corpus["doc_id"], corpus["text"]))

        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Reranker.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            batch_size=args.batch_size,
            fp16=args.fp16,
            device=args.device,
        )
        q_key = "query" if dataset[0].get("query") is not None else "question"

        query_scores = model.rescore(
            [(synt_item[q_key], corpus[synt_item["doc_id"]]) for synt_item in dataset]
        )
        for idx, synt_item in enumerate(dataset):
            synt_item["score"] = query_scores[idx]

    dataset.sort(key=lambda dataset: dataset["score"], reverse=True)

    # Saves only top_k scored queries
    with open(args.output, "w") as fout:
        for row in dataset[: args.keep_top_k]:
            fout.write(json.dumps(row) + "\n")

    # Save all scored queries
    output_file_all = "_all.".join(args.output.rsplit(".", 1))
    with open(output_file_all, "w") as fout:
        for row in dataset:
            fout.write(json.dumps(row) + "\n")

    # Remove model when one is created
    if model is not None:
        del model
        empty_cache()

    print("Done!")