import json
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
from tqdm.auto import tqdm
import pandas as pd
import logging

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .query_eval import QueryEval
from .cpo_dataset import generate_queries  # we should move this.
from . import vllm_inference

logger = logging.getLogger(__name__)


def cpo_eval(
    prompts: List[str],
    doc_ids: List[str],
    model: Union[AutoModelForCausalLM, str],
    tokenizer: AutoTokenizer,
    query_eval: QueryEval,
    output_dir: str,
    use_vllm: bool = False,
    use_wandb: bool = False,
    max_prompt_length: int = 4096,
    max_tokens: int = 256,
    batch_size: int = 256,
    dtype: torch.dtype = torch.float16,
):
    """
    Given a list of prompts and doc_ids, generate queries and evaluate them using QueryEval.

    QueryEval should already be indexed at this point.
    """
    logger.info(f"Generating queries for %d prompts.", len(prompts))

    if use_vllm:
        gen_fn = vllm_inference.generate_queries
        generator_kwargs = {
            "prompts": prompts,
            "doc_ids": doc_ids,
            "model_name": model,
            "max_prompt_length": max_prompt_length,
            "max_tokens": max_tokens,
            "batch_size": batch_size,
            "dtype": dtype,
            "save_folder": output_dir,
        }
        logger.info(f"Using VLLM for inference.")
    else:
        gen_fn = generate_queries
        generator_kwargs = {
            "prompts": prompts,
            "doc_ids": doc_ids,
            "model": model,
            "tokenizer": tokenizer,
            "max_prompt_length": max_prompt_length,
            "max_tokens": max_tokens,
            "batch_size": batch_size,
            "torch_dtype": dtype,
            "save_folder": output_dir,
        }
        logger.info(f"Using {type(model)} for inference.")
    if use_wandb:
        wandb.config.update(generator_kwargs)
    # Generate queries
    generator_output = gen_fn(**generator_kwargs)
    logger.info(f"Generated %d queries.", len(generator_output))
    texts = []
    for doc_id in doc_ids:
        text, _, _ = generator_output[doc_id]
        texts.append(text)

    if use_wandb:
        wandb.log({"num_queries": len(texts), "examples": texts[:5]})

    # Evaluate queries
    scores = {}
    for i in tqdm(
        range(0, len(doc_ids), batch_size),
        desc="Evaluating queries",
        unit="batch",
        total=len(doc_ids) // batch_size,
    ):
        batch_query = texts[i : i + batch_size]
        batch_doc_id = doc_ids[i : i + batch_size]
        batch_scores = query_eval.score(batch_query, batch_doc_id)
        scores.update(
            {doc_id: score.item() for doc_id, score in zip(batch_doc_id, batch_scores)}
        )

    # compute metrics
    metrics = {
        "avg_score": sum(scores.values()) / len(scores),
        "min_score": min(scores.values()),
        "max_score": max(scores.values()),
    }

    if use_wandb:
        wandb.log(metrics)

    return metrics, scores, generator_output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Path to dataset. Could be a JSON or a Dataset",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Path to model or model name"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save the output"
    )
    parser.add_argument(
        "--query_eval_path", nargs="*", required=False, help="Path to query_eval cache"
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=4096, help="Maximum prompt length"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate for each query",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for the query evaluator"
    )
    parser.add_argument(
        "--dtype",
        type=lambda x: exec(x),
        default="torch.bfloat16",
        help="Data type for inference",
    )  # Loophole to get hacked
    parser.add_argument(
        "--use_vllm", action="store_true", help="Use VLLM for inference"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb for logging"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="cpo_eval",
        help="Name of the run so we can identify it later",
    )
    args = parser.parse_args()
    # change dtype to torch.dtype
    logger.info("parsed arguments\n%s", args)
    # wandb
    if args.use_wandb:
        import wandb

        wandb.init(project="cpo_eval", config=args, name=args.run_name)
    # load model and tokenizer if needed
    if args.use_vllm:
        model = args.model_name
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=args.dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    # load dataset
    dataset_paths = args.dataset
    datasets = {}
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"{dataset_path} not found.")
        if dataset_path.suffix == ".json":
            with open(dataset_path, "r") as f:
                dataset = list(json.load(f)["data"].values())
        else:
            dataset = Dataset.load_from_disk(dataset_path)
            logging.warning(
                "Support for loading from a combined dataset has not been implemented"
            )
        datasets[dataset_path.parts[-2:]] = {"dataset": dataset}
    # load query_eval
    if args.query_eval_path and all(Path(qp).exists() for qp in args.query_eval_path):
        for qp, _, dataset in zip(args.query_eval_path, datasets.items()):
            dataset["query_eval"] = QueryEval.load_from_cache(qp)

    elif all((Path(qp) / "query_eval_index.json").exists() for qp in dataset_paths):
        for path in dataset_paths:
            dataset["query_eval"] = QueryEval.load_from_cache(path)

    else:
        # TODO: support query_eval config
        # prepare dataframe for indexing
        for _, dataset in datasets.items():
            corpus = pd.DataFrame(
                [
                    (data["target_doc_id"], data["target_doc_text"])
                    for data in dataset["dataset"]
                ],
                columns=["doc_id", "text"],
            )
            dataset["query_eval"] = QueryEval().load_dataset(corpus, args.batch_size)

    for _, dataset in datasets.items():
        # generate queries and evaluate
        prompts = []
        doc_ids = []
        for data in dataset["dataset"]:
            prompts.append(data["prompt"])
            doc_ids.append(data["target_doc_id"])
        metrics, scores, texts = cpo_eval(
            prompts=prompts,
            doc_ids=doc_ids,
            model=model,
            tokenizer=tokenizer if not args.use_vllm else None,
            query_eval=dataset["query_eval"],
            output_dir=args.output_dir,
            use_vllm=args.use_vllm,
            use_wandb=args.use_wandb,
            max_prompt_length=args.max_prompt_length,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        dataset["metrics"] = metrics
        dataset["scores"] = scores
        dataset["texts"] = texts

    # save output
    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    with open(output_path / "scores.json", "w") as f:
        scores = {dataset: v["scores"] for dataset, v in datasets.items()}
        json.dump(scores, f)
    print(f"Scores saved to {output_path / 'scores.json'}")
    with open(output_path / "metrics.json", "w") as f:
        metrics = {dataset: v["metrics"] for dataset, v in datasets.items()}
        json.dump(metrics, f)
    print(f"Metrics: {metrics}")
    with open(output_path / "doc_text_score_triples.json", "w") as f:
        json.dump(
            {
                dataset: {
                    data["target_doc_id"]: {
                        "document": data["target_doc_text"],
                        "query": v["texts"][data["target_doc_id"]],
                        "score": v["scores"][data["target_doc_id"]],
                    }
                    for data in v["dataset"]
                }
                for dataset, v in datasets.items()
            },
            f,
            indent=2,
        )
    print(
        f"document-query-score triples saved to {output_path / 'doc_text_score_triples.json'}"
    )
    if args.use_wandb:
        wandb.finish()
