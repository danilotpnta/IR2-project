import json
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
from tqdm.auto import tqdm
import pandas as pd

from datasets import Dataset
from transformers import PreTrainedModel, AutoTokenizer

from query_eval import QueryEval
from cpo_dataset import generate_queries # we should move this.
from vllm_inference import generate_queries as vllm_generate

def cpo_eval(
        prompts: List[str],
        doc_ids: List[str],
        model: Union[PreTrainedModel, str],
        tokenizer: AutoTokenizer,
        query_eval: QueryEval,
        batch_size: int = 256,
        **generator_kwargs: Dict[str, Any],
    ):
    """
    Given a list of prompts and doc_ids, generate queries and evaluate them using QueryEval.
    
    QueryEval should already be indexed at this point.
    """
    if isinstance(model, str):
        gen_fn = vllm_generate
    else:
        gen_fn = generate_queries
        generator_kwargs["tokenizer"] = tokenizer
    # Generate queries
    generator_output = gen_fn(prompts, doc_ids, model, **generator_kwargs)
    texts = []
    for doc_id in doc_ids:
        text, _, _ = generator_output[doc_id]
        texts.append(text)

    # Evaluate queries
    scores = {}
    for i in tqdm(
        range(0, len(doc_ids), batch_size),
        desc="Evaluating queries",
        unit="batch",
        total=len(doc_ids),
    ):
        batch_query = texts[i:i+query_eval.batch_size]
        batch_doc_id = doc_ids[i:i+query_eval.batch_size]
        batch_scores = query_eval.score(batch_query, batch_doc_id)
        scores.update({
            doc_id: score
            for doc_id, score in zip(batch_doc_id, batch_scores)
            })

    # compute metrics
    metrics = {
        "avg_score": sum(scores.values()) / len(scores),
        "min_score": min(scores.values()),
        "max_score": max(scores.values())
    }
    return metrics, scores

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset. Could be a JSON or a Dataset")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to model or model name")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save the output")
    parser.add_argument("--query_eval_path", type=str, required=False,
                        help="Path to query_eval cache")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for the query evaluator")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use VLLM for inference")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use wandb for logging")
    args = parser.parse_args()
    # load model and tokenizer if needed
    if args.use_vllm:
        model = args.model_name
    else:
        model = PreTrainedModel.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found.")
    if dataset_path.suffix == ".json":
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    else:
        dataset = Dataset.load_from_disk(dataset_path)
    # load query_eval
    if args.query_eval_path is not None and Path(args.query_eval_path).exists():
        query_eval = QueryEval.load_from_cache(Path(args.query_eval_path))
    else:
        query_eval = QueryEval() # TODO: support query_eval config
        # prepare dataframe for indexing
        corpus = pd.DataFrame(
            [(data["target_doc_id"], data["target_doc_text"])
             for data in dataset],
            columns=["doc_id", "text"])
        query_eval.load_dataset(corpus, args.batch_size)
    # generate queries and evaluate
    prompts = []
    doc_ids = []
    for data in dataset:
        prompts.append(data["prompt"])
        doc_ids.append(data["target_doc_id"])
    metrics, scores = cpo_eval(
        prompts=prompts,
        doc_ids=doc_ids,
        model=model,
        tokenizer=tokenizer if not args.use_vllm else None,
        query_eval=query_eval,
        batch_size=args.batch_size
        # TODO: generator kwargs
    )
    # save output
    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    with open(output_path / "scores.json", "w") as f:
        json.dump(scores, f)
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"Metrics: {metrics}")
    print(f"Scores saved to {output_path / 'scores.json'}")