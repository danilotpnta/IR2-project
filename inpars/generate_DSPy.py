import os
import dspy
import litellm
import argparse
import polars as pl
import pandas as pd
from serve_model import serve_model
from config import MODEL_PORT_MAPPING, STOP_WORDS
from huggingface_hub import list_repo_files, hf_hub_download
import time

litellm.set_verbose = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "nfcorpus",
            "trec-covid",
            "hotpotqa",
            "fiqa",
            "arguana",
            "webis-touche2020",
            "dbpedia-entity",
            "scidocs",
            "fever",
            "climate-fever",
            "scifact",
        ],
        default="trec-covid",
        help="Choose dataset from BEIR to generate queries.",
    )

    parser.add_argument(
        "--generationLLM",
        choices=[
            "EleutherAI/gpt-j-6B",
            "meta-llama/Llama-3.1-8B"
            "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
        ],
        default="meta-llama/Llama-3.1-8B",
        help="Choose query generation model. ",
    )

    parser.add_argument("--data_dir", default="./")
    args = parser.parse_args()

    return args


def download_queries(data_dir: str, repo_id: str = "inpars/generated-data"):
    data_path = os.path.join(data_dir, "data")
    os.makedirs(data_path, exist_ok=True)

    repo_files = list_repo_files(repo_id, repo_type="dataset")
    queries_files = [f for f in repo_files if f.endswith("queries.jsonl")]

    for file_path in queries_files:
        local_file_path = os.path.join(data_path, file_path)
        if not os.path.exists(local_file_path):
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir=data_path,
                repo_type="dataset",
            )
            print(f"Downloaded: {file_path}")

    print(f"InPars queries downloaded and saved to {data_path}")


def generate_queries(model_name: str, dataset: str):

    lm = dspy.HFClientVLLM(
        model=model_name,
        port=MODEL_PORT_MAPPING[model_name],
        url="http://localhost",
        stop=STOP_WORDS,
        max_tokens=64,
        cache=False,
    )
    dspy.configure(lm=lm)
    
    class DocumentToQuery(dspy.Signature):
        """Extract a single relevant query that encompasses the document."""

        document = dspy.InputField(desc="The full document text to analyze.")
        query = dspy.OutputField(
            desc="A single relevant query ending in '?' that represents the document.",
        )

    class SimpleDocumentToQuery(dspy.Signature):
        """Generate a relevant query from the document."""

        document = dspy.InputField(desc="The document text.")
        query = dspy.OutputField(desc="A relevant query ending in '?'")

    # Create predictors for both strategies
    strategies = {
        "CoT": dspy.ChainOfThought(DocumentToQuery),
        "Zero-shot": dspy.Predict(SimpleDocumentToQuery),
    }

    df = pd.read_json(f"data/{dataset}/queries.jsonl", lines=True)
    prompts = df["text"]
    doc_ids = df["doc_id"]

    if not os.path.exists(f"data/{dataset}/prompts.csv"):
        prompts.to_csv(f"data/{dataset}/prompts.csv", index=False)
        doc_ids.to_csv(f"data/{dataset}/doc_ids.csv", index=False)

    # Compute queries on a small subset of the data
    prompts = df["text"].head(100).tolist()
    doc_ids = df["doc_id"].head(100).tolist()

    for strategy_name, predictor in strategies.items():
        start_time = time.time()
        outputs = [predictor(document=prompt, cache=False) for prompt in prompts]
        end_time = time.time()
        elapsed_time = end_time - start_time

        generations = [(d_id, output.query) for d_id, output in zip(doc_ids, outputs)]

        save_path = (
            f"data/{dataset}/queries_{model_name.split('/')[-1]}_{strategy_name}.jsonl"
        )
        save = pl.DataFrame(
            generations,
            schema={"doc_id": str, "query": str},
            strict=False,
            orient="row",
        )
        save.write_ndjson(save_path)
        print(f"Strategy {strategy_name} took {elapsed_time:.2f} seconds")


def main():
    args = parse_args()

    download_queries(args.data_dir)
    serve_model(args.generationLLM)
    generate_queries(args.generationLLM, args.dataset)


if __name__ == "__main__":
    main()
