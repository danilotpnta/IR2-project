import os
import sys
import dspy
import time
import random
import litellm
import argparse
import polars as pl
import pandas as pd
from tqdm import tqdm
from config import MODEL_PORT_MAPPING, STOP_WORDS
from huggingface_hub import list_repo_files, hf_hub_download

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
            "meta-llama/Llama-3.1-8B",
            "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
        ],
        default="meta-llama/Llama-3.1-8B",
        help="Choose query generation model. ",
    )

    parser.add_argument("--data_dir", default="./")
    args = parser.parse_args()

    return args


def download_queries(data_dir: str, repo_id: str = "inpars/generated-data"):
    """Download the InPars queries from the Hugging Face Hub."""

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


def prepare_data(dataset_path, limit=None):
    """Load and prepare the dataset for query generation."""

    df = pd.read_json(dataset_path, lines=True)
    if "text" in df.columns:
        prompts = df["text"]
    elif "doc_text" in df.columns:
        prompts = df["doc_text"]
    else:
        raise ValueError(
            "Neither 'text' nor 'doc_text' column exists in the DataFrame."
        )

    doc_ids = df["doc_id"]

    if not os.path.exists(f"{os.path.dirname(dataset_path)}/prompts.csv"):
        prompts.to_csv(f"{os.path.dirname(dataset_path)}/prompts.csv", index=False)
        doc_ids.to_csv(f"{os.path.dirname(dataset_path)}/doc_ids.csv", index=False)

    if limit is not None:
        prompts = prompts.head(limit)
        doc_ids = doc_ids.head(limit)

    return prompts.tolist(), doc_ids.tolist()


class DocumentToQuery(dspy.Signature):
    """Extract a single relevant query that encompasses the document."""

    document = dspy.InputField(desc="The full document text to analyze.")
    query = dspy.OutputField(
        desc="A single relevant query ending in '?' that represents the document.",
    )


class Prediction:
    def __init__(self, query: str):
        self.query = query

    def __repr__(self):
        return f"Prediction(\n    query='{self.query}'\n)"


class SimpleDocumentToQuery:
    """Generate a single relevant query that encompasses the document."""

    def __init__(self, document=None):
        self.lm = dspy.settings.lm
        assert self.lm is not None, "No LM is loaded."

        self.document = document
        self.query = self._generate()

    def _generate(self):
        """
        Generate a query by appending 'Relevant Question:' to the document.
        """
        prompt = f"{self.document}\nRelevant Question:"
        response = self.lm(prompt=prompt, use_tqdm=True)

        if isinstance(response, list):
            response = response[0]

        return response.strip()

    @classmethod
    def batch(cls, examples):
        """
        Handle batch generation from DSPy Examples.
        """
        lm = dspy.settings.lm
        assert lm is not None, "No LM is loaded."
        documents = [example.document for example in examples]
        prompts = [f"{doc}\nRelevant Question:" for doc in documents]

        responses = lm(prompts, use_tqdm=True)

        return [Prediction(query=response.strip()) for response in responses]


class SimpleDocumentToQuery_batching(dspy.Signature):
    """Extract a single relevant query that encompasses the document."""

    document = dspy.InputField(desc="The full document text to analyze.")
    query = dspy.OutputField(
        desc="A single relevant query ending in '?' that represents the document.",
    )

    def forward(self, **kwargs):
        document = kwargs["document"]
        prompt = f"{document}\nRelevant Question:"
        return prompt


def generate_queries(model_name: str, dataset: str):
    """Generate queries for a given dataset using a specified model."""

    lm = dspy.HFClientVLLM(
        model=model_name,
        port=MODEL_PORT_MAPPING[model_name],
        url="http://localhost",
        stop=STOP_WORDS,
        max_tokens=64,
        cache=False,
    )
    dspy.configure(lm=lm)

    # Define strategies
    strategies = {
        "Zero-shot": SimpleDocumentToQuery(),
        # "Zero-shot_dspyPredict": dspy.Predict(SimpleDocumentToQuery_batching),
        "CoT": dspy.ChainOfThought(DocumentToQuery, use_tqdm=True),
    }

    # Prepare dataset
    dataset_path = f"data/{dataset}/queries.jsonl"
    prompts, doc_ids = prepare_data(
        dataset_path,
        # limit=14
    )

    # select 10 random prompts from the prompts list
    prompts = random.sample(prompts, 1000)

    # Generate queries
    for strategy_name, predictor in strategies.items():
        batch_prompts = [
            dspy.Example(document=p).with_inputs("document") for p in prompts
        ]
        outputs = predictor.batch(batch_prompts)
        # print(f"\n** {strategy_name} **")
        # print(outputs)
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


def main():
    args = parse_args()

    download_queries(args.data_dir)
    generate_queries(args.generationLLM, args.dataset)


if __name__ == "__main__":
    main()
