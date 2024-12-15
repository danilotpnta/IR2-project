import os

# os.environ["DSP_CACHEBOOL"] = "false"

import sys
import dspy
import litellm
import argparse
import polars as pl
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import download_queries, ModelConfig


def disable_warnings():
    litellm.set_verbose = False
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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
        "--model_name",
        choices=[
            "EleutherAI/gpt-j-6B",
            "meta-llama/Llama-3.1-8B",
            "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
        ],
        default="meta-llama/Llama-3.1-8B",
        help="Choose query generation model. ",
    )

    parser.add_argument(
        "--data_dir",
        default="./",
        help="Directory where the generated queries from InPars would be downloaded.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for query generation."
    )
    args = parser.parse_args()

    return args


def prepare_data(
    config,
    dataset_path: str,
    max_new_tokens: int = 200,
    limit: bool = None,
    truncate_max_doc_length: bool = False,
):
    """Load and prepare the dataset for query generation.

    Args:
        config: Model configuration object.
        dataset_path: Path to the dataset file.
        max_new_tokens: Maximum number of new tokens to generate.
        limit: Limit the number of prompts to process.
        truncate_max_doc_length: Whether to truncate documents to fit model context length.

    Returns:
        A tuple containing lists of prompts and document IDs.
    """
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

    if truncate_max_doc_length:
        model_name = dspy.settings.lm.kwargs["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        signature_tokens = 70
        model_context_length = config.get_max_tokens(model_name)
        max_doc_len = model_context_length - 2 * max_new_tokens - signature_tokens
        needs_truncation = prompts.str.len() > model_context_length

        if needs_truncation.any():
            truncated_prompts = prompts[needs_truncation].apply(
                lambda doc: tokenizer.decode(
                    tokenizer(
                        doc,
                        truncation=True,
                        max_length=max_doc_len,
                        add_special_tokens=False,
                    )["input_ids"],
                    skip_special_tokens=True,
                )
            )
            prompts.loc[needs_truncation] = truncated_prompts

    return prompts.tolist(), doc_ids.tolist()


class DocumentToQuery(dspy.Signature):
    """Extract a single relevant query that encompasses the document."""

    document = dspy.InputField(desc="The full document text to analyze.")
    query = dspy.OutputField(
        desc="A single relevant query ending in question mark that represents the document.",
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
        """Generate a query by appending 'Relevant Question:' to the document."""
        prompt = f"{self.document}\nRelevant Question:"
        response = self.lm(prompt=prompt, use_tqdm=True)

        if isinstance(response, list):
            response = response[0]

        return response.strip()

    @classmethod
    def batch(cls, examples):
        """Handle batch generation from DSPy Examples."""
        lm = dspy.settings.lm
        assert lm is not None, "No LM is loaded."
        documents = [example.document for example in examples]
        prompts = [f"{doc}\nRelevant Question:" for doc in documents]

        responses = lm(prompts, use_tqdm=True)

        return [Prediction(query=response.strip()) for response in responses]


class DebugLMWrapper:
    def __init__(self, original_lm):
        self.original_lm = original_lm

    def __call__(self, prompt, **kwargs):
        """Capture and print the prompt before passing it to the original language model."""
        print("\n=== Captured Prompt ===")
        print(prompt)
        print("=======================\n")
        return self.original_lm(prompt, **kwargs)

    def __getattr__(self, name):
        """Pass through attributes not explicitly overridden."""
        return getattr(self.original_lm, name)


def test_some_prompts(prompts, strategies):
    """Test a subset of prompts with different strategies.

    Args:
        prompts: List of document prompts.
        strategies: Dictionary of strategies to test.
    """
    list_prompts = [766, 1966, 1630]
    for i in list_prompts:
        prompt = prompts[i]

        print(" ** Zero-shot **")
        output = strategies["Zero-shot"](document=prompt)
        print(output.query)

        print(" ** CoT **")
        output = strategies["CoT"](document=prompt)
        print(output)
    sys.exit()


def generate_queries(
    model_name: str,
    dataset: str,
    batch_size: int,
    save_interval: int = 10,
    max_new_tokens: int = 200,
):
    """Generate queries for a given dataset using a specified model.

    Args:
        model_name: Name of the model to use for query generation.
        dataset: Name of the dataset to generate queries for.
        batch_size: Batch size for query generation.
        save_interval: Interval to save progress.
        max_new_tokens: Maximum number of new tokens to generate.
    """
    config = ModelConfig()

    lm = dspy.HFClientVLLM(
        model=model_name,
        port=config.get_port(model_name),
        url="http://localhost",
        stop=config.get_stop_words(),
        max_tokens=max_new_tokens,
        cache=False,
    )
    dspy.configure(lm=lm)

    dataset_path = f"data/{dataset}/queries.jsonl"
    prompts, doc_ids = prepare_data(
        config, dataset_path, max_new_tokens, truncate_max_doc_length=True
    )

    total_batches = len(prompts) // batch_size + (
        1 if len(prompts) % batch_size != 0 else 0
    )

    rationale_type_CoT = dspy.OutputField(
        prefix="Reasoning: Let's think step by step in order to",
        desc="${produce one single relevant query}. 1. ... ",
    )

    # Define strategies
    strategies = {
        # "Zero-shot": SimpleDocumentToQuery,
        "CoT": dspy.ChainOfThought(
            DocumentToQuery, use_tqdm=True, rationale_type=rationale_type_CoT
        ),
    }

    # Iterate over strategies
    for strategy_name, predictor in strategies.items():
        intermediate_save_path = (
            f"data/{dataset}/queries_{model_name.split('/')[-1]}_{strategy_name}.jsonl"
        )

        processed_generations = []
        if os.path.exists(intermediate_save_path):
            processed_generations = pl.read_ndjson(intermediate_save_path).to_dicts()
            processed_doc_ids = {gen["doc_id"] for gen in processed_generations}
        else:
            processed_doc_ids = set()

        for batch_idx in tqdm(range(total_batches), desc=f"Processing {strategy_name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(prompts))

            batch_prompts = [
                dspy.Example(document=p).with_inputs("document")
                for p, d_id in zip(
                    prompts[start_idx:end_idx], doc_ids[start_idx:end_idx]
                )
                if d_id not in processed_doc_ids
            ]
            batch_doc_ids = [
                d_id
                for d_id in doc_ids[start_idx:end_idx]
                if d_id not in processed_doc_ids
            ]

            if not batch_prompts:
                continue

            outputs = predictor.batch(batch_prompts)
            batch_generations = [
                {"doc_id": d_id, "query": output.query}
                for d_id, output in zip(batch_doc_ids, outputs)
            ]

            processed_generations.extend(batch_generations)

            # Save progress after every SAVE_INTERVAL batches
            if batch_idx % save_interval == 0 or batch_idx == total_batches - 1:
                save = pl.DataFrame(
                    processed_generations, schema={"doc_id": str, "query": str}
                )
                save.write_ndjson(intermediate_save_path)
                print(f"Saved progress for {strategy_name} at batch {batch_idx}")

        save = pl.DataFrame(processed_generations, schema={"doc_id": str, "query": str})
        save.write_ndjson(intermediate_save_path)
        print(f"Completed and saved results for {strategy_name}")



def main():

    args = parse_args()
    disable_warnings()
    download_queries(args.data_dir)
    generate_queries(args.model_name, args.dataset, args.batch_size)


if __name__ == "__main__":
    main()
