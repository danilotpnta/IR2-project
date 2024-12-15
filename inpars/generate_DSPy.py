import os

os.environ["DSP_CACHEBOOL"] = "false"

import sys
import dspy
import dsp
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature
from dspy.signatures.signature import signature_to_template

import time
import random
import litellm
import argparse
import polars as pl
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from config import MODEL_PORT_MAPPING, STOP_WORDS, MAX_TOKENS
from huggingface_hub import list_repo_files, hf_hub_download

litellm.set_verbose = False


def disable_warnings():
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


def prepare_data(
    dataset_path, max_new_tokens=200, limit=None, truncate_max_doc_length=False
):
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

    if truncate_max_doc_length:
        model_name = dspy.settings.lm.kwargs["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        signature_tokens = 70
        model_context_length = MAX_TOKENS.get(model_name, 2048)
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



class DebugChainOfThought(Module):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        global old_generate  # Move this to the top

        super().__init__()

        self.activated = activated
        self.signature = signature = ensure_signature(signature)
        *_keys, last_key = signature.output_fields.keys()

        prefix = "Reasoning: Let's think step by step in order to"

        if isinstance(dspy.settings.lm, dspy.LM):
            desc = "${reasoning}"
        elif dspy.settings.experimental:
            desc = "${produce the output fields}. We ..."
        else:
            desc = f"${{produce the {last_key}}}. We ..."

        rationale_type = rationale_type or dspy.OutputField(prefix=prefix, desc=desc)

        # Add "rationale" field to the output signature
        if isinstance(dspy.settings.lm, dspy.LM) or dspy.settings.experimental:
            extended_signature = signature.prepend(
                "reasoning", rationale_type, type_=str
            )
        else:
            extended_signature = signature.prepend(
                "rationale", rationale_type, type_=str
            )

        # Store the original generate functions
        self._original_old_generate = old_generate

        # Monkey patch old_generate to capture the prompt
        def patched_old_generate(demos, signature, kwargs, config, lm, stage):
            x = dsp.Example(demos=demos, **kwargs)
            template = signature_to_template(signature)

            print("\n=== Final Prompt Being Sent to LM ===")
            if lm is None:
                with dsp.settings.context(query_only=False):
                    prompt = template(x)
            else:
                with dsp.settings.context(lm=lm, query_only=True):
                    prompt = template(x)
            print(prompt)
            print("=====================================\n")

            return self._original_old_generate(
                demos, signature, kwargs, config, lm, stage
            )

        old_generate = patched_old_generate

        self._predict = dspy.Predict(extended_signature, **config)
        self._predict.extended_signature = extended_signature

    def forward(self, **kwargs):
        assert self.activated in [True, False]
        signature = kwargs.pop(
            "new_signature",
            self._predict.extended_signature if self.activated else self.signature,
        )
        return self._predict(signature=signature, **kwargs)

    @property
    def demos(self):
        return self._predict.demos

    @property
    def extended_signature(self):
        return self._predict.extended_signature


rationale_type = dspy.OutputField(
    prefix="Reasoning: Let's think step by step in order to",
    desc="${produce one single relevant query}. 1. ... ",
    # desc="${produce one single relevant query using terms from the document}. 1. ...",
)
# rationale_type = None


class DebugLMWrapper:
    def __init__(self, original_lm):
        self.original_lm = original_lm

    def __call__(self, prompt, **kwargs):
        print("\n=== Captured Prompt ===")
        print(prompt)
        print("=======================\n")
        return self.original_lm(prompt, **kwargs)

    def __getattr__(self, name):
        # Pass through attributes not explicitly overridden
        return getattr(self.original_lm, name)


def test_some_prompts(prompts, strategies):
    # select 10 random prompts from the prompts list
    # prompts = random.sample(prompts, 1000)
    # prompts = prompts[67237]  # bad long guy
    # prompts = prompts[2456]  # bad long guy

    # list_prompts = [67237, 2456, 350]
    list_prompts = [766, 1966, 1630]
    # list_prompts = [350]
    for i in list_prompts:
        prompt = prompts[i]
        # print("Before truncation: ", len(prompt))
        # prompt = _truncate_max_doc_length(prompt, max_new_tokens)
        # print("After truncation: ", len(prompt))

        # print(" ** Prompt **")
        # print(prompt, "\n")

        print(" ** Zero-shot **")
        output = strategies["Zero-shot"](document=prompt)
        print(output.query)

        print(" ** CoT **")
        output = strategies["CoT"](document=prompt)
        # output = strategies["CoT_debug"](document=prompt)
        print(output)
        # print(output.query)
    sys.exit()


def generate_queries(model_name: str, dataset: str, max_new_tokens: int = 200):
    """Generate queries for a given dataset using a specified model."""

    lm = dspy.HFClientVLLM(
        model=model_name,
        port=MODEL_PORT_MAPPING[model_name],
        url="http://localhost",
        stop=STOP_WORDS,
        max_tokens=max_new_tokens,
        cache=False,
    )
    dspy.configure(lm=lm)
    # debug_lm = DebugLMWrapper(lm)
    # dspy.configure(lm=debug_lm)

    # Define strategies
    strategies = {
        # "Zero-shot": SimpleDocumentToQuery,
        "CoT": dspy.ChainOfThought(
            DocumentToQuery, use_tqdm=True, rationale_type=rationale_type
        ),
        # "CoT_debug":  DebugChainOfThought_(DocumentToQuery, use_tqdm=True),
    }

    # Prepare dataset
    dataset_path = f"data/{dataset}/queries.jsonl"
    prompts, doc_ids = prepare_data(
        dataset_path, max_new_tokens, truncate_max_doc_length=True
    )

    # test_some_prompts(prompts, strategies)

    BATCH_SIZE = 1000
    SAVE_INTERVAL = 10
    total_batches = len(prompts) // BATCH_SIZE + (
        1 if len(prompts) % BATCH_SIZE != 0 else 0
    )

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
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(prompts))

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
            if batch_idx % SAVE_INTERVAL == 0 or batch_idx == total_batches - 1:
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
    # download_queries(args.data_dir)
    generate_queries(args.model_name, args.dataset)


if __name__ == "__main__":
    main()
