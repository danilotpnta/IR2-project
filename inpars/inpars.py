from tqdm.contrib.concurrent import process_map

import os
import ftfy
import torch
import random
import statistics
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import urllib.error
from .prompt import Prompt

import json
import pickle
from pathlib import Path

def _build_prompt(document, doc_id, prompter, cached_indices, n_examples=3):
    return prompter.build(
        document, n_examples=n_examples
    ), doc_id if doc_id not in cached_indices else None


def load_examples(corpus, n_fewshot_examples):
    try:
        df = pd.read_json(
            f"https://huggingface.co/datasets/inpars/fewshot-examples/resolve/main/data/{corpus}.json",
            lines=True,
        )
        # TODO limitar numero de exemplos (?)
        df = df[["query_id", "doc_id", "query", "document"]].values.tolist()
        random_examples = random.sample(df, n_fewshot_examples)
        with open("query_ids_to_remove_from_eval.tsv", "w") as fout:
            for item in random_examples:
                fout.write(f"{item[0]}\t{item[2]}\n")

        return random_examples
    except urllib.error.HTTPError:
        return []


class InPars:
    """
    InPars class for generating prompts and queries using a pre-trained language model.

    Attributes:
        corpus (str): The corpus to use for few-shot examples.
        max_doc_length (int): Maximum document length.
        max_query_length (int): Maximum query length.
        max_prompt_length (int): Maximum prompt length.
        max_new_tokens (int): Maximum number of new tokens to generate.
        n_fewshot_examples (int): Number of few-shot examples to use.
        n_generated_queries (int): Number of queries to generate.
        device (str): Device to run the model on ('cpu', 'cuda', 'tpu', etc).
        verbose (bool): Verbosity mode.
        tokenizer (AutoTokenizer): Tokenizer for the pre-trained model.
        tf (bool): Whether to use TensorFlow.
        fewshot_examples (list): List of few-shot examples.
        prompter (Prompt): Prompt object for building prompts.
    Methods:
        generate(documents, doc_ids, batch_size=1, cache_dir="cache", cache_name="prompts_cache", save_csv=True, **generate_kwargs):
            Generates queries for the given documents and caches the prompts.
    """

    def __init__(
        self,
        base_model="EleutherAI/gpt-j-6B",
        revision=None,
        corpus="msmarco",
        prompt=None,
        n_fewshot_examples=None,
        n_generated_queries=1,
        max_doc_length=None,
        max_query_length=None,
        max_prompt_length=None,
        max_new_tokens=64,
        fp16=False,
        int8=False,
        device=None,
        tf=False,
        torch_compile=False,
        verbose=False,
    ):
        """
        Initializes the InPars object with the given parameters.
        Args:
            base_model (str): The base model to use.
            revision (str): The revision of the model to use.
            corpus (str): The corpus to use for few-shot examples.
            prompt (str): The prompt to use.
            n_fewshot_examples (int): Number of few-shot examples to use.
            max_doc_length (int): Maximum document length.
            max_query_length (int): Maximum query length.
            max_prompt_length (int): Maximum prompt length.
            max_new_tokens (int): Maximum number of new tokens to generate.
            fp16 (bool): Whether to use FP16 precision.
            int8 (bool): Whether to use INT8 precision.
            device (str): Device to run the model on ('cpu' or 'cuda').
            tf (bool): Whether to use TensorFlow.
            torch_compile (bool): Whether to use torch.compile.
            verbose (bool): Verbosity mode.
        """
        self.corpus = corpus
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.n_fewshot_examples = n_fewshot_examples
        self.device = device
        self.verbose = verbose
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"revision": revision}
        if fp16:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True
        if int8:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        if fp16 and base_model == "EleutherAI/gpt-j-6B":
            model_kwargs["revision"] = "float16"

        self.tf = tf

        if self.tf:
            from transformers import TFAutoModelForCausalLM

            self.model = TFAutoModelForCausalLM.from_pretrained(
                base_model,
                revision=revision,
            )
            self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, **model_kwargs
            )
            if torch_compile:
                self.model = torch.compile(self.model)
            self.model.to(self.device)
            self.model.eval()

        self.fewshot_examples = load_examples(corpus, self.n_fewshot_examples)
        self.prompter = Prompt.load(
            name=prompt,
            dataset=corpus,
            examples=self.fewshot_examples,
            n_generated_queries=n_generated_queries,
            tokenizer=self.tokenizer,
            max_query_length=self.max_query_length,
            max_doc_length=self.max_doc_length,
            max_prompt_length=self.max_prompt_length,
            max_new_token=self.max_new_tokens,
        )

    @torch.no_grad()
    def generate(
        self,
        documents,
        doc_ids,
        batch_size=1,
        cache_dir="cache",
        cache_name="prompts_cache",
        save_csv=True,
        **generate_kwargs,
    ):
        """
        Generates queries for the given documents and caches the prompts.
        Args:
            documents (list): List of documents to generate queries for.
            doc_ids (list): List of document IDs.
            batch_size (int): Batch size for generating queries.
            cache_dir (str): Directory to cache prompts.
            cache_name (str): Name of the cache file.
            save_csv (bool): Whether to save the results to a CSV file.
            **generate_kwargs: Additional keyword arguments for the generate method.
        Returns:
            list: List of generated queries and their associated metadata.
        """
        torch.cuda.empty_cache()

        if self.tf:
            import tensorflow as tf

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{cache_name}.pkl"

        prompts_csv = cache_dir / f"{cache_name}_prompts.csv"
        results_csv = cache_dir / f"{cache_name}_results.csv"

        # Try to load cached prompts
        prompts = []
        cached_indices = set()
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    cached_prompts = cache_data.get("prompts", [])
                    cached_doc_ids = cache_data.get("doc_ids", [])

                    prompt_map = dict(zip(cached_doc_ids, cached_prompts))

                    for idx, doc_id in enumerate(doc_ids):
                        if doc_id in prompt_map:
                            prompts.append(prompt_map[doc_id])
                            cached_indices.add(idx)

                    print(f"Loaded {len(cached_indices)} prompts from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")
                prompts = []
                cached_indices = set()

        new_prompts = process_map(
            _build_prompt,
            documents,
            doc_ids,
            [self.prompter] * len(documents),
            [cached_indices] * len(documents),
            [self.n_fewshot_examples] * len(documents),
            chunksize=128,
            total=len(documents),
            desc="Building prompts",
            disable=len(documents) > 1000,
        )
        # Do not reorder the 
        new_doc_ids = [prompt[1] for prompt in new_prompts if prompt is not None]
        new_prompts = [prompt[0] for prompt in new_prompts if prompt is not None]
        prompts.extend(new_prompts)

        # Update cache with new prompts
        if new_prompts:
            try:
                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        cache_data = pickle.load(f)
                else:
                    cache_data = {"prompts": [], "doc_ids": []}

                cache_data["prompts"].extend(new_prompts)
                cache_data["doc_ids"].extend(new_doc_ids)

                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"Cached {len(new_prompts)} new prompts")

                if save_csv:
                    prompts_df = pd.DataFrame(
                        {
                            "doc_id": cache_data["doc_ids"],
                            "prompt": cache_data["prompts"],
                        }
                    )
                    prompts_df.to_csv(prompts_csv, index=False)
                    print(f"Saved prompts to {prompts_csv}")

            except Exception as e:
                print(f"Error updating cache: {e}")

        if self.tf:
            generate = tf.function(self.model.generate, jit_compile=True)
            padding_kwargs = {"pad_to_multiple_of": 8}
        else:
            generate = self.model.generate
            padding_kwargs = {}

        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating queries"):
            batch_prompts = prompts[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_doc_ids = doc_ids[i : i + batch_size]

            tokens = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="tf" if self.tf else "pt",
                **padding_kwargs,
            )

            if not self.tf:
                tokens.to(self.device)

            # TODO: Quick fix, see whether we need this argument
            if "return_dict" in generate_kwargs:
                del generate_kwargs["return_dict"]

            outputs = generate(
                input_ids=tokens["input_ids"].long(),
                attention_mask=tokens["attention_mask"].long(),
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                eos_token_id=198,  # hardcoded Ċ id
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                **generate_kwargs,
            )

            preds = [
                sequence.strip()
                for sequence in self.tokenizer.batch_decode(
                    outputs.sequences[:, tokens["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
            ]

            # Greedy decoding
            gen_sequences = outputs.sequences[:, tokens["input_ids"].shape[-1] :]
            pad_mask = (
                outputs.sequences[:, tokens["input_ids"].shape[-1] :]
                == self.tokenizer.pad_token_id
            )
            probs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
            prob_values, prob_indices = probs.max(dim=2)

            sequence_scores = [
                prob_values[i][~pad_mask[i]].tolist()[:-1]
                for i in range(len(prob_values))
            ]

            for question, log_probs, prompt_text, doc_id, document in zip(
                preds, sequence_scores, batch_prompts, batch_doc_ids, batch_docs
            ):
                results.append(
                    {
                        "query": question,
                        "log_probs": log_probs,
                        "prompt_text": prompt_text,
                        "doc_id": doc_id,
                        "doc_text": document,
                        "fewshot_examples": [
                            example[0] for example in self.fewshot_examples
                        ],
                    }
                )

                # Save intermediate results to CSV
                if save_csv:
                    results_df = pd.DataFrame(results)
                    # Convert log_probs and fewshot_examples to strings for CSV storage
                    results_df["log_probs"] = results_df["log_probs"].apply(
                        lambda x: ",".join(map(str, x))
                    )
                    results_df["fewshot_examples"] = results_df[
                        "fewshot_examples"
                    ].apply(lambda x: ",".join(map(str, x)))
                    results_df.to_csv(results_csv, index=False)

        return results
