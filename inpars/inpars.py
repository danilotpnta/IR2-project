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

def _build_prompt(document, doc_id, prompter, n_examples=3):
    return prompter.build(
        document, n_examples=n_examples
    ), doc_id


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
        bf16=False,
        fp16=False,
        int8=False,
        device=None,
        tf=False,
        torch_compile=False,
        verbose=False,
        only_generate_prompt=False,
        use_vllm=False,
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
        self.tf = tf
        self.only_generate_prompt = only_generate_prompt
        self.use_vllm = use_vllm

        if self.use_vllm:
            try:
                from . import vllm_inference
            except ImportError:
                Warning("""
                        VLLM inference not available. Please install vllm first.
                        
                        Setting use_vllm to False""")
                self.use_vllm = False

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if 'llama' in base_model.lower():
            auth_token = os.environ['HF_TOKEN']
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct", token = auth_token, padding_side='left' 
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model, padding_side="left"
            )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.fewshot_examples = load_examples(corpus, self.n_fewshot_examples)
        
        # Fix : Prompt names are dataset dependent when using promptagator. 
        prompt_name = f'{prompt}-{corpus}' if prompt == 'promptagator' else prompt
        self.prompter = Prompt.load(
            name=prompt_name,
            examples=self.fewshot_examples,
            tokenizer=self.tokenizer,
            max_query_length=self.max_query_length,
            max_doc_length=self.max_doc_length,
            max_prompt_length=self.max_prompt_length,
            max_new_token=self.max_new_tokens,
        )

        model_kwargs = {"revision": revision}
        if bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["low_cpu_mem_usage"] = True
        if fp16:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True
        if int8:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        if fp16 and base_model == "EleutherAI/gpt-j-6B":
            model_kwargs["revision"] = "float16"

        if self.only_generate_prompt:
            return

        # if VLLM is used, the model is loaded in the generate method
        if self.use_vllm:
            self.model = base_model
            self.model_kwargs = {"dtype": model_kwargs["torch_dtype"]}

        elif self.tf:
            from transformers import TFAutoModelForCausalLM

            self.model = TFAutoModelForCausalLM.from_pretrained(
                base_model,
                revision=revision,
            )
            self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            if 'llama' in base_model:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model, token = auth_token, **model_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model, **model_kwargs
                )
            if torch_compile:
                self.model = torch.compile(self.model)
            self.model.to(self.device)
            self.model.eval()

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

        if self.tf and not self.only_generate_prompt:
            import tensorflow as tf

        cache_dir = Path(cache_dir) / self.corpus
        cache_dir.mkdir(exist_ok=True, parents=True)

        prompts_json = cache_dir / f"{cache_name}_prompts.json"
        results_csv = cache_dir / f"{cache_name}_results.csv"

        prompts = None
        if prompts_json.exists():
            try:
                with open(prompts_json, "r") as f:
                    data = json.load(f)
                prompts = [data[doc_id] for doc_id in doc_ids]
            except Exception:
                print("Could not load prompts from cache file, rebuilding ...")
                prompts = None

        if prompts is None:
            print(f"Building prompts for {len(doc_ids)} documents")
            prompts = process_map(
                _build_prompt,
                documents,
                doc_ids,
                [self.prompter] * len(doc_ids),
                [self.n_fewshot_examples] * len(doc_ids),
                chunksize=128,
                total=len(doc_ids),
                desc="Building prompts",
                # disable=len(documents) > 1000,
            )
            # covert to dict for saving
            prompts = {doc_id: prompt for prompt, doc_id in prompts}
            with open(prompts_json, "w") as f:
                json.dump(prompts, f)
            prompts = [prompts[doc_id] for doc_id in doc_ids]

        if self.only_generate_prompt:
            return prompts

        if self.use_vllm:
            from .vllm_inference import generate_queries
            results = generate_queries(
                prompts=prompts,
                doc_ids=doc_ids.values.tolist(),
                model_name=self.model,
                save_folder=cache_dir,
                max_prompt_length=self.max_prompt_length,
                max_tokens=self.max_new_tokens,
                batch_size=batch_size,
                use_tqdm_inner=True,
                force=False,
                logprobs=1,
                **self.model_kwargs,
                **generate_kwargs,
            )
            # results are in the format doc_id: (query, log_probs, cumulative_log_probs)
            results = [
                { # TODO: are we 100% sure we want to save the _same_ prompt for each query?
                  # same for doc_text ...
                    "query": results[doc_id][0],
                    "log_probs": results[doc_id][1],
                    "prompt_text": prompt_text,
                    "doc_id": doc_id,
                    "doc_text": document,
                    "fewshot_examples": [
                        example[0] for example in self.fewshot_examples
                    ],
                }
                for doc_id, prompt_text, document in zip(doc_ids, prompts, documents)
            ]
            return results

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
