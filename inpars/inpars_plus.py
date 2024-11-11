import os
import ftfy
import torch
import random
import statistics
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import urllib.error
import json
import pickle
from pathlib import Path


class Prompt:
    @staticmethod
    def load(
        name,
        examples,
        tokenizer,
        max_query_length=None,
        max_doc_length=None,
        max_prompt_length=None,
        max_new_token=None,
    ):
        return SimplePrompt(
            examples=examples,
            tokenizer=tokenizer,
            max_query_length=max_query_length,
            max_doc_length=max_doc_length,
            max_prompt_length=max_prompt_length,
            max_new_token=max_new_token,
        )


class SimplePrompt:
    def __init__(
        self,
        examples,
        tokenizer,
        max_query_length=None,
        max_doc_length=None,
        max_prompt_length=None,
        max_new_token=None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.max_prompt_length = max_prompt_length
        self.max_new_token = max_new_token

    def build(self, document, n_examples=None):
        examples = (
            random.sample(self.examples, n_examples) if n_examples else self.examples
        )
        prompt = ""

        for query_id, doc_id, query, doc in examples:
            if self.max_doc_length:
                doc = " ".join(doc.split()[: self.max_doc_length])
            if self.max_query_length:
                query = " ".join(query.split()[: self.max_query_length])
            prompt += f"Document: {doc}\nQuery: {query}\n\n"

        if self.max_doc_length:
            document = " ".join(document.split()[: self.max_doc_length])
        prompt += f"Document: {document}\nQuery:"

        if self.max_prompt_length:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > self.max_prompt_length:
                tokens = tokens[-self.max_prompt_length :]
                prompt = self.tokenizer.decode(tokens)

        return prompt


# TODO: Cache the Examples Locally
def load_examples(corpus, n_fewshot_examples):
    try:
        df = pd.read_json(
            f"https://huggingface.co/datasets/inpars/fewshot-examples/resolve/main/data/{corpus}.json",
            lines=True,
        )
        df = df[["query_id", "doc_id", "query", "document"]].values.tolist()
        random_examples = random.sample(df, n_fewshot_examples)
        with open("query_ids_to_remove_from_eval.tsv", "w") as fout:
            for item in random_examples:
                fout.write(f"{item[0]}\t{item[2]}\n")
        return random_examples
    except urllib.error.HTTPError:
        return []


class InPars:
    def __init__(
        self,
        base_model="EleutherAI/gpt-j-6B",
        revision=None,
        corpus="msmarco",
        prompt=None,
        n_fewshot_examples=None,
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
        self.corpus = corpus
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.n_fewshot_examples = n_fewshot_examples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.tf = tf

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {"revision": revision}
        if fp16:
            model_kwargs.update(
                {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
            )
        if int8:
            model_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        if fp16 and base_model == "EleutherAI/gpt-j-6B":
            model_kwargs["revision"] = "float16"

        if self.tf:
            from transformers import TFAutoModelForCausalLM

            self.model = TFAutoModelForCausalLM.from_pretrained(
                base_model, revision=revision
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
            examples=self.fewshot_examples,
            tokenizer=self.tokenizer,
            max_query_length=self.max_query_length,
            max_doc_length=self.max_doc_length,
            max_prompt_length=self.max_prompt_length,
            max_new_token=self.max_new_tokens,
        )

    def _update_cache(self, cache_file, new_prompts, new_doc_ids):
        try:
            cache_data = {"prompts": [], "doc_ids": []}
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)

            cache_data["prompts"].extend(new_prompts)
            cache_data["doc_ids"].extend(new_doc_ids)

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            print(f"Error updating cache: {e}")

    def _save_prompts_csv(self, csv_path, doc_ids, prompts):
        try:
            df = pd.DataFrame({"doc_id": doc_ids, "prompt": prompts})
            df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        except Exception as e:
            print(f"Error saving prompts CSV: {e}")

    def _save_results_csv(self, csv_path, results):
        try:
            df = pd.DataFrame(results)
            df["log_probs"] = df["log_probs"].apply(lambda x: ",".join(map(str, x)))
            df["fewshot_examples"] = df["fewshot_examples"].apply(
                lambda x: ",".join(map(str, x))
            )
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error saving results CSV: {e}")

    @torch.no_grad()
    def generate(
        self,
        documents,
        doc_ids,
        batch_size=1,
        cache_dir="cache_plus",
        cache_name="prompts_cache",
        save_csv=True,
        **generate_kwargs,
    ):
        torch.cuda.empty_cache()
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
                    prompt_map = dict(
                        zip(
                            cache_data.get("doc_ids", []), cache_data.get("prompts", [])
                        )
                    )

                    chunk_size = 1000
                    for i in range(0, len(doc_ids), chunk_size):
                        chunk_doc_ids = doc_ids[i : i + chunk_size]
                        for idx, doc_id in enumerate(chunk_doc_ids, start=i):
                            if doc_id in prompt_map:
                                prompts.append(prompt_map[doc_id])
                                cached_indices.add(idx)

                    if self.verbose:
                        print(f"Loaded {len(cached_indices)} prompts from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")
                prompts = []
                cached_indices = set()

        disable_pbar = len(documents) <= 1000
        new_prompts = []
        new_doc_ids = []

        chunk_size = 100
        for chunk_start in tqdm(
            range(0, len(documents), chunk_size),
            disable=disable_pbar,
            desc="Building prompts",
        ):
            chunk_end = min(chunk_start + chunk_size, len(documents))
            chunk_docs = documents[chunk_start:chunk_end]
            chunk_ids = doc_ids[chunk_start:chunk_end]

            for idx, (document, doc_id) in enumerate(
                zip(chunk_docs, chunk_ids), start=chunk_start
            ):
                if idx not in cached_indices:
                    prompt = self.prompter.build(
                        document, n_examples=self.n_fewshot_examples
                    )
                    new_prompts.append(prompt)
                    new_doc_ids.append(doc_id)
                    prompts.append(prompt)

            # Periodic cache update
            if len(new_prompts) >= 1000:
                self._update_cache(cache_file, new_prompts, new_doc_ids)
                if save_csv:
                    self._save_prompts_csv(prompts_csv, new_doc_ids, new_prompts)
                new_prompts = []
                new_doc_ids = []

        # Final cache update
        if new_prompts:
            self._update_cache(cache_file, new_prompts, new_doc_ids)
            if save_csv:
                self._save_prompts_csv(prompts_csv, new_doc_ids, new_prompts)

        # Generate queries
        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating queries"):
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

            batch_prompts = prompts[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_doc_ids = doc_ids[i : i + batch_size]

            tokens = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                pad_to_multiple_of=8 if not self.tf else None,
            )

            if not self.tf:
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

            outputs = self.model.generate(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                eos_token_id=198,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                **generate_kwargs,
            )

            gen_sequences = outputs.sequences[:, tokens["input_ids"].shape[-1] :]
            pad_mask = gen_sequences == self.tokenizer.pad_token_id

            with torch.no_grad():
                probs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
                prob_values, _ = probs.max(dim=2)

            sequence_scores = [
                prob_values[i][~pad_mask[i]].tolist()[:-1]
                for i in range(len(prob_values))
            ]

            preds = self.tokenizer.batch_decode(
                gen_sequences,
                skip_special_tokens=True,
            )

            batch_results = [
                {
                    "query": pred.strip(),
                    "log_probs": scores,
                    "prompt_text": prompt,
                    "doc_id": doc_id,
                    "doc_text": doc,
                    "fewshot_examples": [
                        example[0] for example in self.fewshot_examples
                    ],
                }
                for pred, scores, prompt, doc_id, doc in zip(
                    preds, sequence_scores, batch_prompts, batch_doc_ids, batch_docs
                )
            ]

            results.extend(batch_results)

            # Periodic save
            if save_csv and len(results) % (batch_size * 10) == 0:
                self._save_results_csv(results_csv, results)

        # Final save
        if save_csv:
            self._save_results_csv(results_csv, results)

        return results
