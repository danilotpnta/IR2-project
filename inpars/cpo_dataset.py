import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re

import ir_datasets
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig
from trl.trainer.utils import pad_to_length
from datasets import concatenate_datasets, Dataset, DatasetInfo
from dataclasses import dataclass

from .prompt import Prompt
from .query_eval import QueryEval
from .utils import get_documents, get_optimal_process_count, get_prompts, get_queries

logger = logging.getLogger(__name__)


# TODO: Expand this with the whole gamma of datasets in ir_datasets
# Dictionary mapping to allowed training datasets where the values represent the designated split to use for training (None means that the dataset is not split)
ALLOWED_DATASETS = {
    "beir/arguana": None,
    "beir/climate-fever": None,
    "beir/cquadupstack/android": None,
    "beir/cquadupstack/english": None,
    "beir/cquadupstack/gaming": None,
    "beir/cquadupstack/gis": None,
    "beir/cquadupstack/mathematica": None,
    "beir/cquadupstack/physics": None,
    "beir/cquadupstack/programmers": None,
    "beir/cquadupstack/stats": None,
    "beir/cquadupstack/tex": None,
    "beir/cquadupstack/unix": None,
    "beir/cquadupstack/webmasters": None,
    "beir/cquadupstack/wordpress": None,
    "beir/dbpedia-entity": "/dev",
    "beir/fever": "/train",
    "beir/fiqa": "/train",
    "beir/hotpotqa": "/train",
    "beir/msmarco": "/train",
    "beir/nfcorpus": "/train",
    "beir/nq": None,
    "beir/quora": "/dev",
    "beir/scidocs": None,
    "beir/scifact": "/train",
    "beir/trec-covid": None,
    "beir/webis-touche2020": "/v2",
    "msmarco-document": "/train",
    "msmarco-document-v2": "/train",
    "msmarco-passage": "/train",
    "msmarco-passage-v2": "/train",
}


def get_corpus(dataset_name: str):
    if dataset_name not in ALLOWED_DATASETS:
        dataset_name = f"beir/{dataset_name}"
        if dataset_name not in ALLOWED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not in the list of allowed datasets."
            )

    return ir_datasets.load(f"{dataset_name}{(ALLOWED_DATASETS[dataset_name]) or ''}")


@dataclass
class DataConfig:
    cpo_data_path: Dict[str, str]
    max_train_samples: Optional[int] = None
    max_source_length: int = 8192
    preprocessing_num_workers: int = get_optimal_process_count()
    overwrite_cache: bool = True
    streaming: bool = False
    dataset_cache: Optional[str] = None


REF_SCORE_KEY = "ref_score"
TEACHER_SCORE_KEY = "teacher_score"
STUDENT_SCORE_KEY = "student_score"

REF_OUTPUT_KEY = "ref_query"
TEACHER_OUTPUT_KEY = "teacher_query"
STUDENT_OUTPUT_KEY = "student_query"


def load_cpo_dataset(data_args: DataConfig, train_args: CPOConfig, tokenizer):
    """
    Implementation adapted from
    https://github.com/fe1ixxu/ALMA/blob/master/utils/utils.py :: `preprocess_cpo_data`
    """

    def get_chosen_reject(example):
        # Finding the indices of the highest and lowest scores
        # Loop unrolling for efficiency

        if example[TEACHER_SCORE_KEY] > example[STUDENT_SCORE_KEY]:
            highest_score_index = (
                TEACHER_OUTPUT_KEY
                if example[TEACHER_SCORE_KEY] > example[REF_SCORE_KEY]
                else REF_OUTPUT_KEY
            )
            lowest_score_index = (
                STUDENT_OUTPUT_KEY
                if example[STUDENT_SCORE_KEY] < example[REF_SCORE_KEY]
                else REF_OUTPUT_KEY
            )
        else:
            highest_score_index = (
                STUDENT_OUTPUT_KEY
                if example[STUDENT_SCORE_KEY] > example[REF_SCORE_KEY]
                else REF_OUTPUT_KEY
            )
            lowest_score_index = (
                TEACHER_OUTPUT_KEY
                if example[TEACHER_SCORE_KEY] < example[REF_SCORE_KEY]
                else REF_OUTPUT_KEY
            )

        # Assigning the corresponding sentences
        highest_score_sentence = example[highest_score_index]
        lowest_score_sentence = example[lowest_score_index]

        return highest_score_sentence, lowest_score_sentence

    def meet_requirements(prompt_tok):
        # TODO: enforce properties about the prompt.
        # should be fine since we already generated the prompt
        return True

    def cpo_prompt_function(examples):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

        # examples is a dict of lists
        # we want a list of dicts. convert
        examples = [
            {k: v[i] for k, v in examples.items()}
            for i in range(len(examples["prompt"]))
        ]
        for example in examples:
            prompt = example["prompt"]
            prompt_tok = tokenizer(
                prompt,
                max_length=data_args.max_source_length,
                padding=True,
                truncation=True,
            ).input_ids
            chosen, rejected = get_chosen_reject(example)
            if meet_requirements(prompt_tok):
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(chosen)
                new_examples["rejected"].append(rejected)
        return new_examples

    # Preprocessing the datasets.
    # NOTE: eval_datasets and test_datasets are not used in the current implementation
    # also note: this is also how the ALMA implementation is done
    train_datasets, eval_datasets, test_datasets = None, None, None
    dataset_path = Path(data_args.dataset_cache) / "llmt_cpo_inparsplus"
    if train_args.do_train:
        if data_args.dataset_cache and dataset_path.exists():
            train_datasets = Dataset.load_from_disk(dataset_path)
            return train_datasets, eval_datasets, test_datasets

        processed_datasets = []
        if data_args.cpo_data_path:
            for dataset_name, path in data_args.cpo_data_path.items():
                with open(path, "r") as f:
                    train_dataset = json.load(f)["data"]
                    train_dataset = Dataset.from_list(
                        list(train_dataset.values()),
                        info=DatasetInfo(dataset_name=dataset_name),
                    )
                if data_args.max_train_samples is not None:
                    max_train_samples = min(
                        len(train_dataset), data_args.max_train_samples
                    )
                    train_dataset = train_dataset.select(range(max_train_samples))
                with train_args.main_process_first(
                    desc=f"CPO train {dataset_name} map pre-processing"
                ):
                    if not data_args.streaming:
                        train_dataset = train_dataset.map(
                            cpo_prompt_function,
                            batched=True,
                            batch_size=1,
                            num_proc=data_args.preprocessing_num_workers,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running CPO preprocessing",
                        )
                    else:
                        train_dataset = train_dataset.map(
                            cpo_prompt_function,
                            batched=True,
                            batch_size=1,
                        )
                processed_datasets.append(train_dataset)

        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets: Dataset = train_datasets.shuffle(seed=train_args.seed)
        train_datasets.save_to_disk(dataset_path)
    return train_datasets, eval_datasets, test_datasets


def continue_from_checkpoint(
    output_path: Path, dataset_name: str, num_samples: Optional[int] = None, **metadata
):
    output = {
        "dataset_name": dataset_name,
        "doc_ids": [],
        "data": {},
        "metadata": dict(
            dataset_name=dataset_name, num_samples=num_samples, **metadata
        ),
    }
    # flags to continue from where we left off
    has_docs_queries = False
    has_prompts = False
    has_teacher_queries = False
    has_student_queries = False
    has_ref_scores = False
    has_teacher_scores = False
    has_student_scores = False

    if output_path.exists():
        with open(output_path, "r") as f:
            output = json.load(f)
        if not "doc_ids" in output and (
            len(output["doc_ids"]) == num_samples
        ):  # start from scratch
            return output, False, False, False, False, False, False, False
        data = next(iter(output["data"].values()))
        has_docs_queries = (
            "target_doc_text" in data
            and "ref_query" in data
            and "target_doc_id" in data
            and "ref_query_id" in data
        )
        has_prompts = "prompt" in data
        has_teacher_queries = "teacher_query" in data
        has_student_queries = "student_query" in data
        has_ref_scores = "ref_score" in data
        has_teacher_scores = "teacher_score" in data
        has_student_scores = "student_score" in data
    return (
        output,
        has_docs_queries,
        has_prompts,
        has_teacher_queries,
        has_student_queries,
        has_ref_scores,
        has_teacher_scores,
        has_student_scores,
    )


def create_prompts(
    dataset: pd.DataFrame, prompt_builder: Prompt, doc_ids: List[str], num_examples: int
):
    """
    Load `num_examples` example document-query pairs for each document.
    Return a dictionary with the following format:
    {
        doc_id: {
            "doc_ids": List[str],
            "doc_texts": List[str],
            "query_ids": List[str],
            "query_texts": List[str]
        }
    }
    """
    # TODO: Make this use multiprocessing like in `build_cpo_dataset`
    prompts = {}
    for doc_id in tqdm(doc_ids, desc="Creating prompts"):
        prompt = prompt_builder.build(
            dataset[dataset["doc_id"] == doc_id]["doc_text"].values[0],
            n_examples=num_examples,
        )
        prompts[doc_id] = prompt
    return prompts


def generate_queries(
    prompts: list[str],
    doc_ids: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_folder="cache",
    max_prompt_length=8192,
    batch_size=256,
    top_k=500,
    top_p=0.9,
    temperature=0.8,
    max_tokens=256,
    logprobs=None,
    stop=["\n", "Example", "Document:"],
    torch_dtype="auto",
    seed=42,
    force=True,
    **kwargs,
):
    """
    Generate queries using the given model and dataset.
    Parameters:
    - model: PreTrainedModel
        The name of the model to use.
    - dataset: dict
        The dataset to generate queries for.
    - prompt: Prompt
        A callable that takes the following positional arguments:
        - example_documents: List[str] -- the example document texts
        - example_queries: List[str] -- the example query texts
        - target_doc_text: str -- the target document text

    The dataset should be a list of dictionaries with the following fields:
    - "target_doc_text": str -- the target document text
    - "example_documents": List[str] -- the example document texts
    - "example_queries": List[str] -- the example query texts
    """
    logger.debug(f"Generating queries for %d documents", len(doc_ids))
    save_folder = Path(save_folder) / model.name_or_path
    save_folder.mkdir(parents=True, exist_ok=True)
    save_file = save_folder / "results_recovery.json"

    if force is True:
        generations = {}

    else:
        try:
            with open(save_file, "r") as f:
                generations = json.load(f)
            logger.debug(f"Found {len(generations)} saved generations.")
            if len(generations) == len(prompts):
                logger.debug("All generations have already been recovered.")
                return generations
        except Exception:
            logger.info("No generated queries were recovered.")
            generations = {}

    # build sampling params
    sampling_params = {
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "logprobs": logprobs,
        "stop_strings": stop,
    }
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    logger.debug("Sampling params: %s", sampling_params)
    # make dataloaders
    loader_docid = DataLoader(doc_ids[len(generations) :], batch_size=batch_size)
    loader_prompts = DataLoader(prompts[len(generations) :], batch_size=batch_size)
    logger.debug("Starting generation for %d document batches", len(loader_docid))
    for d_ids, p in tqdm(
        zip(loader_docid, loader_prompts),
        desc="Generating queries",
        unit="batch",
        total=len(loader_docid),
    ):
        logger.debug("batch doc_ids: %s", d_ids)
        # tokenize the prompts
        inputs = tokenizer(
            p,
            max_length=max_prompt_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        logger.debug("Tokenized prompts")
        # generate queries
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tokenizer=tokenizer,
            use_cache=True,
            **sampling_params,
            **kwargs,
        )
        logger.debug("Generated queries")
        # omit the input
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        # decode
        outputs = pad_to_length(
            outputs, length=max_tokens, pad_value=tokenizer.pad_token_id
        )
        outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Decoded queries\n%s", "\n".join(outputs_decoded))
        # Save the outputs.
        generations |= {
            d_id: (output, None, None) for d_id, output in zip(d_ids, outputs_decoded)
        }

        with open(save_file, "w") as f:
            json.dump(generations, f)

    return generations


def score_queries(
    doc_query_pairs: Dict[str, str], query_eval: QueryEval, batch_size: int = 32
):
    """
    Score the queries in the dataset using the given scoring function.
    Parameters:
    - dataset: Dict[str, str] -- doc_id -> document text map
    - doc_query_pairs: Dict[str, str] -- doc_id -> query_text map
    - query_eval: QueryEval -- the query evaluation object, already initialized

    The dataset should be a dictionary with the following format:
    - "doc_id": str -- the document id
    - "query": str -- the query text
    """
    # convert to list
    doc_ids = []
    queries = []
    doc_id_score_map = {}
    for doc_id, query in doc_query_pairs.items():
        doc_ids.append(doc_id)
        queries.append(query)
    # score the queries
    for i in range(0, len(queries), batch_size):
        queries_batch = queries[i : i + batch_size]
        doc_ids_batch = doc_ids[i : i + batch_size]
        # score the queries
        scores = query_eval.score(queries=queries_batch, doc_indices=doc_ids_batch)
        # save the scores
        for doc_id, score in zip(doc_ids_batch, scores):
            doc_id_score_map[doc_id] = score.item()
    return doc_id_score_map


def build_cpo_dataset(
    model_name: str,
    teacher_model: str,
    output_dir: Path,
    num_samples: int,
    prompt_template_name: str,
    dataset_name: str = "msmarco-document",
    num_examples: int = 3,
    seed: int = 42,
    student_dtype="auto",
    teacher_dtype="auto",
    max_doc_length: int = 512,
    max_query_length: int = 64,
    max_prompt_length: int = 1024,
    max_new_token: int = 16,
    query_eval: Optional[QueryEval] = None,
    max_workers: int = 1,
    use_vllm: bool = False,
    deterministic: bool = True,
    enable_prefix_caching: bool = True,
    enable_chunked_prefill: bool = True,
    temperature=0.3,
):
    """
    TODO: update the docstring
    Build a dataset for CPO training.

    Performs the following steps:
    1. Load the document-query pairs from the dataset.
        Here we only consider the documents that have a relevance score of 1.
    2. Sample `num_samples` document-query pairs.
    3. Load the document and query texts corresponding to the sampled document-query pairs.
        `msmarco-document/train` has url, title and body fields for each document. We concatenate title and body.
    4. Sample `num_examples` example document-query pairs for each document.
    5. Load the example document and query texts.
    6. Generate teacher queries
    7. Generate student queries
    8. Compute reference scores
    9. Compute teacher scores
    10. Compute student scores

    After each of these steps, save the dataset to a JSON file in the following format:
    ```json
    {
        "dataset_name": str -- name of the dataset,
        "doc_ids": List[str] -- list of document IDs,
        "data": {
            {doc_id}: {
                "example_ids": List[str], -- list of example document IDs
                "example_texts": List[str], -- list of example document strings
                "example_query_ids": List[str], -- list of example query IDs
                "example_queries": List[str], -- list of example queries

                "target_doc_id": str, -- the target document ID
                "target_doc_text": str, -- the target document string
                "ref_query_id": str, -- id of the ground truth query
                "ref_query": str, -- the ground truth query string
                "teacher_query": str, -- the teacher query
                "student_query": str, -- the student query

                "ref_score": float, -- the ground truth score
                "teacher_score": float, -- the teacher score
                "student_score": float, -- the student score
            }
        }
    }
    ```

    Parameters:
    - dataset_name: str | list[str]
        The name of the dataset to build. Can also be a list of dataset names in which case a mixed dataset will be built.
    - num_samples: Optional[int]
        The number of samples to build. If None, the entire dataset will be built.
    - num_examples: int
        The number of example document-query pairs to include for each document.
    - seed: int
        The seed for the random number generator.
    - output_dir: Path
        The output directory to save the dataset.
        It will be saved as a JSON file,
        called `llmt_preference_{model_name}_{dataset_name}_{num_samples}.json`.
    """

    output_dir = output_dir / re.sub(r"[^a-zA-Z0-9]", "_", dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"llmt_preference_{model_name.split("/")[-1]}_{num_samples}.json"
    )

    dataset_path = output_dir / "combined_data.csv"

    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
    else:
        # load dataset
        # make a dataframe of query_id, doc_id, document, query.
        # this is unavoidable because we keep using the queries and documents in pairs
        # and we need to sample from the entire dataset for the examples as well.
        dataset = get_corpus(dataset_name)
        documents = {}
        queries = {}
        dq_map = {}

        # Query, Document, Relevance, Iteration
        for query_id, doc_id, rel, _ in tqdm(
            dataset.qrels_iter(), total=dataset.qrels_count(), desc="Loading qrels"
        ):
            if rel > 0:
                documents[doc_id] = None
                queries[query_id] = None
                dq_map[doc_id] = query_id

        # load documents
        documents = get_documents(dataset, documents, max_workers=max_workers)

        # load queries
        queries = get_queries(dataset, queries, max_workers=max_workers)

        q_ids, q_texts, d_ids, d_texts = [], [], [], []
        for doc_id, doc_text in documents.items():
            d_ids.append(doc_id)
            d_texts.append(doc_text)
            q_ids.append(dq_map[doc_id])
            q_texts.append(queries[dq_map[doc_id]])
        dataset = pd.DataFrame(
            {
                "query_id": q_ids,
                "query_text": q_texts,
                "doc_id": d_ids,
                "doc_text": d_texts,
            }
        ).drop_duplicates(subset="doc_id")

        dataset.to_csv(dataset_path, index=False)
        logger.info(f"Saved combined dataset to {dataset_path}")

    (
        output,
        has_docs_queries,  # step 1, 2, 3
        has_prompts,  # step 4, 5
        has_teacher_queries,  # step 6
        has_student_queries,  # step 7
        has_ref_scores,  # step 8
        has_teacher_scores,  # step 9
        has_student_scores,  # step 10
    ) = continue_from_checkpoint(
        output_path,
        dataset_name,
        num_samples,
        model_name=model_name,
        teacher_model=teacher_model,
        prompt_template_name=prompt_template_name,
        num_examples=num_examples,
        max_doc_length=max_doc_length,
        max_query_length=max_query_length,
        max_prompt_length=max_prompt_length,
        max_new_token=max_new_token,
        seed=seed,
        deterministic=deterministic,
    )

    # load the dataset
    if not has_docs_queries:
        # sample num_samples document-query pairs from the dataframe
        if num_samples > len(dataset):
            logging.warning(
                f"Number of samples ({num_samples}) exceeds size of dataset ({len(dataset)}). Only {len(dataset)} examples will be considered."
            )
        samples = (
            dataset
            if num_samples > len(dataset)
            else dataset.sample(num_samples, replace=False, random_state=seed)
        )  # .dropna()

        # TODO: Eventually lift this restriction once we stop using doc_ids as keys?
        #  samples = (
        #     dataset.sample(num_samples, replace=False, random_state=seed)
        #     if num_samples <= len(dataset)
        #     else pd.concat(
        #         [
        #             dataset,
        #             dataset.sample(
        #                 num_samples - len(dataset), replace=True, random_state=seed
        #             ),
        #         ],
        #         ignore_index=True,
        #     )
        # )

        output["doc_ids"] = samples["doc_id"].tolist()
        for row in samples.itertuples():
            output["data"][row.doc_id] = {
                "target_doc_id": str(row.doc_id),
                "target_doc_text": str(row.doc_text),
                "ref_query_id": str(row.query_id),
                "ref_query": row.query_text,
            }
        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Loaded document texts and saved dataset to {output_path}")

    # Step 4,5: Generate the prompts
    if not has_prompts:
        # prompt builder args (with DynamicPromptV2)
        prompt = Prompt.load(
            name=prompt_template_name,
            dataset=dataset_name,
            examples=dataset[["query_id", "doc_id", "query_text", "doc_text"]]
            .sample(num_examples * 100)
            .to_numpy()
            .tolist(),  # maybe we just want to filter out some random population
            tokenizer=AutoTokenizer.from_pretrained(
                model_name
            ),  # we can use the student tokenizer here
            max_doc_length=max_doc_length,
            max_query_length=max_query_length,
            max_prompt_length=max_prompt_length,
            max_new_token=max_new_token,
            deterministic=deterministic,
        )

        # # DynamicPromptV2
        prompts = get_prompts(output, prompt, num_examples, max_workers=max_workers)

        for doc_id, p in prompts:
            output["data"][doc_id]["prompt"] = p

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Generated prompts and saved dataset to {output_path}")

    # generate teacher queries
    if not has_teacher_queries:
        if use_vllm:
            from . import vllm_inference

            prompts = [data["prompt"] for data in output["data"].values()]
            teacher_queries = vllm_inference.generate_queries(
                prompts=prompts,
                doc_ids=output["doc_ids"],
                model_name=teacher_model,
                save_folder=output_dir,
                max_prompt_length=max_prompt_length,
                dtype=teacher_dtype,
                enable_prefix_caching=enable_prefix_caching,
                enable_chunked_prefill=enable_chunked_prefill,
                temperature=temperature,
                force=False,
            )
            vllm_inference.generate_queries.release()

        else:
            # load model and tokenizer
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model)

            prompts = [data["prompt"] for data in output["data"].values()]
            teacher_queries = generate_queries(
                model=teacher_model,
                tokenizer=teacher_tokenizer,
                prompts=prompts,
                doc_ids=output["doc_ids"],
                save_folder=output_dir,
                max_prompt_length=max_prompt_length,
                dtype=teacher_dtype,
                temperature=temperature,
                force=False,
            )

        for doc_id, query in teacher_queries.items():
            text, logprobs, cumlogprob = query

            output["data"][doc_id]["teacher_query"] = text

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Generated teacher queries and saved dataset to {output_path}")

    # generate student queries
    if not has_student_queries:
        if use_vllm:
            from . import vllm_inference

            prompts = [data["prompt"] for data in output["data"].values()]
            student_queries = vllm_inference.generate_queries(
                prompts=prompts,
                doc_ids=output["doc_ids"],
                model_name=model_name,
                save_folder=output_dir,
                max_prompt_length=max_prompt_length,
                dtype=student_dtype,
                temperature=temperature,
                force=True,
            )
            vllm_inference.generate_queries.release()
        else:
            # load model and tokenizer
            student_tokenizer = AutoTokenizer.from_pretrained(model_name)
            student_model = AutoModelForCausalLM.from_pretrained(model_name)

            prompts = [data["prompt"] for data in output["data"].values()]
            student_queries = generate_queries(
                model=student_model,
                tokenizer=student_tokenizer,
                prompts=prompts,
                doc_ids=output["doc_ids"],
                save_folder=output_dir,
                max_prompt_length=max_prompt_length,
                dtype=student_dtype,
                temperature=temperature,
                force=False,
            )

        for doc_id, query in student_queries.items():
            text, logprobs, cumlogprob = query
            output["data"][doc_id]["student_query"] = text

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Generated student queries and saved dataset to {output_path}")

    # initialize or load query evaluator
    if query_eval is None:
        query_eval = QueryEval.load_from_cache(output_dir)
        if query_eval is None:
            query_eval = QueryEval()
            documents = [
                (doc_id, data["target_doc_text"])
                for doc_id, data in output["data"].items()
            ]
            documents = pd.DataFrame(documents, columns=("doc_id", "text"))

            query_eval.load_dataset(documents, batch_size=128)
            # checkpoint
            query_eval.save_to_cache(output_dir)
            logger.info(f"Saved query evaluator to {output_dir}")

    # obtain reference scores
    if not has_ref_scores:
        ref_doc_query_pairs = {
            doc_id: data["ref_query"] for doc_id, data in output["data"].items()
        }

        scores = score_queries(ref_doc_query_pairs, query_eval)

        for doc_id, score in scores.items():
            output["data"][doc_id]["ref_score"] = score

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Generated reference scores and saved dataset to {output_path}")

    # obtain teacher scores
    if not has_teacher_scores:
        teacher_doc_query_pairs = {
            doc_id: data["teacher_query"] for doc_id, data in output["data"].items()
        }
        scores = score_queries(teacher_doc_query_pairs, query_eval)

        for doc_id, score in scores.items():
            output["data"][doc_id]["teacher_score"] = score

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Generated teacher scores and saved dataset to {output_path}")

    # obtain student scores
    if not has_student_scores:
        student_doc_query_pairs = {
            doc_id: data["student_query"] for doc_id, data in output["data"].items()
        }
        scores = score_queries(student_doc_query_pairs, query_eval)

        for doc_id, score in scores.items():
            output["data"][doc_id]["student_score"] = score

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Generated student scores and saved dataset to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Student model")
    parser.add_argument(
        "--teacher_model", type=str, required=True, help="Teacher model"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", default="cache"
    )
    parser.add_argument(
        "--num_samples", type=int, help="Number of samples to build", default=20_000
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        help="Prompt template name",
        default="inparsplus",
        choices=["inparsplus", "inpars", "promptagator"],
    )
    parser.add_argument("--dataset_name", type=str, default="msmarco-document")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_prompt_length", type=int, default=8192)
    parser.add_argument("--max_workers", type=int, default=get_optimal_process_count())
    parser.add_argument("--max_new_token", type=int, default=32)
    parser.add_argument("--student_use_fp16", action="store_true")
    parser.add_argument("--teacher_use_fp16", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()
    logging.debug(f"Arguments: {args}")

    build_cpo_dataset(
        model_name=args.model_name,
        teacher_model=args.teacher_model,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        prompt_template_name=args.prompt_template_name,
        dataset_name=args.dataset_name,
        num_examples=args.num_examples,
        seed=args.seed,
        student_dtype=torch.float16 if args.student_use_fp16 else "auto",
        teacher_dtype=torch.float16 if args.teacher_use_fp16 else "auto",
        max_doc_length=args.max_doc_length,
        max_query_length=args.max_query_length,
        max_prompt_length=args.max_prompt_length,
        max_workers=args.max_workers,
        use_vllm=args.use_vllm,
        max_new_token=args.max_new_token,
        temperature=args.temperature,
    )
