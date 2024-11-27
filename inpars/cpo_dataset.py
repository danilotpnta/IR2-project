from typing import Any, Callable, Dict, Optional, Tuple
import os
import json
import yaml
import logging
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple

import ftfy
import ir_datasets
import numpy as np
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .query_eval import QueryEval
from .prompt import DeterministicPrompt


logger = logging.getLogger(__name__)


def load_cpo_dataset(source_path: Path):
    """
    Implementation adapted from
    https://github.com/fe1ixxu/ALMA/blob/master/utils/utils.py :: `preprocess_cpo_data`
    """
    pass


def continue_from_checkpoint(
    output_path: Path, dataset_name: str, num_samples: Optional[int] = None
):
    output = {"dataset_name": dataset_name, "doc_ids": [], "data": []}
    # flags to continue from where we left off
    has_docs_queries = False
    has_examples = False
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
        data = output["data"].values().next()
        has_docs_queries = (
            "target_doc_text" in data
            and "ref_query" in data
            and "target_doc_id" in data
            and "ref_query_id" in data
        )
        has_examples = "example_ids" in data
        has_teacher_queries = "teacher_query" in data
        has_student_queries = "student_query" in data
        has_ref_scores = "ref_score" in data
        has_teacher_scores = "teacher_score" in data
        has_student_scores = "student_score" in data
    return (
        output,
        has_docs_queries,
        has_examples,
        has_teacher_queries,
        has_student_queries,
        has_ref_scores,
        has_teacher_scores,
        has_student_scores,
    )


def load_examples(dataset, doc_ids, num_examples, seed):
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
    # TODO: should we sample from the whole dataset or just the subset we computed earlier?
    return


def generate_queries(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dict[str, Any],
    prompt_builder: DeterministicPrompt,
    batch_size: int = 32,
):
    """
    Generate queries using the given model and dataset.
    Parameters:
    - model_name: str
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
    # doc_id -> query
    queries: Dict[str, str] = {}

    # construct collator
    # iterate over the dataset

    # NOTE: prompt_builder usage
    data = dataset["data"].values().next()  # just an example
    prompt = prompt_builder.build(
        document=data["target_doc_text"],
        n_examples=len(data["example_texts"]),
        example_queries=data["example_queries"],
        example_docs=data["example_texts"],
    )
    # end of example

    # generate queries
    # save queries
    return queries


def score_queries(
    doc_query_pairs: Dict[str, str], query_eval: QueryEval, batch_size: int = 32
):
    """
    Score the queries in the dataset using the given scoring function.
    Parameters:
    - dataset: Dict[str, str] -- doc_id -> document text map
    - doc_query_pairs: Dict[str, str] -- doc_id -> query_text map
    - query_eval: QueryEval -- the query evaluation object, already initialized

    The dataset should be a list of dictionaries with the following fields:
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
            doc_id_score_map[doc_id] = score
    return doc_id_score_map


def build_cpo_dataset(
    model_name: str,
    teacher_model: str,
    output_dir: Path,
    num_samples: int,
    prompt_template_name: str,
    dataset_name: str = "msmarco-document/train",
    num_examples: int = 3,
    seed: int = 42,
    max_doc_length: int = 512,
    max_query_length: int = 64,
    max_prompt_length: int = 1024,
    max_new_token: int = 16,
    query_eval: Optional[QueryEval] = None,
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
    - dataset_name: str
        The name of the dataset to build. Currently we only support MS-MARCO.
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
    if not dataset_name.startswith("msmarco-document"):
        raise NotImplementedError("Only MS-MARCO is supported for now.")

    output_path = (
        output_dir
        / f"llmt_preference_{model_name}_{dataset_name}{f'_{num_samples}' if num_samples else ''}.json"
    )
    dataset_path = output_dir / "combined_data.csv"

    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
    else:
        # load dataset
        # make a dataframe of query_id, doc_id, document, query.
        # this is unavoidable because we keep using the queries and documents in pairs
        # and we need to sample from the entire dataset for the examples as well.
        dataset = ir_datasets.load(dataset_name)
        documents = {}
        queries = {}
        qd_map = {}

        # Query, Document, Relevance, Iteration
        for query_id, doc_id, rel, _ in tqdm(
            dataset.qrels_iter(), total=dataset.qrels_count(), desc="Loading qrels"
        ):
            if rel == 1:
                documents[doc_id] = None
                queries[query_id] = None
                qd_map[query_id] = doc_id
        # load documents
        for doc_id in tqdm(
            dataset.docs_iter(), total=len(documents), desc="Loading documents"
        ):
            doc_id, url, title, body = doc_id
            # NOTE: body is too long (avg 1000 words) so we only use title
            documents[doc_id] = ftfy.fix_text(title)
        # load queries
        for query_id in tqdm(
            dataset.queries_iter(), total=len(queries), desc="Loading queries"
        ):
            query_id, text = query_id
            queries[query_id] = ftfy.fix_text(text)
        # convert to dataframe
        q_ids, q_texts, d_ids, d_texts = [], [], [], []
        for q_id, q_text in queries.items():
            q_ids.append(q_id)
            q_texts.append(q_text)
            d_ids.append(qd_map[q_id])
            d_texts.append(documents[qd_map[q_id]])
        dataset = pd.DataFrame(
            {
                "query_id": q_ids,
                "query_text": q_texts,
                "doc_id": d_ids,
                "doc_text": d_texts,
            }
        )
        dataset.to_csv(dataset_path, index=False)
        logger.info(f"Saved combined dataset to {dataset_path}")

    (
        output,
        has_docs_queries,  # step 1, 2, 3
        has_examples,  # step 4, 5
        has_teacher_queries,  # step 6
        has_student_queries,  # step 7
        has_ref_scores,  # step 8
        has_teacher_scores,  # step 9
        has_student_scores,  # step 10
    ) = continue_from_checkpoint(output_path, dataset_name, num_samples)

    # load the dataset
    if not has_docs_queries:
        # sample num_samples document-query pairs from the dataframe
        samples = dataset.sample(num_samples, random_state=seed)
        output["doc_ids"] = samples["doc_id"].tolist()
        for i, row in samples.iterrows():
            output["data"][row["doc_id"]] = {
                "target_doc_id": row["doc_id"],
                "target_doc_text": row["doc_text"],
                "ref_query_id": row["query_id"],
                "ref_query": row["query_text"],
            }
        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Loaded document texts and saved dataset to {output_path}")

    # load example document-query pairs
    if not has_examples:
        # sample num_examples example document-query pairs for each document
        for doc_id in output["doc_ids"]:
            # make sure the examples don't include the target document
            while True:
                examples = dataset.sample(num_examples, random_state=seed)
                if doc_id not in examples["doc_id"].tolist():
                    break
            for i, row in examples.iterrows():
                output["data"][doc_id]["example_ids"] = row["doc_id"].tolist()
                output["data"][doc_id]["example_texts"] = row["doc_text"].tolist()
                output["data"][doc_id]["example_query_ids"] = row["query_id"].tolist()
                output["data"][doc_id]["example_queries"] = row["query_text"].tolist()
        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Loaded example texts and saved dataset to {output_path}")

    # prompt builder args
    with open(f"{os.path.dirname(__file__)}/prompts/templates.yaml") as f:
        templates = yaml.safe_load(f)
    prompt_builder_args = namedtuple(
        template=templates["dynamic"][prompt_template_name],
        examples=None,  # no need to this, since we use DeterministicPrompt
        tokenizer=None,  # needs to be replaced for each model
        max_doc_length=max_doc_length,
        max_query_length=max_query_length,
        max_prompt_length=max_prompt_length,
        max_new_token=max_new_token,
    )

    # generate teacher queries
    if not has_teacher_queries and False:
        # load model and tokenizer
        # TODO: we might want to just pass the model and tokenizer
        teacher_tokenizer = PreTrainedTokenizerBase.from_pretrained(teacher_model)
        teacher_model = PreTrainedModel.from_pretrained(teacher_model)
        # build prompter
        prompt_builder_args.tokenizer = teacher_tokenizer
        prompt_builder = DeterministicPrompt(**prompt_builder_args._asdict())

        # TODO: Not complete yet
        teacher_queries = generate_queries(
            teacher_model, teacher_tokenizer, output, prompt_builder
        )
        for doc_id, query in teacher_queries.items():
            output["data"][doc_id]["teacher_query"] = query

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Generated teacher queries and saved dataset to {output_path}")

    # generate student queries
    if not has_student_queries and False:
        # load model and tokenizer
        # TODO: we might want to just pass the model and tokenizer
        student_tokenizer = PreTrainedTokenizerBase.from_pretrained(model_name)
        student_model = PreTrainedModel.from_pretrained(model_name)
        # build prompter
        prompt_builder_args.tokenizer = student_tokenizer
        prompt_builder = DeterministicPrompt(**prompt_builder_args._asdict())

        # TODO: Not complete yet
        student_queries = generate_queries(
            student_model, student_tokenizer, output, prompt_builder
        )

        for doc_id, query in student_queries.items():
            output["data"][doc_id]["student_query"] = query

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Generated student queries and saved dataset to {output_path}")

    # initialize or load query evaluator
    if query_eval is None:
        try:
            query_eval = QueryEval.load_from_cache(output_dir)
        except ValueError:
            query_eval = QueryEval()
            documents = {
                doc_id: data["target_doc_text"]
                for doc_id, data in output["data"].items()
            }
            query_eval.load_dataset(documents)
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
    parser.add_argument("--teacher_model", type=str, required=True, help="Teacher model")
    parser.add_argument("--output_dir", type=str, help="Output directory", default="cache")
    parser.add_argument("--num_samples", type=int, help="Number of samples to build", default=20_000)
    parser.add_argument("--prompt_template_name", type=str, help="Prompt template name", default="inparsplus")
    parser.add_argument("--dataset_name", type=str, default="msmarco-document/train")
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_token", type=int, default=16)
    args = parser.parse_args()

    build_cpo_dataset(
        model_name=args.model_name,
        teacher_model=args.teacher_model,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        prompt_template_name=args.prompt_template_name,
        dataset_name=args.dataset_name,
        num_examples=args.num_examples,
        seed=args.seed,
        max_doc_length=args.max_doc_length,
        max_query_length=args.max_query_length,
        max_prompt_length=args.max_prompt_length,
        max_new_token=args.max_new_token,
    )
