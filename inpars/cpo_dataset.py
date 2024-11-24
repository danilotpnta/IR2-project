from typing import Any, Callable, Dict, Optional
import json
import logging
from tqdm import tqdm
from pathlib import Path

import ftfy
import ir_datasets
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from query_eval import QueryEval

logger = logging.getLogger(__name__)

def load_llmt_dataset(source_path: Path):
    """
        Implementation adapted from 
        https://github.com/fe1ixxu/ALMA/blob/master/utils/utils.py :: `preprocess_cpo_data`
    """
    pass

def continue_from_checkpoint(output_path: Path, dataset_name: str, num_samples: Optional[int] = None):
    output = {
        "dataset_name": dataset_name,
        "doc_ids": [],
        "data": []
    }
    # flags to continue from where we left off
    has_ids = False
    has_doc_query_texts = False
    has_examples = False
    has_teacher_queries = False
    has_student_queries = False
    has_ref_scores = False
    has_teacher_scores = False
    has_student_scores = False

    if output_path.exists():
        with open(output_path, "r") as f:
            output = json.load(f)
            has_ids = "doc_ids" in output and\
                (len(output["doc_ids"]) == num_samples or
                 num_samples is None)
        data = output["data"].values().next()
        has_doc_query_texts = "target_doc_text" in data and "ref_query" in data
        has_examples = "example_ids" in data
        has_teacher_queries = "teacher_query" in data
        has_student_queries = "student_query" in data
        has_ref_scores = "ref_score" in data
        has_teacher_scores = "teacher_score" in data
        has_student_scores = "student_score" in data
    return output, has_ids, has_doc_query_texts, has_examples, has_teacher_queries, has_student_queries, has_ref_scores, has_teacher_scores, has_student_scores

def load_doc_query_pairs(dataset, num_samples, seed):
    # expecting one qrel per query, with relevance 1
    # format is (query_id, doc_id, relevance)
    qrels = dataset.qrels_iter()
    qrels = [
        (qid, doc_id) for qid, doc_id, rel in tqdm(
            qrels, total=dataset.qrels_count(), desc="Loading qrels"
        ) if rel == 1]
    # sample num_samples
    if num_samples is not None and len(qrels) > num_samples:
        qrels = np.random.RandomState(seed).choice(
            qrels, num_samples, replace=False)
    # map doc_ids to query_ids
    return {doc_id: qid for qid, doc_id in qrels}

def load_doc_texts(dataset, doc_ids):
    # format is (doc_id, url, title, body)
    ret = {}
    for doc in tqdm(filter(lambda doc: doc.doc_id in doc_ids,
                           dataset.docs_iter()),
                    total=len(doc_ids),
                    desc="Loading documents"):

        # TODO: 'body' is too long. We can only use 'title'
        ret[doc.doc_id] = ftfy.fix_text(f"{doc.title}")
    return ret

def load_query_texts(dataset, query_ids):
    # load query texts
    # format is (query_id, text)
    ret = {}
    query_ids = set(query_ids)
    queries_iter = tqdm(filter(lambda q: q.query_id in query_ids,
                               dataset.queries_iter()),
                        total=len(query_ids), desc="Loading queries")
    for query in queries_iter:
        ret[query.query_id] = ftfy.fix_text(query.text)
    return ret

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

def generate_queries(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                     dataset: Dict[str, Any], prompt_builder: Callable,
                     batch_size: int = 32):
    """
        Generate queries using the given model and dataset.
        Parameters:
        - model_name: str
            The name of the model to use.
        - dataset: dict
            The dataset to generate queries for.
        - prompt_builder: Callable
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
    # generate queries
    # save queries
    return queries

def score_queries(doc_query_pairs: Dict[str, str],
                  query_eval: QueryEval,
                  batch_size: int = 32):
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
        queries_batch = queries[i:i + batch_size]
        doc_ids_batch = doc_ids[i:i + batch_size]
        # score the queries
        scores = query_eval.score(queries=queries_batch, doc_indices=doc_ids_batch)
        # save the scores
        for doc_id, score in zip(doc_ids_batch, scores):
            doc_id_score_map[doc_id] = score
    return doc_id_score_map

def build_llmt_dataset(
    model_name: str,
    teacher_model: str,
    dataset_name: str = 'msmarco-document/train',
    num_samples: Optional[int] = None,
    prompt_builder: Optional[Callable] = None,
    num_examples: int = 3,
    seed: int = 42,
    output_dir: Optional[Path] = None,
):
    """
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
        - output_dir: Optional[Path]
            The output directory to save the dataset. 
            It will be saved as a JSON file,
            called `llmt_preference_{model_name}_{dataset_name}_{num_samples}.json`.
    """
    if not dataset_name.startswith('msmarco-document'):
        raise NotImplementedError("Only MS-MARCO is supported for now.")

    output_path = output_dir / f"llmt_preference_{model_name}_{dataset_name}{f"_{num_samples}" if num_samples else ""}.json"

    (output,
     has_ids,               # step 1, 2
     has_doc_query_texts,   # step 3
     has_examples,          # step 4, 5
     has_teacher_queries,   # step 6
     has_student_queries,   # step 7
     has_ref_scores,         # step 8
     has_teacher_scores,    # step 9
     has_student_scores     # step 10
     ) = continue_from_checkpoint(output_path, dataset_name, num_samples)

    # load dataset
    dataset = ir_datasets.load(dataset_name)

    # load the dataset
    if not has_ids:
        doc_query_id_map = load_doc_query_pairs(dataset, num_samples, seed)
        output["doc_ids"] = list(doc_query_id_map.keys())
        output["data"] = {doc_id: {
            "target_doc_id": doc_id,
            "ref_query_id": query_id
        } for doc_id, query_id in doc_query_id_map.items()}
        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(
            f"Loaded document and query IDs and Saved dataset to {output_path}")

    # load document texts
    if not has_doc_query_texts:
        doc_ids = output["doc_ids"]
        doc_texts = load_doc_texts(dataset, doc_ids)
        for doc_id, text in doc_texts.items():
            output["data"][doc_id]["target_doc_text"] = text

        query_ids = {data['ref_query_id']: doc_id for doc_id,
                     data in output["data"].items()}
        query_texts = load_query_texts(dataset, query_ids.keys())
        for query_id, text in query_texts.items():
            output["data"][query_ids[query_id]]["ref_query"] = text

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(
            f"Loaded document texts and saved dataset to {output_path}")

    # load example document-query pairs
    if not has_examples:
        examples_map = load_examples(dataset, output["doc_ids"], num_examples, seed)
        for doc_id, examples in examples_map.items():
            output["data"][doc_id]["example_ids"] = examples["doc_ids"]
            output["data"][doc_id]["example_texts"] = examples["doc_texts"]
            output["data"][doc_id]["example_query_ids"] = examples["query_ids"]
            output["data"][doc_id]["example_queries"] = examples["query_texts"]
        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(f"Loaded example texts and saved dataset to {output_path}")

    # generate prompt builder
    if prompt_builder is None:
        prompt_builder = lambda example_documents, example_queries, target_doc_text: f"""
            {'\n'.join(f"Example {i}:\nDocument: {doc}\nQuery {i}: {query}" for i, (doc, query) in enumerate(zip(example_documents, example_queries)))}
            Document: {target_doc_text}
            Query:
        """

    # generate teacher queries
    if not has_teacher_queries:
        # load model and tokenizer
        # TODO: we might want to just pass the model and tokenizer
        teacher_tokenizer = PreTrainedTokenizerBase.from_pretrained(teacher_model)
        teacher_model = PreTrainedModel.from_pretrained(teacher_model)

        teacher_queries = generate_queries(teacher_model, teacher_tokenizer, output, prompt_builder)
        for doc_id, query in teacher_queries.items():
            output["data"][doc_id]["teacher_query"] = query

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(
            f"Generated teacher queries and saved dataset to {output_path}")

    # generate student queries
    if not has_student_queries:
        # load model and tokenizer
        # TODO: we might want to just pass the model and tokenizer
        student_tokenizer = PreTrainedTokenizerBase.from_pretrained(model_name)
        student_model = PreTrainedModel.from_pretrained(model_name)

        student_queries = generate_queries(student_model, student_tokenizer, output, prompt_builder)

        for doc_id, query in student_queries.items():
            output["data"][doc_id]["student_query"] = query

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(
            f"Generated student queries and saved dataset to {output_path}")

    # initialize or load query evaluator
    if query_eval is None:
        try:
            query_eval = QueryEval.load_from_cache(output_dir)
        except ValueError:
            query_eval = QueryEval()
            documents = {doc_id: data["target_doc_text"] for doc_id,
                            data in output["data"].items()}
            query_eval.load_dataset(documents)
            # checkpoint
            query_eval.save_to_cache(output_dir)
            logger.info(f"Saved query evaluator to {output_dir}")

    # generate reference scores
    if not has_ref_scores:
        ref_doc_query_pairs = {doc_id: data["ref_query"] for doc_id,
                               data in output["data"].items()}
        scores = score_queries(ref_doc_query_pairs, query_eval)

        for doc_id, score in scores.items():
            output["data"][doc_id]["ref_score"] = score

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(
            f"Generated reference scores and saved dataset to {output_path}")

    # generate teacher scores
    if not has_teacher_scores:
        teacher_doc_query_pairs = {doc_id: data["teacher_query"] for doc_id,
                                   data in output["data"].items()}
        scores = score_queries(teacher_doc_query_pairs, query_eval)

        for doc_id, score in scores.items():
            output["data"][doc_id]["teacher_score"] = score

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(
            f"Generated teacher scores and saved dataset to {output_path}")

    # generate student scores
    if not has_student_scores:
        student_doc_query_pairs = {doc_id: data["student_query"] for doc_id,
                                   data in output["data"].items()}
        scores = score_queries(student_doc_query_pairs, query_eval)

        for doc_id, score in scores.items():
            output["data"][doc_id]["student_score"] = score

        # checkpoint
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info(
            f"Generated student scores and saved dataset to {output_path}")

    return output
