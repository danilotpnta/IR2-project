from math import prod
import os
import csv
import sys
import ftfy
import requests
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

PREBUILT_RUN_URL = "https://huggingface.co/datasets/unicamp-dl/beir-runs/resolve/main/bm25/run.beir-v1.0.0-{dataset}-flat.trec"
RUNS_CACHE_FOLDER = os.environ["HOME"] + "/.cache/inpars"


# https://stackoverflow.com/a/62113293
def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(fname, "wb") as file,
        tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class TRECRun:
    def __init__(self, run_file, sep=r"\s+"):
        if not os.path.exists(run_file):
            dest_file = os.path.join(
                RUNS_CACHE_FOLDER,
                "runs",
                "run.beir-v1.0.0-{dataset}-flat.trec".format(dataset=run_file),
            )
            if not os.path.exists(dest_file):
                os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
                # TODO handle errors ("Entry not found")
                try:
                    download(PREBUILT_RUN_URL.format(dataset=run_file), dest_file)

                except Exception as e:
                    assert e

            run_file = dest_file

        self.run_file = run_file
        self.df = pd.read_csv(
            run_file,
            sep=sep,
            quoting=csv.QUOTE_NONE,
            keep_default_na=False,
            names=("qid", "_1", "docid", "rank", "score", "ranker"),
            dtype=str,
        )

    def rerank(self, ranker, queries, corpus, top_k=1000):
        # Converts run to float32 and subtracts a large number to ensure the BM25 scores
        # are lower than those provided by the neural ranker.
        self.df["score"] = self.df["score"].astype("float32").apply(lambda x: x - 10000)

        # Reranks only the top-k documents for each query
        subset = (
            self.df[["qid", "docid"]]
            .groupby("qid")
            .head(top_k)
            .apply(lambda x: [queries[x["qid"]], corpus[x["docid"]]], axis=1)
        )
        scores = ranker.rescore(subset.values.tolist())

        self.df.loc[subset.index, "score"] = scores

        self.df["ranker"] = ranker.name
        self.df = (
            self.df.groupby("qid")
            .apply(lambda x: x.sort_values("score", ascending=False))
            .reset_index(drop=True)
        )

        self.df["rank"] = self.df.groupby("qid").cumcount() + 1

    def save(self, path):
        self.df.to_csv(path, index=False, sep="\t", header=False, float_format="%.15f")


def _process_map_q(query):
    return query.query_id, ftfy.fix_text(query.text)


def _process_map_d(document):
    # In case of MSMarco
    d = (
        ftfy.fix_text(document.title + " " + document.body)
        if hasattr(document, "body")
        else ftfy.fix_text(document.text)
    )
    return document.doc_id, d


def _process_map_p(document, doc_id, prompter, n_examples=3):
    return doc_id, prompter.build(document, n_examples=n_examples)


def count_parameters(model, predicate=lambda p: p.requires_grad):
    model_parameters = filter(predicate, model.parameters())
    params = sum([prod(p.size()) for p in model_parameters])
    return params


def get_optimal_process_count() -> int:
    platform, pyver = sys.platform, sys.version_info.minor
    if pyver >= 13:
        return os.process_cpu_count() - 1
    if platform == "linux":
        return len(os.sched_getaffinity(0)) - 1 if pyver >= 12 else os.cpu_count()
    elif platform == "darwin":
        return os.cpu_count()
    elif platform == "win32":
        return os.cpu_count(logical=False)
    else:
        return 1


def get_documents(dataset, documents, max_workers=1):
    if max_workers == 1:
        for doc in tqdm(
            dataset.docs_store().get_many_iter(documents.keys()),
            total=len(documents),
            desc="Loading documents",
        ):
            documents[doc.doc_id] = (
                ftfy.fix_text(doc.title + " " + doc.body)
                if hasattr(doc, "body")
                else ftfy.fix_text(doc.text)
            )
    else:
        _docs = process_map(
            _process_map_d,
            dataset.docs_store().get_many_iter(documents.keys()),
            chunksize=128,
            total=len(documents),
            desc="Loading documents",
            max_workers=max_workers,
        )
        documents = {doc_id: doc for doc_id, doc in _docs}
    return documents


def get_queries(dataset, queries, max_workers=1):
    if max_workers == 1:
        for query in tqdm(
            dataset.queries_iter(), total=len(queries), desc="Loading queries"
        ):
            queries[query.query_id] = ftfy.fix_text(query.text)

    else:
        _queries = process_map(
            _process_map_q,
            dataset.queries_iter(),
            chunksize=128,
            total=len(queries),
            desc="Loading queries",
            max_workers=max_workers,
        )
        queries = {query_id: query for query_id, query in _queries}
    return queries


def get_prompts(output, prompt, num_examples=3, max_workers=1):
    documents = [data["target_doc_text"] for data in output["data"].values()]
    ids = [doc_id for doc_id in output["data"]]

    prompts = []
    if max_workers == 1:
        for _id, doc in tqdm(
            zip(ids, documents), desc="Generating prompts", total=len(documents)
        ):
            prompts.append((_id, prompt.build(doc, n_examples=num_examples)))
    else:
        prompts = process_map(
            _process_map_p,
            documents,
            ids,
            [prompt] * len(documents),
            [num_examples] * len(documents),
            chunksize=128,
            total=len(documents),
            desc="Generating prompts",
            max_workers=max_workers,
        )

    return prompts
