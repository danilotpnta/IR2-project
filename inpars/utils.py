from math import prod
import os
import csv
import sys
import ftfy
import requests
import pandas as pd
from tqdm.auto import tqdm

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
                download(PREBUILT_RUN_URL.format(dataset=run_file), dest_file)
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


def _process_map_q(q):
    return q.query_id, ftfy.fix_text(q.text)


def _process_map_d(d):
    id, d = d
    return id, ftfy.fix_text(d.title + " " + d.body)


def _process_map_p(document, doc_id, prompter, n_examples=3):
    return doc_id, prompter.build(document, n_examples=n_examples)


def count_parameters(model, predicate=lambda p: p.requires_grad):
    model_parameters = filter(predicate, model.parameters())
    params = sum([prod(p.size()) for p in model_parameters])
    return params


def get_optimal_cpu_count():
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
