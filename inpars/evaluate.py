import os
import json
import argparse
import subprocess
from pyserini.search import get_qrels_file
from .utils import TRECRun

import logging

def run_trec_eval(run_file, qrels_file, relevance_threshold=1, remove_unjudged=False):
    args = [
        "python3",
        "-m",
        "pyserini.eval.trec_eval",
        "-c",
        f"-l {relevance_threshold}",
        "-m",
        "all_trec",
        "-m",
        "judged.10",
    ]

    if remove_unjudged:
        args.append("-remove-unjudged")

    args += [qrels_file, run_file]

    result = subprocess.run(args, stdout=subprocess.PIPE)

    metrics = {}
    for line in result.stdout.decode("utf-8").split("\n"):
        for metric in [
            "recip_rank",
            "recall_1000",
            "num_q",
            "num_ret",
            "ndcg_cut_10",
            "ndcg_cut_20",
            "map",
            "P_10",
            "judged_10",
        ]:
            # the space is to avoid getting metrics such as ndcg_cut_100 instead of ndcg_cut_10 as but start with ndcg_cut_10
            if line.startswith(metric + " ") or line.startswith(metric + "\t"):
                metrics[metric] = float(line.split("\t")[-1])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", default="msmarco")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--relevance_threshold", default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--remove_unjudged", action="store_true")

    # Added output path where the results should be saved.
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    if args.dataset == "msmarco":
        dataset_name = "msmarco-passage-dev-subset"
    else:
        dataset_name = f"beir-v1.0.0-{args.dataset}-test"

    qrels_file = get_qrels_file(dataset_name)

    if args.qrels and os.path.exists(args.qrels):
        qrels_file = args.qrels

    run_file = args.run
    if args.run.lower() == "bm25":
        run = TRECRun(args.dataset)
        run_file = run.run_file

    results = run_trec_eval(
        run_file, qrels_file, args.relevance_threshold, args.remove_unjudged
    )
    if args.json:
        with open(args.output_path, 'w') as wf:
            json.dump(results, wf)
    else:
        for (metric, value) in sorted(results.items()):
            logging.info(f"{metric}: {value}")