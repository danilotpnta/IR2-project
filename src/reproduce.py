import subprocess
import argparse
import logging
import time
import json
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        choices=[
            'nfcorpus',
            'trec-covid',
            'hotpotqa',
            'fiqa',
            'arguana',
            'webis-touche2020',
            'dbpedia-entity',
            'scidocs',
            'fever',
            'climate-fever',
            'scifact',
        ],
        default='trec-covid',
        help="Choose dataset from BEIR to reproduce. "
    )

    parser.add_argument(
        '--generationLLM',
        choices=[
            'EleutherAI/gpt-j-6B',
            'meta-llama/Llama-3.2-3B'
        ],
        default='EleutherAI/gpt-j-6B',
        help="Choose query generation model. "
    )

    parser.add_argument(
        '--reranker',
        choices=[
            'castorini/monot5-3b-msmarco-10k',
            'castorini/rankllama-v1-7b-lora-passage'
        ],
        default='castorini/monot5-3b-msmarco-10k',
        help="Choose reranker model."
    )

    parser.add_argument(
        '--root_path',
        required=True
    )

    args = parser.parse_args()

    return args

def generate_queries(prompt_type:str, dataset:str, model:str, output_path:str) -> None:
    logging.info(f'------Starting query generation : {prompt_type}------')
    start_generation = time.time()

    try:
        subprocess.run([
            "python", "-m", "inpars.generate",
            f"--prompt={prompt_type}",
            f"--dataset={dataset}",
            "--dataset_source=ir_datasets",
            f"--base_model={model}",
            f"--output={output_path}",
            '--batch_size=4',
            "--fp16",
        ],  stdout=subprocess.PIPE,
            text=True
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Generating queries took : {generation_time} seconds.\n')


def filter_queries(query_path:str, dataset:str, output_path:str, filtering_strategy:str) -> None:
    logging.info(f'------Starting filtering of : {query_path}------')
    start_generation = time.time()

    try:
        subprocess.run([
            "python", "-m", "inpars.filter",
            f"--input={query_path}",
            f"--dataset={dataset}",
            f"--filter_strategy={filtering_strategy}",
            f"--keep_top_k={4999}",
            f"--output={output_path}"
        ],  stdout=subprocess.PIPE,
            text=True
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Filtering queries took : {generation_time} seconds.\n')

def generate_triples(filtered_path:str, dataset:str, output_path:str):
    logging.info(f'------Starting triple generation of : {filtered_path}------')
    start_generation = time.time()

    try:
        subprocess.run([
            "python", "-m", "inpars.generate_triples",
            f"--input={filtered_path}",
            f"--dataset={dataset}",
            f"--output={output_path}"
        ],  stdout=subprocess.PIPE,
            text=True
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Triple generation took : {generation_time} seconds.\n')

def train_reranker(triples_path:str, ranker_model:str, output_path):
    logging.info(f'------Starting reranker training of : {triples_path}------')
    start_generation = time.time()

    try:
        subprocess.run([
            "python", "-m", "inpars.train",
            f"--triples={triples_path}",
            f"--base_model={ranker_model}",
            f"--output_dir={output_path}",
            "--max_steps=156",
            "--per_device_train_batch_size=2",
            "--gradient_accumulation_steps=16",
            "--fp16",
            "--learning_rate=1e-4",
        ],  stdout=subprocess.PIPE,
            text=True
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Training took : {generation_time} seconds.\n')

def rerank(model_path:str, dataset:str, output_path):
    logging.info(f'------Starting reranking of {dataset} using {model_path}------')
    start_generation = time.time()

    try:
        subprocess.run([
            "python", "-m", "inpars.rerank",
            f"--model={model_path}",
            f"--dataset={dataset}",
            f"--output_run={output_path}",
            "--device=cuda",
            "--fp16",
            "--top_k=100"
        ],  stdout=subprocess.PIPE,
            text=True
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Reranking took : {generation_time} seconds.\n')

def evaluate(dataset:str, trec_path:str, output_path:str):
    logging.info(f'------Starting evaluation of {trec_path} for {dataset}------')
    try:
        subprocess.run([
            "python", "-m", "inpars.evaluate",
            f"--dataset={dataset}",
            f"--run={trec_path}",
            f"--output_path={output_path}",
            "--json"
        ],  stdout=subprocess.PIPE,
            text=True
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    logging.info('Done!\n')


def main(args):
    # Create path where to save data such as queries generated.
    root = args.root_path
    # Start with data dir if it does not exist
    if not os.path.exists(os.path.join(root, 'data')):
        os.mkdir(os.path.join(root, 'data'))

    # Separate directory per dataset
    data_path = os.path.join(root, 'data', args.dataset)

    # Possible prompt templates
    prompt_options = ['inpars', 'promptagator'] #'inpars-gbq']
    # Possible filtering options (inparsV1, inparsV2 respectively)
    filter_options = ['scores', 'reranker']

    # Set up directory structure for saving data
    if not os.path.exists(data_path):
        # Create dataset specific dir
        os.mkdir(data_path)

    # Create generationLLM specific dir
    data_path = os.path.join(data_path, args.generationLLM.split('/')[-1])
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        # Create directory per prompt template (promptagate and inpars)
        for opt in prompt_options:
            os.mkdir(os.path.join(data_path, opt))
            # For each filter option, create directory.
            for filter_opt in filter_options:
                os.mkdir(os.path.join(data_path, opt, filter_opt))


    # For each prompt type, save queries in directory.
    for prompt_type in prompt_options:
        query_output_path = os.path.join(data_path, prompt_type, f'{prompt_type}-queries.jsonl')

        if os.path.exists(query_output_path):
            logging.info(f'{prompt_type} has already been created. Continuing...')
            continue

        generate_queries(prompt_type, args.dataset, args.generationLLM, query_output_path)

    logging.info(f'Done with the query generation stage! \n\n Continuing with the filtering stage...\n')

    for prompt_type in prompt_options:
        input_path = os.path.join(data_path, prompt_type, f'{prompt_type}-queries.jsonl')
        for filter_type in filter_options:
            filter_output_path = os.path.join(data_path, prompt_type, filter_type, f'{prompt_type}-queries-filtered.jsonl')
            if os.path.exists(filter_output_path):
                logging.info(f'({prompt_type},{filter_type}) has already been filtered. Continuing...')
                continue

            filter_queries(input_path, args.dataset, filter_output_path, filter_type)


    logging.info(f'Done with the filtering stage! \n\n Continuing with the negative mining/triple generation stage...\n')

    for prompt_type in prompt_options:
        for filter_type in filter_options:
            input_path = os.path.join(data_path, prompt_type, filter_type, f'{prompt_type}-queries-filtered.jsonl')
            triple_output_path = os.path.join(data_path, prompt_type, filter_type, f'{prompt_type}-triples.tsv')

            if os.path.exists(triple_output_path):
                logging.info(f'({prompt_type},{filter_type}) has already triples generated. Continuing...')
                continue

            generate_triples(input_path, args.dataset, triple_output_path)

    logging.info(f'Done with triple generation stage! \n\n Continuing with the training stage...\n')

    for prompt_type in prompt_options:
        for filter_type in filter_options:
            input_path = os.path.join(data_path, prompt_type, filter_type, f'{prompt_type}-triples.tsv')
            output_path = os.path.join(data_path, prompt_type, filter_type, 'reranker')
            if os.path.exists(output_path):
                logging.info(f'({prompt_type},{filter_type}) has already a reranker trained. Continuing...')
                continue
            logging.info(args.reranker)
            train_reranker(input_path, args.reranker, output_path)

    logging.info(f'Done with training stage! \n\n Continuing with the reranking stage...\n')

    for prompt_type in prompt_options:
        for filter_type in filter_options:
            reranker_path = os.path.join(data_path, prompt_type, filter_type, 'reranker')
            output_path = os.path.join(data_path, prompt_type, filter_type, 'trec-run.txt')

            rerank(reranker_path, args.dataset, output_path)


    logging.info(f'Done with reranking stage! \n\n Continuing with the evaluation stage...\n')

    for prompt_type in prompt_options:
        for filter_type in filter_options:
            trec_path = os.path.join(data_path, prompt_type, filter_type, 'trec-run.txt')
            output_path = os.path.join(data_path, prompt_type, filter_type, 'results.json')

            evaluate(args.dataset, trec_path, output_path)

            with open(output_path, 'r') as rf:
                data = json.loads(rf)

            logging.info(f'--------Results of ({prompt_type, filter_type})--------')
            for key, value in data.items():
                logging.info(f'|{key}\t\t|\t{value}\t|')



if __name__ == '__main__':
    args = parse_args()

    main(args)