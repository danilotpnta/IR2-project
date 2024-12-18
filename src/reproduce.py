from typing import List
import subprocess
import argparse
import logging
import time
import json
import sys
import os
import requests

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
            'meta-llama/Llama-3.2-3B',
            'meta-llama/Llama-3.1-8B',
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'inpars-plus/Meta-Llama-3.1-8B-Instruct_merged-16bit_CPO_BEIR',
            'inpars-plus/Meta-Llama-3.1-8B-Instruct_merged-16bit_CPO_MSMARCO',
            'inpars-plus/Meta-Llama-3.1-8B_merged-16bit_CPO_MSMARCO',
            'neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic',
        ],
        default='EleutherAI/gpt-j-6B',
        help="Choose query generation model. "
    )
    parser.add_argument(
        '--max_generations', type=int, default=100_000, help="Number of queries to generate."
    )

    parser.add_argument(
        '--reranker',
        choices=[
            'castorini/monot5-3b-msmarco-10k',
            'castorini/rankllama-v1-7b-lora-passage',
            'cross-encoder/ms-marco-MiniLM-L-12-v2'
        ],
        default='castorini/monot5-3b-msmarco-10k',
        help="Choose reranker model."
    )
    parser.add_argument(
        '--rerank_topk', type=int, default=1000, help="Number of documents to rerank."
    )

    parser.add_argument(
        '--data_dir',
        default='./'
    )
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--end2end", action="store_true")

    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument(
        "--filter_options",
        nargs='+',
        default=[],
        choices=['scores', 'reranker', 'no_filter'],
        help="""Choose one of more filtering options: 'scores', 'reranker', 'no_filter'.
        If no option is provided, we default to ['scores', 'reranker'].
        If --use_inparsV2_pretrained is set, only 'reranker' is available.
        """
    )
    parser.add_argument(
        "--prompt_options",
        nargs='+',
        default=[],
        choices=['inpars', 'inpars-gbq', 'promptagator'],
        help="""Choose one of more prompt options: 'inpars', 'inpars-gbq', 'promptagator'.
        If no option is provided, we default to ['inpars', 'promptagator'], or, if
        --use_gbq is set, to ['inpars-gbq', 'promptagator'].
        Also, if --use_downloaded is set, only 'inpars' is available.
        """
    )
    parser.add_argument("--triples", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    parser.add_argument("--use_gbq", action="store_true")
    parser.add_argument("--use_downloaded", action="store_true")
    parser.add_argument("--use_inparsV2_pretrained", action="store_true")

    parser.add_argument("--use_vllm", action="store_true",
                        help="Use VLLM for query generation")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use half precision for training and inference (default).")
    parser.add_argument("--no_fp16", dest='fp16', action="store_false",
                        help="Use full precision for training and inference.")

    args = parser.parse_args()
    if args.filter_options == []:
        if not args.use_inparsV2_pretrained:
            args.filter_options = ['scores', 'reranker']
        else:
            args.filter_options = ['reranker']

    if args.prompt_options == []:
        # Possible prompt templates
        inpars_variant = 'inpars-gbq' if args.use_gbq else 'inpars'
        args.prompt_options = [inpars_variant, 'promptagator']
    # Only inpars generations are available
    if args.use_downloaded:
        if args.prompt_options != []:
            logging.warning('Only inpars prompt is available when using downloaded model.')
        args.prompt_options = ['inpars']


    return args


def generate_queries(prompt_type:str, dataset:str, model:str, output_path:str, fp16:bool, use_vllm:bool, seed:int, max_generations:int) -> None:
    logging.info(f'------Starting query generation : {prompt_type}------')
    start_generation = time.time()

    cache_path = f"{os.path.dirname(output_path)}/cache"
    args = [
        "python", "-m", "inpars.generate",
        f"--prompt={prompt_type}",
        f"--dataset={dataset}",
        "--dataset_source=ir_datasets",
        f"--max_generations={max_generations}",
        f"--batch_size=2048",
        f"--base_model={model}",
        f"--output={output_path}",
        f"--seed={seed}",
        f"--cache_dir={cache_path}",
    ]
    if use_vllm:
        args.append("--use_vllm")
    if fp16:
        args.append("--fp16")

    try:
        process = subprocess.run(args, stdout=subprocess.PIPE, text=True)
        logging.info(process.stdout)
        logging.error(process.stderr)

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Generating queries took : {generation_time} seconds.\n')


def filter_queries(query_path:str, dataset:str, output_path:str, filtering_strategy:str,
                   fp16: bool, seed: int, filtering_model:str = None) -> None:
    logging.info(f'------Starting filtering of : {query_path}------')
    start_generation = time.time()
    if filtering_strategy == 'no_filter':
        # copy the file
        try:
            process = subprocess.run([
                "cp", query_path, output_path
            ],  stdout=subprocess.PIPE,
                text=True
                )
            logging.info(process.stdout)
            logging.error(process.stderr)

        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess failed with exit code {e.returncode}")
            return
    
        generation_time = time.time() - start_generation
        logging.info(f'Queries copied without filtering in {generation_time} seconds.\n')
        return

    args = [
        "python", "-m", "inpars.filter",
        f"--input={query_path}",
        f"--dataset={dataset}",
        f"--filter_strategy={filtering_strategy}",
        f"--keep_top_k={10_000}",
        f"--output={output_path}",
        f"--seed={seed}",
    ]
    if fp16:
        args.append("--fp16")
    if filtering_strategy == 'reranker' and filtering_model is not None:
        args.append(f"--model_name_or_path={filtering_model}")

    try:
        process = subprocess.run(args,  stdout=subprocess.PIPE,
            text=True
            )
        logging.info(process.stdout)
        logging.error(process.stderr)

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Filtering queries took : {generation_time} seconds.\n')

def generate_triples(filtered_path:str, dataset:str, output_path:str, seed:int) -> None:
    logging.info(f'------Starting triple generation of : {filtered_path}------')
    start_generation = time.time()

    try:
        process = subprocess.run([
            "python", "-m", "inpars.generate_triples",
            f"--input={filtered_path}",
            f"--dataset={dataset}",
            f"--output={output_path}",
            f"--seed={seed}",
        ],  stdout=subprocess.PIPE,
            text=True
            )
        logging.info(process.stdout)
        logging.error(process.stderr)

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Triple generation took : {generation_time} seconds.\n')

def train_reranker(triples_path:str, ranker_model:str, output_path, fp16:bool, seed:int) -> None:
    logging.info(f'------Starting reranker training of : {triples_path}------')
    start_generation = time.time()
    
    args = [
        "python", "-m", "inpars.train",
        f"--triples={triples_path}",
        f"--base_model={ranker_model}",
        f"--output_dir={output_path}",
        "--max_steps=156",
        f"--seed={seed}",
    ]
    if fp16:
        args.append("--fp16")

    try:
        process = subprocess.run(args, stdout=subprocess.PIPE, text=True)
        logging.info(process.stdout)
        logging.error(process.stderr)

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Training took : {generation_time} seconds.\n')

def rerank(model_path:str, dataset:str, output_path:str, top_k:int, fp16:bool, seed:int) -> None:
    logging.info(f'------Starting reranking of {dataset} using {model_path}------')
    start_generation = time.time()

    args = [
        "python", "-m", "inpars.rerank",
        f"--model={model_path}",
        f"--dataset={dataset}",
        f"--output_run={output_path}",
        f"--top_k={top_k}",
        f"--seed={seed}",
    ]
    if fp16:
        args.append("--fp16")

    try:
        process = subprocess.run(args, stdout=subprocess.PIPE, text=True)
        logging.info(process.stdout)
        logging.error(process.stderr)

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    generation_time = time.time() - start_generation
    logging.info(f'Reranking took : {generation_time} seconds.\n')

def evaluate(dataset:str, trec_path:str, output_path:str):
    logging.info(f'------Starting evaluation of {trec_path} for {dataset}------')
    try:
        process = subprocess.run([
            "python", "-m", "inpars.evaluate",
            f"--dataset={dataset}",
            f"--run={trec_path}",
            f"--output_path={output_path}",
            "--json"
        ],  stdout=subprocess.PIPE,
            text=True
            )
        logging.info(process.stdout)
        logging.error(process.stderr)

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        return

    logging.info('Done!\n')


class InParsExperiment:
    def __init__(
            self, 
            data_path:str,
            dataset:str,
            generation_model:str,
            max_generations:int,
            reranker_model:str,
            rerank_topk:int,
            seed:int,
            filter_options:List[str],
            prompt_options:List[str],
            use_gbq:bool = False,
            use_inparsV2_pretrained:bool = False,
            use_downloaded:bool = False,
            use_vllm:bool = False,
            fp16:bool = True,
            ):
        self.dataset = dataset 
        self.generation_model = generation_model
        self.max_generations = max_generations
        self.reranker_model = reranker_model 
        self.rerank_topk = rerank_topk
        self.seed = seed
        self.use_inparsV2_pretrained = use_inparsV2_pretrained
        self.use_downloaded = use_downloaded
        self.pretrained_model = None
        self.use_vllm = use_vllm
        self.fp16 = fp16
        self.filter_options = filter_options
        self.prompt_options = prompt_options

        # Create path where to save data such as queries generated.
        self.root = args.data_dir
        # Start with data dir if it does not exist
        if not os.path.exists(os.path.join(self.root, 'data')):
            os.mkdir(os.path.join(self.root, 'data'))

        # Separate directory per dataset
        if use_downloaded and use_inparsV2_pretrained:
            result_directory = 'downloaded_pretrained'
        elif use_downloaded:
            result_directory = 'downloaded'
        elif use_inparsV2_pretrained:
            result_directory = 'pretrained'
        else:
            result_directory = 'data'
        
        data_path = os.path.join(self.root, result_directory, dataset)


        # Set up directory structure for saving data
        if not os.path.exists(data_path):
            # Create dataset specific dir
            os.mkdir(data_path)

        # Create generationLLM specific dir
        self.data_path = os.path.join(data_path, args.generationLLM.split('/')[-1])

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            # Create directory per prompt template (promptagate and inpars)
            for opt in self.prompt_options:
                os.mkdir(os.path.join(self.data_path, opt))
                # For each filter option, create directory.
                for filter_opt in self.filter_options:
                    os.mkdir(os.path.join(self.data_path, opt, filter_opt))
        
        if use_downloaded or use_inparsV2_pretrained:
            synthetic_datasets_v1 = {
                'trec-covid' : 'https://zav-public.s3.amazonaws.com/inpars/trec_covid_synthetic_queries_100k.jsonl',
                'nfcorpus' : 'https://zav-public.s3.amazonaws.com/inpars/nfcorpus_synthetic_queries_100k.jsonl',
                'hotpotqa' : 'https://zav-public.s3.amazonaws.com/inpars/hotpotqa_synthetic_queries_100k.jsonl',
                'fiqa' : 'https://zav-public.s3.amazonaws.com/inpars/fiqa_synthetic_queries_100k.jsonl',
                'arguana' : 'https://zav-public.s3.amazonaws.com/inpars/arguana_synthetic_queries_100k.jsonl',
                'webis-touche2020' : 'https://zav-public.s3.amazonaws.com/inpars/touche_synthetic_queries_100k.jsonl',
                'dbpedia-entity' : 'https://zav-public.s3.amazonaws.com/inpars/dbpedia_synthetic_queries_100k.jsonl',
                'scidocs' : 'https://zav-public.s3.amazonaws.com/inpars/scidocs_synthetic_queries_100k.jsonl',
                'fever' : 'https://zav-public.s3.amazonaws.com/inpars/fever_synthetic_queries_100k.jsonl',
                'climate-fever' : 'https://zav-public.s3.amazonaws.com/inpars/climate_fever_synthetic_queries_100k.jsonl',
                'scifact' : 'https://zav-public.s3.amazonaws.com/inpars/scifacts_synthetic_queries_100k.jsonl',
            }
            # synthetic_datasets_v2 = {
            #     'trec-covid' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/nfcorpus/queries.jsonl',
            #     'nfcorpus' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/nfcorpus/queries.jsonl',
            #     'hotpotqa' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/hotpotqa/queries.jsonl',
            #     'fiqa' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/fiqa/queries.jsonl',
            #     'arguana' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/arguana/queries.jsonl',
            #     'webis-touche2020' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/webis-touche2020/queries.jsonl',
            #     'dbpedia-entity' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/dbpedia-entity/queries.jsonl',
            #     'scidocs' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/scidocs/queries.jsonl',
            #     'fever' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/fever/queries.jsonl',
            #     'climate-fever' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/climate-fever/queries.jsonl',
            #     'scifact' : 'https://huggingface.co/datasets/inpars/generated-data/resolve/main/scifact/queries.jsonl',
            # }

            # /scratch-shared/InPars-data/data/trec-covid/gpt-j-6B/inpars-gbq
            target_path = os.path.join(self.data_path, 'inpars', 'inpars-queries.jsonl')
            if not os.path.exists(target_path):
                downloaded_data = requests.get(synthetic_datasets_v1[dataset], stream=True)
                if downloaded_data.status_code == 200:
                    with open(target_path, 'wb') as wf:
                        for chunk in downloaded_data.iter_content(chunk_size=8192):
                            wf.write(chunk)
                        
                    print(f'Successfully downloaded {synthetic_datasets_v1[dataset]}!')

                else:
                    print(f'Failed downloading {synthetic_datasets_v1[dataset]}...')

            
        if use_inparsV2_pretrained:
            model_name_inpars = {
                'trec-covid' : 'trec_covid' ,
                'nfcorpus' : 'nfcorpus',
                'hotpotqa' : 'hotpotqa',
                'fiqa' : 'fiqa',
                'arguana' : 'arguana',
                'webis-touche2020' : 'touche',
                'dbpedia-entity' : 'dbpedia',
                'scidocs' : 'scidocs',
                'fever' : 'fever',
                'climate-fever' : 'climate_fever',
                'scifact' : 'scifact',
            }[dataset]

            model_name_promptagator = {
                'trec-covid' : 'trec-covid' ,
                'nfcorpus' : 'nfcorpus',
                'hotpotqa' : 'hotpotqa',
                'fiqa' : 'fiqa',
                'arguana' : 'arguana',
                'webis-touche2020' : 'webis-touche2020',
                'dbpedia-entity' : 'dbpedia',
                'scidocs' : 'scidocs',
                'fever' : 'fever',
                'scifact' : 'scifact',
            }[dataset]

            self.pretrained_model_inpars = f'zeta-alpha-ai/monot5-3b-inpars-v2-{model_name_inpars}'
            self.pretrained_model_prompt = f'inpars/monot5-3b-inpars-v2-{model_name_promptagator}-promptagator'


    def run_generation(self):
        logging.info(f'Starting query generation stage...')

        # For each prompt type, save queries in directory.
        for prompt_type in self.prompt_options:
            query_output_path = os.path.join(self.data_path, prompt_type, f'{prompt_type}-queries.jsonl')

            if os.path.exists(query_output_path):
                logging.info(f'{prompt_type} has already been created. Continuing...')
                continue

            generate_queries(
                prompt_type=prompt_type,
                dataset=self.dataset,
                model=self.generation_model,
                output_path=query_output_path,
                fp16=self.fp16,
                use_vllm=self.use_vllm,
                seed=self.seed,
                max_generations=self.max_generations
            )

        logging.info(f'Done with the query generation stage!')


    def run_filtering(self):

        for prompt_type in self.prompt_options:
            input_path = os.path.join(self.data_path, prompt_type, f'{prompt_type}-queries.jsonl')
            for filter_type in self.filter_options:
                filter_output_path = os.path.join(self.data_path, prompt_type, filter_type, f'{prompt_type}-queries-filtered.jsonl')
                if os.path.exists(filter_output_path):
                    logging.info(f'({prompt_type},{filter_type}) has already been filtered. Continuing...')
                    continue

                filter_queries(
                    query_path=input_path,
                    dataset=self.dataset,
                    output_path=filter_output_path,
                    filtering_strategy=filter_type,
                    filtering_model=self.reranker_model,
                    fp16=self.fp16,
                    seed=self.seed
                )


        logging.info(f'Done with the filtering stage! \n')

    def run_triple_generation(self):
        for prompt_type in self.prompt_options:
            for filter_type in self.filter_options:
                input_path = os.path.join(self.data_path, prompt_type, filter_type, f'{prompt_type}-queries-filtered.jsonl')
                triple_output_path = os.path.join(self.data_path, prompt_type, filter_type, f'{prompt_type}-triples.tsv')

                if os.path.exists(triple_output_path):
                    logging.info(f'({prompt_type},{filter_type}) has already triples generated. Continuing...')
                    continue

                generate_triples(
                    filtered_path=input_path,
                    dataset=self.dataset,
                    output_path=triple_output_path,
                    seed=self.seed
                )

        logging.info(f'Done with triple generation stage! \n\n')

    def run_training(self):
        for prompt_type in self.prompt_options:
            for filter_type in self.filter_options:
                input_path = os.path.join(self.data_path, prompt_type, filter_type, f'{prompt_type}-triples.tsv')
                output_path = os.path.join(self.data_path, prompt_type, filter_type, 'reranker')
                if os.path.exists(output_path):
                    logging.info(f'({prompt_type},{filter_type}) has already a reranker trained. Continuing...')
                    continue
                
                train_reranker(
                    triples_path=input_path,
                    ranker_model=self.reranker_model,
                    output_path=output_path,
                    fp16=self.fp16,
                    seed=self.seed
                )

        logging.info(f'Done with training stage! \n\n')
    
    def run_reranking(self):
        for prompt_type in self.prompt_options:
            for filter_type in self.filter_options:
                reranker_path = os.path.join(self.data_path, prompt_type, filter_type, 'reranker')
                output_path = os.path.join(self.data_path, prompt_type, filter_type, 'trec-run.txt')
                if os.path.exists(output_path):
                    logging.info(f'({prompt_type},{filter_type}) has already been reranked. Continuing...')
                    continue
                
                if self.use_inparsV2_pretrained:
                    reranker_path = self.pretrained_model_inpars if 'inpars' in prompt_type else self.pretrained_model_prompt
                rerank(
                    model_path=reranker_path,
                    dataset=self.dataset,
                    output_path=output_path,
                    top_k=self.rerank_topk,
                    fp16=self.fp16,
                    seed=self.seed
                )


        logging.info(f'Done with reranking stage! \n\n')

    def run_evaluation(self):
        for prompt_type in self.prompt_options:
            for filter_type in self.filter_options:
                trec_path = os.path.join(self.data_path, prompt_type, filter_type, 'trec-run.txt')
                output_path = os.path.join(self.data_path, prompt_type, filter_type, 'results.json')

                evaluate(self.dataset, trec_path, output_path)

                with open(output_path, 'r') as rf:
                    data = json.load(rf)

                logging.info(f'--------Results of ({prompt_type, filter_type})--------')
                for key, value in data.items():
                    logging.info(f'|---{key}---|---{value}---|')

        print('Evaluating BM25')
        if not os.path.exists(os.path.join(self.data_path, 'BM25')):
            os.mkdir(os.path.join(self.data_path, 'BM25'))
        output_path = os.path.join(self.data_path, 'BM25', 'results.json')
        evaluate(self.dataset, 'BM25', output_path)

        with open(output_path, 'r') as rf:
            data = json.load(rf)

        logging.info(f'--------Results of (BM25)--------')
        for key, value in data.items():
            logging.info(f'|---{key}---|---{value}---|')


if __name__ == '__main__':
    args = parse_args()

    inpars = InParsExperiment(
        data_path = args.data_dir,
        dataset = args.dataset,
        generation_model = args.generationLLM,
        max_generations=args.max_generations,
        reranker_model = args.reranker,
        rerank_topk=args.rerank_topk,
        seed = args.seed,
        filter_options = args.filter_options,
        prompt_options=args.prompt_options,
        use_gbq = args.use_gbq,
        use_downloaded = args.use_downloaded,
        use_inparsV2_pretrained = args.use_inparsV2_pretrained,
        use_vllm = args.use_vllm,
        fp16=args.fp16,
    )

    if args.end2end:
        inpars.run_generation()
        inpars.run_filtering()
        inpars.run_triple_generation()
        inpars.run_training()
        inpars.run_reranking()
        inpars.run_evaluation()
    else:
        if args.generate:
            inpars.run_generation()
        if args.filter:
            inpars.run_filtering()
        if args.triples:
            inpars.run_triple_generation()
        if args.finetune:
            inpars.run_training()
        if args.rerank:
            inpars.run_reranking()
        if args.evaluate:
            inpars.run_evaluation()


    print('Finished process')
