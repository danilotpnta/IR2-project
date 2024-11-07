import subprocess
import argparse 
import time 
import json
import os 

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
        ],
        default='castorini/monot5-3b-msmarco-10k',
        help="Choose reranker model."
    )
    
    args = parser.parse_args()
    print(f'------Starting reproduction experiment {args.experiment}------')

    return args 

def generate_queries(prompt_type:str, dataset:str, model:str, output_path:str) -> None:
    print(f'------Starting query generation : {prompt_type}------')
    start_generation = time.time()

    subprocess.run([
        "python", "-m", "inpars.generate",
        f"--prompt='{prompt_type}'",
        f"--dataset='{dataset}'",
        "--dataset_source='ir_datasets'",
        f"--base_model='{model}'",
        f"--output='{output_path}'"
    ])

    generation_time = time.time() - start_generation
    print(f'Generating queries took : {generation_time} seconds.\n')


def filter_queries(query_path:str, dataset:str, output_path:str, filtering_strategy:str) -> None:
    print(f'------Starting filtering of : {query_path}------')
    start_generation = time.time()

    subprocess.run([
        "python", "-m", "inpars.filter",
        f"--input='{query_path}'",
        f"--dataset='{dataset}'",
        f"--filter_strategy='{filtering_strategy}'",
        f"--keep_top_k='{10_000}'",
        f"--output='{output_path}'"
    ])

    generation_time = time.time() - start_generation
    print(f'Filtering queries took : {generation_time} seconds.\n')

def generate_triples(filtered_path:str, dataset:str, output_path:str):
    print(f'------Starting triple generation of : {filtered_path}------')
    start_generation = time.time()

    subprocess.run([
        "python", "-m", "inpars.generate_triples",
        f"--input='{filtered_path}'",
        f"--dataset='{dataset}'",
        f"--output='{output_path}'"
    ])

    generation_time = time.time() - start_generation
    print(f'Triple generation took : {generation_time} seconds.\n')

def train_reranker(triples_path:str, ranker_model:str, output_path):
    print(f'------Starting reranker training of : {triples_path}------')
    start_generation = time.time()

    subprocess.run([
        "python", "-m", "inpars.generate_triples",
        f"--triples='{triples_path}'",
        f"--base_model='{ranker_model}'",
        f"--output_dir='{output_path}'",
        "--max_steps='156'"
    ])

    generation_time = time.time() - start_generation
    print(f'Training took : {generation_time} seconds.\n')

def rerank(model_path:str, dataset:str, output_path):
    print(f'------Starting reranking of {dataset} using {model_path}------')
    start_generation = time.time()

    subprocess.run([
        "python", "-m", "inpars.rerank",
        f"--model='{model_path}'",
        f"--dataset='{dataset}'",
        f"--output_run='{output_path}'",
    ])

    generation_time = time.time() - start_generation
    print(f'Reranking took : {generation_time} seconds.\n')

def evaluate(dataset:str, trec_path:str, output_path:str):
    print(f'------Starting evaluation of {trec_path} for {dataset}------')
    subprocess.run([
        "python", "-m", "inpars.evaluate",
        f"--dataset='{dataset}'",
        f"--run='{trec_path}'",
        f"--output_path='{output_path}'",
        "--json"
    ])
    print('Done!\n')


def main(args):
    # Create path where to save data such as queries generated. 
    data_path = os.path.join(os.curdir, 'data', args.dataset)
    
    prompt_options = ['inpars', 'promptagator'] #'inpars-gbq']
    filter_options = ['scores', 'reranker']

    if not os.path.exists(data_path):
        os.mkdir(data_path)
        for opt in filter_options:
            os.mkdir(os.path.join(data_path, opt))

    for prompt_type in prompt_options:
        prompt_path = os.path.join(data_path, prompt_type)
        query_output_path = os.path.join(prompt_path, f'{prompt_type}-queries.jsonl')

        if not os.path.exists(prompt_path):
            os.mkdir(prompt_path)
        elif os.path.exists(query_output_path):
            print(f'{prompt_type} has already been created. Continuing...')
            continue
        
        generate_queries(prompt_type, args.dataset, args.model, query_output_path)
    
    print(f'Done with the query generation stage! \n\n Continuing with the filtering stage...\n')

    for prompt_type in prompt_options:
        input_path = os.path.join(data_path, prompt_type, f'{prompt_type}-queries.jsonl')
        for filter_type in filter_options:
            filter_output_path = os.path.join(data_path, prompt_type, filter_type, f'{prompt_type}-queries-filtered.jsonl')
            if os.path.exists(filter_output_path):
                print(f'({prompt_type},{filter_type}) has already been filtered. Continuing...')
                continue
            
            filter_queries(input_path, args.dataset, filter_output_path, filter_type)


    print(f'Done with the filtering stage! \n\n Continuing with the negative mining/triple generation stage...\n')

    for prompt_type in prompt_options:
        for filter_type in filter_options:
            input_path = os.path.join(data_path, prompt_type, filter_type, f'{prompt_type}-queries.jsonl')
            triple_output_path = os.path.join(data_path, prompt_type, filter_type, f'{prompt_type}-triples.tsv')

            if os.path.exists(triple_output_path):
                print(f'({prompt_type},{filter_type}) has already triples generated. Continuing...')
                continue 
            
            generate_triples(input_path, args.dataset, triple_output_path)
        
    print(f'Done with triple generation stage! \n\n Continuing with the training stage...\n')

    for prompt_type in prompt_options:
        for filter_type in filter_options:
            input_path = os.path.join(data_path, prompt_type, filter_type, f'{prompt_type}-triples.tsv')
            output_path = os.path.join(data_path, prompt_type, filter_type, 'reranker')
            if os.path.exists(output_path):
                print(f'({prompt_type},{filter_type}) has already a reranker trained. Continuing...')
                continue 

            train_reranker(input_path, args.reranker, output_path)

    print(f'Done with training stage! \n\n Continuing with the reranking stage...\n')

    for prompt_type in prompt_options:
        for filter_type in filter_options:  
            reranker_path = os.path.join(data_path, prompt_type, filter_type, 'reranker')
            output_path = os.path.join(data_path, prompt_type, filter_type, 'trec-run.txt')

            rerank(reranker_path, args.dataset, output_path)


    print(f'Done with reranking stage! \n\n Continuing with the evaluation stage...\n')

    for prompt_type in prompt_options:
        for filter_type in filter_options:  
            trec_path = os.path.join(data_path, prompt_type, filter_type, 'trec-run.txt')
            output_path = os.path.join(data_path, prompt_type, filter_type, 'results.json')

            evaluate(args.dataset, trec_path, output_path)

            with open(output_path, 'r') as rf:
                data = json.loads(rf)
            
            print(f'--------Results of ({prompt_type, filter_type})--------')
            for key, value in data.items():
                print(f'|{key}\t\t|\t{value}\t|')
    


if __name__ == '__main__':
    args = parse_args()

    main(args)

        













    
