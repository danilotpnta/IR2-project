import requests 
import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        default='./'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    synthetic_datasets = {
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

    synthetic_data_dir = os.path.join(args.data_dir, 'synthetic_data_inpars')
    if not os.path.exists(synthetic_data_dir):
        os.mkdir(synthetic_data_dir)

    for key, value in synthetic_datasets.items():
        downloaded_data = requests.get(value, stream=True)

        if os.path.exists(os.path.join(synthetic_data_dir, f'{key}.jsonl')):
            continue

        if downloaded_data.status_code == 200:
            with open(os.path.join(synthetic_data_dir, f'{key}.jsonl'), 'wb') as wf:
                for chunk in downloaded_data.iter_content(chunk_size=8192):
                    wf.write(chunk)
                
            print(f'Successfully downloaded {key}!')

        else:
            print(f'Failed downloading {key}...')
                
