import os
import ftfy
import json
import pandas as pd
from tqdm.auto import tqdm
import ir_datasets
import pyarrow.parquet as pq

def load_corpus(dataset_name, source='ir_datasets', chunk_size=10000, summary_dataset=False):

    cache_dir = ir_datasets.util.home_path()
    parquet_path = os.path.join(cache_dir, f"beir/{dataset_name}/docs.parquet")
    
    if os.path.exists(parquet_path):
        print(f"Found cached Parquet dataset at: {parquet_path}.")

        parquet_file = pq.ParquetFile(parquet_path)
        num_rows = parquet_file.metadata.num_rows
        rows = []
        for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), total=num_rows // chunk_size, desc=f"Loading {num_rows} documents"):
            rows.append(batch.to_pandas())
        corpus_df = pd.concat(rows, ignore_index=True)

        if summary_dataset:
            text_lengths = corpus_df['text'].str.len()
            memory_usage = corpus_df.memory_usage(deep=True).sum() / (1024 ** 2)  
            print(f"\n-- Dataset Summary: {dataset_name.capitalize()} --")
            print(f"Total documents: {len(corpus_df)}")
            print(f"Average text length: {text_lengths.mean():.2f} characters")
            print(f"Text length [min - max]: [{text_lengths.min()} - {text_lengths.max()}] characters")
            print(f"Memory usage: {memory_usage:.2f} MB\n")
                
        return corpus_df

    texts = []
    docs_ids = []

    if source == 'ir_datasets':
        print(f"Loading corpus from ir_datasets for dataset: {dataset_name}")

        identifier = f'beir/{dataset_name}'
        if identifier in ir_datasets.registry._registered:
            dataset = ir_datasets.load(identifier)
        else:
            dataset = ir_datasets.load(dataset_name)

        for doc in tqdm(
            dataset.docs_iter(), total=dataset.docs_count(), desc="Loading documents from ir_datasets"
        ):
            texts.append(
                ftfy.fix_text(
                    f"{doc.title} {doc.text}"
                    if "title" in dataset.docs_cls()._fields
                    else doc.text
                )
            )
            docs_ids.append(doc.doc_id)
    else:
        from pyserini.search.lucene import LuceneSearcher
        from pyserini.prebuilt_index_info import TF_INDEX_INFO

        identifier = f'beir-v1.0.0-{dataset_name}.flat'
        if identifier in TF_INDEX_INFO:
            dataset = LuceneSearcher.from_prebuilt_index(identifier)
        else:
            dataset = LuceneSearcher.from_prebuilt_index(dataset_name)

        for idx in tqdm(range(dataset.num_docs), desc="Loading documents from Pyserini"):
            doc = json.loads(dataset.doc(idx).raw())
            texts.append(
                ftfy.fix_text(
                    f"{doc['title']} {doc['text']}"
                    if doc['title']
                    else doc['text']
                )
            )
            docs_ids.append(doc['_id'])
    

    corpus_df = pd.DataFrame({'doc_id': docs_ids, 'text': texts})
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True) 
    corpus_df.to_parquet(parquet_path, engine="pyarrow")
    print(f"Corpus saved to {parquet_path}")

    return corpus_df



    
def load_queries(dataset_name, source='ir_datasets'):
    queries = {}

    if source == 'ir_datasets':
        import ir_datasets
        dataset = ir_datasets.load(f'beir/{dataset_name}')

        for query in dataset.queries_iter():
            queries[query.query_id] = ftfy.fix_text(query.text)
    else:
        from pyserini.search import get_topics

        for (qid, data) in get_topics(f'beir-v1.0.0-{dataset_name}-test').items():
            queries[str(qid)] = ftfy.fix_text(data["title"])  # assume 'title' is the query

    return queries
