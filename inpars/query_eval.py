from typing import List, Optional, Union, Dict, Tuple
import logging
from pathlib import Path
import json

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from collections import OrderedDict
from bm25s import BM25
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class QueryEval(torch.nn.Module):
    def __init__(
        self,
        dense: str = "sentence-transformers/all-mpnet-base-v2",
        bm25: bool = True,
        weights: List[float] = [0.5, 0.5],
        device: Optional[str] = None
    ):
        super(QueryEval, self).__init__()
        assert sum(weights) == 1, "Weights should sum to 1."
        self.weights = torch.tensor(weights).cpu()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dense_name = dense
        self.dense =  (
            AutoModel.from_pretrained(dense).to('cpu'),
            AutoTokenizer.from_pretrained(dense)
        )
        self.bm25 = bm25 # to be instantiated when the dataset is loaded
        self.doc_embeddings = None
        self.doc_id2idx = None

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings using attention mask"""
        token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
        return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
            Compute embeddings for a list of documents.
            NOTE: Loads the model and tokenizer to the device and does not release it.
            Inputs:
                texts: A batch of texts to embed for each model
            Returns:
                A Tensor of shape (len(texts), embedding_dim)
                containing the embeddings for the dense encoder model.
        """
        with torch.no_grad():
            model, tokenizer = self.dense
            model.to(self.device)

            inputs = tokenizer(texts, padding=True,
                               truncation=True, return_tensors='pt').to(self.device) # B x T
            outputs = model(**inputs) # B x D
            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask']).to(self.device) # B x D
        return embeddings # B x M x D_max

    def load_dataset(self,
                     documents: pd.DataFrame,
                     batch_size: int = 32
                     ) -> None:
        """
            Precompute and store embeddings for the document dataset.
            Inputs:
                documents: DataFrame with columns ['doc_id', 'text']
                batch_size: Batch size for processing
            Loads the embeddings into self.doc_embeddings as 
            a dictionary of model_name ->
                pd.DataFrame objects with columns ['doc_id', 'embedding']
            if bm25 is used, the plain text documents are stored in self.doc_embeddings["bm25"]
        """
        if documents is None:
            raise ValueError("No documents nor a valid path provided to load_dataset.")

        logger.info("Computing embeddings for %s%s", self.dense_name,
                    " and BM25" if self.bm25 else "")
        # save the mapping of doc_id to index
        self.doc_id2idx = {doc_id: idx for idx, doc_id in enumerate(documents['doc_id'])}
        # compute embeddings for each model and store them on cpu
        documents = documents['text'].tolist()
        self.doc_embeddings = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Computing embeddings"):
            batch = documents[i:i+batch_size]
            embeddings = self._get_embeddings(batch)
            self.doc_embeddings.append(embeddings)
        # concatenate
        self.doc_embeddings = torch.cat(self.doc_embeddings, dim=1)
        logger.debug("nr of documents: %d", len(documents))
        logger.debug("Document embeddings shape: %s", self.doc_embeddings.shape)

        # initialize the bm25 model
        if self.bm25:
            docs = [doc.split() for doc in documents]
            self.bm25 = BM25()
            self.bm25.index(docs)

        logger.info("Document corpus loaded successfully")

    def _prepare_cache(self, cache_path: Path) -> None:
        index_path = cache_path / "index.json"
        embedding_path = cache_path / "embeddings.pt"
        bm25_path = cache_path / "bm25.pkl"
        return index_path, embedding_path, bm25_path

    def save_to_cache(self, cache_path: Path) -> None:
        """Save precomputed embeddings to cache.
        Args:
            cache_path: Path to the cache directory
        Saves a pandas DataFrame with columns ['doc_id', 'embedding'] to the cache file.
        """
        if not cache_path.exists():
            cache_path.mkdir(parents=True)
        index_path, embedding_path, bm25_path = self._prepare_cache(cache_path)
        with open(index_path, "w") as f:
            json.dump(self.doc_id2idx, f)
        torch.save(self.doc_embeddings, embedding_path)
        if self.bm25:
            self.bm25.save(bm25_path)
        logger.info("Saved embeddings to %s", cache_path)

    def load_from_cache(self, cache_path: Path) -> None:
        """Load precomputed embeddings from cache.
        Args:
            cache_path: Path to the cache file
        Loads a pandas DataFrame with columns ['doc_id', 'embedding'] from the cache file.
        """
        if not cache_path.exists() or not cache_path.is_dir():
            raise ValueError("Cache directory does not exist.")
        index_path, embedding_path, bm25_path = self._prepare_cache(cache_path)
        if not index_path.exists():
            raise ValueError("Index file not found in cache.")
        if not embedding_path.exists():
            raise ValueError("Embedding file not found in cache.")
        if self.bm25 and not bm25_path.exists():
            raise ValueError("BM25 file not found in cache.")
        with open(index_path, "r") as f:
            self.doc_id2idx = json.load(f)
        self.doc_embeddings = torch.load(embedding_path)
        if self.bm25:
            self.bm25 = BM25.load(bm25_path)
        logger.info("Loaded embeddings from %s", cache_path)

    def score(self,
              queries: Union[str, List[str]],
              doc_indices: Union[str, List[str]]) -> torch.Tensor:
        """Compute similarity scores between query and documents.

        Args:
            query: a batch of queries
            doc_indices: a batch of document indices, each corresponding to an input query

        Returns:
            For each query-document pair in the batch, a similarity score, i.e.
            (q1 ~ d1, q2 ~ d2, ..., qn ~ dn)
        """
        if self.doc_embeddings is None:
            raise ValueError("No document embeddings found. Call load_dataset first.")

        logger.info("Computing similarity scores")
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(doc_indices, str):
            doc_indices = [doc_indices]
        if len(queries) != len(doc_indices):
            raise ValueError("Number of queries and document indices should be the same.")

        # find the embeddings corresponding to the doc_indices in the dataframes
        doc_indices = [self.doc_id2idx[doc_id] for doc_id in doc_indices]
        # load relevant embeddings to device
        doc_embeddings = self.doc_embeddings[doc_indices].to(self.device)

        # embed the queries and load to device.
        query_embeddings = self._get_embeddings(queries).to(self.device)

        # compute cosine similarity and scale to [0, 1]
        # shape is len(queries)
        similarities: List[torch.Tensor] = [(
            1 + torch.cosine_similarity(query_embeddings, doc_embeddings)
        ) / 2]

        if self.bm25:
            scores = []
            for q, d in zip(queries, doc_indices):
                bm25: BM25 = self.bm25
                # compute raw bm25 scores for all the documents in the corpus
                alldocs = torch.tensor(bm25.get_scores(q.split()))
                # send to device
                alldocs = alldocs.to(self.device).squeeze()
                # normalise to get "probabilities"
                # opposed to dense vectors, here we have
                # no choice but to consider the whole corpus.
                alldocs = alldocs.softmax(dim=0)
                # select the score of the requested document
                scores.append(alldocs[d])
            similarities.append(torch.tensor(scores).to(self.device))

        similarities = torch.stack(similarities, dim=1)

        weighted_sums = similarities @ self.weights.to(self.device)

        return weighted_sums.squeeze()

    def forward(self, query: Union[str, List[str]], doc_indices: Optional[List[int]] = None) -> torch.Tensor:
        return self.score(query, doc_indices)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    documents = pd.DataFrame({'doc_id': ['d1', 'd2', 'd3'],
                              'text': [
                                "The quick brown fox jumps over the lazy dog.",
                                "A fast brown fox leaps over a sleepy dog.",
                                "The lazy dog lies in the sun."
                             ]})
    query = "quick brown fox"

    query_eval = QueryEval()
    query_eval.load_dataset(documents)
    scores = [query_eval(query, [i]) for i in range(3)]

    assert scores[0] > scores[1] and scores[0] > scores[2], "First document should be more relevant than the other two."
    print("Test passed. Scores:", scores)

