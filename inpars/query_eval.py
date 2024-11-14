from typing import List, Optional, Union, Dict, Tuple
import logging
from pathlib import Path

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
        models: List[str] = ["bm25", "sentence-transformers/all-mpnet-base-v2"],
        weights: List[float] = [0.5, 0.5],
        device: Optional[str] = None
    ):
        super(QueryEval, self).__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = OrderedDict()
        self.tokenizers = OrderedDict()
        for model in models:
            if model == "bm25":
                self.models["bm25"] = None # initialised with the corpus
            else:
                self.tokenizers[model] = AutoTokenizer.from_pretrained(model)
                self.models[model] = AutoModel.from_pretrained(model).to(self.device)
        self.weights = OrderedDict({model: weight for model, weight in zip(models, weights)})
        self.doc_embeddings = OrderedDict()

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings using attention mask"""
        token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
        return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_embeddings(self, model_name, documents: List[str], batch_size: int) -> torch.Tensor:
        """
            Compute embeddings for a list of documents
            Inputs:
                model_name: Name of the model to use
                documents: A list of D documents
            Returns:
                List of embeddings for the given documents computed by the given model.
        """
        embeddings = []
        if model_name not in self.tokenizers:
            raise ValueError(f"No tokenizer for {model_name}.")

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        with torch.no_grad():
            inputs = tokenizer(documents, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = model(**inputs)
            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask']).to(self.device)

        # sanity check
        assert embeddings.shape[0] == len(documents), f"sanity check failed: {embeddings.shape[0]} != {len(documents)}"
        assert embeddings.shape[1] == model.config.hidden_size, f"sanity check failed: {embeddings.shape[1]} != {outputs.size(-1)}"
        return embeddings

    def load_from_cache(self, cache_path: Path) -> None:
        """Load precomputed embeddings from cache.
        Args:
            cache_path: Path to the cache file
        Loads a pandas DataFrame with columns ['doc_id', 'embedding'] from the cache file.
        """
        self.doc_embeddings = torch.load(cache_path)
        if "bm25" in self.models:
            self.models["bm25"] = BM25.load(
                cache_path.parent.absolute() / f"{cache_path.stem}_bm25{cache_path.suffix}")

    def save_to_cache(self, cache_path: Path) -> None:
        """Save precomputed embeddings to cache.
        Args:
            cache_path: Path to the cache file
        Saves a pandas DataFrame with columns ['doc_id', 'embedding'] to the cache file.
        """
        torch.save(self.doc_embeddings, cache_path)
        if "bm25" in self.models:
            self.models["bm25"].save(cache_path.parent.absolute(
            ) / f"{cache_path.stem}_bm25{cache_path.suffix}")

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

        def embed_for_model(model_name) -> Union[Dict[str, Tuple[int, List[str]]], pd.DataFrame]:
            """
            Compute embeddings for the given corpus and model.
            - If the model is a BM25 model, return the tokenized text, 
                including the integer index of the documents in the corpus.
                We need these indices to identify the documents in the output
                of the BM25.get_scores method.
            - Otherwise, embed the documents in batches and
                store the embeddings in a pd.DataFrame.
            """
            if model_name == "bm25":
                texts = OrderedDict({
                    doc_id: (idx, text.split()) for idx, (doc_id, text) in enumerate(documents[['doc_id', 'text']].values)
                })
                # split the text into tokens
                return texts
            # otherwise, iterate over the list of documents,
            # embed them in batches and
            # store the embeddings in a pd.DataFrame
            embeddings = pd.DataFrame(columns=['doc_id', 'embedding'])
            for i in tqdm(range(0, len(documents), batch_size), desc=f"Computing embeddings for {model_name}"):
                batch = documents[i:i + batch_size]
                embeddings_batch = self._get_embeddings(model_name, batch['text'].tolist(), batch_size)
                embeddings = pd.concat([embeddings, pd.DataFrame({
                    'doc_id': batch['doc_id'].tolist(),
                    'embedding': embeddings_batch.cpu().tolist()
                })], ignore_index=True)
            return embeddings

        logger.info("Computing embeddings for %d models", len(self.models))
        self.doc_embeddings = OrderedDict({model_name: embed_for_model(model_name)
                               for model_name in self.models.keys()})

        if "bm25" in self.doc_embeddings:
            self.models["bm25"] = BM25()
            self.models["bm25"].index([text for _, text in self.doc_embeddings["bm25"].values()])

        # sanity
        assert all([len(embeddings["embedding"].to_numpy()[0]) == self.models[model_name].config.hidden_size
                    for model_name, embeddings in self.doc_embeddings.items()
                    if model_name != "bm25"]), f"sanity check: model embedding doesn't match with the model's hidden size"

        logger.info("Document embeddings computed successfully")

    def score(self,
                queries: Union[str, List[str]],
                doc_indices: Union[str, List[int]] = None,
                batch_size: int = 32) -> torch.Tensor:
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
        if isinstance(doc_indices, int):
            doc_indices = [doc_indices]
        if len(queries) != len(doc_indices):
            raise ValueError("Number of queries and document indices should be the same.")

        # find the embeddings corresponding to the doc_indices in the dataframes
        doc_embeddings = OrderedDict({
            model_name: torch.tensor(
                self.doc_embeddings[model_name].loc[
                    self.doc_embeddings[model_name]['doc_id'].isin(doc_indices), 'embedding'
                ].values.tolist(),
                device=self.device
            )
            for model_name in self.doc_embeddings.keys()
            if model_name != "bm25"
        })

        # embed the queries in the same order
        query_embeddings = OrderedDict({model_name: self._get_embeddings(model_name, queries, batch_size)
                                        for model_name in self.doc_embeddings.keys()
                                        if model_name != "bm25"})

        # embedding dimensions differ from model to model,
        # so we have to do one by one
        similarities = OrderedDict({
            # compute cosine similarity between query and document embeddings
            # and rescale it to [0, 1] so we can regard it as a probability
            # NOTE: this is a coarse approximation because we don't want to compute softmax over the whole corpus.
            #       Instead, we can adjust the weights to make the scores comparable.
            model_name: 
                (1 + torch.cosine_similarity(
                        query_embeddings[model_name], doc_embeddings[model_name]).to(self.device)
                 ) / 2
            for model_name in self.doc_embeddings.keys()
            if model_name != "bm25"
        })
        # do the same for BM25
        if "bm25" in self.models:
            # select the numeric indices of the requested documents
            doc_idxs = [self.doc_embeddings["bm25"][d][0] for d in doc_indices]
            scores = []
            for q, d in zip(queries, doc_idxs):
                bm25: BM25 = self.models["bm25"]
                # compute raw bm25 scores for all the documents in the corput
                alldocs = torch.tensor(bm25.get_scores(q.split()))
                # send to device
                alldocs = alldocs.to(self.device).squeeze()
                # normalise to get "probabilities" 
                # opposed to dense vectors, here we have no choice but to consider the whole corpus.
                alldocs = alldocs.softmax(dim=0)
                # select the score of the requested document
                scores.append(alldocs[d])
            similarities["bm25"] = torch.tensor(scores).to(self.device)

        # apply weights
        similarities = torch.stack([
                similarities[model_name] * self.weights[model_name]
                for model_name in similarities.keys()
            ]).to(self.device)
        # compute weighted average of similarities
        weighted_mean = torch.mean(similarities, dim=0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Overall similarity score: %s", weighted_mean)
            for model_name, sim in similarities.items():
                logger.debug("Similarity score using %s: %s", model_name, sim)
            if "bm25" in self.models:
                logger.debug("Similarity score using BM25: %s", similarities["bm25"])

        return weighted_mean.squeeze()

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

