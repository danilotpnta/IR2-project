from typing import List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel
from collections import OrderedDict
import numpy as np
from multiprocessing import Pool
from rank_bm25 import BM25Okapi as BM25
from tqdm.auto import tqdm
from pathlib import Path

class QueryEval:
    def __init__(
        self,
        models: str = ["bm25", "sentence-transformers/all-mpnet-base-v2"],
        weights: List[float] = [0.5, 0.5],
        device: Optional[str] = None
    ):
        """Initialize the dense retriever with a pre-trained model.

        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = OrderedDict()
        self.tokenizers = OrderedDict()
        for model in models:
            if model == "bm25":
                self.models["bm25"] = None # initialised with the corpus
            else:
                self.tokenizers[model] = AutoTokenizer.from_pretrained(model)
                self.models[model] = AutoModel.from_pretrained(model).to(self.device)
        self.weights = weights
        self.doc_embeddings = None

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings using attention mask"""
        token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
        return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_embeddings(self, model_name, documents: List[str], batch_size: int) -> np.ndarray:
        """
            Compute embeddings for a list of documents
            Inputs:
                documents: List of document texts
                batch_size: Batch size for processing
            Returns:
                List of embeddings for each (dense) model and document in the input list
                so a list of shape (num_models, num_documents, model_embedding_dim)
        """
        # different models may have different embedding dimensions
        # so we handle them separately (i.e. we don't stack them on the first axis)
        embeddings = []
        if model_name not in self.tokenizers:
            raise ValueError(f"No tokenizer for {model_name}.")

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(documents), batch_size),
                          desc=f"Compute embeddings using {model_name}"):
                batch = documents[i:i+batch_size] # B x str
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device) # B x T
                outputs = model(**inputs) # B x T x D
                outputs = self._mean_pooling(
                    outputs.last_hidden_state, inputs['attention_mask']).cpu().numpy() # B x D
                embeddings.append(outputs)
        embeddings = np.concatenate(embeddings, dim=1) # N x [D_model]
        # sanity check
        assert embeddings.shape[0] == len(documents), f"sanity check failed: {embeddings.shape[0]} != {len(documents)}"
        assert embeddings.shape[1] == outputs.size(-1), f"sanity check failed: {embeddings.shape[1]} != {outputs.size(-1)}"
        return embeddings

    def _load_cached_embeddings(self, cache_path: Path) -> None:
        """Load precomputed embeddings from cache"""
        self.doc_embeddings = torch.load(cache_path)

    def load_dataset(self,
                     documents: Optional[List[List[str]]] = None,
                     cache_path: Optional[Path] = None,
                     batch_size: int = 32,
                     num_workers: int = 4) -> None:
        """Precompute and store embeddings for the document dataset"""
        # if cache exists, load it first
        if cache_path is not None and cache_path.exists():
            self._load_cached_embeddings(cache_path)
            return
        if documents is None:
            raise ValueError("No documents nor a valid path provided to load_dataset.")

        def load_for_model(model_name) -> Union[BM25, np.ndarray]:
            if model_name == "bm25":
                # type mismatch but whatever. Handling it below.
                return BM25([doc.split() for doc in documents])
            return self._get_embeddings(model_name, documents, batch_size)

        # compute embeddings for each model
        with Pool(num_workers) as p:
            embeddings = p.map(load_for_model, self.models.keys())

        # store embeddings
        self.doc_embeddings = dict(zip(self.models.keys(), embeddings))
        # store bm25 separately
        if "bm25" in self.doc_embeddings:
            self.models["bm25"] = self.doc_embeddings["bm25"]
            del self.doc_embeddings["bm25"]
        # sanity
        assert all([embeddings.shape[1] == self.models[model_name].config.hidden_size
                    for model_name, embeddings in self.doc_embeddings.items()]), f"sanity check: model embedding doesn't match with the model's hidden size"

    def score(self,
              query: Union[str, List[str]],
              doc_indices: Optional[List[int]] = None,
              batch_size: int = 32) -> np.ndarray:
        """Compute similarity scores between query and documents.
        
        Args:
            query: Query text or list of query texts
            doc_indices: Optional list of document indices to score against.
                        If None, scores against all documents.
        
        Returns:
            Array of similarity scores
        """
        if self.doc_embeddings is None:
            raise ValueError("No document embeddings found. Call load_dataset first.")

        if isinstance(query, str):
            query = [query]

        # Get query embeddings
        query_embeddings = [self._get_embeddings(model_name, query, batch_size)
                            for model_name in self.doc_embeddings]

        # Get relevant document embeddings
        if doc_indices is not None:
            # list of model embeddings. Embedding dims may mismatch, so not stacking them
            doc_embeddings = [self.doc_embeddings[model_name] for model_name in self.doc_embeddings]
        else:
            doc_embeddings = [emb for emb in self.doc_embeddings.values()]

        # Compute dot product similarities
        similarities = [np.dot(q_emb, d_emb.T)
                        for q_emb, d_emb in zip(query_embeddings, doc_embeddings)]
        if "bm25" in self.models:
            for q in query:
                similarities.append(
                    self.models["bm25"].get_batch_scores(q.split(), doc_indices))
        # compute weighted sum of similarities
        similarities = np.average(similarities, axis=0, weights=self.weights)
        # Return scores (remove singleton dimension if single query)
        return similarities.squeeze()
