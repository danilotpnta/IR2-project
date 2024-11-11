from typing import List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel
from collections import OrderedDict
from multiprocessing import Pool
from rank_bm25 import BM25Okapi as BM25
from tqdm.auto import tqdm
from pathlib import Path
import logging

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
        self.weights = torch.tensor(weights, device=self.device)
        self.doc_embeddings = None

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings using attention mask"""
        token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
        return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_embeddings(self, model_name, documents: List[str], batch_size: int) -> torch.Tensor:
        """
            Compute embeddings for a list of documents
            Inputs:
                documents: List of document texts
                batch_size: Batch size for processing
            Returns:
                List of embeddings for each (dense) model and document in the input list
                so a list of shape (num_models, num_documents, model_embedding_dim)
        """
        embeddings = []
        if model_name not in self.tokenizers:
            raise ValueError(f"No tokenizer for {model_name}.")

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        embeddings = torch.zeros((len(documents), model.config.hidden_size), device=self.device)
        with torch.no_grad():
            for i in tqdm(range(0, len(documents), batch_size), desc=f"Compute embeddings using {model_name}"):
                batch = documents[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
                outputs = model(**inputs)
                outputs = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask']).to(self.device)
                embeddings[i:i+batch_size] = outputs

        # sanity check
        assert embeddings.shape[0] == len(documents), f"sanity check failed: {embeddings.shape[0]} != {len(documents)}"
        assert embeddings.shape[1] == outputs.size(-1), f"sanity check failed: {embeddings.shape[1]} != {outputs.size(-1)}"
        return embeddings

    def _load_cached_embeddings(self, cache_path: Path) -> None:
        """Load precomputed embeddings from cache"""
        self.doc_embeddings = torch.load(cache_path)

    def load_dataset(self,
                     documents: Optional[List[str]] = None,
                     cache_path: Optional[Path] = None,
                     batch_size: int = 32,
                     num_workers: int = 4) -> None:
        """Precompute and store embeddings for the document dataset"""
        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading embeddings from cache: {cache_path}")
            self._load_cached_embeddings(cache_path)
            return
        if documents is None:
            raise ValueError("No documents nor a valid path provided to load_dataset.")

        def load_for_model(model_name) -> Union[BM25, torch.Tensor]:
            if model_name == "bm25":
                return BM25([doc.split() for doc in documents])
            return self._get_embeddings(model_name, documents, batch_size)

        logger.info("Computing embeddings using a pool of %d workers", num_workers)
        with Pool(num_workers) as p:
            embeddings = p.map(load_for_model, self.models.keys())

        self.doc_embeddings = dict(zip(self.models.keys(), embeddings))
        if "bm25" in self.doc_embeddings:
            self.models["bm25"] = self.doc_embeddings["bm25"]
            del self.doc_embeddings["bm25"]
        # sanity
        assert all([embeddings.shape[1] == self.models[model_name].config.hidden_size
                    for model_name, embeddings in self.doc_embeddings.items()]), f"sanity check: model embedding doesn't match with the model's hidden size"
        logger.info("Document embeddings computed successfully")
        # cache embeddings
        if cache_path is not None:
            torch.save(self.doc_embeddings, cache_path)
            logger.info(f"Embeddings saved to {cache_path}")

    def score(self,
                query: Union[str, List[str]],
                doc_indices: Optional[List[int]] = None,
                batch_size: int = 32) -> torch.Tensor:
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

        logger.info("Computing similarity scores")
        if isinstance(query, str):
            query = [query]

        query_embeddings = [self._get_embeddings(model_name, query, batch_size)
                            for model_name in self.doc_embeddings]

        if doc_indices is not None:
            doc_embeddings = [self.doc_embeddings[model_name] for model_name in self.doc_embeddings]
        else:
            doc_embeddings = [emb for emb in self.doc_embeddings.values()]

        similarities = []
        for q_emb, d_emb in zip(query_embeddings, doc_embeddings):
            q_emb = q_emb.to(self.device)
            d_emb = d_emb.to(self.device)
            similarities.append(torch.einsum('ij,kj->ik', q_emb, d_emb).cpu())
        if "bm25" in self.models:
            for q in query:
                # BM25Okapi uses np arrays
                similarities.append(
                    torch.tensor(self.models["bm25"].get_batch_scores(q.split(), doc_indices), device='cpu'))

        similarities = torch.stack(similarities)
        # compute weighted average of similarities
        weighted_mean = torch.mean(similarities, dim=0, weights=self.weights)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logger.debug("Overall similarity score: %s", weighted_mean)
            for model_name, sim in zip(
                    filter(lambda m: m != "bm25", self.models.keys()),
                    similarities):
                logger.debug("Similarity score using %s: %s", model_name, sim)
            if "bm25" in self.models:
                logger.debug("Similarity score using BM25: %s", similarities[-1])

        return weighted_mean.squeeze()

    def forward(self, query: Union[str, List[str]], doc_indices: Optional[List[int]] = None) -> torch.Tensor:
        return self.score(query, doc_indices)

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy dog.",
        "The lazy dog lies in the sun."
    ]
    query = "quick brown fox"

    query_eval = QueryEval()
    query_eval.load_dataset(documents)
    scores = query_eval(query)

    assert scores[0] > scores[1] and scores[0] > scores[2], "First document should be more relevant than the other two."
    print("Test passed. Scores:", scores)

