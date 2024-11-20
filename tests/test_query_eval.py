
import pytest
import pandas as pd
import torch
from pathlib import Path
from query_eval import QueryEval
from bm25s import BM25

@pytest.fixture
def documents():
    return pd.DataFrame({
        'doc_id': ['d1', 'd2', 'd3'],
        'text': [
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown fox leaps over a sleepy dog.",
            "The lazy dog lies in the sun."
        ]
    })

@pytest.fixture
def queries():
    return [
        "quick brown fox",
        "fast brown fox",
        "lazy dog"
    ]

@pytest.fixture
def query_eval():
    return QueryEval()

def test_load_dataset(documents, query_eval):
    query_eval.load_dataset(documents)
    assert isinstance (query_eval.bm25, BM25)
    assert query_eval.doc_embeddings is not None
    assert query_eval.doc_embeddings.shape[0] == 3

def test_save_and_load_cache(documents, query_eval, tmp_path):
    query_eval.load_dataset(documents)
    cache_path = tmp_path / "cache"
    query_eval.save_to_cache(cache_path)
    assert cache_path.exists()
    index_path = cache_path / "index.json"
    embedding_path = cache_path / "embeddings.pt"
    bm25_path = cache_path / "bm25.pkl"
    assert index_path.exists()
    assert embedding_path.exists()
    assert bm25_path.exists()

    new_query_eval = QueryEval()
    new_query_eval.load_from_cache(cache_path)
    assert torch.all(new_query_eval.doc_embeddings == query_eval.doc_embeddings)

def test_score_single_query_single_doc(documents, queries, query_eval):
    query_eval.load_dataset(documents)
    score = query_eval.score(queries[0], 'd1')
    assert isinstance(score, torch.Tensor)
    assert score.dim() == 0  # should be a scalar
    bad_score = query_eval.score(queries[0], 'd2')
    assert score > bad_score, "Score should be higher for relevant document"

def test_score_batch_queries(documents, queries, query_eval):
    query_eval.load_dataset(documents)
    scores = query_eval.score(queries, ['d1', 'd2', 'd3'])
    assert isinstance(scores, torch.Tensor)
    assert scores.shape[0] == 3  # should have three scalars
    bad_scores = query_eval.score(queries, ['d2', 'd3', 'd1'])
    assert all(scores > bad_scores), "Scores should be higher for relevant documents"


def test_query_doc_len_mismatch(queries, query_eval):
    with pytest.raises(ValueError):
        query_eval.score(queries[0], ['d1', 'd2', 'd3'])
