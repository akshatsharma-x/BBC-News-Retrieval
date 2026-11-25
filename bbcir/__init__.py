"""bbcir - a small information-retrieval toolkit for the BBC News corpus.

Modules
-------
data        load the raw BBC corpus and the processed article table
preprocess  cleaning, tokenisation, stop-word removal and Porter stemming
index       inverted index (postings lists) built from preprocessed tokens
models      TF-IDF, BM25, Sentence-Transformer and Reciprocal Rank Fusion rankers
metrics     P@k, Recall@k, AP/MAP, nDCG@k, RR and interpolated precision-recall
engine      a SearchEngine facade used by the evaluation scripts and the Streamlit UI
"""

__version__ = "1.0.0"
