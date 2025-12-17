import math
import pickle
from collections import defaultdict, Counter
from typing import DefaultDict, List, Optional, Set
from pathlib import Path

from .keyword_search import tokenize
from .search_utils import load_movies, CACHE_PATH, BM25_K1, BM25_B

class InvertedIndex():
    def __init__(self):
        self.index: DefaultDict[str, Set[int]] = defaultdict(set)
        self.docmap: DefaultDict[int, dict] = defaultdict(str)
        self.term_frequencies: DefaultDict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict = {}
    
    def __get_avg_doc_length(self) -> float:
        total_length = 0
        for k, v in self.doc_lengths.items():
            total_length += v
        doc_length = len(self.doc_lengths)
        if doc_length == 0:
            return 0.0
        else:
            return total_length / len(self.doc_lengths)


    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)
            token_length = len(tokens)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = token_length

    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        IDs = []
        for id in self.index[term]:
            IDs.append(id)
        return sorted(IDs)
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise Exception("Only one token expected")
        term_frequency = self.term_frequencies[doc_id][tokens[0]]
        return term_frequency

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise Exception("Only one token expected")        
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(tokens[0], set()))
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise Exception("Only one token expected")          
        df = len(self.index.get(tokens[0], set()))
        N = len(self.docmap)
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return bm25_tf

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            id, title, description = movie["id"], movie["title"], movie["description"]
            input_text = f"{title} {description}"
            self.__add_document(id, input_text)
            self.docmap[id] = movie

    def save(self) -> None:
        CACHE_PATH.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH / "index.pkl", 'wb') as f:
            pickle.dump(self.index, f)
        with open(CACHE_PATH / "docmap.pkl", 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(CACHE_PATH / "term_frequencies.pkl", 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(CACHE_PATH / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(CACHE_PATH / "index.pkl", 'rb') as f:
            self.index = pickle.load(f)
        with open(CACHE_PATH / "docmap.pkl", 'rb') as f:
            self.docmap = pickle.load(f)
        with open(CACHE_PATH / "term_frequencies.pkl", 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(CACHE_PATH / "doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float | None = None, b: float | None = None) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1)