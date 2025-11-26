import pickle
from collections import defaultdict, Counter
from typing import DefaultDict, List, Set
from pathlib import Path

from .keyword_search import tokenize
from .search_utils import load_movies, CACHE_PATH

class InvertedIndex():
    def __init__(self):
        self.index: DefaultDict[str, Set[int]] = defaultdict(set)
        self.docmap: DefaultDict[int, dict] = defaultdict(str)
        self.term_frequencies: DefaultDict[int, Counter] = defaultdict(Counter)
     
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id].update([token])

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

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            id, title, description = movie["id"], movie['title'], movie['description']
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

    def load(self) -> None:
        with open(CACHE_PATH / "index.pkl", 'rb') as f:
            self.index = pickle.load(f)
        with open(CACHE_PATH / "docmap.pkl", 'rb') as f:
            self.docmap = pickle.load(f)
        with open(CACHE_PATH / "term_frequencies.pkl", 'rb') as f:
            self.term_frequencies = pickle.load(f)