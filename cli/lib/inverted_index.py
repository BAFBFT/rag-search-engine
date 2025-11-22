import pickle
from collections import defaultdict
from typing import DefaultDict, List, Set
from pathlib import Path

from .keyword_search import tokenize
from .search_utils import load_movies

class InvertedIndex():
    def __init__(self):
        self.index: DefaultDict[str, Set[int]] = defaultdict(set)
        self.docmap: DefaultDict[int, str] = defaultdict(str)
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)
    
    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        IDs = []
        for id in self.index[term]:
            IDs.append(id)
        return IDs.sort()
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            id, title, description = movie["id"], movie['title'], movie['description']
            input_text = f"{title} {description}"
            self.__add_document(id, input_text)
            self.docmap[id] = (title + description)

    def save(self):
        cache_path = Path.cwd().parent / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.index, cache_path / "index.pkl")
        pickle.dump(self.docmap, cache_path / "docmap.pkl")
