import numpy as np

from numpy import ndarray
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import DefaultDict, List, Set, Tuple
from .search_utils import load_movies, CACHE_PATH, DEFAULT_SEARCH_LIMIT

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None 
        self.document_map = {}
    
    def generate_embedding(self, text: str) -> ndarray:
        text_list = [text]
        embeddings = self.model.encode(text_list)
        return embeddings[0]
    
    def build_embeddings(self, documents:  List[str]):
        self.documents = documents
        doc_list = []
        for doc in documents:
            id = doc["id"]
            self.document_map[id] = doc
            doc_repr = f"{doc['title']}: {doc['description']}"
            doc_list.append(doc_repr)
        embeddings = self.model.encode(doc_list, show_progress_bar=True)
        np.save(CACHE_PATH / "movie_embeddings.npy", embeddings)
        return embeddings
    
    def load_or_create_embeddings(self, documents: List[str]):
        self.documents = documents
        doc_list = []
        for doc in documents:
            id = doc["id"]
            self.document_map[id] = doc
            doc_repr = f"{doc['title']}: {doc['description']}"
            doc_list.append(doc_repr)
        movie_embeddings = CACHE_PATH / "movie_embeddings.npy"
        if movie_embeddings.is_file():
            self.embeddings = np.load(movie_embeddings)
            if len(self.embeddings) == len(documents):
                return
            else:
                raise ValueError("Cached movie embeddings is not consistent with documents.")
        else:
            self.embeddings = self.build_embeddings(documents)
    
    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
        if self.embeddings.all():
            pass
        else:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        
        similarity_scores = []
        for i, e in enumerate(self.embeddings, start=1):
            similarity = cosine_similarity(e, query_embedding)
            similarity_scores.append((similarity, i))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        top_k_scores = similarity_scores[:DEFAULT_SEARCH_LIMIT]
        search_results = []
        for res in top_k_scores:
            res_info = {}
            res_info["score"] = res[0] # get the similarity score
            res_info["title"] = self.document_map[res[1]]["title"] # get the title by query documents
            res_info["description"] = self.documents[res[1]]["description"] # get the descripition by query documents
            search_results.append(res_info)
        return search_results
    
def verify_model() -> None:
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    search = SemanticSearch()
    movies = load_movies()
    movies_list = [m for m in movies]
    search.load_or_create_embeddings(movies_list)
    print(f"Number of docs:   {len(movies_list)}")
    print(f"Embeddings shape: {search.embeddings.shape[0]} vectors in {search.embeddings.shape[1]} dimensions")

def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedded_text = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedded_text[:5]}")
    print(f"Shape: {embedded_text.shape}")    

def cosine_similarity(vec1: ndarray, vec2: ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if (norm1 * norm2) == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_text(text: str, 
               chunk_size: int, 
               overlap: int | None = None) -> List[str]:
    words = text.split(' ')
    chunks = []
    i = 0
    count = 0
    chunk = ""
    while i < len(words):               
        if count < chunk_size:
            chunk += (words[i] + " ")
            count += 1
            i += 1
        else:
            chunk.rstrip()
            chunks.append(chunk)
            chunk = ""               
            count = 0
            if overlap:
                i -= overlap
    # final chunk
    chunks.append(chunk)
    return chunks