import string 

from typing import List
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from nltk.stem import PorterStemmer

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[dict]:
    movie_count = 0
    movies = load_movies()

    tokenized_query = tokenize(query)
    results = []
    for movie in movies:
        movie_title = tokenize(movie["title"])
        # check if any of the tokens match any in the movie db
        for token in tokenized_query:
            if token in movie_title:
                results.append(movie["title"])
                movie_count += 1
        if movie_count >= limit:
            break
    return results


def preprocess_text(text: str) -> str:
    # translation table to remove punctuation
    punc_table = str.maketrans("", "", string.punctuation)
    make_case_insesitive = text.lower()
    remove_punctuation = make_case_insesitive.translate(punc_table)
    return remove_punctuation

def stem_word(text: str) -> str:
    stemmer = PorterStemmer()
    return stemmer.stem(text)

def tokenize(text: str) -> List[str]:
    stop_words = load_stopwords()
    processed_text = preprocess_text(text)
    tokenized = processed_text.split()
    for i, s in enumerate(tokenized):
        tokenized[i] = stem_word(s)
        if s == "":
            del tokenized[i]
        if s in stop_words:
            del tokenized[i]
    return tokenized