import json 

from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.parent
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
DEFAULT_SEARCH_LIMIT = 5

def load_movies() -> List[dict]:
    with open("data/movies.json", 'r') as f:
        movies_data = json.load(f)
    return movies_data["movies"]

def load_stopwords() -> List[str]:
    with open(STOPWORDS_PATH, 'r') as f:
            stop_words = f.read()
            stop_words_list = stop_words.splitlines()
    return stop_words_list
