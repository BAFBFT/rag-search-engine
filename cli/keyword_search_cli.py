#!/usr/bin/env python3

import argparse
import json
import string
from typing import List


# translation table to remove punctuation
punc_table = str.maketrans("", "", string.punctuation)

def process_string(input_str: str) -> List[str]:
    
    make_case_insesitive = input_str.lower()
    remove_punctuation = make_case_insesitive.translate(punc_table)
    tokenized = remove_punctuation.split()
    for i, s in enumerate(tokenized):
        if s == "":
            del tokenized[i]
    return tokenized

def main() -> None:


    with open("data/movies.json", 'r') as f:
        movies_data = json.load(f)
    
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movie_count = 1

            for movie in movies_data["movies"]:
                processed_query = process_string(args.query)
                processed_movie_str = ("").join(process_string(movie["title"]))
                # check if any of the tokens match any in the movie db
                for token in processed_query:
                    if token in processed_movie_str:
                        print(f"{str(movie_count)}. {movie["title"]}")
                        movie_count += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main() 