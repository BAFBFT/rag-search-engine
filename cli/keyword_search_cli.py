#!/usr/bin/env python3

import argparse
import json
import string

def main() -> None:
    # translation table to remove punctuation
    punc_table = str.maketrans("", "", string.punctuation)

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
                if args.query.lower().translate(punc_table) in movie["title"].lower().translate(punc_table):
                    print(f"{str(movie_count)}. {movie["title"]}")
                    movie_count += 1
                if movie_count == 6:
                    break
        case _:
            parser.print_help()

if __name__ == "__main__":
    main() 