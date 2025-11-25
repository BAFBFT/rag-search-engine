#!/usr/bin/env python3

import argparse
import json
import string
from typing import List
from nltk.stem import PorterStemmer

from lib.keyword_search import search_command, search_index
from lib.inverted_index import InvertedIndex

def main() -> None:


    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build an inverted index from movies.json")

    args = parser.parse_args()
    
    match args.command:
        case "search":
            index = InvertedIndex()
            index.load()
            print(f"Searching for: {args.query}")
            results = search_index(args.query, index.index, index.docmap)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res["title"]}, ID: {res["id"]}")
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main() 