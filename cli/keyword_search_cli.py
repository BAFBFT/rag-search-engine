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

    subparsers.add_parser("build", help="Build an inverted index from movies.json")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a term")
    tf_parser.add_argument("doc_ID", type=int, help="document ID")
    tf_parser.add_argument("term", type=str, help="Term for querying")

    args = parser.parse_args()
    
    match args.command:
        case "search":
            index = InvertedIndex()
            index.load()
            print(f"Searching for: {args.query}")
            results = search_index(args.query, index.index, index.docmap)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res["title"]}, ID: {res["id"]}")
        case "tf":
            index = InvertedIndex()
            index.load()
            print(index.term_frequencies[424])
            print(f"Searching for: {args.term} in document with ID {args.doc_ID}")
            tf = index.get_tf(args.doc_ID, args.term)
            if tf > 0:
                print(f"The term, {args.term} appears {tf} in the document with ID {args.doc_ID}.")
            else:
                print(tf)
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main() 