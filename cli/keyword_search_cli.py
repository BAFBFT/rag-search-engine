#!/usr/bin/env python3

import argparse
import json
import string
import math

from lib.keyword_search import search_index, tokenize
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

    idf_parser = subparsers.add_parser("idf", help="Get the idf of a term")
    idf_parser.add_argument("term", type=str, help="Term for querying")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Get the idf of a term")
    tf_idf_parser.add_argument("doc_ID", type=int, help="document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term for querying")
    
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
            print(f"Searching for: {args.term} in document with ID {args.doc_ID}")
            tf = index.get_tf(args.doc_ID, args.term)
            print(f"The term, {args.term} appears {tf} in the document with ID {args.doc_ID}.")
        case "idf":
            index = InvertedIndex()
            index.load()
            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            idf = index.get_idf(args.term)
            tf = index.get_tf(args.doc_ID, args.term)
            tfidf = tf * idf
            print(f"The TF-IDF value for '{args.term}' in the document with ID {args.doc_ID} is {tfidf:.2f}.")
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main() 