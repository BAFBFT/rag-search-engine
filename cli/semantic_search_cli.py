#!/usr/bin/env python3

import argparse

from lib.search_utils import load_movies
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query_text
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model has loaded correctly.")
    
    embed_parser = subparsers.add_parser("embed_text", help="Embed text using embedding model.")
    embed_parser.add_argument("text", type=str, help="Text for embedding.")

    verify_embedings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings are represented correctly.")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed the query text.")
    embed_query_parser.add_argument("text", type=str, help="Text for embedding.")

    search_parser = subparsers.add_parser("search", help="Search the vectorDB using cosine similarity")
    search_parser.add_argument("query", type=str, help="Query for searching.")
    search_parser.add_argument("--limit", type=int, help="Number of results to return")

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.text)
        case "search":
            search = SemanticSearch()
            movies = load_movies()
            search.load_or_create_embeddings(movies)
            res = search.search(args.query, args.limit)
            for i, r in enumerate(res, start=1):
                print(f"{i}. {r["title"]} (score: {r["score"]:.4f})\n")
                print(f"{r["description"]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()