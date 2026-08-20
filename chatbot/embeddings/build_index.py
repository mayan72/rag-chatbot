"""
Build ChromaDB index from CSV.

Reads:
    data/result.csv

Embeds:
    description column

Stores:
    Chroma Persistent DB
"""

from pathlib import Path

import pandas as pd

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CSV_FILE,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents():

    df = pd.read_csv(CSV_FILE)

    print(f"Loaded {len(df)} rows")

    docs = []

    for idx, row in df.iterrows():

        description = str(row.get("description", "")).strip()

        if not description:
            continue

        metadata = {}

        for col in df.columns:

            if col == "description":
                continue

            metadata[col] = str(row[col])

        metadata["row_number"] = idx + 1

        docs.append(
            Document(
                page_content=description,
                metadata=metadata,
            )
        )

    return docs


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def build_vector_db(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    print()

    print("=" * 60)

    print("Vector DB Created Successfully")

    print(f"Documents : {len(documents)}")

    print(f"Database  : {CHROMA_DB_PATH}")

    print("=" * 60)

    return vectordb


def main():

    print()

    print("=" * 60)

    print("Loading CSV...")

    documents = load_documents()

    print(f"Documents Loaded : {len(documents)}")

    print()

    print("Splitting Documents...")

    chunks = split_documents(documents)

    print(f"Chunks Created : {len(chunks)}")

    print()

    print("Creating Embeddings...")

    build_vector_db(chunks)


if __name__ == "__main__":

    main()