from pathlib import Path

import chromadb
from chromadb import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from src.utils import methods


def connect_to_chroma_client() -> ClientAPI:
    chroma_client = chromadb.Client()
    return chroma_client


def create_chroma_collection(
    chroma_client: ClientAPI,
    collection_name: str,
    embedding_function: embedding_functions,
) -> Collection:
    chroma_collection = chroma_client.create_collection(
        name=collection_name, embedding_function=embedding_function
    )
    return chroma_collection


def read_data_and_insert_into_collection(
    chroma_collection: Collection, file_path: Path
) -> None:
    texts = methods.read_pdf(file_path)
    chunks = methods.chunk_texts(texts)
    ids = [str(i) for i in range(len(chunks))]
    chroma_collection.add(ids=ids, documents=chunks)
