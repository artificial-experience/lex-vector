from pathlib import Path
from typing import Optional

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from pypdf import PdfReader
from tqdm import tqdm


def read_pdf(filename: Path) -> list:
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts


def chunk_texts(
    texts, chunk_overlap: Optional[int] = 0, tokens_per_chunk: Optional[int] = 256
):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text("\n\n".join(texts))
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap, tokens_per_chunk=tokens_per_chunk
    )

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


def word_wrap(string, n_chars: Optional[int] = 72):
    if len(string) < n_chars:
        return string
    else:
        return (
            string[:n_chars].rsplit(" ", 1)[0]
            + "\n"
            + word_wrap(string[len(string[:n_chars].rsplit(" ", 1)[0]) + 1 :], n_chars)
        )


def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings
