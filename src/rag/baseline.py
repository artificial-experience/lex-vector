from typing import Optional

import numpy as np
import openai
from chromadb.utils import embedding_functions
from omegaconf import OmegaConf
from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.client import chroma


class RagBaseline:
    def __init__(self, conf: OmegaConf) -> None:
        self._conf = conf

        # llm state
        self._api_key = None
        self._llm_model = None

        # clients and collections
        self._chroma_client = None
        self._openai_client = None
        self._collection = None

        # re-ranking
        self._cross_encoder = None

    def ensemble_rag(self, api_key: str) -> None:
        self._api_key = api_key
        assert self._api_key is not None
        openai.api_key = self._api_key

        self._openai_client = OpenAI()

        self._chroma_client = chroma.connect_to_chroma_client()
        assert self._chroma_client is not None

        collection_name = self._conf["collection_name"]
        embed_func_name = self._conf["embed_func_name"]
        embed_func = getattr(embedding_functions, embed_func_name)()
        self._collection = chroma.create_chroma_collection(
            chroma_client=self._chroma_client,
            collection_name=collection_name,
            embedding_function=embed_func,
        )
        assert self._collection is not None

        self._llm_model = self._conf["llm_model"]
        assert self._llm_model is not None

        self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def feed_data_instances(self, feed: list) -> None:
        for feed_entity_path in feed:
            chroma.read_data_and_insert_into_collection(
                chroma_collection=self._collection, file_path=feed_entity_path
            )

    def forward_query(
        self,
        query,
        n_results: Optional[int] = 5,
        n_document_to_accept: Optional[int] = 4,
    ):
        augmented_query = self._augment_multiple_query(
            query=query, model=self._llm_model
        )
        queries = [query] + augmented_query

        results = self._collection.query(query_texts=queries, n_results=n_results)
        assert results is not None

        retrieved_doc_chunks = results["documents"][0]

        unique_documents = set()
        for documents in retrieved_doc_chunks:
            unique_documents.add(documents)

        unique_documents = list(unique_documents)
        ranked_documents = self._cross_encoder_ranking(query, unique_documents)
        chosen_ranked_document_ids = ranked_documents[:n_document_to_accept]
        chosen_documents = [unique_documents[idx] for idx in chosen_ranked_document_ids]

        final_query = [query] + chosen_documents
        final_query_squashed = "\n".join(final_query)

        messages = [
            {
                "role": "system",
                "content": "Jesteś systemem ekspertowym, którego zadaniem jest analizowanie i odpowiadanie na pytania związane ze sprawami alimentacyjnymi, opierając się wyłącznie na informacjach zawartych w treści zapytania"
                "Twoim zadaniem jest dokładne przetworzenie treści zapytania i wydobycie z niego kluczowych informacji, które umożliwią udzielenie precyzyjnej i adekwatnej odpowiedzi"
                "Pytanie użytkownika jest prezentowane jako pierwsze, po czym następuje analiza treści dokumentów, które są podstawą do udzielenia odpowiedzi",
            },
            {"role": "user", "content": final_query_squashed},
        ]

        response = self._openai_client.chat.completions.create(
            model=self._llm_model,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content

    def _augment_multiple_query(self, query: list, model: str):
        messages = [
            {
                "role": "system",
                "content": "Jesteś pomocnym ekspertem i asystentem sędziego. Twoi użytkownicy zadają pytania dotyczące sprawy sądowej"
                "Zaproponuj do pięciu dodatkowych pokrewnych pytań, aby pomóc im znaleźć potrzebne informacje, w oparciu o zadane pytanie"
                "Proponuj tylko krótkie pytania bez zdań złożonych. Zaproponuj różnorodne pytania, które obejmują różne aspekty tematu"
                "Upewnij się, że są to pełne pytania i że są związane z oryginalnym pytaniem"
                "Wypisz jedno pytanie w każdej linii. Nie numeruj pytań.",
            },
            {"role": "user", "content": query},
        ]

        response = self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content
        content = content.split("\n")
        return content

    def _cross_encoder_ranking(
        self, original_query: str, unique_docs: list
    ) -> np.ndarray:
        pairs = []
        for doc in unique_docs:
            pairs.append([original_query, doc])

        scores = self._cross_encoder.predict(pairs)
        article_ids = np.argsort(scores)[::-1]
        return article_ids
