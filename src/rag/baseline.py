from typing import Optional

import openai
from chromadb.utils import embedding_functions
from omegaconf import OmegaConf
from openai import OpenAI

from src.client import chroma
from src.utils import methods


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

    def feed_data_instances(self, feed: list) -> None:
        for feed_entity_path in feed:
            chroma.read_data_and_insert_into_collection(
                chroma_collection=self._collection, file_path=feed_entity_path
            )

    def forward_query(self, query, n_results: Optional[int] = 5):
        results = self._collection.query(query_texts=[query], n_results=n_results)
        assert results is not None

        retrieved_doc_chunks = results["documents"][0]
        information = "\n\n".join(retrieved_doc_chunks)

        messages = [
            {
                "role": "system",
                "content": "Jesteś pomocnym ekspertem i asystentem sędziego. Twoi użytkownicy będą zadawać pytania dotyczące informacji zawartych w części dokumentu sądowego."
                "Będziesz widział pytanie użytkownika oraz odpowiednie fragmenty dokumentu sądowego. Odpowiedz na pytanie użytkownika, korzystając wyłącznie z tych informacji.",
            },
            {
                "role": "user",
                "content": f"Question: {query}. \n Information: {information}",
            },
        ]

        llm_response = self._openai_client.chat.completions.create(
            model=self._llm_model,
            messages=messages,
        )
        content = llm_response.choices[0].message.content
        return content
