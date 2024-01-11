import os
from pathlib import Path

import hydra
import streamlit as st
from dotenv import find_dotenv
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from rag import RagBaseline

from src.utils import constants


def fetch_pdf_file_paths(data_directory: Path) -> list:
    if not data_directory.exists():
        raise ValueError("The provided directory does not exist.")

    if not data_directory.is_dir():
        raise ValueError("The provided path is not a directory.")

    pdf_file_paths = []
    for file_path in data_directory.rglob("*.pdf"):
        pdf_file_paths.append(file_path)

    return pdf_file_paths


def extract_api_key() -> str:
    _ = load_dotenv(find_dotenv())
    api_key = os.environ["OPENAI_API_KEY"]
    return api_key


@st.cache_resource
def get_rag_instance(_cfg):
    rag_configuration = _cfg["rag"]
    api_key = extract_api_key()
    rag = RagBaseline(rag_configuration)
    rag.ensemble_rag(api_key=api_key)
    return rag


@st.cache_resource
def load_and_feed_data(_cfg):
    rag = get_rag_instance(_cfg)
    data_directory = constants.DATA_DIR / "examples"
    files = fetch_pdf_file_paths(data_directory)
    rag.feed_data_instances(files)
    return rag


def process_query(user_query: str, rag: RagBaseline) -> str:
    response = rag.forward_query(user_query)
    return response


def streamlit_ui(cfg):
    st.title("PDF Data Query Interface")
    user_query = st.text_input("Enter your query:")

    if st.button("Process Query") and user_query:
        rag = load_and_feed_data(cfg)
        response = process_query(user_query, rag)
        st.write(response)


def run_with_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    @hydra.main(version_base=None, config_path="conf", config_name="trial")
    def runner(cfg: DictConfig) -> None:
        streamlit_ui(cfg)

    runner()


if __name__ == "__main__":
    run_with_hydra()
