import os
from pathlib import Path

import hydra
from dotenv import find_dotenv
from dotenv import load_dotenv
from omegaconf import OmegaConf
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


@hydra.main(version_base=None, config_path="conf", config_name="trial")
def runner(cfg: OmegaConf) -> None:
    rag_configuration = cfg["rag"]
    api_key = extract_api_key()
    rag = RagBaseline(rag_configuration)
    rag.ensemble_rag(api_key=api_key)

    data_directory = constants.DATA_DIR / "examples"
    files = fetch_pdf_file_paths(data_directory)
    rag.feed_data_instances(files)

    user_query = "Jaka jest kwota alimentow"
    response = rag.forward_query(user_query)
    print(response)


if __name__ == "__main__":
    runner()
