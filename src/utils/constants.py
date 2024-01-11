import os
from pathlib import Path


ROOT_DIR = Path(os.getenv("ROOT", "."))
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
