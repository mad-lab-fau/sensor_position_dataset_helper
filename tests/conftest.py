from pathlib import Path

import joblib
import pandas as pd
import pytest

# change this path, if your dataset is stored somewhere else!
DATASET_PATH = Path(__file__).parent.parent.parent / "dataset_2019"

# Remember to delete the cache if you made changes to the code you want to test.
CACHE = joblib.Memory("./.cache")


@pytest.fixture()
def dataset_path():
    return DATASET_PATH


def load_or_store_snapshot(name, data: pd.DataFrame) -> pd.DataFrame:
    file_name = Path(__file__).parent / "snapshots" / (name + ".json")
    if not file_name.is_file():
        data.to_json(file_name, orient="table")
    out = pd.read_json(file_name, orient="table")
    return out
