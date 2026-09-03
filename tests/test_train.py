"""Testes de validacao do modelo treinado e do conjunto de teste processado.

Operam sobre os artefatos do pipeline (``models/model.pkl`` e
``data/processed/test.csv``). Se o pipeline ainda nao foi executado, os testes
sao pulados com uma mensagem orientando a rodar ``dvc repro``.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

import train

ACCURACY_THRESHOLD = 0.8
REQUIRED_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture(scope="module")
def test_dataframe() -> pd.DataFrame:
    """DataFrame completo de ``data/processed/test.csv`` (features + alvo)."""
    if not train.TEST_PATH.exists():
        pytest.skip(f"{train.TEST_PATH} ausente - rode `dvc repro` (estagio prepare).")
    return pd.read_csv(train.TEST_PATH)


@pytest.fixture(scope="module")
def test_data(test_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Par ``(X_test, y_test)`` derivado do CSV de teste."""
    x_test = test_dataframe.drop(columns=[train.TARGET_COLUMN])
    y_test = test_dataframe[train.TARGET_COLUMN]
    return x_test, y_test


@pytest.fixture(scope="module")
def model() -> object:
    """Modelo treinado carregado de ``models/model.pkl``."""
    if not train.MODEL_PATH.exists():
        pytest.skip(f"{train.MODEL_PATH} ausente - rode `dvc repro` (estagio train).")
    return joblib.load(train.MODEL_PATH)


def test_model_accuracy_threshold(model: object, test_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """A accuracy do modelo no conjunto de teste deve superar o limiar minimo."""
    x_test, y_test = test_data
    accuracy = accuracy_score(y_test, model.predict(x_test))
    assert accuracy > ACCURACY_THRESHOLD, (
        f"accuracy {accuracy:.4f} <= limiar {ACCURACY_THRESHOLD}"
    )


def test_model_output_shape(model: object, test_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """As predicoes devem ser um vetor 1-D com uma entrada por amostra."""
    x_test, y_test = test_data
    predictions = np.asarray(model.predict(x_test))

    assert predictions.ndim == 1
    assert predictions.shape == (len(x_test),)
    assert set(np.unique(predictions)).issubset(set(y_test.unique()))


def test_required_features(
    test_dataframe: pd.DataFrame,
    model: object,
    test_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """O CSV de teste deve conter as features exigidas e casar com o modelo."""
    columns = set(test_dataframe.columns)
    missing = set(REQUIRED_FEATURES) - columns
    assert not missing, f"colunas ausentes em {train.TEST_PATH}: {sorted(missing)}"
    assert train.TARGET_COLUMN in columns

    x_test, _ = test_data
    assert list(x_test.columns) == REQUIRED_FEATURES
    assert getattr(model, "n_features_in_", len(REQUIRED_FEATURES)) == len(REQUIRED_FEATURES)
