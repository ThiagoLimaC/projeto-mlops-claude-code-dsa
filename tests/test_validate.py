"""Testes do schema de validacao de dados (src/validate.py)."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

import validate


def _valid_frame() -> pd.DataFrame:
    """DataFrame minimo que satisfaz o schema (3 linhas, uma por classe)."""
    return pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 6.0, 6.3],
            "sepal width (cm)": [3.5, 2.7, 3.3],
            "petal length (cm)": [1.4, 4.5, 6.0],
            "petal width (cm)": [0.2, 1.5, 2.5],
            "target": [0, 1, 2],
        }
    )


def test_valida_dataframe_correto():
    result = validate.validate_dataframe(_valid_frame())
    assert len(result) == 3


def test_rejeita_valores_nulos():
    frame = _valid_frame()
    frame.loc[0, "sepal length (cm)"] = None

    with pytest.raises(SchemaErrors):
        validate.validate_dataframe(frame)


def test_rejeita_feature_fora_de_range():
    frame = _valid_frame()
    frame.loc[1, "petal width (cm)"] = 42.0

    with pytest.raises(SchemaErrors):
        validate.validate_dataframe(frame)


def test_rejeita_alvo_fora_do_dominio():
    frame = _valid_frame()
    frame.loc[2, "target"] = 7

    with pytest.raises(SchemaErrors):
        validate.validate_dataframe(frame)


def test_rejeita_coluna_inesperada():
    frame = _valid_frame()
    frame["extra"] = 1

    with pytest.raises(SchemaErrors):
        validate.validate_dataframe(frame)
