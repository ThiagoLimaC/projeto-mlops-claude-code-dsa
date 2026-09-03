"""Testes de fumaca: garantem que os modulos de src importam e expoem a API basica."""

import pandas as pd
import pytest

import evaluate
import preprocess
import train


def _iris_like_frame(rows_per_class: int = 20) -> pd.DataFrame:
    """DataFrame sintetico com 3 classes balanceadas e a coluna alvo 'target'."""
    n = rows_per_class * 3
    return pd.DataFrame(
        {
            "sepal length (cm)": range(n),
            "sepal width (cm)": range(n, 2 * n),
            "petal length (cm)": range(2 * n, 3 * n),
            "petal width (cm)": range(3 * n, 4 * n),
            "target": [0, 1, 2] * rows_per_class,
        }
    )


def test_modulos_importam():
    assert hasattr(preprocess, "run")
    assert hasattr(train, "run")
    assert hasattr(evaluate, "run")


def test_load_params_le_secao_data(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        "data:\n  test_size: 0.3\n  random_state: 7\n  normalize: true\n",
        encoding="utf-8",
    )

    config = preprocess.load_params(params_file)

    assert config.test_size == 0.3
    assert config.random_state == 7
    assert config.normalize is True


def test_split_dataset_estratificado():
    frame = _iris_like_frame(rows_per_class=20)
    config = preprocess.PreprocessConfig(test_size=0.2, random_state=42)

    train_df, test_df = preprocess.split_dataset(frame, config)

    assert len(train_df) == 48
    assert len(test_df) == 12
    assert set(test_df[preprocess.TARGET_COLUMN].unique()) == {0, 1, 2}


def test_load_raw_data_le_csv(tmp_path):
    csv_path = tmp_path / "iris.csv"
    _iris_like_frame(rows_per_class=5).to_csv(csv_path, index=False)

    frame = preprocess.load_raw_data(csv_path)

    assert len(frame) == 15
    assert preprocess.TARGET_COLUMN in frame.columns


def test_load_raw_data_sem_coluna_alvo(tmp_path):
    csv_path = tmp_path / "sem_alvo.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

    with pytest.raises(KeyError):
        preprocess.load_raw_data(csv_path)


def test_load_raw_data_arquivo_ausente(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess.load_raw_data(tmp_path / "nao_existe.csv")


_PARAMS_YAML = """\
model:
  type: random_forest
  random_forest:
    n_estimators: 250
    max_depth: 10
  xgboost:
    n_estimators: 400
    max_depth: 6
    learning_rate: 0.1
data:
  random_state: 7
"""


def test_train_load_params_random_forest(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(_PARAMS_YAML, encoding="utf-8")

    config = train.load_params(params_file)

    assert config.model_type == "random_forest"
    assert config.params == {"n_estimators": 250, "max_depth": 10}
    assert config.random_state == 7
    assert config.model_params["random_state"] == 7


def test_train_load_params_seleciona_xgboost(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(_PARAMS_YAML, encoding="utf-8")

    config = train.load_params(params_file, model_type="xgboost")

    assert config.model_type == "xgboost"
    assert config.params["n_estimators"] == 400
    assert config.params["learning_rate"] == 0.1


def test_train_build_config_cli_sobrescreve_params(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(_PARAMS_YAML, encoding="utf-8")
    args = train.parse_args(
        ["--model_type", "xgboost", "--n-estimators", "999", "--params", str(params_file)]
    )

    config = train.build_config(args)

    assert config.model_type == "xgboost"  # veio da CLI
    assert config.params["n_estimators"] == 999  # sobrescrito pela CLI
    assert config.params["max_depth"] == 6  # veio do bloco xgboost do params.yaml


def test_train_build_model_por_tipo(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    rf = train.build_model(train.TrainingConfig(model_type="random_forest", params={}))
    xgb = train.build_model(train.TrainingConfig(model_type="xgboost", params={}))

    assert isinstance(rf, RandomForestClassifier)
    assert isinstance(xgb, XGBClassifier)


def test_compute_metrics_multiclasse():
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([0, 1, 2, 0, 2, 1])

    metrics = evaluate.compute_metrics(y_true, y_pred)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert {"accuracy", "precision", "recall", "f1"} == metrics.keys()
