"""Testes de fumaca: garantem que os modulos de src importam e expoem a API basica."""

import pandas as pd

import evaluate
import preprocess
import train


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
    frame = preprocess.load_dataset()
    config = preprocess.PreprocessConfig(test_size=0.2, random_state=42)

    train_df, test_df = preprocess.split_dataset(frame, config)

    assert len(train_df) == 120
    assert len(test_df) == 30
    assert set(test_df[preprocess.TARGET_COLUMN].unique()) == {0, 1, 2}


def test_compute_metrics_multiclasse():
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([0, 1, 2, 0, 2, 1])

    metrics = evaluate.compute_metrics(y_true, y_pred)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert {"accuracy", "precision", "recall", "f1"} == metrics.keys()
