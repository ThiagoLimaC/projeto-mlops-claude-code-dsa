"""Avaliacao do modelo do dsa_mlops_project (estagio ``evaluate`` do DVC).

Carrega o modelo treinado de ``models/model.pkl`` e o conjunto de teste de
``data/processed/test.csv``, calcula as metricas de classificacao, salva o
relatorio em ``reports/metrics.json``, gera a matriz de confusao em
``reports/confusion_matrix.png`` e registra as metricas no MLflow.

Exemplo:
    python src/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MODEL_PATH = Path("models/model.pkl")
TEST_PATH = Path("data/processed/test.csv")
REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
TARGET_COLUMN = "target"
EXPERIMENT_NAME = "DSA MLOps"


def load_model(path: str | Path = MODEL_PATH) -> object:
    """Carrega o modelo treinado.

    Args:
        path: Caminho do arquivo ``.pkl``.

    Returns:
        O estimador treinado.

    Raises:
        FileNotFoundError: Se o arquivo do modelo nao existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo '{path}' nao encontrado. Rode o train antes.")
    return joblib.load(path)


def load_test_data(path: str | Path = TEST_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega o conjunto de teste processado.

    Args:
        path: Caminho do CSV de teste.

    Returns:
        Tupla ``(x_test, y_test)``.

    Raises:
        FileNotFoundError: Se o arquivo de teste nao existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Conjunto de teste '{path}' nao encontrado. Rode o preprocess antes."
        )
    frame = pd.read_csv(path)
    x_test = frame.drop(columns=[TARGET_COLUMN])
    y_test = frame[TARGET_COLUMN]
    return x_test, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula accuracy, precision, recall e f1 (media macro).

    Args:
        y_true: Rotulos verdadeiros.
        y_pred: Rotulos previstos.

    Returns:
        Dicionario ``{nome_metrica: valor}``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_metrics(metrics: dict[str, float], path: Path = METRICS_PATH) -> Path:
    """Grava as metricas em JSON.

    Args:
        metrics: Metricas calculadas.
        path: Caminho do arquivo de saida.

    Returns:
        O caminho gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path = CONFUSION_MATRIX_PATH,
) -> Path:
    """Gera e salva a matriz de confusao como PNG.

    Args:
        y_true: Rotulos verdadeiros.
        y_pred: Rotulos previstos.
        path: Caminho do arquivo PNG de saida.

    Returns:
        O caminho gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusao - avaliacao")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run() -> dict[str, float]:
    """Executa a avaliacao completa e registra os resultados no MLflow.

    Returns:
        Dicionario com as metricas calculadas.
    """
    model = load_model()
    x_test, y_test = load_test_data()
    y_pred = model.predict(x_test)

    metrics = compute_metrics(y_test, y_pred)
    save_metrics(metrics)
    save_confusion_matrix(y_test, y_pred)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        mlflow.log_artifact(str(CONFUSION_MATRIX_PATH), artifact_path="reports")

    print("Metricas de avaliacao:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.4f}")
    print(f"\nRelatorio: {METRICS_PATH}")
    print(f"Matriz de confusao: {CONFUSION_MATRIX_PATH}")
    return metrics


if __name__ == "__main__":
    run()
