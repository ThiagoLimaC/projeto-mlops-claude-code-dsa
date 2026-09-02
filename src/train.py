"""Treino do modelo de classificacao do dsa_mlops_project com integracao ao MLflow.

Treina um ``RandomForestClassifier`` no dataset Iris, registra parametros,
metricas, matriz de confusao e o modelo (com assinatura) no MLflow, publica o
modelo no Model Registry como ``DSAClassifier`` e salva uma copia local em
``models/model.pkl`` para versionamento com DVC.

Exemplo:
    python -m train --n-estimators 300 --max-depth 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.pkl"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
EXPERIMENT_NAME = "DSA MLOps"
REGISTERED_MODEL_NAME = "DSAClassifier"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuracao de um experimento de treino.

    Attributes:
        n_estimators: Numero de arvores da floresta aleatoria.
        max_depth: Profundidade maxima de cada arvore (``None`` = sem limite).
        test_size: Fracao dos dados reservada para teste.
        random_state: Semente para reprodutibilidade.
    """

    n_estimators: int = 100
    max_depth: int | None = None
    test_size: float = 0.2
    random_state: int = 42

    @property
    def model_params(self) -> dict[str, int | None]:
        """Retorna apenas os hiperparametros do estimador."""
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth}


@dataclass
class DataSplit:
    """Conjuntos de treino e teste mais os metadados do dataset.

    Attributes:
        x_train: Features de treino.
        x_test: Features de teste.
        y_train: Alvo de treino.
        y_test: Alvo de teste.
        feature_names: Nomes das colunas de features.
        target_names: Nomes das classes do alvo.
    """

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    target_names: list[str] = field(default_factory=list)


def load_data(test_size: float, random_state: int) -> DataSplit:
    """Carrega o dataset Iris e divide em treino/teste de forma estratificada.

    Args:
        test_size: Fracao dos dados reservada para teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        Um :class:`DataSplit` com os conjuntos e metadados.
    """
    dataset = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )
    return DataSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(dataset.feature_names),
        target_names=list(dataset.target_names),
    )


def build_model(config: TrainingConfig) -> RandomForestClassifier:
    """Instancia o classificador com os hiperparametros da configuracao.

    Args:
        config: Configuracao do treino.

    Returns:
        Estimador ``RandomForestClassifier`` ainda nao treinado.
    """
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )


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


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path = CONFUSION_MATRIX_PATH,
) -> Path:
    """Gera e salva a matriz de confusao como imagem PNG.

    Args:
        y_true: Rotulos verdadeiros.
        y_pred: Rotulos previstos.
        class_names: Nomes das classes, para os eixos.
        output_path: Caminho do arquivo PNG de saida.

    Returns:
        O caminho da imagem gravada.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusao - DSAClassifier")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_model_local(model: RandomForestClassifier, output_path: Path = MODEL_PATH) -> Path:
    """Persiste o modelo treinado localmente para versionamento com DVC.

    Args:
        model: Estimador treinado.
        output_path: Caminho do arquivo ``.pkl`` de saida.

    Returns:
        O caminho do arquivo gravado.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def run(config: TrainingConfig) -> dict[str, float]:
    """Executa o treino completo com rastreamento no MLflow.

    Passos: carrega Iris, treina o modelo, calcula metricas, loga parametros,
    metricas, matriz de confusao e o modelo (com assinatura) no MLflow,
    registra o modelo como ``DSAClassifier`` e salva a copia local.

    Args:
        config: Configuracao do treino.

    Returns:
        Dicionario com as metricas de teste.
    """
    data = load_data(config.test_size, config.random_state)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        model = build_model(config)
        model.fit(data.x_train, data.y_train)

        y_pred = model.predict(data.x_test)
        metrics = compute_metrics(data.y_test, y_pred)

        mlflow.log_params(config.model_params)
        mlflow.log_param("test_size", config.test_size)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_metrics(metrics)

        cm_path = save_confusion_matrix(data.y_test, y_pred, data.target_names)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        signature = infer_signature(data.x_train, model.predict(data.x_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=data.x_train[:5],
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        local_path = save_model_local(model)
        mlflow.log_artifact(str(local_path), artifact_path="pickle")

    _print_summary(metrics)
    return metrics


def _print_summary(metrics: dict[str, float]) -> None:
    """Imprime um resumo legivel das metricas no terminal.

    Args:
        metrics: Metricas de teste calculadas em :func:`run`.
    """
    print("\nMetricas de teste:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.4f}")
    print(f"\nModelo salvo em: {MODEL_PATH}")
    print(f"Registrado no MLflow como: {REGISTERED_MODEL_NAME}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando.

    Args:
        argv: Lista de argumentos (usa ``sys.argv`` quando ``None``).

    Returns:
        Namespace com os hiperparametros informados.
    """
    parser = argparse.ArgumentParser(
        description="Treina o DSAClassifier (RandomForest) no Iris com tracking MLflow."
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="Numero de arvores.")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profundidade maxima das arvores (padrao: sem limite).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fracao para teste.")
    parser.add_argument("--random-state", type=int, default=42, help="Semente aleatoria.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada de linha de comando.

    Args:
        argv: Lista de argumentos (usa ``sys.argv`` quando ``None``).
    """
    args = parse_args(argv)
    config = TrainingConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    run(config)


if __name__ == "__main__":
    main()
