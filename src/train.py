"""Treino do modelo de classificacao do dsa_mlops_project (estagio ``train`` do DVC).

Consome os dados processados em ``data/processed/`` (gerados pelo estagio
``prepare``) e a configuracao de ``params.yaml``. Suporta dois algoritmos,
selecionaveis por ``model.type`` no ``params.yaml`` ou pela flag
``--model_type``:

* ``random_forest`` -> :class:`sklearn.ensemble.RandomForestClassifier`
* ``xgboost``       -> :class:`xgboost.XGBClassifier`

Registra parametros, metricas (treino e teste), matriz de confusao e o modelo
(com assinatura) no MLflow, publica no Model Registry como ``DSAClassifier`` e
salva a copia local em ``models/model.pkl`` para o DVC.

Exemplo:
    python src/train.py                          # usa params.yaml
    python src/train.py --model_type xgboost     # troca o algoritmo
    python src/train.py --n-estimators 500       # sobrescreve um hiperparametro
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

PARAMS_PATH = Path("params.yaml")
PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.pkl"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
TARGET_COLUMN = "target"
EXPERIMENT_NAME = "DSA MLOps"
REGISTERED_MODEL_NAME = "DSAClassifier"

RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"
SUPPORTED_MODELS = (RANDOM_FOREST, XGBOOST)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuracao do estagio de treino.

    Attributes:
        model_type: Algoritmo a treinar (``random_forest`` ou ``xgboost``).
        params: Hiperparametros especificos do algoritmo escolhido.
        random_state: Semente para reprodutibilidade do estimador.
    """

    model_type: str = RANDOM_FOREST
    params: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42

    def __post_init__(self) -> None:
        """Valida o ``model_type`` informado."""
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"model_type invalido: '{self.model_type}'. Use um de {SUPPORTED_MODELS}."
            )

    @property
    def model_params(self) -> dict[str, Any]:
        """Hiperparametros + ``random_state``, prontos para o construtor do estimador."""
        return {**self.params, "random_state": self.random_state}


def load_params(
    path: str | Path = PARAMS_PATH,
    model_type: str | None = None,
) -> TrainingConfig:
    """Le a secao ``model`` de ``params.yaml``.

    Args:
        path: Caminho do arquivo de parametros.
        model_type: Se informado, sobrescreve ``model.type`` do arquivo (e faz a
            leitura dos hiperparametros do bloco correspondente).

    Returns:
        Um :class:`TrainingConfig` com os valores encontrados (ou os padroes).

    Raises:
        FileNotFoundError: Se o arquivo de parametros nao existir.
        KeyError: Se nao houver bloco de hiperparametros para o ``model_type``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de parametros nao encontrado: {path}")

    params = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model_cfg = params.get("model", {})
    data_cfg = params.get("data", {})

    resolved_type = model_type or model_cfg.get("type", RANDOM_FOREST)
    if resolved_type not in model_cfg:
        raise KeyError(
            f"params.yaml nao tem o bloco 'model.{resolved_type}' com os hiperparametros."
        )

    return TrainingConfig(
        model_type=resolved_type,
        params=dict(model_cfg.get(resolved_type) or {}),
        random_state=int(data_cfg.get("random_state", 42)),
    )


def load_processed(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega um CSV processado e separa features do alvo.

    Args:
        path: Caminho do CSV (ex.: ``data/processed/train.csv``).

    Returns:
        Tupla ``(x, y)``.

    Raises:
        FileNotFoundError: Se o arquivo nao existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo processado '{path}' nao encontrado. Rode o preprocess antes."
        )
    frame = pd.read_csv(path)
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


def build_model(config: TrainingConfig) -> ClassifierMixin:
    """Instancia o classificador conforme ``config.model_type``.

    Args:
        config: Configuracao do treino.

    Returns:
        Estimador scikit-learn compativel, ainda nao treinado.
    """
    if config.model_type == RANDOM_FOREST:
        return RandomForestClassifier(**config.model_params)
    return XGBClassifier(**config.model_params)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
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
    y_true: pd.Series,
    y_pred: pd.Series,
    model_type: str,
    output_path: Path = CONFUSION_MATRIX_PATH,
) -> Path:
    """Gera e salva a matriz de confusao (conjunto de teste) como PNG.

    Args:
        y_true: Rotulos verdadeiros do teste.
        y_pred: Rotulos previstos para o teste.
        model_type: Algoritmo treinado, usado no titulo do grafico.
        output_path: Caminho do arquivo PNG de saida.

    Returns:
        O caminho da imagem gravada.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de Confusao - {model_type} (teste)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_model_local(model: ClassifierMixin, output_path: Path = MODEL_PATH) -> Path:
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


def run(config: TrainingConfig | None = None) -> dict[str, float]:
    """Executa o treino completo com rastreamento no MLflow.

    Args:
        config: Configuracao do treino. Se ``None``, le de ``params.yaml``.

    Returns:
        Dicionario com as metricas do conjunto de teste.
    """
    config = config or load_params()

    x_train, y_train = load_processed(TRAIN_PATH)
    x_test, y_test = load_processed(TEST_PATH)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        model = build_model(config)
        model.fit(x_train, y_train)

        train_metrics = compute_metrics(y_train, model.predict(x_train))
        test_metrics = compute_metrics(y_test, model.predict(x_test))

        mlflow.set_tag("model_type", config.model_type)
        mlflow.log_param("model_type", config.model_type)
        mlflow.log_params(config.model_params)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        cm_path = save_confusion_matrix(y_test, model.predict(x_test), config.model_type)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        signature = infer_signature(x_train, model.predict(x_train))
        _log_model(model, config.model_type, signature, x_train.head())

        local_path = save_model_local(model)
        mlflow.log_artifact(str(local_path), artifact_path="pickle")

    _print_summary(config, train_metrics, test_metrics)
    return test_metrics


def _log_model(
    model: ClassifierMixin,
    model_type: str,
    signature: Any,
    input_example: pd.DataFrame,
) -> None:
    """Loga o modelo no MLflow com o flavor adequado e o registra no Model Registry.

    Usa ``mlflow.xgboost`` para o XGBoost (o flavor sklearn nao serializa tipos
    do XGBoost via skops) e ``mlflow.sklearn`` para o Random Forest.

    Args:
        model: Estimador treinado.
        model_type: Algoritmo treinado (``random_forest`` ou ``xgboost``).
        signature: Assinatura inferida do modelo.
        input_example: Amostra de entrada para o artefato do modelo.
    """
    flavor = mlflow.xgboost if model_type == XGBOOST else mlflow.sklearn
    flavor.log_model(
        model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
    )


def _print_summary(
    config: TrainingConfig,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    """Imprime um resumo legivel do treino no terminal.

    Args:
        config: Configuracao usada no treino.
        train_metrics: Metricas no conjunto de treino.
        test_metrics: Metricas no conjunto de teste.
    """
    print(f"Modelo: {config.model_type}")
    print(f"Hiperparametros: {config.model_params}")
    print("\n{:<10} {:>8} {:>8}".format("metrica", "treino", "teste"))
    for name in test_metrics:
        print(f"{name:<10} {train_metrics[name]:>8.4f} {test_metrics[name]:>8.4f}")
    print(f"\nModelo salvo em: {MODEL_PATH}")
    print(f"Registrado no MLflow como: {REGISTERED_MODEL_NAME}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando (todos opcionais).

    Args:
        argv: Lista de argumentos (usa ``sys.argv`` quando ``None``).

    Returns:
        Namespace; valores nao informados ficam ``None`` e caem no ``params.yaml``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Treina o DSAClassifier a partir de data/processed. Sem flags, usa "
            "os valores de params.yaml."
        )
    )
    parser.add_argument(
        "--model_type",
        choices=SUPPORTED_MODELS,
        default=None,
        help="Algoritmo a treinar (padrao: model.type do params.yaml).",
    )
    parser.add_argument("--n-estimators", type=int, default=None, help="Numero de arvores.")
    parser.add_argument(
        "--max-depth", type=int, default=None, help="Profundidade maxima das arvores."
    )
    parser.add_argument("--random-state", type=int, default=None, help="Semente aleatoria.")
    parser.add_argument(
        "--params", type=Path, default=PARAMS_PATH, help="Caminho do params.yaml."
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Combina ``params.yaml`` com as sobrescritas de linha de comando.

    Args:
        args: Namespace retornado por :func:`parse_args`.

    Returns:
        Configuracao final do treino.
    """
    config = load_params(args.params, model_type=args.model_type)

    param_overrides = {
        key: value
        for key, value in (
            ("n_estimators", args.n_estimators),
            ("max_depth", args.max_depth),
        )
        if value is not None
    }
    if param_overrides:
        config = replace(config, params={**config.params, **param_overrides})
    if args.random_state is not None:
        config = replace(config, random_state=args.random_state)
    return config


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada de linha de comando.

    Args:
        argv: Lista de argumentos (usa ``sys.argv`` quando ``None``).
    """
    run(build_config(parse_args(argv)))


if __name__ == "__main__":
    main()
