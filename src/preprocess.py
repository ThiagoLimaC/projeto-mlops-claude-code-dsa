"""Pre-processamento de dados do dsa_mlops_project (estagio ``prepare`` do DVC).

Carrega o dataset Iris, le os parametros de ``params.yaml``, aplica um
feature engineering basico (normalizacao opcional) e grava os conjuntos de
treino e teste em ``data/processed/train.csv`` e ``data/processed/test.csv``.

Exemplo:
    python src/preprocess.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PARAMS_PATH = Path("params.yaml")
PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
SCALER_PATH = PROCESSED_DATA_DIR / "scaler.pkl"
TARGET_COLUMN = "target"


@dataclass(frozen=True)
class PreprocessConfig:
    """Parametros do estagio de pre-processamento.

    Attributes:
        test_size: Fracao dos dados reservada para teste.
        random_state: Semente para reprodutibilidade.
        normalize: Se ``True``, padroniza as features com ``StandardScaler``
            (ajustado apenas no treino) e salva o scaler em disco.
    """

    test_size: float = 0.2
    random_state: int = 42
    normalize: bool = False


def load_params(path: str | Path = PARAMS_PATH) -> PreprocessConfig:
    """Le a secao ``data`` de ``params.yaml``.

    Args:
        path: Caminho do arquivo de parametros.

    Returns:
        Um :class:`PreprocessConfig` com os valores encontrados (ou os padroes).

    Raises:
        FileNotFoundError: Se o arquivo de parametros nao existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de parametros nao encontrado: {path}")

    params = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data_params = params.get("data", {})
    return PreprocessConfig(
        test_size=float(data_params.get("test_size", 0.2)),
        random_state=int(data_params.get("random_state", 42)),
        normalize=bool(data_params.get("normalize", False)),
    )


def load_dataset() -> pd.DataFrame:
    """Carrega o dataset Iris como um DataFrame rotulado.

    Returns:
        DataFrame com as 4 features originais e a coluna alvo ``target``
        (codificada como inteiro: 0, 1, 2).
    """
    dataset = load_iris(as_frame=True)
    frame = dataset.frame.copy()
    frame = frame.rename(columns={"target": TARGET_COLUMN})
    return frame


def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica o feature engineering basico aos conjuntos.

    Atualmente: normalizacao opcional das features numericas com
    ``StandardScaler``, ajustado somente no conjunto de treino para evitar
    vazamento de dados. O scaler e persistido em ``data/processed/scaler.pkl``.

    Args:
        train_df: Conjunto de treino (features + alvo).
        test_df: Conjunto de teste (features + alvo).
        config: Configuracao do pre-processamento.

    Returns:
        Tupla ``(train_df, test_df)`` transformada.
    """
    if not config.normalize:
        return train_df, test_df

    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    scaler = StandardScaler()

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return train_df, test_df


def split_dataset(
    frame: pd.DataFrame,
    config: PreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o DataFrame em treino e teste de forma estratificada.

    Args:
        frame: DataFrame completo (features + alvo).
        config: Configuracao do pre-processamento.

    Returns:
        Tupla ``(train_df, test_df)``.
    """
    train_df, test_df = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=frame[TARGET_COLUMN],
    )
    return train_df, test_df


def run(config: PreprocessConfig | None = None) -> tuple[Path, Path]:
    """Executa o pre-processamento de ponta a ponta.

    Args:
        config: Configuracao do pre-processamento. Se ``None``, le de
            ``params.yaml``.

    Returns:
        Tupla ``(train_path, test_path)`` com os caminhos gravados.
    """
    config = config or load_params()

    frame = load_dataset()
    train_df, test_df = split_dataset(frame, config)
    train_df, test_df = engineer_features(train_df, test_df, config)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Treino: {len(train_df)} linhas -> {TRAIN_PATH}")
    print(f"Teste:  {len(test_df)} linhas -> {TEST_PATH}")
    print(f"Normalizacao: {'ativada' if config.normalize else 'desativada'}")
    return TRAIN_PATH, TEST_PATH


if __name__ == "__main__":
    run()
