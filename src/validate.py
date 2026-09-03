"""Validacao de dados do dsa_mlops_project (estagio ``validate`` do DVC).

Primeiro estagio do pipeline: valida ``data/raw/iris.csv`` com um schema
``pandera`` antes de qualquer pre-processamento. Verifica:

1. ausencia de valores nulos em todas as colunas;
2. tipos de dados de cada coluna (4 features ``float``, alvo ``int``);
3. ranges plausiveis dos features e dominio do alvo (``{0, 1, 2}``);
4. ausencia de colunas inesperadas (``strict``).

Em caso de sucesso grava ``reports/validation.json`` (consumido pelo estagio
``prepare``); em caso de falha lista todas as violacoes e sai com codigo 1.

Exemplo:
    python src/validate.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

try:  # pandera >= 0.20 expoe o namespace dedicado ao pandas
    import pandera.pandas as pa
except ModuleNotFoundError:  # pragma: no cover - fallback para versoes antigas
    import pandera as pa

from pandera.errors import SchemaErrors

RAW_DATA_PATH = Path("data/raw/iris.csv")
REPORT_PATH = Path("reports/validation.json")
TARGET_COLUMN = "target"

# Limites plausiveis (cm) para o Iris, com folga sobre os valores observados.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "sepal length (cm)": (3.5, 9.0),
    "sepal width (cm)": (1.5, 5.0),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.05, 3.0),
}
TARGET_CLASSES = (0, 1, 2)


def build_schema() -> pa.DataFrameSchema:
    """Monta o schema de validacao do dataset bruto.

    Returns:
        Schema ``pandera`` com checagens de nulos, tipos e ranges.
    """
    feature_columns = {
        name: pa.Column(
            float,
            checks=pa.Check.in_range(low, high),
            nullable=False,
            required=True,
        )
        for name, (low, high) in FEATURE_RANGES.items()
    }
    feature_columns[TARGET_COLUMN] = pa.Column(
        int,
        checks=pa.Check.isin(TARGET_CLASSES),
        nullable=False,
        required=True,
    )
    return pa.DataFrameSchema(feature_columns, strict=True, coerce=False)


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Le o CSV bruto a ser validado.

    Args:
        path: Caminho do arquivo.

    Returns:
        DataFrame com os dados brutos.

    Raises:
        FileNotFoundError: Se o arquivo nao existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset bruto '{path}' nao encontrado. Rode `dvc pull` ou gere o arquivo."
        )
    return pd.read_csv(path)


def validate_dataframe(df: pd.DataFrame, schema: pa.DataFrameSchema | None = None) -> pd.DataFrame:
    """Valida o DataFrame contra o schema, coletando todas as violacoes.

    Args:
        df: DataFrame a validar.
        schema: Schema a usar. Se ``None``, usa :func:`build_schema`.

    Returns:
        O DataFrame validado (inalterado, pois ``coerce=False``).

    Raises:
        pandera.errors.SchemaErrors: Se qualquer checagem falhar.
    """
    schema = schema or build_schema()
    return schema.validate(df, lazy=True)


def write_report(df: pd.DataFrame, path: Path = REPORT_PATH) -> Path:
    """Grava o relatorio de validacao bem-sucedida.

    Args:
        df: DataFrame validado.
        path: Caminho do relatorio JSON.

    Returns:
        O caminho gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "success",
        "source": str(RAW_DATA_PATH),
        "rows": int(len(df)),
        "columns": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "null_counts": {name: int(count) for name, count in df.isna().sum().items()},
        "validated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run() -> Path:
    """Executa a validacao de ponta a ponta.

    Returns:
        O caminho do relatorio em caso de sucesso.

    Raises:
        SystemExit: Com codigo 1 se a validacao falhar.
    """
    df = load_raw_data()
    try:
        validate_dataframe(df)
    except SchemaErrors as err:
        print(f"Validacao FALHOU para {RAW_DATA_PATH}:\n", file=sys.stderr)
        print(err.failure_cases.to_string(index=False), file=sys.stderr)
        raise SystemExit(1) from err

    report_path = write_report(df)
    print(f"Validacao OK: {len(df)} linhas, {len(df.columns)} colunas -> {report_path}")
    return report_path


if __name__ == "__main__":
    run()
