# dsa_mlops_project

Projeto completo de **MLOps** para um problema de **classificação**, construído de
forma *AI-first* com o **Claude Code** no âmbito do curso *Engenharia de Software
com IA Generativa e Vibe Coding* da **Data Science Academy**.

O objetivo não é apenas treinar um modelo, mas entregar um **template
reutilizável e profissional**: engenharia de software desde o primeiro commit
(PEP 8, type hints, docstrings, testes), rastreamento de experimentos com
**MLflow**, versionamento de dados e pipeline com **DVC**, e um `CLAUDE.md` com
contexto persistente de regras de negócio e estilo de código.

## Estrutura do projeto

```
dsa_mlops_project/
├── src/                  # Código-fonte
│   ├── __init__.py
│   ├── preprocess.py     # Carga e transformação dos dados brutos
│   ├── train.py          # Treino do modelo + tracking MLflow
│   └── evaluate.py       # Avaliação e métricas
├── tests/                # Testes com pytest
├── configs/              # Configurações (config.yaml)
├── data/
│   ├── raw/              # Dados brutos (versionados via DVC)
│   └── processed/        # Dados processados (gerados pelo pipeline)
├── models/               # Artefatos de modelo (versionados via DVC/MLflow)
├── notebooks/            # Notebooks exploratórios
├── pyproject.toml        # Metadados e dependências (fonte de verdade)
├── requirements.txt      # Dependências de runtime (compatibilidade pip)
└── .gitignore
```

## Ambiente e dependências

O projeto usa [`uv`](https://docs.astral.sh/uv/) para ambiente e dependências.

```bash
uv sync --extra dev
```

Alternativa com pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Fluxo de trabalho

```bash
# 1. Pré-processamento: data/raw -> data/processed
python -m preprocess

# 2. Treino do modelo (registra params/métricas/modelo no MLflow)
python -m train

# 3. Avaliação no conjunto de teste
python -m evaluate

# Testes e lint
pytest
ruff check .
```

> As etapas de dados, DVC, pipeline e CI/CD serão adicionadas nas próximas
> fases do projeto. O dataset de classificação ainda será definido.

## Ferramentas de MLOps

| Ferramenta | Papel no projeto |
|------------|------------------|
| **MLflow** | Rastreamento de experimentos: parâmetros, métricas e artefatos de modelo. |
| **DVC**    | Versionamento de dados e definição do pipeline como DAG. |
| **pytest** | Testes unitários e de regressão. |
| **ruff**   | Lint e formatação (PEP 8, imports, docstrings). |
