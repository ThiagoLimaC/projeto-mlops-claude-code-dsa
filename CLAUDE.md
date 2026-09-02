# CLAUDE.md

Contexto persistente para o Claude Code neste repositório: **dsa_mlops_project**,
um template de MLOps para um problema de classificação (Data Science Academy —
curso *Engenharia de Software com IA Generativa e Vibe Coding*).

## Comandos de Build

- `python -m pytest tests/ -v` — Rodar testes
- `python -m pip install -e .` — Instalar pacote em modo editável
- `dvc repro` — Reproduzir pipeline completo
- `mlflow ui --port 5000` — Iniciar interface MLflow

## Estilo de Código

- Use type hints em todas as funções
- Siga convenções PEP 8
- Use dataclasses ou Pydantic para modelos de dados
- Docstrings no formato Google

## Arquitetura

- MLflow para tracking de experimentos
- DVC para versionamento de dados e pipelines
- pytest para testes unitários
- Hydra ou YAML para configuração

## Workflow

- Sempre rode testes após mudanças no código
- Faça commits frequentes com mensagens descritivas
- Use `dvc add` para novos datasets
