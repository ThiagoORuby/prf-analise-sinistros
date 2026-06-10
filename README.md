# Análise de Gravidade de Sinistros nas Rodovias Federais Brasileiras

[Dashboard interativo aqui](https://dashboard-prf.streamlit.app/)

Este repositório contém o projeto de análise e modelagem preditiva de sinistros de trânsito ocorridos nas rodovias federais brasileiras (recorte temporal de 2022 a 2026), baseado nos dados abertos fornecidos pela Polícia Rodoviária Federal (PRF).

O objetivo principal é identificar padrões e fatores (temporais, espaciais, meteorológicos, causais e infraestruturais) associados à gravidade dos acidentes e implementar um modelo preditivo capaz de classificar a gravidade das ocorrências, disponibilizando os resultados em um painel interativo.

**Equipe:**

- Davi da Silva Romão
- Diogo Tallys da Mota Amorim
- Gabriel Gomes de Oliveira
- Thiago Ribeiro da Silva

---

## Estrutura do Repositório

O projeto segue uma estrutura modular projetada para possibilitar o uso do núcleo do projeto como um SDK reutilizável:

- **`app/`**: Painel interativo (dashboard) desenvolvido para exibição dos resultados.
- **`data/`**: Contém as bases de dados brutos (`data/raw/`) e processados (`data/processed/`).
- **`docs/`**: Documentação de apoio do projeto, incluindo a proposta inicial do projeto (`proposta_inicial.pdf`), figuras auxiliares (`figures/`) e o relatório final/lartigo (`artigo/`).
- **`labs/`**: Notebooks Jupyter organizados para experimentação rápida e prototipação.
- **`models/`**: Pasta para armazenamento dos arquivos binários dos modelos treinados (serializados em formato `.pkl`).
- **`scripts/`**: Scripts utilitários de linha de comando (CLI) que consomem as funções do SDK do projeto.
- **`src/`**: Diretório que contém o código fonte do SDK do projeto:
  - **`src/prf_sdk/`**: Pacote principal do SDK Python contendo os módulos reutilizáveis:
    - `preprocessing/`: Limpeza de dados (`cleaner.py`), engenharia de features (`features.py`) e loaders de carga (`loader.py`).
    - `analysis/`: Módulos de análises específicas (como testes de hipóteses estatísticas em `initial.py`).
    - `utils/`: Funções utilitárias e rotinas matemáticas/estatísticas (como `stats.py`).
    - `settings.py`: Configuração central do SDK utilizando `pydantic-settings` (gerenciamento do `BASE_DIR`, caminhos e variáveis de ambiente).
- **`tests/`**: Testes unitários e de integração automatizados (utilizando `pytest`).

---

## Instalação e Execução

Este projeto utiliza o gerenciador de pacotes `uv`.

### 1. Clonar o Repositório

```bash
git clone https://github.com/ThiagoORuby/prf-analise-sinistros.git
cd prf-analise-sinistros
```

### 2. Sincronizar o Ambiente Virtual e Dependências

Para criar o ambiente virtual e instalar todas as dependências de forma rápida e segura:

```bash
uv sync
```

### 3. Executar Scripts de Linha de Comando (CLI)

O projeto disponibiliza scripts utilitários na pasta `scripts/` que importam as funcionalidades do SDK em `src/prf_sdk/`. Para rodar os scripts:

```bash
uv run scripts/algum_script.py
```

### 4. Executar Testes Automatizados

O projeto utiliza o `pytest` para garantir a integridade das rotinas do projeto. Para rodar a suíte de testes:

```bash
uv run pytest
```
