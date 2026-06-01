# Análise de Gravidade de Sinistros nas Rodovias Federais Brasileiras

Este repositório contém o projeto de análise e modelagem preditiva de sinistros de trânsito ocorridos nas rodovias federais brasileiras (recorte temporal de 2022 a 2026), baseado nos dados abertos fornecidos pela Polícia Rodoviária Federal (PRF).

O objetivo principal é identificar padrões e fatores (temporais, espaciais, meteorológicos, causais e infraestruturais) associados à gravidade dos acidentes e implementar um modelo preditivo capaz de classificar a gravidade das ocorrências, disponibilizando os resultados em um painel interativo.

**Equipe:**

- Davi da Silva Romão
- Diogo Tallys da Mota Amorim
- Gabriel Gomes de Oliveira
- Thiago Ribeiro da Silva

---

## Estrutura do Repositório

O projeto segue uma estrutura modular projetada para facilitar a colaboração em equipe:

- **`app/`**: Painel interativo desenvolvido para exibição dos resultados.
- **`data/`**: Contém as bases de dados brutos (`data/raw/`) e processados (`data/processed/`).
- **`docs/`**: Documentação de apoio do projeto, incluindo a proposta de trabalho (`initial_proposal.pdf`), figuras auxiliares (`figures/`) e o projeto latext do relatório final (`artigo/`).
- **`labs/`**: Notebooks Jupyter organizados cronologicamente para experimentação, análises estatísticas, modelagem experimental e exploração geoespacial.
- **`models/`**: Pasta para armazenamento dos arquivos binários dos modelos treinados (serializados em formato `.pkl`).
- **`src/`**: Pacote Python modular contendo as funções de pré-processamento de dados (`preprocessing/`), análise de hipóteses (`analysis/`), modelagem preditiva (`models/`) e utilitários gerais (`utils/`).

---

## Instalação e Execução

Este projeto utiliza o gerenciador de pacotes `uv`.

### 1. Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd prf-analise-sinistros
```

### 2. Sincronizar o Ambiente Virtual e Dependências

Para criar o ambiente virtual e instalar as dependências de forma otimizada com o `uv`:

```bash
uv sync
```
