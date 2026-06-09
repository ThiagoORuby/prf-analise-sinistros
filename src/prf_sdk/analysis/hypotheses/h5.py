import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def create_fatality_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variável binária indicando
    ocorrência de acidente fatal.
    """
    df = df.copy()
    df["fatal"] = (df["classificacao_acidente"] == "Com Vítimas Fatais").astype(int)
    return df


def severity_profile_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribuição percentual dos níveis
    de gravidade por região.
    """
    return pd.crosstab(df["regiao"], df["classificacao_acidente"], normalize="index")


def road_type_distribution_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribuição percentual dos tipos de pista
    dentro de cada região.
    """
    return pd.crosstab(df["regiao"], df["tipo_pista"], normalize="index")


def fatality_rate_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Taxa de acidentes fatais por região.
    """
    return (
        df.groupby("regiao")
        .agg(acidentes=("fatal", "count"), taxa_fatalidade=("fatal", "mean"))
        .sort_values("taxa_fatalidade", ascending=False)
    )


def fatality_rate_by_road_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Taxa de acidentes fatais por tipo de pista.
    """
    return (
        df.groupby("tipo_pista")
        .agg(acidentes=("fatal", "count"), taxa_fatalidade=("fatal", "mean"))
        .sort_values("taxa_fatalidade", ascending=False)
    )


def chi_square_region_severity(df: pd.DataFrame, alpha: float = 0.05) -> dict:

    table = pd.crosstab(df["regiao"], df["classificacao_acidente"])

    chi2, p, dof, expected = chi2_contingency(table)

    return {
        "chi2": chi2,
        "pvalue": p,
        "dof": dof,
        "alpha": alpha,
        "reject_h0": p < alpha,
        "observed": table,
        "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
    }


def chi_square_road_type_severity(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    table = pd.crosstab(df["tipo_pista"], df["classificacao_acidente"])
    chi2, p, dof, expected = chi2_contingency(table)

    return {
        "chi2": chi2,
        "pvalue": p,
        "dof": dof,
        "alpha": alpha,
        "reject_h0": p < alpha,
        "observed": table,
        "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
    }


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    fatalidade = fatality_rate_by_region(df)
    pistas = road_type_distribution_by_region(df)

    resumo = pd.DataFrame(
        {
            "Pista Simples (%)": pistas["Simples"] * 100,
            "Pista Dupla (%)": pistas["Dupla"] * 100,
            "Pista Múltipla (%)": pistas["Múltipla"] * 100,
            "Taxa Fatalidade (%)": fatalidade["taxa_fatalidade"] * 100,
        }
    )

    resumo = resumo.round(2)

    return resumo


def show_chi_square_result(result: dict, title: str) -> None:
    """
    Exibe resultado resumido de um teste Qui-Quadrado.
    """

    print("=" * 60)
    print(title)
    print("=" * 60)

    print(f"Qui²: {result['chi2']:.2f}")
    print(f"Graus de liberdade: {result['dof']}")
    print(f"p-value: {result['pvalue']:.6f}")

    if result["reject_h0"]:
        print("\nResultado: REJEITA H0")
        print("Há evidências de associação estatisticamente significativa.")
    else:
        print("\nResultado: NÃO REJEITA H0")
        print("Não foram encontradas evidências de associação.")


def plot_h5_summary(df):
    """
    Exibe gráfico comparando:
    - proporção de pista simples
    - taxa de fatalidade

    por região.
    """
    fatalidade = fatality_rate_by_region(df)
    pistas = road_type_distribution_by_region(df)

    resumo = pd.DataFrame(
        {
            "Pista Simples (%)": pistas["Simples"] * 100,
            "Taxa Fatalidade (%)": fatalidade["taxa_fatalidade"] * 100,
        }
    )

    resumo = resumo.round(2)

    resumo = resumo.sort_values("Pista Simples (%)", ascending=False)

    plt.figure(figsize=(10, 6))

    plt.plot(
        resumo.index,
        resumo["Pista Simples (%)"],
        marker="o",
        linewidth=2,
        label="Pista Simples (%)",
    )

    plt.plot(
        resumo.index,
        resumo["Taxa Fatalidade (%)"],
        marker="o",
        linewidth=2,
        label="Fatalidade (%)",
    )

    plt.title("Pista Simples vs Taxa de Fatalidade por Região")

    plt.ylabel("Percentual")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.show()

def show_h5_results(df):
    """
    Exibe resumo visual e estatístico
    da hipótese H5.
    """

    plot_h5_summary(df)

    print("\n")

    show_chi_square_result(
        chi_square_region_severity(df),
        "Teste Qui-Quadrado: Região x Gravidade"
    )

    print("\n")

    show_chi_square_result(
        chi_square_road_type_severity(df),
        "Teste Qui-Quadrado: Tipo de Pista x Gravidade"
    )