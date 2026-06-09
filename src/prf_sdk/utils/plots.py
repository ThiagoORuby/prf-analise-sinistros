from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL


def set_plotting_theme():
    """
    Aplica um padrão visual unificado para todos os plots do projeto.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def build_h1b_fatality_figure(
    df: pd.DataFrame,
    output_path: Path,
    behavioral_causes: list[str] | None = None,
    night_periods: tuple[str, ...] = ("Madrugada", "Noite"),
) -> Path:
    """
    Gera gráfico de barras agrupadas com taxa de fatalidade por grupo de causa
    (comportamental vs. outras) e período do dia (diurno vs. noturno).

    Visualiza o padrão de inversão do OR de H1b: causas comportamentais
    são relativamente mais letais durante o dia do que à noite, quando
    comparadas com as demais causas no mesmo período.

    :param df: DataFrame com colunas ``causa_acidente``, ``horario`` e
        ``classificacao_acidente``.
    :param output_path: Caminho de saída da figura.
    :param behavioral_causes: Lista de causas comportamentais. Se ``None``,
        usa as quatro causas padrão de H1.
    :param night_periods: Faixas horárias classificadas como noturnas.
    :returns: Caminho da figura salva.
    """
    if behavioral_causes is None:
        behavioral_causes = [
            "Condutor Dormindo",
            "Velocidade Incompatível",
            "Ingestão de álcool pelo condutor",
            "Ingestão de substâncias psicoativas pelo condutor",
        ]

    df_temp = df.copy()
    df_temp["causa_grupo"] = (
        df_temp["causa_acidente"]
        .isin(behavioral_causes)
        .map({True: "Comportamental", False: "Outras causas"})
    )
    df_temp["periodo"] = (
        df_temp["horario"].isin(night_periods).map({True: "Noturno", False: "Diurno"})
    )
    df_temp["is_fatal"] = df_temp["classificacao_acidente"].isin(
        ["Com Vítimas Fatais", "Com Vitimas Fatais"]
    )

    rates = df_temp.groupby(["periodo", "causa_grupo"])["is_fatal"].mean().reset_index()
    rates["taxa_fatalidade"] = rates["is_fatal"] * 100

    set_plotting_theme()
    fig, ax = plt.subplots(figsize=(9, 6))

    palette = {"Comportamental": "#ef4444", "Outras causas": "#64748b"}
    sns.barplot(
        data=rates,
        x="periodo",
        y="taxa_fatalidade",
        hue="causa_grupo",
        order=["Diurno", "Noturno"],
        hue_order=["Comportamental", "Outras causas"],
        ax=ax,
        palette=palette,
    )
    ax.set_xlabel("Período do Dia")
    ax.set_ylabel("Taxa de Fatalidade (%)")
    ax.legend(title="Grupo de causa")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_eda_categoricas_figure(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Gera gráfico de barras com taxa de fatalidade por tipo de acidente (top 10
    por frequência) e por grupo de causa do acidente.

    :param df: DataFrame com colunas ``tipo_acidente``, ``causa_acidente_grupo``
        e ``classificacao_acidente``.
    :param output_path: Caminho de saída da figura.
    :returns: Caminho da figura salva.
    """
    df_temp = df.copy()
    df_temp["is_fatal"] = df_temp["classificacao_acidente"] == "Com Vítimas Fatais"

    top_tipos = df_temp["tipo_acidente"].value_counts().nlargest(10).index
    fatalidade_tipo = (
        df_temp[df_temp["tipo_acidente"].isin(top_tipos)]
        .groupby("tipo_acidente")["is_fatal"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    fatalidade_causa = (
        df_temp.groupby("causa_acidente_grupo")["is_fatal"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    set_plotting_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.barplot(
        x=fatalidade_tipo.values,
        y=fatalidade_tipo.index,
        hue=fatalidade_tipo.index,
        legend=False,
        ax=axes[0],
        palette="Reds_r",
    )
    axes[0].set_xlabel("Taxa de Fatalidade (%) — Tipo de Acidente (Top 10 Frequentes)")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, fatalidade_tipo.max() * 1.2)
    for p in axes[0].patches:
        axes[0].annotate(
            f"{p.get_width():.1f}%",
            (p.get_width() + 0.1, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
        )

    sns.barplot(
        x=fatalidade_causa.values,
        y=fatalidade_causa.index,
        hue=fatalidade_causa.index,
        legend=False,
        ax=axes[1],
        palette="Reds_r",
    )
    axes[1].set_xlabel("Taxa de Fatalidade (%) — Grupo de Causa")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, fatalidade_causa.max() * 1.2)
    for p in axes[1].patches:
        axes[1].annotate(
            f"{p.get_width():.1f}%",
            (p.get_width() + 0.1, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_stl_seasonality_figure(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Gera figura comparando as componentes sazonais normalizadas da decomposição
    STL do volume mensal de sinistros e da proporção fatal mensal.

    Permite verificar visualmente se a sazonalidade é um fenômeno de volume,
    de gravidade, ou de ambos de forma independente.

    :param df: DataFrame com colunas ``data_hora``, ``mortos`` e
        ``classificacao_acidente``.
    :param output_path: Caminho de saída da figura.
    :returns: Caminho da figura salva.
    """
    df_temp = df.copy()
    df_temp["data_hora"] = pd.to_datetime(df_temp["data_hora"], errors="coerce")
    df_temp = df_temp.dropna(subset=["data_hora"])
    df_temp["is_fatal"] = (
        (df_temp["mortos"].fillna(0) > 0)
        | df_temp["classificacao_acidente"].isin(
            ["Com Vítimas Fatais", "Com Vitimas Fatais"]
        )
    ).astype(int)

    monthly = (
        df_temp.set_index("data_hora")
        .resample("ME")
        .agg(total=("is_fatal", "count"), fatais=("is_fatal", "sum"))
    )
    monthly["proporcao_fatal"] = monthly["fatais"] / monthly["total"]

    stl_vol = STL(monthly["total"], period=12, seasonal=13).fit()
    stl_fat = STL(monthly["proporcao_fatal"].dropna(), period=12, seasonal=13).fit()

    vol_norm = stl_vol.seasonal / stl_vol.seasonal.std()
    fat_norm = stl_fat.seasonal / stl_fat.seasonal.std()

    set_plotting_theme()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        vol_norm.index,
        vol_norm,
        label="Volume de Sinistros",
        color="#3b82f6",
        linewidth=2,
    )
    ax.plot(
        fat_norm.index,
        fat_norm,
        label="Proporção Fatal",
        color="#ef4444",
        linewidth=2,
        linestyle="--",
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Sazonalidade (desvios padrão)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_lorenz_spatial_figure(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Gera curva de Lorenz espacial e ranking das faixas de 5 km com maior
    concentração de sinistros fatais (top 10).

    :param df: DataFrame com colunas ``uf``, ``br``, ``faixa_km``,
        ``classificacao_acidente`` e ``mortos``.
    :param output_path: Caminho de saída da figura.
    :returns: Caminho da figura salva.
    """
    df_temp = df.copy()
    df_temp["id_trecho"] = (
        df_temp["uf"]
        + " - BR-"
        + df_temp["br"].astype(str)
        + " (km "
        + df_temp["faixa_km"].astype(int).astype(str)
        + ")"
    )
    df_temp["acidente_fatal"] = df_temp["classificacao_acidente"].str.contains(
        "Fatais", case=False, na=False
    ) | (df_temp["mortos"] > 0)

    trechos_agg = (
        df_temp.groupby("id_trecho")
        .agg(
            total_sinistros=("id_trecho", "count"),
            sinistros_fatais=("acidente_fatal", "sum"),
        )
        .reset_index()
    )
    trechos_agg = trechos_agg[trechos_agg["total_sinistros"] > 0]
    trechos = trechos_agg.sort_values("sinistros_fatais", ascending=False).reset_index(
        drop=True
    )

    total_trechos = len(trechos)
    total_fatais = trechos["sinistros_fatais"].sum()
    trechos["pct_trechos"] = np.arange(1, total_trechos + 1) / total_trechos
    trechos["pct_fatais"] = trechos["sinistros_fatais"].cumsum() / total_fatais

    top_5pct_fatais = trechos[trechos["pct_trechos"] <= 0.05]["pct_fatais"].max()
    top_10 = trechos.head(10)

    set_plotting_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(
        [0, 1], [0, 1], linestyle="--", color="gray", label="Distribuição Uniforme"
    )
    axes[0].plot(
        trechos["pct_trechos"],
        trechos["pct_fatais"],
        color="red",
        linewidth=2,
        label="Concentração Real",
    )
    axes[0].axvline(x=0.05, color="orange", linestyle=":", alpha=0.7)
    axes[0].axhline(y=top_5pct_fatais, color="orange", linestyle=":", alpha=0.7)
    axes[0].plot(0.05, top_5pct_fatais, "ro")
    axes[0].text(
        0.06,
        top_5pct_fatais - 0.05,
        f"Top 5% trechos\n→ {top_5pct_fatais:.1%} dos fatais",
        color="darkred",
    )
    axes[0].set_xlabel("% Acumulada de Trechos")
    axes[0].set_ylabel("% Acumulada de Sinistros Fatais")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sns.barplot(
        data=top_10, x="sinistros_fatais", y="id_trecho", ax=axes[1], palette="Reds_r"
    )
    axes[1].set_xlabel("Número de Sinistros Fatais")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, top_10["sinistros_fatais"].max() * 1.15)
    for p in axes[1].patches:
        axes[1].annotate(
            f"{int(p.get_width())}",
            (p.get_width() + 0.3, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_regional_h5_figure(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Gera gráfico de barras agrupadas com taxa de fatalidade por região
    e tipo de pista.

    :param df: DataFrame com colunas ``regiao``, ``tipo_pista`` e
        ``classificacao_acidente``.
    :param output_path: Caminho de saída da figura.
    :returns: Caminho da figura salva.
    """
    df_temp = df.copy()
    df_temp["is_fatal"] = df_temp["classificacao_acidente"] == "Com Vítimas Fatais"

    grouped = df_temp.groupby(["regiao", "tipo_pista"])["is_fatal"].mean().reset_index()
    grouped["taxa_fatalidade"] = grouped["is_fatal"] * 100

    region_order = (
        df_temp.groupby("regiao")["is_fatal"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    set_plotting_theme()
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=grouped,
        x="regiao",
        y="taxa_fatalidade",
        hue="tipo_pista",
        order=region_order,
        ax=ax,
        palette="Blues",
    )
    ax.set_xlabel("Região")
    ax.set_ylabel("Taxa de Fatalidade (%)")
    ax.legend(title="Tipo de Pista")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
