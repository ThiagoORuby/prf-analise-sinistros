import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _build_id_trecho(df: pd.DataFrame) -> pd.Series:
    return (
        df["uf"]
        + " - BR-"
        + df["br"].astype(str)
        + " (km "
        + df["faixa_km"].astype(int).astype(str)
        + ")"
    )


def get_h4_metrics(df: pd.DataFrame) -> dict:
    """
    Valida a Hipótese 4: A distribuição dos sinistros não é uniforme, com concentração em
    trechos específicos que respondem por parcela desproporcional das ocorrências fatais.

    Utiliza a coluna ``faixa_km`` (bins de 5 km) criada pelo pipeline de pré-processamento.
    Retorna métricas de concentração e dataframe agregado dos trechos.
    """
    df_temp = df.copy()
    df_temp["id_trecho"] = _build_id_trecho(df_temp)
    df_temp["acidente_fatal"] = (
        df_temp["classificacao_acidente"].str.contains("Fatais", case=False, na=False)
    ) | (df_temp["mortos"] > 0)

    trechos_agg = (
        df_temp.groupby("id_trecho")
        .agg(
            total_sinistros=("id_trecho", "count"),
            sinistros_fatais=("acidente_fatal", "sum"),
            total_mortos=("mortos", "sum"),
            uf=("uf", "first"),
            br=("br", "first"),
            km_inicial=("faixa_km", "first"),
        )
        .reset_index()
    )
    trechos_agg = trechos_agg[trechos_agg["total_sinistros"] > 0]

    trechos_fatais_desc = trechos_agg.sort_values(
        by="sinistros_fatais", ascending=False
    ).reset_index(drop=True)

    total_trechos = len(trechos_fatais_desc)
    total_fatais = trechos_fatais_desc["sinistros_fatais"].sum()

    trechos_fatais_desc["cum_trechos"] = np.arange(1, total_trechos + 1)
    trechos_fatais_desc["pct_trechos"] = trechos_fatais_desc["cum_trechos"] / total_trechos
    trechos_fatais_desc["cum_fatais"] = trechos_fatais_desc["sinistros_fatais"].cumsum()
    trechos_fatais_desc["pct_fatais"] = trechos_fatais_desc["cum_fatais"] / total_fatais

    pct_trechos_50_fatais = (
        len(trechos_fatais_desc[trechos_fatais_desc["pct_fatais"] <= 0.50]) / total_trechos
    )
    top_5pct_fatais = (
        trechos_fatais_desc[trechos_fatais_desc["pct_trechos"] <= 0.05]["pct_fatais"].max()
    )
    top_10pct_fatais = (
        trechos_fatais_desc[trechos_fatais_desc["pct_trechos"] <= 0.10]["pct_fatais"].max()
    )

    return {
        "total_trechos": total_trechos,
        "total_sinistros_fatais": total_fatais,
        "pct_trechos_que_concentram_50pct_fatais": pct_trechos_50_fatais,
        "pct_fatais_no_top_5pct_trechos": top_5pct_fatais,
        "pct_fatais_no_top_10pct_trechos": top_10pct_fatais,
        "top_10_trechos_fatais": trechos_fatais_desc.head(10),
        "dados_plot_lorenz": trechos_fatais_desc[["pct_trechos", "pct_fatais"]],
    }


def plot_lorenz_curve(metrics_dict: dict, save_path: str = None):
    """
    Plota a curva de Lorenz para mostrar a concentração de acidentes fatais.
    """
    lorenz_data = metrics_dict["dados_plot_lorenz"]

    plt.figure(figsize=(8, 6))
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--", color="gray", label="Distribuição Uniforme (Teórica)",
    )
    plt.plot(
        lorenz_data["pct_trechos"], lorenz_data["pct_fatais"],
        color="red", linewidth=2, label="Concentração Real",
    )
    plt.axvline(x=0.05, color="orange", linestyle=":", alpha=0.7)
    plt.axhline(
        y=metrics_dict["pct_fatais_no_top_5pct_trechos"],
        color="orange", linestyle=":", alpha=0.7,
    )
    plt.plot(0.05, metrics_dict["pct_fatais_no_top_5pct_trechos"], "ro")
    plt.text(
        0.06,
        metrics_dict["pct_fatais_no_top_5pct_trechos"] - 0.05,
        f"Top 5% trechos\n-> {metrics_dict['pct_fatais_no_top_5pct_trechos']:.1%} dos fatais",
        color="darkred",
    )
    plt.title("Curva de Concentração (Lorenz) — Sinistros Fatais por Faixa de 5 km")
    plt.xlabel("% Acumulada de Trechos de Rodovia")
    plt.ylabel("% Acumulada de Sinistros Fatais")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_top_10_dangerous_segments(metrics_dict: dict, save_path: str = None):
    """
    Plota um gráfico de barras com os 10 trechos mais perigosos.
    """
    top_10 = metrics_dict["top_10_trechos_fatais"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_10, x="sinistros_fatais", y="id_trecho", palette="Reds_r")

    plt.title("Top 10 Faixas de 5 km com Mais Sinistros Fatais (2022–2026)")
    plt.xlabel("Número de Sinistros Fatais")
    plt.ylabel("Trecho da Rodovia")
    for p in plt.gca().patches:
        plt.gca().annotate(
            f"{int(p.get_width())}",
            (p.get_width() + 0.5, p.get_y() + p.get_height() / 2.0),
            ha="left", va="center",
        )
    plt.xlim(0, top_10["sinistros_fatais"].max() * 1.1)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
