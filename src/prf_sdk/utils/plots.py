from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL


_NE_STATES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
    "/master/geojson/ne_50m_admin_1_states_provinces.geojson"
)
_BR_LAT = (-34, 6)
_BR_LON = (-74, -34)


def _load_brazil_states(cache_dir: Path):
    """Carrega limites dos estados brasileiros, baixando uma vez se necessário."""
    import geopandas as gpd

    cache_path = cache_dir / "brazil_states.geojson"
    if not cache_path.exists():
        import requests

        cache_dir.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_NE_STATES_URL, timeout=30)
        resp.raise_for_status()
        import json

        data = resp.json()
        br_features = [
            f for f in data["features"] if f["properties"].get("iso_a2") == "BR"
        ]
        cache_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": br_features})
        )

    return gpd.read_file(cache_path)


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


def _build_top10_trechos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega top-10 trechos fatais com coordenadas medianas."""
    d = df.copy()
    d["is_fatal"] = d["classificacao_acidente"].str.contains(
        "Fatais", case=False, na=False
    ) | (d["mortos"].fillna(0) > 0)
    d["id_trecho"] = (
        d["uf"]
        + " — BR-"
        + d["br"].astype(str)
        + " (km "
        + d["faixa_km"].astype(int).astype(str)
        + ")"
    )
    return (
        d[d["is_fatal"]]
        .groupby("id_trecho")
        .agg(
            sinistros_fatais=("is_fatal", "sum"),
            lat=("latitude", "median"),
            lon=("longitude", "median"),
        )
        .reset_index()
        .sort_values("sinistros_fatais", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


def _draw_map_panel(ax, fig, fatais, top10, states, legend_fontsize: int = 6):
    """Desenha hexbin, bordas estaduais, marcadores numerados e legenda Amazônia."""
    states.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#b0b0b0", linewidth=0.5)

    hb = ax.hexbin(
        fatais["longitude"],
        fatais["latitude"],
        gridsize=120,
        mincnt=1,
        bins="log",
        cmap="YlOrRd",
        alpha=0.85,
        zorder=2,
    )

    # Colorbar como inset dentro do painel — não redimensiona os eixos principais
    # Posição: [x0, y0, largura, altura] em coordenadas normalizadas dos eixos
    cbax = ax.inset_axes([0.22, 0.025, 0.52, 0.022])
    cb = fig.colorbar(hb, cax=cbax, orientation="horizontal")
    cb.set_label("Sinistros fatais (log)", fontsize=legend_fontsize, labelpad=2)
    cb.ax.tick_params(labelsize=legend_fontsize - 0.5)
    cbax.set_zorder(9)

    # Estrelas + círculo com rank (sem texto longo no mapa)
    for i, row in enumerate(top10.itertuples()):
        ax.plot(
            row.lon,
            row.lat,
            marker="*",
            color="black",
            markersize=9,
            zorder=5,
            markeredgewidth=0.3,
        )
        ax.text(
            row.lon + 0.22,
            row.lat + 0.22,
            str(i + 1),
            fontsize=5.5,
            ha="center",
            va="center",
            zorder=7,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.12", fc="#1a1a1a", ec="none"),
        )

    # Legenda numerada na Amazônia (canto sup-esq do mapa, sem dados de rodovias)
    legend_lines = ["★  Top-10 trechos (sinistros fatais):"] + [
        f"  {i + 1:2d}. {row.id_trecho}  ({int(row.sinistros_fatais)})"
        for i, row in enumerate(top10.itertuples())
    ]
    ax.text(
        -73.5,
        3.0,
        "\n".join(legend_lines),
        fontsize=legend_fontsize,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", alpha=0.92, ec="#999", lw=0.5),
        zorder=8,
    )

    ax.set_xlim(-74, -33)
    ax.set_ylim(-34, 6)
    ax.set_xlabel("Longitude", fontsize=legend_fontsize + 1)
    ax.set_ylabel("Latitude", fontsize=legend_fontsize + 1)
    ax.tick_params(labelsize=legend_fontsize)


def build_lorenz_spatial_figure(
    df: pd.DataFrame,
    output_path: Path,
    geo_cache_dir: Path | None = None,
) -> Path:
    """
    Gera figura com dois painéis: curva de Lorenz espacial (esquerda) e mapa
    de calor hexagonal de sinistros fatais (direita) com os dez trechos
    críticos marcados.

    :param df: DataFrame com colunas ``uf``, ``br``, ``faixa_km``,
        ``classificacao_acidente``, ``mortos``, ``latitude`` e ``longitude``.
    :param output_path: Caminho de saída da figura.
    :param geo_cache_dir: Diretório de cache do shapefile de estados.
        Se ``None``, usa ``output_path.parent.parent / 'geo'``.
    :returns: Caminho da figura salva.
    """
    if geo_cache_dir is None:
        geo_cache_dir = output_path.parent.parent / "geo"

    # --- Lorenz data ---
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
    trechos = (
        trechos_agg[trechos_agg["total_sinistros"] > 0]
        .sort_values("sinistros_fatais", ascending=False)
        .reset_index(drop=True)
    )
    total_trechos = len(trechos)
    total_fatais = trechos["sinistros_fatais"].sum()
    trechos["pct_trechos"] = np.arange(1, total_trechos + 1) / total_trechos
    trechos["pct_fatais"] = trechos["sinistros_fatais"].cumsum() / total_fatais
    top_5pct_fatais = trechos[trechos["pct_trechos"] <= 0.05]["pct_fatais"].max()

    # --- map data ---
    top10 = _build_top10_trechos(df)
    df_temp2 = df.copy()
    df_temp2["is_fatal"] = df_temp2["classificacao_acidente"].str.contains(
        "Fatais", case=False, na=False
    ) | (df_temp2["mortos"].fillna(0) > 0)
    fatais = df_temp2[
        df_temp2["is_fatal"]
        & df_temp2["latitude"].between(*_BR_LAT)
        & df_temp2["longitude"].between(*_BR_LON)
    ]
    states = _load_brazil_states(geo_cache_dir)

    # --- figura ---
    set_plotting_theme()
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.25)
    ax_lorenz = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])

    # Lorenz curve
    ax_lorenz.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", label="Distribuição Uniforme"
    )
    ax_lorenz.plot(
        trechos["pct_trechos"],
        trechos["pct_fatais"],
        color="red",
        linewidth=2,
        label="Concentração Real",
    )
    ax_lorenz.axvline(x=0.05, color="orange", linestyle=":", alpha=0.7)
    ax_lorenz.axhline(y=top_5pct_fatais, color="orange", linestyle=":", alpha=0.7)
    ax_lorenz.plot(0.05, top_5pct_fatais, "ro")
    ax_lorenz.text(
        0.06,
        top_5pct_fatais - 0.05,
        f"Top 5% trechos\n→ {top_5pct_fatais:.1%} dos fatais",
        color="darkred",
    )
    ax_lorenz.set_xlabel("% Acumulada de Trechos")
    ax_lorenz.set_ylabel("% Acumulada de Sinistros Fatais")
    ax_lorenz.legend()
    ax_lorenz.grid(True, alpha=0.3)

    # Map panel
    _draw_map_panel(ax_map, fig, fatais, top10, states, legend_fontsize=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
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


def build_fatality_map_figure(
    df: pd.DataFrame,
    output_path: Path,
    geo_cache_dir: Path | None = None,
) -> Path:
    """
    Gera mapa de calor hexagonal dos sinistros fatais nas rodovias federais
    brasileiras, sobreposto ao contorno dos estados.

    A intensidade é proporcional à concentração de ocorrências em escala
    logarítmica (``hexbin`` com ``bins='log'``). Os dez trechos de 5 km com
    maior número absoluto de sinistros fatais são marcados com estrela
    numerada; a legenda completa é posicionada na Amazônia, onde a densidade
    de rodovias é mínima.

    :param df: DataFrame com colunas ``classificacao_acidente``, ``mortos``,
        ``latitude``, ``longitude``, ``uf``, ``br`` e ``faixa_km``.
    :param output_path: Caminho de saída da figura (recomendado: ``.png``).
    :param geo_cache_dir: Diretório onde o shapefile de estados é cacheado.
        Se ``None``, usa ``output_path.parent.parent / 'geo'``.
    :returns: Caminho da figura salva.
    """
    if geo_cache_dir is None:
        geo_cache_dir = output_path.parent.parent / "geo"

    df_temp = df.copy()
    df_temp["is_fatal"] = df_temp["classificacao_acidente"].str.contains(
        "Fatais", case=False, na=False
    ) | (df_temp["mortos"].fillna(0) > 0)
    fatais = df_temp[
        df_temp["is_fatal"]
        & df_temp["latitude"].between(*_BR_LAT)
        & df_temp["longitude"].between(*_BR_LON)
    ]
    top10 = _build_top10_trechos(df)
    states = _load_brazil_states(geo_cache_dir)

    set_plotting_theme()
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_map_panel(ax, fig, fatais, top10, states, legend_fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
