import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from prf_sdk.settings import settings


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard PRF — Sinistros",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] { background-color: #0F172A; }
    [data-testid="stSidebar"]          { background-color: #1E293B; }
    .main-header { font-size: 2.4rem; color: #F8FAFC; font-weight: 700; margin-bottom: 0; }
    .sub-header  { font-size: 1.15rem; color: #94A3B8; margin-bottom: 18px; }
    .hyp-card {
        background-color: #1E293B;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .badge-confirmed { background:#10b981; color:#fff; padding:.2rem .7rem;
                       border-radius:9999px; font-weight:700; font-size:.85rem; }
    .badge-partial   { background:#f59e0b; color:#fff; padding:.2rem .7rem;
                       border-radius:9999px; font-weight:700; font-size:.85rem; }
    .badge-refuted   { background:#ef4444; color:#fff; padding:.2rem .7rem;
                       border-radius:9999px; font-weight:700; font-size:.85rem; }
    div[data-testid="stMetricValue"] { color: #60A5FA; }
    div[data-testid="stMetricLabel"] { color: #94A3B8; }
    h1,h2,h3,p,label { color: #F8FAFC !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PALETA = ["#60A5FA", "#FBBF24", "#34D399", "#F87171", "#A78BFA", "#3B82F6", "#94A3B8"]
COR_1 = [PALETA[0]]

DATA_PATH = settings.BASE_DIR / "data/processed/datatran_2022_2026_processed_v1.csv"
FIGURES_DIR = settings.BASE_DIR / "docs/figures"

_ORDEM_HORARIO = ["Madrugada", "Manhã", "Tarde", "Noite"]
_ORDEM_SEMANA = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]


# ---------------------------------------------------------------------------
# Carregamento e cache de dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando dados…")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df["ano"] = df["data_hora"].dt.year
    df["hora"] = df["data_hora"].dt.hour
    df["is_fatal"] = df["classificacao_acidente"].str.contains(
        "Fatais", case=False, na=False
    )
    # Coordenadas fora dos limites continentais do Brasil → descartadas do mapa
    lat_ok = df["latitude"].between(-35, 6)
    lon_ok = df["longitude"].between(-75, -29)
    df.loc[~(lat_ok & lon_ok), ["latitude", "longitude"]] = np.nan
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def img(filename: str) -> str | None:
    p = FIGURES_DIR / filename
    return str(p) if p.exists() else None


def fmt(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
    )
    fig.update_xaxes(showgrid=False, color="#94A3B8")
    fig.update_yaxes(showgrid=True, gridcolor="#334155", color="#94A3B8")
    return fig


def taxa_df(df: pd.DataFrame, *groupby: str) -> pd.DataFrame:
    """Agrupa e calcula taxa de fatalidade (%) por coluna(s)."""
    g = (
        df.groupby(list(groupby))["is_fatal"]
        .agg(total="count", fatais="sum")
        .reset_index()
    )
    g["taxa"] = g["fatais"] / g["total"] * 100
    return g


def hyp_card(titulo: str, badge_html: str) -> None:
    st.markdown(
        f'<div class="hyp-card"><h3>{titulo}</h3>'
        f"<strong>Status:</strong> {badge_html}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: navegação
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/"
    "Bras%C3%A3o_da_Pol%C3%ADcia_Rodovi%C3%A1ria_Federal.svg/"
    "1200px-Bras%C3%A3o_da_Pol%C3%ADcia_Rodovi%C3%A1ria_Federal.svg.png",
    width=140,
)
st.sidebar.title("Navegação")
PAGES = {
    "Visão Geral": "geral",
    "Análise Exploratória (EDA)": "eda",
    "Mapa de Acidentes": "mapa",
    "H1 — Causas Comportamentais": "h1",
    "H2 — Sazonalidade e Feriados": "h2",
    "H3 — Fatores Meteorológicos": "h3",
    "H4 — Concentração Espacial": "h4",
    "H5 — Impacto Regional": "h5",
}
selection = st.sidebar.radio("Selecione a Página", list(PAGES.keys()))
page = PAGES[selection]

# ---------------------------------------------------------------------------
# Carrega dados
# ---------------------------------------------------------------------------
df_all = load_data()

st.markdown(
    '<div class="main-header">Polícia Rodoviária Federal — Sinistros 2022-2026</div>',
    unsafe_allow_html=True,
)

if df_all.empty:
    st.error("Base de dados não encontrada. Execute o pipeline de pré-processamento.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: filtros globais
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("Filtros")

anos_disp = sorted(df_all["ano"].dropna().unique().astype(int).tolist())
anos_sel = st.sidebar.multiselect("Anos", anos_disp, default=anos_disp)

regioes_disp = sorted(df_all["regiao"].dropna().unique().tolist())
regioes_sel = st.sidebar.multiselect("Regiões", regioes_disp, default=regioes_disp)

ufs_disp = sorted(
    df_all[df_all["regiao"].isin(regioes_sel)]["uf"].dropna().unique().tolist()
)
ufs_sel = st.sidebar.multiselect("UF", ufs_disp, default=ufs_disp)

grav_disp = sorted(df_all["classificacao_acidente"].dropna().unique().tolist())
grav_sel = st.sidebar.multiselect("Gravidade", grav_disp, default=grav_disp)

df = df_all[
    df_all["ano"].isin(anos_sel)
    & df_all["regiao"].isin(regioes_sel)
    & df_all["uf"].isin(ufs_sel)
    & df_all["classificacao_acidente"].isin(grav_sel)
].copy()

if len(df) == 0:
    st.info("Nenhum dado para os filtros selecionados.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs globais (topo de todas as páginas)
# ---------------------------------------------------------------------------
total = len(df)
mortos = int(df["mortos"].sum())
feridos = int(df["feridos"].sum())
sev = mortos / total * 100 if total > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total de Acidentes", f"{total:,}".replace(",", "."))
k2.metric("Mortos", f"{mortos:,}".replace(",", "."))
k3.metric("Feridos", f"{feridos:,}".replace(",", "."))
k4.metric("Taxa de Fatalidade", f"{sev:.2f}%".replace(".", ","))

st.markdown("---")

# ===========================================================================
# VISÃO GERAL
# ===========================================================================
if page == "geral":
    st.markdown(
        '<div class="sub-header">Visão Geral do Projeto</div>', unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Evolução Mensal de Acidentes")
        ev = df.copy()
        ev["Mes_Ano"] = ev["data_hora"].dt.to_period("M").dt.to_timestamp()
        ev_cnt = ev.groupby("Mes_Ano").size().reset_index(name="Quantidade")
        fig = fmt(
            px.line(
                ev_cnt,
                x="Mes_Ano",
                y="Quantidade",
                color_discrete_sequence=COR_1,
                labels={"Mes_Ano": "Mês/Ano", "Quantidade": "Acidentes"},
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Gravidade dos Sinistros")
        grav = df["classificacao_acidente"].value_counts().reset_index()
        grav.columns = ["Classificação", "Quantidade"]
        fig = fmt(
            px.pie(
                grav,
                values="Quantidade",
                names="Classificação",
                hole=0.4,
                color_discrete_sequence=PALETA,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    #### Objetivos do Estudo
    - **Identificar fatores de risco** comportamentais, temporais, espaciais, meteorológicos e de infraestrutura.
    - **Validar empiricamente 5 hipóteses** por meio de testes estatísticos rigorosos.
    - **Prover insights acionáveis** para fiscalização e alocação de recursos da PRF.

    *Use a barra lateral para navegar entre as hipóteses, o mapa e a análise exploratória.*
    """)

    c3, c4, c5 = st.columns(3)
    with c3:
        st.subheader("Acidentes por Região")
        reg = df["regiao"].value_counts().reset_index()
        reg.columns = ["Região", "Quantidade"]
        fig = fmt(
            px.bar(reg, x="Região", y="Quantidade", color_discrete_sequence=COR_1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Tipo de Pista")
        pista = df["tipo_pista"].value_counts().reset_index()
        pista.columns = ["Pista", "Quantidade"]
        fig = fmt(
            px.pie(
                pista,
                values="Quantidade",
                names="Pista",
                color_discrete_sequence=PALETA,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        st.subheader("Uso do Solo")
        uso = df["uso_solo"].value_counts().reset_index()
        uso.columns = ["Uso", "Quantidade"]
        fig = fmt(
            px.pie(
                uso,
                values="Quantidade",
                names="Uso",
                hole=0.4,
                color_discrete_sequence=PALETA,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# EDA
# ===========================================================================
elif page == "eda":
    st.markdown(
        '<div class="sub-header">Análise Exploratória de Dados</div>',
        unsafe_allow_html=True,
    )

    # --- Linha 1: Causas + UF ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Principais Causas (Top 10)")
        col_causa = (
            "causa_acidente_grupo"
            if "causa_acidente_grupo" in df.columns
            else "causa_acidente"
        )
        top = df[col_causa].value_counts().nlargest(10).reset_index()
        top.columns = ["Causa", "Quantidade"]
        fig = fmt(
            px.bar(
                top,
                x="Quantidade",
                y="Causa",
                orientation="h",
                color_discrete_sequence=COR_1,
            )
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Acidentes por UF")
        uf_cnt = df["uf"].value_counts().reset_index()
        uf_cnt.columns = ["UF", "Quantidade"]
        fig = fmt(px.bar(uf_cnt, x="UF", y="Quantidade", color_discrete_sequence=COR_1))
        st.plotly_chart(fig, use_container_width=True)

    # --- Linha 2: Tipos de acidente + Condição meteorológica ---
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Tipos de Acidente (Top 10)")
        tipos = df["tipo_acidente"].value_counts().nlargest(10).reset_index()
        tipos.columns = ["Tipo", "Quantidade"]
        fig = fmt(
            px.bar(
                tipos,
                x="Quantidade",
                y="Tipo",
                orientation="h",
                color_discrete_sequence=COR_1,
            )
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Condição Meteorológica")
        cond = df["condicao_meteorologica"].value_counts().reset_index()
        cond.columns = ["Condição", "Quantidade"]
        fig = fmt(
            px.pie(
                cond,
                values="Quantidade",
                names="Condição",
                hole=0.45,
                color_discrete_sequence=PALETA,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Linha 3: Fase do dia + BRs ---
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Fase do Dia")
        fase = df["fase_dia"].value_counts().reset_index()
        fase.columns = ["Fase", "Quantidade"]
        ordem_fase = [
            f
            for f in ["Pleno dia", "Amanhecer", "Anoitecer", "Plena Noite"]
            if f in fase["Fase"].values
        ]
        if ordem_fase:
            fase["Fase"] = pd.Categorical(
                fase["Fase"], categories=ordem_fase, ordered=True
            )
            fase = fase.sort_values("Fase")
        fig = fmt(px.bar(fase, x="Fase", y="Quantidade", color_discrete_sequence=COR_1))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.subheader("Rodovias com Mais Acidentes (Top 10)")
        brs = df.copy()
        brs["br"] = brs["br"].astype(str).str.replace(r"\.0$", "", regex=True)
        brs_cnt = brs["br"].value_counts().nlargest(10).reset_index()
        brs_cnt.columns = ["BR", "Quantidade"]
        brs_cnt["BR"] = "BR-" + brs_cnt["BR"]
        fig = fmt(
            px.bar(brs_cnt, x="BR", y="Quantidade", color_discrete_sequence=COR_1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap: Hora × Dia da Semana ---
    st.markdown("---")
    st.subheader("Padrão de Acidentes: Hora × Dia da Semana")
    heat = df.groupby(["dia_semana", "hora"]).size().reset_index(name="Quantidade")
    pivot = heat.pivot(index="dia_semana", columns="hora", values="Quantidade").fillna(
        0
    )
    # Reordena dias se possível
    dias_presentes = [d for d in _ORDEM_SEMANA if d in pivot.index]
    if dias_presentes:
        pivot = pivot.reindex(dias_presentes)
    fig_heat = px.imshow(
        pivot,
        labels=dict(y="Dia da Semana", x="Hora do Dia", color="Qtd"),
        color_continuous_scale=[[0, "#1E293B"], [0.5, "#FBBF24"], [1, "#F87171"]],
        aspect="auto",
    )
    fig_heat = fmt(fig_heat)
    fig_heat.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Taxa de fatalidade por tipo de pista (Visão rápida H5) ---
    st.markdown("---")
    st.subheader("Taxa de Fatalidade por Tipo de Pista e Região")
    taxa_pista_reg = taxa_df(df, "regiao", "tipo_pista")
    fig = fmt(
        px.bar(
            taxa_pista_reg,
            x="regiao",
            y="taxa",
            color="tipo_pista",
            barmode="group",
            color_discrete_sequence=PALETA,
            labels={
                "regiao": "Região",
                "taxa": "Taxa Fatalidade (%)",
                "tipo_pista": "Pista",
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# MAPA DE ACIDENTES
# ===========================================================================
elif page == "mapa":
    st.markdown(
        '<div class="sub-header">Mapa de Calor de Acidentes no Brasil</div>',
        unsafe_allow_html=True,
    )

    tipo_mapa = st.radio(
        "Exibir:",
        ["Todos os acidentes", "Apenas acidentes fatais"],
        horizontal=True,
    )

    df_mapa = df.copy()
    if tipo_mapa == "Apenas acidentes fatais":
        df_mapa = df_mapa[df_mapa["is_fatal"]]

    df_coords = df_mapa[["latitude", "longitude"]].dropna()

    if df_coords.empty:
        st.warning("Sem coordenadas para os filtros selecionados.")
        st.stop()

    # Agrega em células de 0,05° para reduzir ruído
    df_coords = df_coords.copy()
    df_coords["lat_r"] = df_coords["latitude"].round(2)
    df_coords["lon_r"] = df_coords["longitude"].round(2)
    dens = df_coords.groupby(["lat_r", "lon_r"]).size().reset_index(name="count")

    st.caption(
        f"{len(df_coords):,} registros com coordenadas válidas "
        f"({len(dens):,} células de ~0,05°)".replace(",", ".")
    )

    fig_mapa = go.Figure(
        go.Densitymapbox(
            lat=dens["lat_r"],
            lon=dens["lon_r"],
            z=dens["count"],
            radius=14,
            colorscale=[
                [0.00, "rgba(15,23,42,0)"],
                [0.15, "#1d4ed8"],
                [0.40, "#facc15"],
                [0.70, "#f97316"],
                [1.00, "#ef4444"],
            ],
            zmin=0,
            zmax=dens["count"].quantile(0.97),
            showscale=True,
            colorbar=dict(
                title=dict(text="Acidentes", font=dict(color="#94A3B8")),
                tickfont=dict(color="#94A3B8"),
                bgcolor="rgba(0,0,0,0)",
            ),
            hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Acidentes: %{z}<extra></extra>",
        )
    )
    fig_mapa.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=-15.5, lon=-51.5),
            zoom=3.6,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        margin=dict(l=0, r=0, t=0, b=0),
        height=580,
    )
    st.plotly_chart(fig_mapa, use_container_width=True)

    # Top-10 hotspots
    st.markdown("---")
    st.subheader("Top-20 Trechos de 5 km com Mais Sinistros Fatais")
    df_trechos = df_all[  # usa df_all para não filtrar por gravidade aqui
        df_all["ano"].isin(anos_sel)
        & df_all["regiao"].isin(regioes_sel)
        & df_all["uf"].isin(ufs_sel)
        & df_all["is_fatal"]
    ].copy()
    df_trechos["faixa_km"] = pd.to_numeric(df_trechos["faixa_km"], errors="coerce")
    df_trechos = df_trechos.dropna(subset=["faixa_km", "br"])
    df_trechos["id_trecho"] = (
        df_trechos["uf"]
        + " — BR-"
        + df_trechos["br"].astype(str).str.replace(r"\.0$", "", regex=True)
        + " (km "
        + df_trechos["faixa_km"].astype(int).astype(str)
        + ")"
    )
    top20 = (
        df_trechos.groupby("id_trecho")
        .size()
        .reset_index(name="Sinistros Fatais")
        .sort_values("Sinistros Fatais", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top20.index += 1
    fig_top = fmt(
        px.bar(
            top20,
            x="Sinistros Fatais",
            y="id_trecho",
            orientation="h",
            color_discrete_sequence=["#F87171"],
            labels={"id_trecho": "Trecho"},
        )
    )
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)

# ===========================================================================
# H1 — Causas Comportamentais
# ===========================================================================
elif page == "h1":
    hyp_card(
        "H1: Fatores Comportamentais e Gravidade Noturna",
        '<span class="badge-partial">Parcialmente Confirmada</span>',
    )
    st.markdown("""
    **Enunciado:** Fatores comportamentais de alto risco (álcool, sono, substâncias psicoativas,
    excesso de velocidade) concentram-se no período noturno e potencializam a gravidade dos acidentes.

    **Resultados:**
    - **Concentração noturna confirmada** (χ² = 8029,3; p < 0,001): infrações comportamentais são
      proporcionalmente maiores na madrugada e à noite.
    - **Potencialização da gravidade refutada**: à noite, causas comportamentais apresentam *menor*
      chance de letalidade que demais causas do mesmo período (OR = 0,662). Durante o dia, são mais
      letais que o referencial (OR = 1,280).
    """)

    p = img("fig1.jpg")
    if p:
        st.image(
            p,
            caption="Taxa de fatalidade por tipo de causa e período do dia",
            use_column_width=True,
        )

    st.markdown("---")
    st.subheader("Taxa de Fatalidade por Causa × Período do Dia")

    col_causa = (
        "causa_acidente_grupo"
        if "causa_acidente_grupo" in df.columns
        else "causa_acidente"
    )
    h1_g = taxa_df(df, col_causa, "horario")
    # mantém apenas causas com pelo menos 50 acidentes no grupo
    causas_validas = df[col_causa].value_counts()[lambda s: s >= 50].index.tolist()
    h1_g = h1_g[h1_g[col_causa].isin(causas_validas)]

    horarios_presentes = [h for h in _ORDEM_HORARIO if h in h1_g["horario"].values]
    if horarios_presentes:
        h1_g["horario"] = pd.Categorical(
            h1_g["horario"], categories=horarios_presentes, ordered=True
        )
        h1_g = h1_g.sort_values("horario")

    pivot_h1 = h1_g.pivot(index=col_causa, columns="horario", values="taxa").fillna(0)
    fig_h1 = px.imshow(
        pivot_h1,
        color_continuous_scale=[[0, "#1E293B"], [0.5, "#FBBF24"], [1, "#F87171"]],
        labels=dict(x="Período do Dia", y="Causa", color="Taxa Fatal. (%)"),
        aspect="auto",
        text_auto=".1f",
    )
    fig_h1 = fmt(fig_h1)
    fig_h1.update_layout(height=420)
    st.plotly_chart(fig_h1, use_container_width=True)

    st.subheader("Volume de Acidentes por Causa × Período do Dia")
    h1_vol = df.groupby([col_causa, "horario"]).size().reset_index(name="Quantidade")
    h1_vol = h1_vol[h1_vol[col_causa].isin(causas_validas)]
    if horarios_presentes:
        h1_vol["horario"] = pd.Categorical(
            h1_vol["horario"], categories=horarios_presentes, ordered=True
        )
    fig_vol = fmt(
        px.bar(
            h1_vol.sort_values("horario"),
            x=col_causa,
            y="Quantidade",
            color="horario",
            barmode="group",
            color_discrete_sequence=PALETA,
            labels={col_causa: "Causa", "horario": "Período"},
        )
    )
    fig_vol.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_vol, use_container_width=True)

# ===========================================================================
# H2 — Sazonalidade e Feriados
# ===========================================================================
elif page == "h2":
    hyp_card(
        "H2: Sazonalidade, Fins de Semana e Feriados",
        '<span class="badge-partial">Parcialmente Confirmada</span>',
    )
    st.markdown("""
    **Enunciado:** A proporção de sinistros com vítimas fatais é significativamente maior em
    finais de semana, feriados nacionais e meses de férias.

    **Resultados:**
    - **Fins de semana e feriados confirmados:** taxa fatal salta de 6,62% (dias normais) para
      8,32% (fins de semana) e 7,84% (feriados).
    - **Meses de férias refutado:** Janeiro, Julho e Dezembro elevam o **volume** de sinistros,
      mas a proporção fatal não muda estatisticamente (7,22% vs 7,16%).
    """)

    p = img("fig2.jpg")
    if p:
        st.image(
            p,
            caption="Decomposição STL: componente sazonal da frequência e proporção fatal",
            use_column_width=True,
        )

    st.markdown("---")
    st.subheader("Taxa de Fatalidade por Tipo de Dia")

    df_h2 = df.copy()
    cond_list = [df_h2["feriado_nacional"] == 1, df_h2["fim_de_semana"] == 1]
    choices = ["Feriado Nacional", "Fim de Semana"]
    df_h2["tipo_dia"] = np.select(cond_list, choices, default="Dia Útil")
    taxa_tipo = (
        df_h2.groupby("tipo_dia")["is_fatal"]
        .agg(total="count", fatais="sum")
        .reset_index()
    )
    taxa_tipo["taxa"] = taxa_tipo["fatais"] / taxa_tipo["total"] * 100
    taxa_tipo = taxa_tipo.sort_values("taxa", ascending=False)

    fig_td = fmt(
        px.bar(
            taxa_tipo,
            x="tipo_dia",
            y="taxa",
            text=taxa_tipo["taxa"].apply(lambda v: f"{v:.2f}%"),
            color="tipo_dia",
            color_discrete_sequence=PALETA,
            labels={"tipo_dia": "Tipo de Dia", "taxa": "Taxa de Fatalidade (%)"},
        )
    )
    fig_td.update_traces(textposition="outside")
    fig_td.update_layout(showlegend=False)
    st.plotly_chart(fig_td, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Taxa de Fatalidade por Mês")
        taxa_mes = taxa_df(df, "mes")
        taxa_mes = taxa_mes.sort_values("mes")
        fig = fmt(
            px.line(
                taxa_mes,
                x="mes",
                y="taxa",
                markers=True,
                color_discrete_sequence=COR_1,
                labels={"mes": "Mês", "taxa": "Taxa Fatal. (%)"},
            )
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Taxa de Fatalidade por Estação")
        if "estacao" in df.columns:
            taxa_est = taxa_df(df, "estacao").sort_values("taxa", ascending=False)
            fig = fmt(
                px.bar(
                    taxa_est,
                    x="estacao",
                    y="taxa",
                    text=taxa_est["taxa"].apply(lambda v: f"{v:.2f}%"),
                    color_discrete_sequence=COR_1,
                    labels={"estacao": "Estação", "taxa": "Taxa Fatal. (%)"},
                )
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Volume de Acidentes por Horário do Dia (hora exata)")
    vol_hora = df.groupby("hora").size().reset_index(name="Quantidade")
    fig_hora = fmt(
        px.bar(
            vol_hora,
            x="hora",
            y="Quantidade",
            color_discrete_sequence=COR_1,
            labels={"hora": "Hora do Dia"},
        )
    )
    fig_hora.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_hora, use_container_width=True)

# ===========================================================================
# H3 — Fatores Meteorológicos
# ===========================================================================
elif page == "h3":
    hyp_card(
        "H3: Condições Meteorológicas Adversas",
        '<span class="badge-refuted">Refutada</span>',
    )
    st.markdown("""
    **Enunciado:** Condições meteorológicas adversas (chuva, neblina, etc.) elevam a probabilidade
    de acidentes fatais por redução de visibilidade e aderência.

    **Resultados:**
    - Hipótese **refutada** — fenômeno de *risk compensation* (compensação de risco).
    - Em condições adversas: taxa fatal = **6,47%** vs céu claro = **7,30%** (OR = 0,879; p < 0,001).
    - Efeito homogêneo em todas as regiões, descartando anomalias geográficas.
    """)

    st.markdown("---")
    st.subheader("Taxa de Fatalidade por Condição Meteorológica")

    taxa_meteo = taxa_df(df, "condicao_meteorologica").sort_values("taxa")
    taxa_meteo["cor"] = taxa_meteo["condicao_meteorologica"].apply(
        lambda v: (
            "#F87171" if "Ceu Claro" in str(v) or "Céu Claro" in str(v) else "#60A5FA"
        )
    )
    fig = fmt(
        px.bar(
            taxa_meteo,
            x="taxa",
            y="condicao_meteorologica",
            orientation="h",
            text=taxa_meteo["taxa"].apply(lambda v: f"{v:.2f}%"),
            color="condicao_meteorologica",
            color_discrete_sequence=PALETA,
            labels={"condicao_meteorologica": "Condição", "taxa": "Taxa Fatal. (%)"},
        )
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Taxa de Fatalidade por Condição Meteorológica × Região")
    taxa_meteo_reg = taxa_df(df, "regiao", "condicao_meteorologica")
    # mantém apenas condições com ≥ 100 acidentes por região
    cond_validas = (
        df.groupby("condicao_meteorologica").size()[lambda s: s >= 100].index.tolist()
    )
    taxa_meteo_reg = taxa_meteo_reg[
        taxa_meteo_reg["condicao_meteorologica"].isin(cond_validas)
    ]
    pivot_meteo = taxa_meteo_reg.pivot(
        index="condicao_meteorologica", columns="regiao", values="taxa"
    ).fillna(0)
    fig_m = px.imshow(
        pivot_meteo,
        color_continuous_scale=[[0, "#1E293B"], [0.5, "#FBBF24"], [1, "#F87171"]],
        labels=dict(x="Região", y="Condição Meteorológica", color="Taxa Fatal. (%)"),
        text_auto=".1f",
        aspect="auto",
    )
    fig_m = fmt(fig_m)
    st.plotly_chart(fig_m, use_container_width=True)

# ===========================================================================
# H4 — Concentração Espacial
# ===========================================================================
elif page == "h4":
    hyp_card(
        "H4: Concentração Espacial de Fatalidades",
        '<span class="badge-confirmed">Confirmada</span>',
    )
    st.markdown("""
    **Enunciado:** A letalidade viária não é espacialmente homogênea, estando concentrada em
    segmentos críticos ("hotspots") da malha rodoviária.

    **Resultados:**
    - Curva de Lorenz espacial demonstrou altíssima concentração.
    - **10%** dos trechos mais críticos acumulam **41,3%** de todas as ocorrências fatais.
    - BR-381/SP (km 80–85) destaca-se isoladamente com 63 acidentes fatais em 10 km.
    """)

    p = img("fig3.jpg")
    if p:
        st.image(
            p,
            caption="Curva de Lorenz espacial (faixas de 5 km) e mapa de calor hexagonal dos sinistros fatais",
            use_column_width=True,
        )

    st.markdown("---")
    st.subheader("Top-20 Trechos de 5 km com Mais Sinistros Fatais")

    df_sp = df[df["is_fatal"]].copy()
    df_sp["faixa_km"] = pd.to_numeric(df_sp["faixa_km"], errors="coerce")
    df_sp = df_sp.dropna(subset=["faixa_km", "br"])
    df_sp["id_trecho"] = (
        df_sp["uf"]
        + " — BR-"
        + df_sp["br"].astype(str).str.replace(r"\.0$", "", regex=True)
        + " (km "
        + df_sp["faixa_km"].astype(int).astype(str)
        + ")"
    )
    top20_h4 = (
        df_sp.groupby("id_trecho")
        .size()
        .reset_index(name="Sinistros Fatais")
        .sort_values("Sinistros Fatais", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top20_h4.index += 1

    fig_sp = fmt(
        px.bar(
            top20_h4,
            x="Sinistros Fatais",
            y="id_trecho",
            orientation="h",
            color_discrete_sequence=["#F87171"],
            text="Sinistros Fatais",
            labels={"id_trecho": "Trecho (5 km)"},
        )
    )
    fig_sp.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_sp.update_traces(textposition="outside")
    st.plotly_chart(fig_sp, use_container_width=True)

    st.dataframe(top20_h4, use_container_width=True)

# ===========================================================================
# H5 — Impacto Regional / Infraestrutura
# ===========================================================================
elif page == "h5":
    hyp_card(
        "H5: Infraestrutura Viária e Desigualdade Regional",
        '<span class="badge-confirmed">Confirmada</span>',
    )
    st.markdown("""
    **Enunciado:** A falta de duplicação viária (pista simples) é o vetor estrutural que explica
    as disparidades na letalidade dos sinistros entre as regiões brasileiras.

    **Resultados** (χ² = 2023,65; p < 0,001; V de Cramér = 0,082):
    - Pistas simples: taxa fatal = **9,88%** (> 2× pistas duplas: 4,71%).
    - Norte (69,9% simples → 11,02% fatal) e Nordeste (55,5% simples → 10,33% fatal) lideram.
    - Sudeste (40,4% simples → 5,74%) e Sul (46,9% simples → 5,48%) têm menor exposição.
    - Post-hoc Bonferroni (k = 10): apenas Norte–Nordeste (p_adj = 0,111) e Sudeste–Sul
      (p_adj = 0,182) não apresentam diferença significativa entre si.
    """)

    p = img("fig4.jpg")
    if p:
        st.image(
            p,
            caption="Taxa de fatalidade por região e tipo de pista",
            use_column_width=True,
        )

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Taxa de Fatalidade por Região × Tipo de Pista")
        taxa_rp = taxa_df(df, "regiao", "tipo_pista")
        fig = fmt(
            px.bar(
                taxa_rp,
                x="regiao",
                y="taxa",
                color="tipo_pista",
                barmode="group",
                text=taxa_rp["taxa"].apply(lambda v: f"{v:.1f}%"),
                color_discrete_sequence=PALETA,
                labels={
                    "regiao": "Região",
                    "taxa": "Taxa Fatal. (%)",
                    "tipo_pista": "Pista",
                },
            )
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Proporção de Pista Simples por Região")
        pista_reg = df.groupby(["regiao", "tipo_pista"]).size().reset_index(name="n")
        total_reg = df.groupby("regiao").size().reset_index(name="total")
        pista_reg = pista_reg.merge(total_reg, on="regiao")
        pista_reg["pct"] = pista_reg["n"] / pista_reg["total"] * 100
        pista_simples = pista_reg[pista_reg["tipo_pista"] == "Simples"].sort_values(
            "pct", ascending=False
        )
        fig = fmt(
            px.bar(
                pista_simples,
                x="regiao",
                y="pct",
                text=pista_simples["pct"].apply(lambda v: f"{v:.1f}%"),
                color_discrete_sequence=["#F87171"],
                labels={"regiao": "Região", "pct": "% em Pista Simples"},
            )
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap: Taxa de Fatalidade — Região × Tipo de Pista")
    pivot_h5 = taxa_rp.pivot(
        index="tipo_pista", columns="regiao", values="taxa"
    ).fillna(0)
    fig_hm = px.imshow(
        pivot_h5,
        color_continuous_scale=[[0, "#1E293B"], [0.5, "#FBBF24"], [1, "#F87171"]],
        labels=dict(x="Região", y="Tipo de Pista", color="Taxa Fatal. (%)"),
        text_auto=".2f",
        aspect="auto",
    )
    fig_hm = fmt(fig_hm)
    fig_hm.update_layout(height=280)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Distribuição de Acidentes por Tipo de Pista × Região")
    dist_pista = (
        df.groupby(["regiao", "tipo_pista"]).size().reset_index(name="Quantidade")
    )
    fig_dist = fmt(
        px.bar(
            dist_pista,
            x="regiao",
            y="Quantidade",
            color="tipo_pista",
            barmode="stack",
            color_discrete_sequence=PALETA,
            labels={"regiao": "Região", "tipo_pista": "Pista"},
        )
    )
    st.plotly_chart(fig_dist, use_container_width=True)
