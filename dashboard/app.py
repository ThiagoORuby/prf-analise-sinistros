import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Painel PRF: Sinistros 2022-2026",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0F172A; }
    [data-testid="stSidebar"] { background-color: #1E293B; }
    .main-header { font-size: 2.5rem; color: #F8FAFC; font-weight: 700; margin-bottom: 0px; }
    .sub-header { font-size: 1.2rem; color: #94A3B8; margin-bottom: 20px; }
    div[data-testid="stMetricValue"] { color: #60A5FA; }
    div[data-testid="stMetricLabel"] { color: #94A3B8; }
    h1, h2, h3, p, label { color: #F8FAFC !important; }
</style>
""", unsafe_allow_html=True)

PALETA_CATEGORICA = ['#60A5FA', '#3B82F6', '#FBBF24', '#F87171', '#34D399', '#A78BFA', '#94A3B8']
COR_PRINCIPAL = [PALETA_CATEGORICA[0]]

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('datatran_2022_2026_processed_v1.csv')
        df['data_hora'] = pd.to_datetime(df['data_hora'], errors='coerce')
        df['ano'] = df['data_hora'].dt.year
        df['hora'] = df['data_hora'].dt.hour
        df['dia_semana'] = df['data_hora'].dt.day_name(locale='pt_BR.utf8') if hasattr(df['data_hora'].dt, 'day_name') else df['data_hora'].dt.dayofweek
        return df
    except Exception:
        return pd.DataFrame()

data = load_data()

if data.empty:
    st.warning('Nenhum dado encontrado.')
    st.stop()

st.sidebar.title("Filtros")
anos_disponiveis = sorted(data['ano'].dropna().unique().astype(int).tolist())
anos_selecionados = st.sidebar.multiselect('Anos', anos_disponiveis, default=anos_disponiveis)
regioes_disponiveis = sorted(data['regiao'].dropna().unique().tolist())
regioes_selecionadas = st.sidebar.multiselect('Regiões', regioes_disponiveis, default=regioes_disponiveis)
ufs_disponiveis = sorted(data[data['regiao'].isin(regioes_selecionadas)]['uf'].dropna().unique().tolist())
ufs_selecionadas = st.sidebar.multiselect('UF', ufs_disponiveis, default=ufs_disponiveis)

df_filtrado = data[
    (data['ano'].isin(anos_selecionados)) &
    (data['regiao'].isin(regioes_selecionadas)) &
    (data['uf'].isin(ufs_selecionadas))
]

st.markdown('<div class="main-header">Painel de Sinistros PRF (2022-2026)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análise interativa dos dados em rodovias federais.</div>', unsafe_allow_html=True)

total_acidentes = len(df_filtrado)
total_mortos = df_filtrado['mortos'].sum() if 'mortos' in df_filtrado.columns else 0
total_feridos = df_filtrado['feridos'].sum() if 'feridos' in df_filtrado.columns else 0
total_vitimas = total_mortos + total_feridos
severidade = (total_mortos / total_acidentes * 100) if total_acidentes > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Acidentes", f"{total_acidentes:,}".replace(',', '.'))
col2.metric("Total Vítimas", f"{total_vitimas:,}".replace(',', '.'))
col3.metric("Total Mortos", f"{total_mortos:,}".replace(',', '.'))
col4.metric("Severidade", f"{severidade:.2f}".replace('.', ','))

st.markdown("---")

if total_acidentes == 0:
    st.info("Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

def formatar_figura(fig):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#F8FAFC")
    fig.update_xaxes(showgrid=False, color="#94A3B8")
    fig.update_yaxes(showgrid=True, gridcolor="#334155", color="#94A3B8")
    return fig

c1, c2 = st.columns(2)
with c1:
    st.subheader("Evolução Temporal")
    df_temp = df_filtrado.copy()
    df_temp['Mes_Ano'] = df_temp['data_hora'].dt.to_period('M')
    evolucao = df_temp.groupby('Mes_Ano').size().reset_index(name='Quantidade')
    evolucao['Mes_Ano'] = evolucao['Mes_Ano'].dt.to_timestamp()
    fig = formatar_figura(px.line(evolucao, x='Mes_Ano', y='Quantidade', color_discrete_sequence=COR_PRINCIPAL))
    st.plotly_chart(fig, width="stretch")

with c2:
    st.subheader("Gravidade")
    if 'classificacao_acidente' in df_filtrado.columns:
        gravidade = df_filtrado['classificacao_acidente'].value_counts().reset_index()
        gravidade.columns = ['Classificação', 'Quantidade']
        fig = formatar_figura(px.pie(gravidade, values='Quantidade', names='Classificação', hole=0.4, color_discrete_sequence=PALETA_CATEGORICA))
        st.plotly_chart(fig, width="stretch")

c3, c4 = st.columns(2)
with c3:
    st.subheader("Principais Causas")
    coluna = 'causa_acidente_grupo' if 'causa_acidente_grupo' in df_filtrado.columns else 'causa_acidente'
    if coluna in df_filtrado.columns:
        top = df_filtrado[coluna].value_counts().nlargest(10).reset_index()
        top.columns = ['Causa', 'Quantidade']
        fig = formatar_figura(px.bar(top, x='Quantidade', y='Causa', orientation='h', color_discrete_sequence=COR_PRINCIPAL))
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width="stretch")

with c4:
    st.subheader("Acidentes por UF")
    acidentes_uf = df_filtrado['uf'].value_counts().reset_index()
    acidentes_uf.columns = ['UF', 'Quantidade']
    fig = formatar_figura(px.bar(acidentes_uf, x='UF', y='Quantidade', color_discrete_sequence=COR_PRINCIPAL))
    st.plotly_chart(fig, width="stretch")

c5, c6 = st.columns(2)
with c5:
    st.subheader("Tipos de Acidente")
    if 'tipo_acidente' in df_filtrado.columns:
        tipos = df_filtrado['tipo_acidente'].value_counts().nlargest(10).reset_index()
        tipos.columns = ['Tipo', 'Quantidade']
        fig = formatar_figura(px.bar(tipos, x='Quantidade', y='Tipo', orientation='h', color_discrete_sequence=COR_PRINCIPAL))
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width="stretch")

with c6:
    st.subheader("Condição Meteorológica")
    if 'condicao_meteorologica' in df_filtrado.columns:
        cond = df_filtrado['condicao_meteorologica'].value_counts().reset_index()
        cond.columns = ['Condição', 'Quantidade']
        fig = formatar_figura(px.pie(cond, values='Quantidade', names='Condição', hole=0.5, color_discrete_sequence=PALETA_CATEGORICA))
        st.plotly_chart(fig, width="stretch")

c7, c8 = st.columns(2)
with c7:
    st.subheader("Fase do Dia")
    if 'fase_dia' in df_filtrado.columns:
        fase = df_filtrado['fase_dia'].value_counts().reset_index()
        fase.columns = ['Fase', 'Quantidade']
        fig = formatar_figura(px.bar(fase, x='Fase', y='Quantidade', color_discrete_sequence=COR_PRINCIPAL))
        st.plotly_chart(fig, width="stretch")

with c8:
    st.subheader("Tipo de Pista")
    if 'tipo_pista' in df_filtrado.columns:
        pista = df_filtrado['tipo_pista'].value_counts().reset_index()
        pista.columns = ['Pista', 'Quantidade']
        fig = formatar_figura(px.pie(pista, values='Quantidade', names='Pista', color_discrete_sequence=PALETA_CATEGORICA))
        st.plotly_chart(fig, width="stretch")

c9, c10 = st.columns(2)
with c9:
    st.subheader("Uso do Solo (Urbano x Rural)")
    if 'uso_solo' in df_filtrado.columns:
        uso = df_filtrado['uso_solo'].value_counts().reset_index()
        uso.columns = ['Uso', 'Quantidade']
        fig = formatar_figura(px.pie(uso, values='Quantidade', names='Uso', hole=0.4, color_discrete_sequence=PALETA_CATEGORICA))
        st.plotly_chart(fig, width="stretch")

with c10:
    st.subheader("Rodovias com Mais Acidentes (BRs)")
    if 'br' in df_filtrado.columns:
        df_temp2 = df_filtrado.copy()
        df_temp2['br'] = df_temp2['br'].astype(str).str.replace(r'\.0$', '', regex=True)
        brs = df_temp2['br'].value_counts().nlargest(10).reset_index()
        brs.columns = ['BR', 'Quantidade']
        brs['BR'] = "BR-" + brs['BR']
        fig = formatar_figura(px.bar(brs, x='BR', y='Quantidade', color_discrete_sequence=COR_PRINCIPAL))
        st.plotly_chart(fig, width="stretch")

if 'dia_semana' in df_filtrado.columns and 'hora' in df_filtrado.columns:
    st.markdown("---")
    st.subheader("Padrão de Acidentes: Hora vs Dia da Semana")
    heat = df_filtrado.groupby(['dia_semana', 'hora']).size().reset_index(name='Quantidade')
    pivot = heat.pivot(index='dia_semana', columns='hora', values='Quantidade').fillna(0)
    fig = px.imshow(pivot, labels=dict(y="Dia da Semana", x="Hora do Dia", color="Quantidade"),
                    color_continuous_scale=[[0, '#1E293B'], [1, '#F87171']], aspect="auto")
    fig = formatar_figura(fig)
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch")

if 'latitude' in df_filtrado.columns and 'longitude' in df_filtrado.columns:
    st.markdown("---")
    st.subheader("Mapa de Calor de Acidentes no Brasil")

    import plotly.graph_objects as go

    df_mapa = (
        df_filtrado[['latitude', 'longitude']]
        .dropna()
        .rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    )

    # Agrupa em células de ~0,05° para reduzir ruído e melhorar a densidade
    df_mapa['lat_r'] = df_mapa['lat'].round(2)
    df_mapa['lon_r'] = df_mapa['lon'].round(2)
    densidade = df_mapa.groupby(['lat_r', 'lon_r']).size().reset_index(name='count')

    fig_mapa = go.Figure(go.Densitymapbox(
        lat=densidade['lat_r'],
        lon=densidade['lon_r'],
        z=densidade['count'],
        radius=12,
        colorscale=[
            [0.0,  'rgba(15,23,42,0)'],
            [0.15, '#1d4ed8'],
            [0.40, '#facc15'],
            [0.70, '#f97316'],
            [1.0,  '#ef4444'],
        ],
        zmin=0,
        zmax=densidade['count'].quantile(0.97),   # evita que poucos pontos extremos "apaguem" a variação
        showscale=True,
        colorbar=dict(
            title=dict(text="Acidentes", font=dict(color="#94A3B8")),
            tickfont=dict(color="#94A3B8"),
            bgcolor='rgba(0,0,0,0)',
        ),
        hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Qtd: %{z}<extra></extra>",
    ))

    fig_mapa.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=-15.5, lon=-47.5),
            zoom=3.5,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#F8FAFC",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )

    st.plotly_chart(fig_mapa, use_container_width=True)