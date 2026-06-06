from datetime import timedelta

import holidays
import pandas as pd
from dateutil.easter import easter
from unidecode import unidecode

from src.utils.constants import CAUSE_MAPPING, UF_TO_REGION


def one_hot_encode_tracado_via(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis indicadoras para características do traçado da via.

    Decompõe as classificações multivaloradas separadas por ponto e vírgula na
    coluna 'tracado_via' em colunas binárias independentes e gera uma variável
    auxiliar para contabilizar o número de características concorrentes.

    :param df: DataFrame de entrada contendo a coluna 'tracado_via'.
    :type df: pd.DataFrame
    :return: DataFrame com o traçado de via codificado e coluna original removida.
    :rtype: pd.DataFrame
    """
    df_encoded = df.copy()
    if "tracado_via" not in df_encoded.columns:
        return df_encoded

    road_layouts = df_encoded["tracado_via"].dropna().unique()
    unique_layouts = set(";".join(road_layouts).split(";"))

    for layout in unique_layouts:
        layout_cleaned = layout.strip()
        if not layout_cleaned:
            continue
        col_name = f"tr_{unidecode(layout_cleaned.lower().replace(' ', '_'))}"
        df_encoded[col_name] = df_encoded["tracado_via"].apply(
            lambda x, lc=layout_cleaned: (
                1
                if isinstance(x, str)
                and lc in [t.strip() for t in x.split(";") if t.strip()]
                else 0
            )
        )

    cols_tr = [c for c in df_encoded.columns if c.startswith("tr_")]
    df_encoded["n_caracteristicas_tracado"] = df_encoded[cols_tr].sum(axis=1)

    df_encoded = df_encoded.drop(columns=["tracado_via"])
    return df_encoded


def group_causa_acidente(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida as causas de acidentes em grupos semânticos/funcionais.

    Agrupa as 76 causas originais de acidentes nos 8 grandes grupos causais
    definidos no mapeamento para evitar problemas de esparsidade dos dados.

    :param df: DataFrame de entrada contendo a coluna 'causa_acidente'.
    :type df: pd.DataFrame
    :return: DataFrame contendo a nova variável 'causa_acidente_grupo'.
    :rtype: pd.DataFrame
    """
    df_mapped = df.copy()
    if "causa_acidente" not in df_mapped.columns:
        return df_mapped

    df_mapped["causa_acidente_grupo"] = df_mapped["causa_acidente"].map(CAUSE_MAPPING)  # type: ignore

    # Mapeia causas não listadas no dicionário para a categoria 'outros'
    df_mapped["causa_acidente_grupo"] = df_mapped["causa_acidente_grupo"].fillna(
        "outros"
    )

    return df_mapped


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a feature indicativa de feriados nacionais fixos e móveis no Brasil.

    Calcula dinamicamente os feriados móveis calculados a partir da Páscoa (Carnaval,
    Sexta-feira Santa, Corpus Christi) e combina-os com os feriados nacionais fixos.

    :param df: DataFrame contendo a coluna datetime 'data_hora'.
    :type df: pd.DataFrame
    :return: DataFrame com a nova coluna binária 'feriado_nacional'.
    :rtype: pd.DataFrame
    """
    df_holidays = df.copy()
    if "data_hora" not in df_holidays.columns:
        return df_holidays

    years = df_holidays["data_hora"].dt.year.dropna().unique()
    years = [int(y) for y in years]

    if not years:
        df_holidays["feriado_nacional"] = 0
        return df_holidays

    def get_mobile_holidays(years_list):
        dates = []
        for year in years_list:
            pascoa = easter(year)
            dates += [
                pascoa - timedelta(days=48),  # Segunda de Carnaval
                pascoa - timedelta(days=47),  # Terça de Carnaval
                pascoa - timedelta(days=2),  # Sexta-feira Santa
                pascoa + timedelta(days=60),  # Corpus Christi
            ]
        return set(dates)

    mobile_holidays = get_mobile_holidays(years)
    fixed_holidays = holidays.Brazil(years=years)

    all_holidays = set(fixed_holidays.keys()) | mobile_holidays

    df_holidays["feriado_nacional"] = (
        df_holidays["data_hora"].dt.date.isin(all_holidays).astype(int)
    )
    return df_holidays


def get_season(month: int) -> str:
    """Identifica a estação do ano associada ao mês no hemisfério sul.

    :param month: O número correspondente ao mês (1-12).
    :type month: int
    :return: Nome da estação do ano (Verão, Outono, Inverno, Primavera).
    :rtype: str
    """
    if month in [12, 1, 2]:
        return "Verão"
    elif month in [3, 4, 5]:
        return "Outono"
    elif month in [6, 7, 8]:
        return "Inverno"
    else:
        return "Primavera"


def get_time_of_day_category(hour: int) -> str:
    """Categoriza a hora do dia em uma das quatro faixas horárias do projeto.

    As faixas horárias são: Madrugada (0h-5h59), Manhã (6h-11h59), Tarde (12h-17h59)
    e Noite (18h-23h59).

    :param hour: Hora do dia (0 a 23).
    :type hour: int
    :return: A faixa horária correspondente.
    :rtype: str
    """
    if 0 <= hour < 6:
        return "Madrugada"
    elif 6 <= hour < 12:
        return "Manhã"
    elif 12 <= hour < 18:
        return "Tarde"
    else:
        return "Noite"


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Executa a engenharia de features completa nos dados limpos da PRF.

    Esta função combina campos de data e horário, calcula variáveis derivadas
    (estação, faixa horária, feriado, fim de semana, região geográfica),
    decompõe a coluna multilabel de traçado e consolida causas de acidentes.

    :param df: DataFrame com os dados previamente limpos.
    :type df: pd.DataFrame
    :return: DataFrame contendo todas as variáveis derivadas calculadas.
    :rtype: pd.DataFrame
    """
    df_features = df.copy()

    # Criação do campo data_hora unificado
    if "data_inversa" in df_features.columns and "horario" in df_features.columns:
        df_features["data_hora"] = pd.to_datetime(
            df_features["data_inversa"] + " " + df_features["horario"],
            errors="coerce",
        )
        df_features = df_features.drop(columns=["data_inversa", "horario"])

    # Calcula pessoas não classificadas (total de pessoas - soma de desfechos
    # individuais)
    outcome_cols = ["feridos", "mortos", "ilesos", "ignorados"]
    if "pessoas" in df_features.columns and all(
        c in df_features.columns for c in outcome_cols
    ):
        df_features["nao_classificados"] = df_features["pessoas"] - df_features[
            outcome_cols
        ].sum(axis=1)

    # Decompõe características de traçado da via
    df_features = one_hot_encode_tracado_via(df_features)

    # Aplica agrupamento semântico nas causas
    df_features = group_causa_acidente(df_features)

    # Adiciona a feature de feriados nacionais
    df_features = add_holiday_features(df_features)

    # Criação de features temporais adicionais
    if "data_hora" in df_features.columns:
        df_features["mes"] = df_features["data_hora"].dt.month
        df_features["estacao"] = df_features["mes"].apply(get_season)
        df_features["horario"] = df_features["data_hora"].dt.hour.apply(
            get_time_of_day_category
        )
        df_features["fim_de_semana"] = (
            df_features["data_hora"].dt.weekday.isin([5, 6]).astype(int)
        )

    # Mapeamento de região baseado no estado (UF)
    if "uf" in df_features.columns:
        df_features["regiao"] = df_features["uf"].map(UF_TO_REGION)  # type: ignore

    # Discretização da quilometragem (KM) em faixas de 5km
    if "km" in df_features.columns:
        df_features["faixa_km"] = (df_features["km"] // 5) * 5

    return df_features
