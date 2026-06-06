import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa a base de dados de sinistros da PRF.

    Esta função remove metadados administrativos irrelevantes, trata registros
    com valores ausentes na coluna classificacao_acidente, converte colunas
    geográficas de string para numérico e remove registros geograficamente
    inconsistentes (como km <= 0).

    :param df: O DataFrame bruto com os dados de sinistros da PRF.
    :type df: pd.DataFrame
    :return: O DataFrame limpo e com as colunas corrigidas.
    :rtype: pd.DataFrame
    """
    df_clean = df.copy()

    # Remove colunas administrativas irrelevantes
    columns_to_remove = ["id", "regional", "delegacia", "uop"]
    df_clean = df_clean.drop(columns=columns_to_remove, errors="ignore")

    # Remove linhas duplicadas mascaradas pelo id
    df_clean = df_clean.drop_duplicates()

    # Remove valores nulos na variável alvo classificacao_acidente
    df_clean = df_clean.dropna(subset=["classificacao_acidente"])

    # Converte colunas com separador em vírgula para tipo numérico correto
    for col in ["km", "latitude", "longitude"]:
        if col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace(",", ".")
                )

    # Remove valores de km inconsistentes (ex: km <= 0)
    if "km" in df_clean.columns:
        df_clean = df_clean[df_clean["km"] > 0]

    # Corrige a ortografia do nome da coluna de condição meteorológica
    if "condicao_metereologica" in df_clean.columns:
        df_clean = df_clean.rename(
            columns={"condicao_metereologica": "condicao_meteorologica"}  # type: ignore
        )

    return df_clean  # type: ignore
