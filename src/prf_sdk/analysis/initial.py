from typing import Any

import numpy as np
import pandas as pd

from prf_sdk.utils.stats import wilson_interval


def diagnose_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza um diagnóstico estrutural e estatístico inicial da base de dados.

    Exibe informações de dimensões, contagem de linhas duplicadas (desconsiderando
    a coluna 'id') e monta um DataFrame estatístico de resumo contendo o tipo
    de dado, contagem de valores nulos e cardinalidade de cada coluna.

    :param df: DataFrame bruto a ser diagnosticado.
    :type df: pd.DataFrame
    :return: DataFrame contendo tipo, nulos e cardinalidade de cada coluna.
    :rtype: pd.DataFrame
    """
    rows, cols = df.shape
    print("======= DIAGNÓSTICO ESTRUTURAL DA BASE =======")
    print(f"Dimensões dos dados: {rows} linhas, {cols} colunas")

    # Remove a coluna id se existir para contagem correta de registros duplicados
    df_temp = df.drop(columns=["id"], errors="ignore")
    duplicates = df_temp.duplicated().sum()
    print(f"Registros duplicados (excluindo coluna 'id'): {duplicates}\n")

    summary_df = pd.DataFrame(
        {
            "data_type": df.dtypes.astype(str),
            "missing_values": df.isnull().sum(),
            "cardinality": df.nunique(),
        }
    )

    return summary_df


def evaluate_target_sensitivity(df: pd.DataFrame, target: str) -> None:
    """Aplica análise de sensibilidade teórica para remoção de nulos.

    Analisa o impacto de remover registros nulos na variável alvo
    em relação à classe crítica (de menor frequência amostral),
    confrontando esse impacto máximo com o erro padrão da amostra. Se o
    impacto for menor que o erro padrão, a remoção é aceita estatisticamente.

    :param df: DataFrame com os dados brutos.
    :type df: pd.DataFrame
    :param target: Nome da variável alvo de análise.
    :type target: str
    :raises KeyError: Se a coluna alvo não existir no DataFrame.
    """
    if target not in df.columns:
        raise KeyError(f"A variável alvo '{target}' não está presente no DataFrame.")

    total = len(df)
    missing_count = df[target].isna().sum()
    df_valid = df.dropna(subset=[target])
    valid_count = len(df_valid)

    if missing_count == 0:
        print(f"## AVALIAÇÃO DA VARIÁVEL ALVO: {target} ##")
        print("Nenhum registro com valor nulo encontrado nesta variável.\n")
        return

    # Pega classe crítica (menor proporção na base)
    critic_class = df[target].value_counts(sort=True, ascending=True).index[0]
    critic_count = (df[target] == critic_class).sum()
    sample_proportion = critic_count / valid_count

    # Calcula Erro Padrão da proporção amostral
    standard_error = np.sqrt(
        (sample_proportion * (1 - sample_proportion)) / valid_count
    )

    # CASO 1: todos os nulos pertencem à classe crítica
    max_proportion = (critic_count + missing_count) / total
    # CASO 2: nenhum nulo pertence à classe crítica
    min_proportion = critic_count / total

    max_impact = max(
        abs(max_proportion - sample_proportion),
        abs(sample_proportion - min_proportion),
    )
    can_remove = max_impact < standard_error

    print(f"## AVALIAÇÃO DA VARIÁVEL ALVO: {target} ##")
    print(f"Classe crítica identificada: '{critic_class}'")
    print(
        f"Valores nulos: {missing_count} de {total} "
        f"({missing_count / total * 100:.4f}%)"
    )
    print(f"Erro padrão da amostra (SE): {standard_error * 100:.5f} p.p.")
    print(
        f"Impacto máximo teórico dos nulos: {max_impact * 100:.5f} p.p. "
        f"(diferença absoluta)"
    )
    print(f"A remoção é estatisticamente recomendada (Impacto < SE)? {can_remove}\n")


def evaluate_variable_inconsistency(
    df: pd.DataFrame,
    var: str,
    invalid_value: Any,
    target: str,
    conf: float = 0.95,
) -> bool:
    """Avalia inconsistência de variável usando Intervalo de Wilson.

    Compara a taxa de classe crítica da variável alvo entre o grupo com
    valores válidos e o grupo com valores inválidos (inconsistentes). Se a
    proporção do grupo válido cair fora do intervalo de Wilson do grupo
    inconsistente, assume-se comportamento anômalo.

    :param df: DataFrame com os dados.
    :type df: pd.DataFrame
    :param var: Nome da variável analisada (ex: 'br', 'km').
    :type var: str
    :param invalid_value: Valor considerado inconsistente a ser filtrado.
    :type invalid_value: Any
    :param target: Nome da variável alvo para cruzamento.
    :type target: str
    :param conf: Nível de confiança estatística (default 0.95).
    :type conf: float
    :return: True se as distribuições forem homogêneas, False se houver anomalia.
    :rtype: bool
    """
    df_clean = df.dropna(subset=[target]).copy()
    if target not in df_clean.columns or var not in df_clean.columns:
        print(f"Colunas '{target}' ou '{var}' ausentes no DataFrame.")
        return True

    # Coerce a coluna da variável para tipo numérico se aplicável
    # (evita erro 'str' vs 'int' na comparação)
    if not pd.api.types.is_numeric_dtype(df_clean[var]):
        df_clean[var] = pd.to_numeric(
            df_clean[var].astype(str).str.replace(",", "."), errors="coerce"
        )

    # Pega classe crítica (menor proporção na base)
    critic_class = df_clean[target].value_counts(sort=True, ascending=True).index[0]

    # Grupo A: Dados Válidos
    group_a = df_clean[df_clean[var] != invalid_value]
    n_a = len(group_a)
    if n_a == 0:
        print("Grupo válido vazio. Não há registros para comparação.")
        return True
    p_a = (group_a[target] == critic_class).mean()

    # Grupo B: Dados Inconsistentes
    group_b = df_clean[df_clean[var] == invalid_value]
    n_b = len(group_b)

    if n_b == 0:
        print(
            f"Nenhum registro com valor inválido {invalid_value} encontrado em {var}."
        )
        return True

    critic_count_b = (group_b[target] == critic_class).sum()
    p_b = critic_count_b / n_b

    # Intervalo de Confiança do Grupo B
    lower_bound, upper_bound = wilson_interval(p_b, n_b, conf)

    # Verifica se a probabilidade do grupo A está contida no IC do grupo B
    is_homogeneous = lower_bound <= p_a <= upper_bound

    print(f"## AVALIAÇÃO DE INCONSISTÊNCIA DA VARIÁVEL: {var} ##")
    print(f"Variável alvo associada: {target} (Classe crítica: {critic_class})")
    print(f"Taxa no Grupo Válido ({var} != {invalid_value}): {p_a * 100:.3f}%")
    print(f"Taxa no Grupo Inválido ({var} == {invalid_value}): {p_b * 100:.3f}%")
    print(
        f"IC de Wilson ({conf * 100:.1f}%) do Grupo Inválido: "
        f"[{lower_bound * 100:.3f}%, {upper_bound * 100:.3f}%]"
    )
    print(f"Os grupos são estatisticamente homogêneos? {is_homogeneous}")
    if not is_homogeneous:
        print(
            "  => Comportamento anômalo confirmado. "
            "Recomenda-se a eliminação desses registros.\n"
        )
    else:
        print(
            "  => Comportamento estatisticamente homogêneo. "
            "Sem evidência clara para descarte.\n"
        )

    return is_homogeneous


def evaluate_best_km_bucket(df: pd.DataFrame) -> None:
    """Avalia o suporte estatístico de diferentes tamanhos de buckets para 'km'.

    Calcula a cobertura (proporção de intervalos com ao menos 30 registros)
    para tamanhos de bucket de 1km, 5km, 10km e 20km.

    :param df: DataFrame contendo os registros de sinistros.
    :type df: pd.DataFrame
    """
    if "km" not in df.columns:
        print("A variável 'km' não está presente no DataFrame.")
        return

    # Coerce km para numérico se for string
    km_series = df["km"]
    if not pd.api.types.is_numeric_dtype(km_series):
        km_series = pd.to_numeric(
            km_series.astype(str).str.replace(",", "."), errors="coerce"
        )

    # Remove nulos ou valores menores/iguais a zero para focar em
    # trechos mapeados válidos
    km_series = km_series.dropna()  # type: ignore
    km_series = km_series[km_series > 0]  # type: ignore

    print("======= AVALIAÇÃO DE BUCKETS PARA A VARIÁVEL KM =======")
    for bucket in [1, 5, 10, 20]:
        faixas = (km_series // bucket) * bucket  # type: ignore
        counts = faixas.value_counts()  # type: ignore

        adequate_support_ratio = (counts >= 30).mean() * 100
        total_buckets = len(counts)

        print(
            f"Bucket {bucket:2d}km -> {total_buckets:5d} intervalos únicos | "
            f"{adequate_support_ratio:.1f}% de cobertura adequada (>= 30 registros)"
        )
    print("")
