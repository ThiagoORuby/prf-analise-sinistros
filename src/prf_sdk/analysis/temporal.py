"""
Analise temporal de sinistros.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal runtimes
    seasonal_decompose = None


VACATION_MONTHS = (1, 7, 12)


def _normal_two_tailed_p_value(z_statistic: float) -> float:
    """Calcula p-valor bicaudal da normal padrao sem depender de scipy."""
    return erfc(abs(z_statistic) / sqrt(2))


@dataclass(frozen=True)
class ProportionTestResult:
    """Resultado do teste de proporcoes entre dois grupos."""

    group: str
    reference: str
    group_n: int
    reference_n: int
    group_fatal: int
    reference_fatal: int
    group_rate: float
    reference_rate: float
    difference: float
    relative_risk: float
    z_statistic: float
    p_value: float


def prepare_h2_temporal_features(
    df: pd.DataFrame,
    vacation_months: Iterable[int] = VACATION_MONTHS,
) -> pd.DataFrame:
    """Prepara variaveis temporais usadas na avaliacao da hipotese H2.

    A hipotese H2 considera fins de semana, feriados nacionais e meses tipicos
    de ferias escolares/viagens no Brasil (janeiro, julho e dezembro).
    """
    required = {"data_hora", "classificacao_acidente", "mortos"}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Colunas obrigatorias ausentes: {missing_cols}")

    h2_df = df.copy()
    h2_df["data_hora"] = pd.to_datetime(h2_df["data_hora"], errors="coerce")
    h2_df = h2_df.dropna(subset=["data_hora"])

    h2_df["data"] = h2_df["data_hora"].dt.date
    h2_df["mes"] = h2_df["data_hora"].dt.month

    if "fim_de_semana" not in h2_df.columns:
        h2_df["fim_de_semana"] = h2_df["data_hora"].dt.weekday.isin([5, 6])
    h2_df["fim_de_semana"] = h2_df["fim_de_semana"].astype(int)

    if "feriado_nacional" not in h2_df.columns:
        h2_df["feriado_nacional"] = 0
    h2_df["feriado_nacional"] = h2_df["feriado_nacional"].fillna(0).astype(int)

    vacation_months = set(vacation_months)
    h2_df["periodo_ferias"] = h2_df["mes"].isin(vacation_months).astype(int)
    h2_df["periodo_intenso"] = (
        (h2_df["fim_de_semana"] == 1)
        | (h2_df["feriado_nacional"] == 1)
        | (h2_df["periodo_ferias"] == 1)
    ).astype(int)
    h2_df["sinistro_fatal"] = (
        (h2_df["mortos"].fillna(0) > 0)
        | (h2_df["classificacao_acidente"] == "Com Vitimas Fatais")
        | (h2_df["classificacao_acidente"] == "Com Vítimas Fatais")
    ).astype(int)

    return h2_df


def summarize_h2_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Resume frequencia e letalidade por recortes temporais da H2."""
    groups = {
        "Fim de semana": "fim_de_semana",
        "Feriado nacional": "feriado_nacional",
        "Ferias (jan/jul/dez)": "periodo_ferias",
        "Periodo intenso (qualquer)": "periodo_intenso",
    }

    rows = []
    total_days = df["data"].nunique()
    for label, column in groups.items():
        for value, bucket in [(1, label), (0, f"Demais dias - {label}")]:
            subset = df[df[column] == value]
            days = subset["data"].nunique()
            accidents = len(subset)
            fatal_accidents = int(subset["sinistro_fatal"].sum())
            deaths = int(subset["mortos"].sum())
            rows.append(
                {
                    "recorte": label,
                    "grupo": bucket,
                    "dias_observados": days,
                    "percentual_dias": days / total_days if total_days else np.nan,
                    "sinistros": accidents,
                    "sinistros_por_dia": accidents / days if days else np.nan,
                    "sinistros_fatais": fatal_accidents,
                    "sinistros_fatais_por_dia": (
                        fatal_accidents / days if days else np.nan
                    ),
                    "proporcao_fatal": (
                        fatal_accidents / accidents if accidents else np.nan
                    ),
                    "mortos": deaths,
                    "mortos_por_100_sinistros": (
                        deaths / accidents * 100 if accidents else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def two_proportion_z_test(
    df: pd.DataFrame,
    group_column: str,
    group_label: str,
    reference_label: str = "Demais dias",
) -> ProportionTestResult:
    """Compara a proporcao de sinistros fatais entre grupo e referencia."""
    group = df[df[group_column] == 1]
    reference = df[df[group_column] == 0]

    group_n = len(group)
    reference_n = len(reference)
    group_fatal = int(group["sinistro_fatal"].sum())
    reference_fatal = int(reference["sinistro_fatal"].sum())

    group_rate = group_fatal / group_n
    reference_rate = reference_fatal / reference_n
    pooled = (group_fatal + reference_fatal) / (group_n + reference_n)
    standard_error = np.sqrt(pooled * (1 - pooled) * (1 / group_n + 1 / reference_n))
    z_statistic = (group_rate - reference_rate) / standard_error
    p_value = _normal_two_tailed_p_value(z_statistic)
    relative_risk = group_rate / reference_rate

    return ProportionTestResult(
        group=group_label,
        reference=reference_label,
        group_n=group_n,
        reference_n=reference_n,
        group_fatal=group_fatal,
        reference_fatal=reference_fatal,
        group_rate=group_rate,
        reference_rate=reference_rate,
        difference=group_rate - reference_rate,
        relative_risk=relative_risk,
        z_statistic=z_statistic,
        p_value=p_value,
    )


def run_h2_proportion_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Executa testes de proporcao para os recortes da hipotese H2."""
    tests = [
        two_proportion_z_test(df, "fim_de_semana", "Fim de semana"),
        two_proportion_z_test(df, "feriado_nacional", "Feriado nacional"),
        two_proportion_z_test(df, "periodo_ferias", "Ferias (jan/jul/dez)"),
        two_proportion_z_test(
            df,
            "periodo_intenso",
            "Periodo intenso (fim de semana, feriado ou ferias)",
        ),
    ]
    return pd.DataFrame([test.__dict__ for test in tests])
