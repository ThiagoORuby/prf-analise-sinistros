from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

from prf_sdk.schemas import H3aResult, H3bResult, H3OddsRatioResult, H3Result
from prf_sdk.utils.stats import cramers_v_corrected


ADVERSE_CONDITIONS: frozenset[str] = frozenset(
    {
        "Chuva",
        "Granizo",
        "Neve",
        "Nevoeiro/Neblina",
        "Vento",
        "Garoa/Chuvisco",
        "Tempestade",
    }
)

FATAL_CLASS: str = "Com Vítimas Fatais"

REGION_COL: str = "regiao"

WEATHER_COL: str = "condicao_meteorologica"

TARGET_COL: str = "classificacao_acidente"


def _odds_ratio_with_ci(
    a: int,
    b: int,
    c: int,
    d: int,
    n_comparisons: int = 1,
    alpha: float = 0.05,
) -> H3OddsRatioResult:
    """
    Calcula OR com IC log-normal e correção de Bonferroni no p-valor.

    Tabela de contingência esperada::

                  Fatal    Não Fatal
        Adverso     a          b
        Favorável   c          d

    :param a: Registros adversos com desfecho fatal.
    :param b: Registros adversos sem desfecho fatal.
    :param c: Registros favoráveis com desfecho fatal.
    :param d: Registros favoráveis sem desfecho fatal.
    :param n_comparisons: Número de comparações para correção de
        Bonferroni. Use ``1`` para o cálculo global.
    :param alpha: Nível de significância antes da correção.
    :return: :class:`H3OddsRatioResult` preenchido.
    """
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)

    z = stats.norm.ppf(1 - alpha / 2)
    log_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_lower = float(np.exp(np.log(or_val) - z * log_se))
    ci_upper = float(np.exp(np.log(or_val) + z * log_se))

    contingency = np.array([[a, b], [c, d]])
    _, p_value, _, _ = chi2_contingency(contingency, correction=False)
    p_bonferroni = min(float(p_value) * n_comparisons, 1.0)

    significant = (not (ci_lower <= 1.0 <= ci_upper)) and (p_bonferroni < alpha)

    return H3OddsRatioResult(
        odds_ratio=float(or_val),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(p_value),
        p_value_bonferroni=p_bonferroni,
        significant=significant,
        n_adverse=int(a + b),
        n_clear=int(c + d),
    )


def _build_summary_h3(result: H3Result) -> str:
    """
    Gera texto descritivo dos principais resultados de H3.

    :param result: :class:`H3Result` já preenchido (exceto ``summary``).
    :return: String com o resumo narrativo.
    """
    h3a = result.h3a
    h3b = result.h3b
    status = "CONFIRMADA" if result.confirmed else "REFUTADA"

    or_str_global = (
        f"OR = {h3a.or_global.odds_ratio:.3f} "
        f"[IC 95%: {h3a.or_global.ci_lower:.3f}–{h3a.or_global.ci_upper:.3f}]"
    )

    regional_lines = []
    for region, or_res in sorted(h3b.regional_ors.items()):
        sig_marker = "✓" if or_res.significant else "✗"
        regional_lines.append(
            f"  {region:<15} OR = {or_res.odds_ratio:.3f} "
            f"[{or_res.ci_lower:.3f}–{or_res.ci_upper:.3f}]  "
            f"p_adj = {or_res.p_value_bonferroni:.4f}  {sig_marker}"
        )

    or_values = [v.odds_ratio for v in h3b.regional_ors.values()]
    or_range = max(or_values) - min(or_values)

    lines = [
        f"H3 — {status}",
        "",
        "Efeito global (H3a):",
        f"  χ² = {h3a.chi2:.1f}  |  V de Cramér = {h3a.cramers_v:.4f}  "
        f"|  p = {h3a.p_value:.2e}",
        f"  {or_str_global}",
        f"  Taxa de fatalidade: adverso = {h3a.adverse_fatal_rate:.3%}  "
        f"| favorável = {h3a.clear_fatal_rate:.3%}",
        "",
        f"Heterogeneidade regional (H3b) — correção de Bonferroni "
        f"(k = {h3b.n_bonferroni}):",
        *regional_lines,
        "",
        f"  Amplitude dos ORs regionais: {or_range:.3f}  "
        f"({'heterogêneo' if h3b.heterogeneous else 'homogêneo'})",
    ]
    return "\n".join(lines)


def verify_h3(
    df: pd.DataFrame,
    adverse_conditions: frozenset[str] | set[str] = ADVERSE_CONDITIONS,
    fatal_class: str = FATAL_CLASS,
    region_col: str = REGION_COL,
    weather_col: str = WEATHER_COL,
    target_col: str = TARGET_COL,
    heterogeneity_threshold: float = 0.3,
) -> H3Result:
    """
    Verifica a Hipótese H3 sobre o impacto diferenciado das condições
    meteorológicas adversas por região do país.

    **H3a** testa se a meteorologia adversa tem associação global com
    a fatalidade (qui-quadrado + V de Cramér + OR global). Dado o Cramér
    V observado na EDA (0,031), espera-se efeito negligenciável.

    **H3b** testa se esse efeito é *heterogêneo* entre regiões: para
    cada uma das cinco regiões, calcula-se o OR de ``is_adversa`` ×
    ``is_fatal`` com p-valor corrigido por Bonferroni (k = número de
    regiões). A hipótese é confirmada se a amplitude entre o maior e o
    menor OR regional superar ``heterogeneity_threshold`` ou se pelo
    menos uma região apresentar OR significativo enquanto outra não.

    :param df: DataFrame processado pelo pipeline de features do projeto.
        Deve conter as colunas ``condicao_meteorologica``, ``regiao`` e
        ``classificacao_acidente``.
    :param adverse_conditions: Conjunto de strings correspondentes às
        condições classificadas como adversas em ``condicao_meteorologica``.
        Padrão: :data:`ADVERSE_CONDITIONS`.
    :param fatal_class: Valor de ``classificacao_acidente`` que representa
        desfecho fatal. Padrão: ``"Com Vítimas Fatais"``.
    :param region_col: Nome da coluna de região. Padrão: ``"regiao"``.
    :param weather_col: Nome da coluna de condição meteorológica.
        Padrão: ``"condicao_meteorologica"``.
    :param target_col: Nome da coluna da variável-alvo.
        Padrão: ``"classificacao_acidente"``.
    :param heterogeneity_threshold: Amplitude mínima entre o maior e o
        menor OR regional para classificar o efeito como heterogêneo.
        Padrão: ``0,3``.
    :return: :class:`H3Result` com todos os resultados e campo ``summary``
        com texto descritivo.
    :raises KeyError: Se alguma das colunas obrigatórias estiver ausente.
    :raises ValueError: Se nenhum valor de ``adverse_conditions`` for
        encontrado nos dados.
    """
    required_cols = {weather_col, region_col, target_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no DataFrame: {missing}")

    found_conditions = set(df[weather_col].unique()) & set(adverse_conditions)
    if not found_conditions:
        raise ValueError(
            f"Nenhum valor de 'adverse_conditions' encontrado em '{weather_col}'. "
            f"Condições disponíveis: {sorted(df[weather_col].unique())}"
        )

    is_adverse = df[weather_col].isin(adverse_conditions)
    is_fatal = df[target_col] == fatal_class

    ct_global = pd.crosstab(is_adverse, is_fatal)
    chi2_val, p_global, _, _ = chi2_contingency(ct_global, correction=False)
    n_total = len(df)
    v_global = cramers_v_corrected(chi2_val, n_total, 2, 2)

    n_adverse_total = int(is_adverse.sum())
    n_clear_total = n_total - n_adverse_total

    adverse_fatal_rate = (
        float((is_adverse & is_fatal).sum() / n_adverse_total)
        if n_adverse_total > 0
        else 0.0
    )
    clear_fatal_rate = (
        float((~is_adverse & is_fatal).sum() / n_clear_total)
        if n_clear_total > 0
        else 0.0
    )

    a_g = int((is_adverse & is_fatal).sum())
    b_g = int((is_adverse & ~is_fatal).sum())
    c_g = int((~is_adverse & is_fatal).sum())
    d_g = int((~is_adverse & ~is_fatal).sum())
    or_global = _odds_ratio_with_ci(a_g, b_g, c_g, d_g, n_comparisons=1)

    h3a_result = H3aResult(
        chi2=float(chi2_val),
        p_value=float(p_global),
        cramers_v=v_global,
        or_global=or_global,
        adverse_fatal_rate=adverse_fatal_rate,
        clear_fatal_rate=clear_fatal_rate,
        n_adverse=n_adverse_total,
        n_clear=n_clear_total,
    )

    regions = sorted(df[region_col].dropna().unique())
    n_regions = len(regions)

    regional_ors: dict[str, H3OddsRatioResult] = {}
    rate_records: list[dict] = []

    for region in regions:
        mask = df[region_col] == region
        sub_adverse = is_adverse[mask]
        sub_fatal = is_fatal[mask]

        a = int((sub_adverse & sub_fatal).sum())
        b = int((sub_adverse & ~sub_fatal).sum())
        c = int((~sub_adverse & sub_fatal).sum())
        d = int((~sub_adverse & ~sub_fatal).sum())

        or_res = _odds_ratio_with_ci(a, b, c, d, n_comparisons=n_regions)
        regional_ors[region] = or_res

        n_adv_reg = a + b
        n_clr_reg = c + d
        rate_records.append(
            {
                "regiao": region,
                "n_total": int(mask.sum()),
                "n_adverso": n_adv_reg,
                "n_favoravel": n_clr_reg,
                "taxa_fatal_adverso": a / n_adv_reg if n_adv_reg > 0 else np.nan,
                "taxa_fatal_favoravel": c / n_clr_reg if n_clr_reg > 0 else np.nan,
                "odds_ratio": or_res.odds_ratio,
                "ci_lower": or_res.ci_lower,
                "ci_upper": or_res.ci_upper,
                "p_value_bonferroni": or_res.p_value_bonferroni,
                "significativo": or_res.significant,
            }
        )

    regional_rates = pd.DataFrame(rate_records).set_index("regiao")

    or_values = [v.odds_ratio for v in regional_ors.values()]
    or_range = max(or_values) - min(or_values)
    sig_flags = [v.significant for v in regional_ors.values()]

    heterogeneous = or_range >= heterogeneity_threshold or (
        any(sig_flags) and not all(sig_flags)
    )

    h3b_result = H3bResult(
        regional_ors=regional_ors,
        regional_rates=regional_rates,
        n_bonferroni=n_regions,
        heterogeneous=heterogeneous,
    )

    confirmed = h3b_result.heterogeneous

    result = H3Result(
        h3a=h3a_result,
        h3b=h3b_result,
        adverse_conditions=frozenset(adverse_conditions),
        confirmed=confirmed,
    )
    result.summary = _build_summary_h3(result)
    return result
