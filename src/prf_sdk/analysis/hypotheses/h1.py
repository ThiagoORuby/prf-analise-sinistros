from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency

from prf_sdk.schemas import H1aResult, H1bResult, H1Result, OddsRatioResult
from prf_sdk.utils.stats import cramers_v_corrected, odds_ratio_with_ci


BEHAVIORAL_CAUSES: list[str] = [
    "Condutor Dormindo",
    "Velocidade Incompatível",
    "Ingestão de álcool pelo condutor",
    "Ingestão de substâncias psicoativas pelo condutor",
]

NIGHT_PERIODS: tuple[str, ...] = ("Madrugada", "Noite")

FATAL_CLASS: str = "Com Vítimas Fatais"


def _build_summary_h1(result: H1Result) -> str:
    """
    Gera um texto descritivo dos principais resultados de H1.

    :param result: Objeto :class:`H1Result` já preenchido (exceto ``summary``).
    :return: String com o resumo narrativo.
    """
    h1a = result.h1a
    h1b = result.h1b
    orn = h1b.or_nocturnal
    ord_ = h1b.or_diurnal

    night_str = " e ".join(result.night_periods)
    causes_str = ", ".join(f"'{c}'" for c in result.behavioral_causes)

    if result.confirmed:
        status = "CONFIRMADA"
    elif h1a.confirmed or h1b.confirmed:
        status = "PARCIALMENTE CONFIRMADA"
    else:
        status = "REFUTADA"

    lines = [
        f"H1 — {status}",
        "",
        f"Causas comportamentais: {causes_str}",
        f"({result.behavioral_n:,} registros; {result.behavioral_pct:.1f}% da base).",
        "",
        "H1a (concentração noturna):",
        f"  χ² = {h1a.chi2:.1f}  |  V de Cramér = {h1a.cramers_v:.4f}  "
        f"|  p = {h1a.p_value:.2e}",
        f"  {'✓ Confirmada' if h1a.confirmed else '✗ Não confirmada'}: causas "
        f"comportamentais {'são' if h1a.confirmed else 'não são'} "
        f"proporcionalmente mais frequentes em {night_str}.",
        "",
        "H1b (maior gravidade noturna):",
        f"  OR noturno  = {orn.odds_ratio:.3f}  "
        f"[IC 95%: {orn.ci_lower:.3f}–{orn.ci_upper:.3f}]  "
        f"{'✓' if orn.significant else '✗'}",
        f"  OR diurno   = {ord_.odds_ratio:.3f}  "
        f"[IC 95%: {ord_.ci_lower:.3f}–{ord_.ci_upper:.3f}]  "
        f"{'✓' if ord_.significant else '✗'}",
        f"  {'✓ Confirmada' if h1b.confirmed else '✗ Não confirmada'}: OR noturno "
        f"{'>' if orn.odds_ratio > ord_.odds_ratio else '<='} OR diurno.",
    ]
    return "\n".join(lines)


def verify_h1(
    df: pd.DataFrame,
    behavioral_causes: list[str] = BEHAVIORAL_CAUSES,
    night_periods: tuple[str, ...] = NIGHT_PERIODS,
    fatal_class: str = FATAL_CLASS,
) -> H1Result:
    """
    Verifica a Hipótese H1 sobre causas comportamentais, período noturno
    e gravidade dos sinistros.

    A verificação é composta por dois sub-testes:

    **H1a** — qui-quadrado de independência entre a flag
    ``is_comportamental`` e a faixa horária do sinistro. Confirma se
    causas comportamentais ocorrem proporcionalmente mais no período
    noturno.

    **H1b** — razão de chances (OR) entre ``is_comportamental`` e
    desfecho fatal, estratificada por período noturno e diurno. Confirma
    se o efeito de gravidade do grupo comportamental é mais pronunciado
    à noite.

    :param df: DataFrame já processado pelo pipeline de features do
        projeto. Deve conter as colunas ``causa_acidente``,
        ``horario`` e ``classificacao_acidente``.
    :param behavioral_causes: Lista de valores de ``causa_acidente`` que
        representam causas comportamentais. Padrão: :data:`BEHAVIORAL_CAUSES`.
    :param night_periods: Faixas horárias classificadas como noturnas.
        Padrão: ``("Madrugada", "Noite")``.
    :param fatal_class: Valor de ``classificacao_acidente`` que
        representa desfecho fatal. Padrão: ``"Com Vítimas Fatais"``.
    :return: :class:`H1Result` com todos os resultados e um campo
        ``summary`` com texto descritivo dos principais achados.
    :raises KeyError: Se alguma das colunas obrigatórias estiver ausente.
    :raises ValueError: Se algum valor de ``behavioral_causes`` não existir
        em ``causa_acidente``, ou se ``night_periods`` não corresponderem
        a valores de ``horario``.
    """
    required_cols = {"causa_acidente", "horario", "classificacao_acidente"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no DataFrame: {missing}")

    available_causes = set(df["causa_acidente"].unique())
    invalid_causes = [c for c in behavioral_causes if c not in available_causes]
    if invalid_causes:
        raise ValueError(
            f"Causas não encontradas em 'causa_acidente': {invalid_causes}"
        )

    invalid_periods = set(night_periods) - set(df["horario"].unique())
    if invalid_periods:
        raise ValueError(
            f"Faixas horárias não encontradas em 'horario': {invalid_periods}. "
            f"Valores disponíveis: {sorted(df['horario'].unique())}"
        )

    is_behavioral = df["causa_acidente"].isin(behavioral_causes)
    is_nocturnal = df["horario"].isin(night_periods)
    is_fatal = df["classificacao_acidente"] == fatal_class

    behavioral_n = int(is_behavioral.sum())
    behavioral_pct = behavioral_n / len(df) * 100

    ct_h1a = pd.crosstab(
        is_behavioral.map({True: "comportamental", False: "outras_causas"}),
        df["horario"],
    )
    chi2_val, p_h1a, _, _ = chi2_contingency(ct_h1a, correction=False)
    n_total = len(df)
    r, c = ct_h1a.shape
    v_h1a = cramers_v_corrected(chi2_Preciso que refatore o h1.py, ja comecei a refatoração mudando alguns arquivos de lugar, mas tem um problema enorme aqui: nao da pra usar os grupos de causa, pq eles sao gerais demais pra condicao de comportamento norturno descrito na H1, como bebida, sono, drogas, velocidade. Por isso troquei por causa_acidente e ja listei as necessarias. Refatore o resto do codigo do verify_h1.py para fazer sentidoval, n_total, r, c)

    rate_table = ct_h1a.div(ct_h1a.sum(axis=1), axis=0)
    night_rate_behavioral = float(
        rate_table.loc["comportamental", list(night_periods)].sum()
    )
    night_rate_others = float(
        rate_table.loc["outras_causas", list(night_periods)].sum()
    )
    h1a_confirmed = (night_rate_behavioral > night_rate_others) and (p_h1a < 0.05)

    h1a_result = H1aResult(
        contingency_table=ct_h1a,
        rate_table=rate_table,
        chi2=float(chi2_val),
        p_value=float(p_h1a),
        cramers_v=v_h1a,
        confirmed=h1a_confirmed,
    )

    def _compute_or_for_mask(mask: pd.Series) -> OddsRatioResult:
        """Calcula OR de is_comportamental × is_fatal dentro de uma máscara."""
        beh_sub = is_behavioral[mask]
        fat_sub = is_fatal[mask]
        a = int((beh_sub & fat_sub).sum())
        b = int((beh_sub & ~fat_sub).sum())
        c = int((~beh_sub & fat_sub).sum())
        d = int((~beh_sub & ~fat_sub).sum())
        return odds_ratio_with_ci(a, b, c, d)

    or_nocturnal = _compute_or_for_mask(is_nocturnal)
    or_diurnal = _compute_or_for_mask(~is_nocturnal)
    or_overall = _compute_or_for_mask(pd.Series(True, index=df.index))

    h1b_confirmed = (
        or_nocturnal.significant
        and or_nocturnal.odds_ratio > 1.0
        and or_nocturnal.odds_ratio > or_diurnal.odds_ratio
    )

    h1b_result = H1bResult(
        or_nocturnal=or_nocturnal,
        or_diurnal=or_diurnal,
        or_overall=or_overall,
        confirmed=h1b_confirmed,
    )

    confirmed = h1a_confirmed and h1b_confirmed

    result = H1Result(
        h1a=h1a_result,
        h1b=h1b_result,
        behavioral_causes=behavioral_causes,
        night_periods=night_periods,
        behavioral_n=behavioral_n,
        behavioral_pct=behavioral_pct,
        confirmed=confirmed,
    )
    result.summary = _build_summary_h1(result)
    return result
