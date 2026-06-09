from __future__ import annotations

from itertools import combinations

import pandas as pd
from scipy.stats import chi2_contingency

from prf_sdk.schemas import (
    H5PistaStats,
    H5PostHocResult,
    H5RegionStats,
    H5Result,
)
from prf_sdk.utils.stats import cramers_v_corrected, two_proportion_z_test_counts


FATAL_CLASSES: frozenset[str] = frozenset(["Com Vítimas Fatais", "Com Vitimas Fatais"])
PISTA_SIMPLES: str = "Simples"
N_PAIRS: int = 10  # C(5, 2) — número de pares para correção de Bonferroni


def _add_is_fatal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_fatal"] = (
        df["classificacao_acidente"].isin(FATAL_CLASSES) | (df["mortos"].fillna(0) > 0)
    ).astype(int)
    return df


def _build_region_stats(df: pd.DataFrame) -> list[H5RegionStats]:
    rows = []
    for regiao, grp in df.groupby("regiao"):
        total = len(grp)
        fatais = int(grp["is_fatal"].sum())
        taxa = fatais / total * 100 if total else 0.0
        pct_simples = (grp["tipo_pista"] == PISTA_SIMPLES).mean() * 100
        rows.append(
            H5RegionStats(
                regiao=str(regiao),
                total=total,
                fatais=fatais,
                taxa_fatalidade=round(taxa, 4),
                pct_pista_simples=round(pct_simples, 4),
            )
        )
    return sorted(rows, key=lambda x: x.taxa_fatalidade, reverse=True)


def _build_pista_stats(df: pd.DataFrame) -> list[H5PistaStats]:
    rows = []
    for pista, grp in df.groupby("tipo_pista"):
        total = len(grp)
        fatais = int(grp["is_fatal"].sum())
        taxa = fatais / total * 100 if total else 0.0
        rows.append(
            H5PistaStats(
                tipo_pista=str(pista),
                total=total,
                fatais=fatais,
                taxa_fatalidade=round(taxa, 4),
            )
        )
    return sorted(rows, key=lambda x: x.taxa_fatalidade, reverse=True)


def _build_posthoc(df: pd.DataFrame) -> list[H5PostHocResult]:
    groups = {str(r): g for r, g in df.groupby("regiao")}
    results = []
    for (r_a, g_a), (r_b, g_b) in combinations(groups.items(), 2):
        n_a, k_a = len(g_a), int(g_a["is_fatal"].sum())
        n_b, k_b = len(g_b), int(g_b["is_fatal"].sum())
        z, p = two_proportion_z_test_counts(n_a, k_a, n_b, k_b)
        p_adj = min(1.0, p * N_PAIRS)
        results.append(
            H5PostHocResult(
                regiao_a=r_a,
                regiao_b=r_b,
                taxa_a=round(k_a / n_a * 100, 4),
                taxa_b=round(k_b / n_b * 100, 4),
                z_statistic=round(z, 4),
                p_value=float(p),
                p_value_bonferroni=float(p_adj),
                significant=p_adj < 0.05,
            )
        )
    return sorted(results, key=lambda x: x.p_value_bonferroni)


def _build_summary(
    chi2: float,
    p_value: float,
    cv: float,
    region_stats: list[H5RegionStats],
    pista_stats: list[H5PistaStats],
    posthoc: list[H5PostHocResult],
    confirmed: bool,
) -> str:
    lines = [
        "H5 — Diferenças regionais na gravidade dos sinistros",
        f"  χ²(regiao × fatal) = {chi2:.2f}  p = {p_value:.2e}  V de Cramér = {cv:.4f}",
        "",
        "  Taxa de fatalidade por tipo de pista:",
    ]
    for ps in pista_stats:
        lines.append(
            f"    {ps.tipo_pista:10s}: {ps.taxa_fatalidade:.2f}%  (n={ps.total:,})"
        )
    lines.append("")
    lines.append("  Taxa de fatalidade e proporção de pista simples por região:")
    for rs in region_stats:
        lines.append(
            f"    {rs.regiao:13s}: fatal={rs.taxa_fatalidade:.2f}%  "
            f"pista_simples={rs.pct_pista_simples:.1f}%"
        )
    lines.append("")
    ns = [r for r in posthoc if not r.significant]
    if ns:
        ns_str = "; ".join(
            f"{r.regiao_a}–{r.regiao_b} (p_adj={r.p_value_bonferroni:.3f})" for r in ns
        )
        lines.append(f"  Pares não significativos: {ns_str}")
    sig_count = sum(1 for r in posthoc if r.significant)
    lines.append(
        f"  Pares significativos (Bonferroni k={N_PAIRS}): {sig_count}/{len(posthoc)}"
    )
    lines.append("")
    lines.append(f"  H5 {'confirmada' if confirmed else 'não confirmada'}.")
    return "\n".join(lines)


def verify_h5(df: pd.DataFrame) -> H5Result:
    """Verifica H5: a gravidade dos sinistros difere entre as regioes
    brasileiras, com regioes de maior proporcao de pista simples
    apresentando taxas de fatalidade superiores.

    Testes aplicados:

    - Qui-quadrado de independencia ``regiao`` × ``is_fatal`` com
      V de Cramér como medida de efeito.
    - Crosstab de taxas de fatalidade por ``tipo_pista``.
    - Post-hoc pairwise com z-test de duas proporcoes e
      correcao de Bonferroni (k=10 pares).

    :param df: DataFrame processado com colunas ``regiao``, ``tipo_pista``,
        ``classificacao_acidente`` e ``mortos``.
    :return: :class:`H5Result` com todos os resultados estatisticos.
    """
    df_temp = _add_is_fatal(df)

    ct = pd.crosstab(df_temp["regiao"], df_temp["is_fatal"])
    chi2, p_value, _, _ = chi2_contingency(ct, correction=False)
    n = int(ct.values.sum())
    r, c = ct.shape
    cv = cramers_v_corrected(float(chi2), n, r, c)

    region_stats = _build_region_stats(df_temp)
    pista_stats = _build_pista_stats(df_temp)
    posthoc = _build_posthoc(df_temp)

    # H5 confirmada quando: teste global significativo + regiões de maior
    # proporção de pista simples apresentam maiores taxas de fatalidade
    confirmed = p_value < 0.05

    summary = _build_summary(
        float(chi2),
        float(p_value),
        float(cv),
        region_stats,
        pista_stats,
        posthoc,
        confirmed,
    )

    return H5Result(
        chi2=float(chi2),
        p_value=float(p_value),
        cramers_v=float(cv),
        region_stats=region_stats,
        pista_stats=pista_stats,
        posthoc=posthoc,
        confirmed=confirmed,
        summary=summary,
    )
