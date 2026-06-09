from math import erfc, sqrt

import numpy as np
from scipy import stats

from prf_sdk.schemas import OddsRatioResult


def wilson_interval(p, n, conf=0.95):
    """Calcula Intervalo de Confiança de Wilson para uma propoção (p) e
    montante (n) de ocorrências
    """
    z = stats.norm.ppf(1 - (1 - conf) / 2)

    denominator = 1 + (z**2 / n)
    centre_adjusted_probability = p + (z**2 / (2 * n))
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + (z**2 / (4 * n))) / n)

    lower_bound = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator

    return lower_bound, upper_bound


def cramers_v_corrected(chi2: float, n: int, r: int, c: int) -> float:
    """
    Calcula o V de Cramér com correção de viés (Bergsma, 2013).

    :param chi2: Estatística qui-quadrado observada.
    :param n: Tamanho total da amostra.
    :param r: Número de linhas da tabela de contingência.
    :param c: Número de colunas da tabela de contingência.
    :return: V de Cramér corrigido no intervalo [0, 1].
    """
    phi2 = chi2 / n
    r_tilde = r - (r - 1) ** 2 / (n - 1)
    c_tilde = c - (c - 1) ** 2 / (n - 1)
    phi2_tilde = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    denom = min(r_tilde - 1, c_tilde - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_tilde / denom))


def odds_ratio_with_ci(
    a: int, b: int, c: int, d: int, alpha: float = 0.05
) -> OddsRatioResult:
    """
    Calcula o OR com IC pelo método de Haldane-Anscombe (correção +0,5
    quando alguma célula é zero) e p-valor pelo qui-quadrado 2×2.

    Tabela de contingência esperada::

                  Fatal   Não Fatal
        Comport.    a         b
        Outros      c         d

    :param a: Casos comportamentais fatais.
    :param b: Casos comportamentais não fatais.
    :param c: Casos não comportamentais fatais.
    :param d: Casos não comportamentais não fatais.
    :param alpha: Nível de significância para o IC (padrão 0,05).
    :return: :class:`OddsRatioResult` com OR, IC e p-valor.
    """
    # Correção de Haldane-Anscombe quando alguma célula é zero
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)

    # IC pelo método log-normal
    z = stats.norm.ppf(1 - alpha / 2)
    log_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_lower = float(np.exp(np.log(or_val) - z * log_se))
    ci_upper = float(np.exp(np.log(or_val) + z * log_se))

    # P-valor pelo qui-quadrado 2×2
    contingency = np.array([[a, b], [c, d]])
    _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)

    n = int(a + b + c + d)
    significant = not (ci_lower <= 1.0 <= ci_upper)

    return OddsRatioResult(
        odds_ratio=float(or_val),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(p_value),
        significant=significant,
        n=n,
    )


def two_proportion_z_test_counts(
    n1: int, k1: int, n2: int, k2: int
) -> tuple[float, float]:
    """Z-test bicaudal de duas proporcoes a partir de contagens brutas.

    :param n1: Tamanho do grupo 1.
    :param k1: Sucessos (eventos fatais) no grupo 1.
    :param n2: Tamanho do grupo 2.
    :param k2: Sucessos (eventos fatais) no grupo 2.
    :return: Tupla ``(z_statistic, p_value)``.
    """
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = erfc(abs(z) / sqrt(2))
    return float(z), float(p)
