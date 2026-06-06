import numpy as np
from scipy import stats


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
