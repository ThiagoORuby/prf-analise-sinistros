from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class H1aResult:
    """
    Resultado da verificação de H1a (concentração noturna de causas
    comportamentais).

    :param contingency_table: Tabela de contingência
        ``is_comportamental`` × ``faixa_horaria`` (contagens absolutas).
    :param rate_table: Proporção de cada faixa horária dentro do grupo
        comportamental e do grupo restante (linha = grupo, coluna = faixa).
    :param chi2: Estatística qui-quadrado.
    :param p_value: P-valor do teste (sem correção; tamanho amostral
        grande torna o p-valor menos informativo — use ``cramers_v``).
    :param cramers_v: V de Cramér com correção de viés (Bergsma, 2013)
        como medida de tamanho de efeito.
    :param confirmed: ``True`` se a proporção noturna é maior no grupo
        comportamental do que no restante e p_value < 0,05.
    """

    contingency_table: pd.DataFrame
    rate_table: pd.DataFrame
    chi2: float
    p_value: float
    cramers_v: float
    confirmed: bool


@dataclass
class OddsRatioResult:
    """
    Resultado de um teste de razão de chances com IC de Wilson.

    :param odds_ratio: Razão de chances pontual.
    :param ci_lower: Limite inferior do IC 95%.
    :param ci_upper: Limite superior do IC 95%.
    :param p_value: P-valor do teste qui-quadrado 2×2.
    :param significant: ``True`` se o IC não inclui 1,0.
    :param n: Tamanho da sub-amostra usada no cálculo.
    """

    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    n: int


@dataclass
class H1bResult:
    """
    Resultado da verificação de H1b (maior gravidade noturna de causas
    comportamentais em relação a outras causas no mesmo horário).

    :param or_nocturnal: OR de ``is_comportamental`` × ``is_fatal``
        restrito ao período noturno.
    :param or_diurnal: OR equivalente para o período diurno (controle).
    :param or_overall: OR global, sem estratificação por horário.
    :param confirmed: ``True`` se o OR noturno for > 1, significativo e
        superior ao OR diurno.
    """

    or_nocturnal: OddsRatioResult
    or_diurnal: OddsRatioResult
    or_overall: OddsRatioResult
    confirmed: bool


@dataclass
class H1Result:
    """
    Resultado consolidado da verificação de H1.

    :param h1a: Resultado da sub-hipótese de concentração noturna.
    :param h1b: Resultado da sub-hipótese de maior gravidade noturna.
    :param behavioral_causes: Lista de causas individuais utilizadas como grupo comportamental.
    :param night_periods: Faixas horárias classificadas como noturnas.
    :param behavioral_n: Número de registros do grupo comportamental.
    :param behavioral_pct: Percentual do grupo comportamental na base.
    :param confirmed: ``True`` se H1a **e** H1b forem confirmadas.
    :param summary: Texto descritivo dos principais resultados.
    """

    h1a: H1aResult
    h1b: H1bResult
    behavioral_causes: list[str]
    night_periods: tuple[str, ...]
    behavioral_n: int
    behavioral_pct: float
    confirmed: bool
    summary: str = field(default="", repr=False)


@dataclass
class H3OddsRatioResult:
    """
    Resultado de um teste de razão de chances para H3, com correção de
    Bonferroni e contagens por grupo meteorológico.

    :param odds_ratio: Razão de chances pontual.
    :param ci_lower: Limite inferior do IC 95%.
    :param ci_upper: Limite superior do IC 95%.
    :param p_value: P-valor do qui-quadrado 2×2 antes da correção.
    :param p_value_bonferroni: P-valor após correção de Bonferroni pelo
        número de regiões testadas.
    :param significant: ``True`` se o IC não inclui 1,0 **e** o
        p-valor corrigido é inferior a 0,05.
    :param n_adverse: Registros com meteorologia adversa no grupo.
    :param n_clear: Registros com meteorologia favorável no grupo.
    """

    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_bonferroni: float
    significant: bool
    n_adverse: int
    n_clear: int


@dataclass
class H3aResult:
    """
    Resultado da análise global de H3 (efeito médio da meteorologia
    adversa sobre a fatalidade, sem estratificação regional).

    :param chi2: Estatística qui-quadrado.
    :param p_value: P-valor do teste global.
    :param cramers_v: V de Cramér com correção de viés.
    :param or_global: OR global ``is_adversa`` × ``is_fatal``.
    :param adverse_fatal_rate: Taxa de fatalidade em condições adversas.
    :param clear_fatal_rate: Taxa de fatalidade em condições favoráveis.
    :param n_adverse: Total de registros com meteorologia adversa.
    :param n_clear: Total de registros com meteorologia favorável.
    """

    chi2: float
    p_value: float
    cramers_v: float
    or_global: H3OddsRatioResult
    adverse_fatal_rate: float
    clear_fatal_rate: float
    n_adverse: int
    n_clear: int


@dataclass
class H3bResult:
    """
    Resultado da análise regional de H3 (heterogeneidade do efeito
    meteorológico entre as cinco regiões brasileiras).

    :param regional_ors: Dict mapeando cada região ao seu
        :class:`H3OddsRatioResult`.
    :param regional_rates: DataFrame com taxas de fatalidade em
        condições adversas e favoráveis por região.
    :param n_bonferroni: Número de comparações usadas na correção
        de Bonferroni (igual ao número de regiões).
    :param heterogeneous: ``True`` se os ORs variam substancialmente
        entre regiões (ao menos uma região com OR significativo e ao
        menos uma sem, ou amplitude dos ORs > ``heterogeneity_threshold``).
    """

    regional_ors: dict[str, H3OddsRatioResult]
    regional_rates: pd.DataFrame
    n_bonferroni: int
    heterogeneous: bool


@dataclass
class H3Result:
    """
    Resultado consolidado da verificação de H3.

    :param h3a: Resultado do efeito global.
    :param h3b: Resultado da análise de heterogeneidade regional.
    :param adverse_conditions: Conjunto de condições classificadas
        como adversas.
    :param confirmed: ``True`` se H3b evidenciar heterogeneidade
        regional relevante no efeito da meteorologia adversa.
    :param summary: Texto descritivo dos principais resultados.
    """

    h3a: H3aResult
    h3b: H3bResult
    adverse_conditions: frozenset[str]
    confirmed: bool
    summary: str = field(default="", repr=False)


@dataclass
class H5RegionStats:
    """
    Estatísticas de fatalidade e infraestrutura para uma região brasileira.

    :param regiao: Nome da região (Norte, Nordeste, Centro-Oeste, Sudeste, Sul).
    :param total: Total de sinistros registrados na região.
    :param fatais: Número de sinistros fatais.
    :param taxa_fatalidade: Taxa de fatalidade em percentual.
    :param pct_pista_simples: Percentual de sinistros ocorridos em pista simples.
    """

    regiao: str
    total: int
    fatais: int
    taxa_fatalidade: float
    pct_pista_simples: float


@dataclass
class H5PistaStats:
    """
    Estatísticas de fatalidade por tipo de pista.

    :param tipo_pista: Categoria da pista (Simples, Dupla, Múltipla).
    :param total: Total de sinistros no tipo de pista.
    :param fatais: Número de sinistros fatais.
    :param taxa_fatalidade: Taxa de fatalidade em percentual.
    """

    tipo_pista: str
    total: int
    fatais: int
    taxa_fatalidade: float


@dataclass
class H5PostHocResult:
    """
    Resultado do teste post-hoc pairwise entre duas regiões (z-test de proporções
    com correção de Bonferroni).

    :param regiao_a: Primeira região da comparação.
    :param regiao_b: Segunda região da comparação.
    :param taxa_a: Taxa de fatalidade da região A (%).
    :param taxa_b: Taxa de fatalidade da região B (%).
    :param z_statistic: Estatística z do teste de duas proporções.
    :param p_value: P-valor bruto (bicaudal).
    :param p_value_bonferroni: P-valor após correção de Bonferroni (k=10).
    :param significant: ``True`` se p_value_bonferroni < 0,05.
    """

    regiao_a: str
    regiao_b: str
    taxa_a: float
    taxa_b: float
    z_statistic: float
    p_value: float
    p_value_bonferroni: float
    significant: bool


@dataclass
class H5Result:
    """
    Resultado consolidado da verificação de H5.

    :param chi2: Estatística qui-quadrado do teste de independência
        ``regiao`` × ``is_fatal``.
    :param p_value: P-valor do teste global.
    :param cramers_v: V de Cramér com correção de viés.
    :param region_stats: Lista de :class:`H5RegionStats` por região,
        ordenada por taxa de fatalidade decrescente.
    :param pista_stats: Lista de :class:`H5PistaStats` por tipo de pista,
        ordenada por taxa de fatalidade decrescente.
    :param posthoc: Lista de :class:`H5PostHocResult` para todos os pares
        de regiões (k=10).
    :param confirmed: ``True`` se o teste global for significativo e o
        padrão regional acompanhar a proporção de pista simples.
    :param summary: Texto descritivo dos principais resultados.
    """

    chi2: float
    p_value: float
    cramers_v: float
    region_stats: list[H5RegionStats]
    pista_stats: list[H5PistaStats]
    posthoc: list[H5PostHocResult]
    confirmed: bool
    summary: str = field(default="", repr=False)


@dataclass
class CVScores:
    """
    Métricas agregadas de um processo de cross-validation.

    :param f1_macro_mean: Média do F1-macro entre os folds.
    :param f1_macro_std: Desvio padrão do F1-macro entre os folds.
    :param roc_auc_ovr_mean: Média do AUC-ROC (OvR, macro) entre os folds.
    :param roc_auc_ovr_std: Desvio padrão do AUC-ROC entre os folds.
    """

    f1_macro_mean: float
    f1_macro_std: float
    roc_auc_ovr_mean: float
    roc_auc_ovr_std: float


@dataclass
class ModelEvalResult:
    """
    Resultado da avaliação de um classificador no conjunto de teste.

    :param f1_macro: F1-macro no conjunto de teste.
    :param f1_per_class: F1 por classe (dict rótulo → score).
    :param roc_auc_ovr: AUC-ROC one-vs-rest (macro) no conjunto de teste.
    :param confusion_matrix: Matriz de confusão normalizada por linha (``np.ndarray``
        de shape 3×3), linhas = verdadeiro, colunas = predito.
    :param feature_importance: Série com importância das features, ordenada
        decrescentemente. Vazia para estimadores sem ``feature_importances_``.
    """

    f1_macro: float
    f1_per_class: dict[str, float]
    roc_auc_ovr: float
    confusion_matrix: Any
    feature_importance: pd.Series


@dataclass
class ModelResult:
    """
    Resultado completo do pipeline de treinamento e avaliação.

    :param model: Classificador treinado (``DecisionTreeClassifier``).
    :param dummy: Baseline treinado (``DummyClassifier``).
    :param cv_scores: Métricas de cross-validation do classificador.
    :param cv_dummy_scores: Métricas de cross-validation do baseline.
    :param eval_result: Avaliação do classificador no conjunto de teste.
    :param eval_dummy_result: Avaliação do baseline no conjunto de teste.
    :param X_train: Features do conjunto de treino (após encoding).
    :param X_test: Features do conjunto de teste (após encoding).
    :param y_train: Target do conjunto de treino.
    :param y_test: Target do conjunto de teste.
    :param classes: Rótulos das classes na ordem usada pelo modelo.
    :param cv_folds: Número de folds usados na cross-validation.
    :param summary: Texto descritivo dos principais resultados.
    """

    model: Any
    dummy: Any
    cv_scores: CVScores
    cv_dummy_scores: CVScores
    eval_result: ModelEvalResult
    eval_dummy_result: ModelEvalResult
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    classes: list[str]
    cv_folds: int
    summary: str = field(default="", repr=False)
