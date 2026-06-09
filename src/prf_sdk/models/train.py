from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from prf_sdk.models.evaluate import build_summary, evaluate_model
from prf_sdk.schemas import CVScores, ModelResult


LEAKAGE_COLS: frozenset[str] = frozenset(
    {
        "mortos",
        "feridos_leves",
        "feridos_graves",
        "ilesos",
        "pessoas",
        "feridos",
        "ignorados",
    }
)

COLS_TO_DROP: frozenset[str] = frozenset(
    {
        "data_hora",
        "data_inversa",
        "municipio",
        "causa_acidente",
        "causa_acidente_grupo",
        "uf",
        "km",
        "latitude",
        "longitude",
        "tipo_acidente",
        "veiculos",
    }
)

TARGET_COL: str = "classificacao_acidente"

CV_FOLDS: int = 5
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


@dataclass
class ModelConfig:
    """
    Configuração de uma variante de ``DecisionTreeClassifier``.

    :param name: Nome descritivo exibido em tabelas e gráficos.
    :param params: Parâmetros passados ao construtor do classificador
        (além de ``class_weight='balanced'`` e ``random_state``, que são
        fixados pelo pipeline).
    :param plottable: ``True`` indica que a árvore é rasa o suficiente
        para ser visualizada com ``plot_decision_tree``.
    """

    name: str
    params: dict = field(default_factory=dict)
    plottable: bool = False


DEFAULT_CONFIGS: list[ModelConfig] = [
    ModelConfig(
        name="DT Irrestrita",
        params={},
    ),
    ModelConfig(
        name="DT Profundidade 3",
        params={"max_depth": 3},
        plottable=True,
    ),
    ModelConfig(
        name="DT Profundidade 5",
        params={"max_depth": 5},
    ),
    ModelConfig(
        name="DT Podada (min_leaf=500)",
        params={"min_samples_leaf": 500},
    ),
]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepara features e target para o modelo de classificação.

    Remove colunas de leakage (contagens de vítimas que determinam a
    variável-alvo), colunas de alta cardinalidade ou redundantes, e aplica
    one-hot encoding nas variáveis categóricas restantes.

    Colunas removidas como leakage:
    ``mortos``, ``feridos_leves``, ``feridos_graves``, ``ilesos``,
    ``pessoas``, ``feridos``, ``ignorados``.

    Colunas removidas por alta cardinalidade ou redundância:
    ``municipio``, ``causa_acidente`` (substituída por
    ``causa_acidente_grupo``), ``uf`` (substituída por ``regiao``),
    ``km`` (substituída por ``faixa_km``), ``latitude``, ``longitude``
    (coordenadas contínuas sem valor interpretativo para DT),
    ``data_hora``, ``data_inversa``.

    :param df: DataFrame processado pelo pipeline de features do projeto.
    :return: Tupla ``(X, y, feature_names)`` onde ``X`` é a matriz de
        features codificada, ``y`` é o vetor-alvo e ``feature_names`` é
        a lista de nomes das colunas de ``X``.
    :raises KeyError: Se a coluna-alvo ``classificacao_acidente`` estiver
        ausente no DataFrame.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Coluna-alvo '{TARGET_COL}' ausente no DataFrame.")

    y = df[TARGET_COL].copy()

    cols_to_remove = (LEAKAGE_COLS | COLS_TO_DROP | {TARGET_COL}) & set(df.columns)
    X = df.drop(columns=list(cols_to_remove))

    datetime_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
    X = X.drop(columns=datetime_cols)

    X = pd.get_dummies(X, drop_first=True)

    return X, y, X.columns.tolist()


def _extract_cv_scores(scores: dict) -> CVScores:
    """
    Agrega os scores dos folds de cross-validation em médias e desvios.

    :param scores: Dict retornado por
        :func:`sklearn.model_selection.cross_validate`.
    :return: :class:`CVScores` com médias e desvios padrão.
    """
    f1 = scores["test_f1_macro"]
    auc = scores["test_roc_auc_ovr"]
    return CVScores(
        f1_macro_mean=float(f1.mean()),
        f1_macro_std=float(f1.std()),
        roc_auc_ovr_mean=float(auc.mean()),
        roc_auc_ovr_std=float(auc.std()),
    )


def train_decision_tree(
    df: pd.DataFrame,
    dt_params: dict | None = None,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> ModelResult:
    """
    Treina e avalia uma Árvore de Decisão para classificação de gravidade
    de sinistros, comparando com um baseline ``DummyClassifier``.

    O pipeline executa as seguintes etapas em ordem:

    1. Preparação de features (remoção de leakage + one-hot encoding).
    2. Divisão treino/teste estratificada (padrão 80%/20%).
    3. Stratified K-Fold cross-validation (padrão k=5) no treino.
    4. Treinamento final no conjunto de treino completo.
    5. Avaliação no conjunto de teste.

    :param df: DataFrame processado pelo pipeline de features do projeto.
    :param dt_params: Parâmetros adicionais para ``DecisionTreeClassifier``
        (ex.: ``{"max_depth": 3}``). ``class_weight`` e ``random_state``
        são sempre fixados pelo pipeline. Padrão: ``None`` (árvore irrestrita).
    :param test_size: Proporção reservada para o conjunto de teste.
        Padrão: ``0.2``.
    :param cv_folds: Número de folds para a cross-validation.
        Padrão: ``5``.
    :param random_state: Semente para reprodutibilidade. Padrão: ``42``.
    :return: :class:`ModelResult` com modelo treinado, baseline, scores
        de CV, avaliação no teste e metadados do experimento.
    :raises KeyError: Se ``classificacao_acidente`` estiver ausente no
        DataFrame.
    """
    X, y, feature_names = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=random_state,
        **(dt_params or {}),
    )
    dummy = DummyClassifier(strategy="stratified", random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {"f1_macro": "f1_macro", "roc_auc_ovr": "roc_auc_ovr"}

    cv_raw = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_dummy_raw = cross_validate(dummy, X_train, y_train, cv=cv, scoring=scoring)

    model.fit(X_train, y_train)
    dummy.fit(X_train, y_train)

    classes = list(model.classes_)

    eval_result = evaluate_model(model, X_test, y_test, feature_names, classes)
    eval_dummy_result = evaluate_model(dummy, X_test, y_test, feature_names, classes)

    result = ModelResult(
        model=model,
        dummy=dummy,
        cv_scores=_extract_cv_scores(cv_raw),
        cv_dummy_scores=_extract_cv_scores(cv_dummy_raw),
        eval_result=eval_result,
        eval_dummy_result=eval_dummy_result,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classes=classes,
        cv_folds=cv_folds,
    )
    result.summary = build_summary(result)
    return result


def compare_decision_trees(
    df: pd.DataFrame,
    configs: list[ModelConfig] = DEFAULT_CONFIGS,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[ModelConfig, ModelResult]]:
    """
    Treina e avalia múltiplas variantes de ``DecisionTreeClassifier``
    sobre o mesmo split treino/teste, permitindo comparação justa.

    Features e divisão treino/teste são computadas uma única vez e
    compartilhadas entre todas as configurações. O ``DummyClassifier``
    também é treinado uma vez e reutilizado como baseline de cada
    ``ModelResult``.

    :param df: DataFrame processado pelo pipeline de features do projeto.
    :param configs: Lista de :class:`ModelConfig` com as variantes a
        comparar. Padrão: :data:`DEFAULT_CONFIGS`.
    :param test_size: Proporção reservada para o conjunto de teste.
        Padrão: ``0.2``.
    :param cv_folds: Número de folds para a cross-validation.
        Padrão: ``5``.
    :param random_state: Semente para reprodutibilidade. Padrão: ``42``.
    :return: Lista de pares ``(ModelConfig, ModelResult)`` na mesma ordem
        que ``configs``.
    :raises KeyError: Se ``classificacao_acidente`` estiver ausente no
        DataFrame.
    """
    X, y, feature_names = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {"f1_macro": "f1_macro", "roc_auc_ovr": "roc_auc_ovr"}

    dummy = DummyClassifier(strategy="stratified", random_state=random_state)
    cv_dummy_raw = cross_validate(dummy, X_train, y_train, cv=cv, scoring=scoring)
    dummy.fit(X_train, y_train)
    cv_dummy_scores = _extract_cv_scores(cv_dummy_raw)

    results: list[tuple[ModelConfig, ModelResult]] = []

    for config in configs:
        model = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
            **config.params,
        )
        cv_raw = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        model.fit(X_train, y_train)

        classes = list(model.classes_)

        eval_result = evaluate_model(model, X_test, y_test, feature_names, classes)
        eval_dummy_result = evaluate_model(
            dummy, X_test, y_test, feature_names, classes
        )

        result = ModelResult(
            model=model,
            dummy=dummy,
            cv_scores=_extract_cv_scores(cv_raw),
            cv_dummy_scores=cv_dummy_scores,
            eval_result=eval_result,
            eval_dummy_result=eval_dummy_result,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            classes=classes,
            cv_folds=cv_folds,
        )
        result.summary = build_summary(result)
        results.append((config, result))

    return results
