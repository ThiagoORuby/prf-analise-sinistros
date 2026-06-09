from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.tree import plot_tree

from prf_sdk.schemas import ModelEvalResult, ModelResult


if TYPE_CHECKING:
    from prf_sdk.models.train import ModelConfig


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    classes: list[str],
    top_n_features: int = 15,
) -> ModelEvalResult:
    """
    Avalia um classificador sklearn no conjunto de teste.

    Calcula F1-macro, F1 por classe, AUC-ROC one-vs-rest e matriz de
    confusão normalizada. Extrai feature importance quando disponível
    (``DecisionTreeClassifier``); retorna série vazia para estimadores
    sem esse atributo (``DummyClassifier``).

    :param model: Estimador sklearn já treinado com ``predict`` e
        ``predict_proba``.
    :param X_test: Features do conjunto de teste.
    :param y_test: Target do conjunto de teste.
    :param feature_names: Lista de nomes das features, na mesma ordem das
        colunas de ``X_test``.
    :param classes: Rótulos das classes na ordem usada pelo modelo.
    :param top_n_features: Número de features mais importantes a reter na
        série de feature importance. Padrão: ``15``.
    :return: :class:`ModelEvalResult` com todas as métricas de avaliação.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_per_class = {
        cls: float(
            f1_score(y_test, y_pred, labels=[cls], average="macro", zero_division=0)
        )
        for cls in classes
    }

    roc_auc = float(
        roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro", labels=classes
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")

    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
        feature_importance = importance.sort_values(ascending=False).head(
            top_n_features
        )
    else:
        feature_importance = pd.Series(dtype=float)

    return ModelEvalResult(
        f1_macro=f1_macro,
        f1_per_class=f1_per_class,
        roc_auc_ovr=roc_auc,
        confusion_matrix=cm,
        feature_importance=feature_importance,
    )


def build_summary(result: ModelResult) -> str:
    """
    Gera texto descritivo dos principais resultados do pipeline de modelagem.

    :param result: :class:`ModelResult` já preenchido (exceto ``summary``).
    :return: String com o resumo narrativo comparando classificador e baseline.
    """
    cv = result.cv_scores
    cv_d = result.cv_dummy_scores
    ev = result.eval_result
    ev_d = result.eval_dummy_result

    class_lines = "\n".join(
        f"    {cls}: F1 = {score:.3f}" for cls, score in ev.f1_per_class.items()
    )

    fi_lines = (
        "\n".join(
            f"    {feat}: {imp:.4f}" for feat, imp in ev.feature_importance.items()
        )
        if not ev.feature_importance.empty
        else "    (não disponível)"
    )

    lines = [
        "Árvore de Decisão — Resultados",
        "",
        f"Cross-validation (Stratified K-Fold, k={result.cv_folds}):",
        f"  Classificador  F1-macro = {cv.f1_macro_mean:.3f} ± {cv.f1_macro_std:.3f}"
        f"  |  AUC-ROC = {cv.roc_auc_ovr_mean:.3f} ± {cv.roc_auc_ovr_std:.3f}",
        f"  Baseline       F1-macro = {cv_d.f1_macro_mean:.3f} ± {cv_d.f1_macro_std:.3f}"
        f"  |  AUC-ROC = {cv_d.roc_auc_ovr_mean:.3f} ± {cv_d.roc_auc_ovr_std:.3f}",
        "",
        "Avaliação no conjunto de teste:",
        f"  Classificador  F1-macro = {ev.f1_macro:.3f}  |  AUC-ROC = {ev.roc_auc_ovr:.3f}",
        f"  Baseline       F1-macro = {ev_d.f1_macro:.3f}  |  AUC-ROC = {ev_d.roc_auc_ovr:.3f}",
        "",
        "F1 por classe (classificador):",
        class_lines,
        "",
        f"Top-{len(ev.feature_importance)} features mais importantes:",
        fi_lines,
    ]
    return "\n".join(lines)


def build_comparison_table(
    comparisons: list[tuple[ModelConfig, ModelResult]],
) -> pd.DataFrame:
    """
    Constrói um DataFrame comparativo com as métricas de todas as variantes.

    Inclui as métricas de CV (média ± dp) e de teste para cada configuração,
    além de uma linha de baseline (``DummyClassifier``) extraída do primeiro
    resultado da lista.

    :param comparisons: Lista de pares ``(ModelConfig, ModelResult)`` retornada
        por :func:`~prf_sdk.models.train.compare_decision_trees`.
    :return: DataFrame indexado pelo nome do modelo, com colunas de F1-macro
        e AUC-ROC (CV e teste) e F1 da classe ``Com Vítimas Fatais``.
    """
    rows = []

    for config, result in comparisons:
        ev = result.eval_result
        cv = result.cv_scores
        rows.append(
            {
                "Modelo": config.name,
                "CV F1-macro": cv.f1_macro_mean,
                "CV F1-macro dp": cv.f1_macro_std,
                "CV AUC-ROC": cv.roc_auc_ovr_mean,
                "CV AUC-ROC dp": cv.roc_auc_ovr_std,
                "Teste F1-macro": ev.f1_macro,
                "Teste AUC-ROC": ev.roc_auc_ovr,
                "F1 Fatais": ev.f1_per_class.get("Com Vítimas Fatais", 0.0),
            }
        )

    _, first_result = comparisons[0]
    ev_d = first_result.eval_dummy_result
    cv_d = first_result.cv_dummy_scores
    rows.append(
        {
            "Modelo": "Dummy (baseline)",
            "CV F1-macro": cv_d.f1_macro_mean,
            "CV F1-macro dp": cv_d.f1_macro_std,
            "CV AUC-ROC": cv_d.roc_auc_ovr_mean,
            "CV AUC-ROC dp": cv_d.roc_auc_ovr_std,
            "Teste F1-macro": ev_d.f1_macro,
            "Teste AUC-ROC": ev_d.roc_auc_ovr,
            "F1 Fatais": ev_d.f1_per_class.get("Com Vítimas Fatais", 0.0),
        }
    )

    return pd.DataFrame(rows).set_index("Modelo")


def plot_decision_tree(
    model,
    feature_names: list[str],
    classes: list[str],
    max_depth: int | None = None,
    figsize: tuple[int, int] = (22, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plota a estrutura da Árvore de Decisão.

    Usa :func:`sklearn.tree.plot_tree` com nós coloridos por classe majoritária
    e proporções de amostra em cada nó. Funciona com qualquer profundidade, mas
    é mais legível para árvores rasas (``max_depth`` ≤ 4).

    :param model: ``DecisionTreeClassifier`` já treinado.
    :param feature_names: Nomes das features (mesmo comprimento que
        ``model.n_features_in_``).
    :param classes: Rótulos das classes na ordem usada pelo modelo.
    :param max_depth: Profundidade máxima exibida no plot. ``None`` exibe a
        árvore completa. Para árvores profundas, recomenda-se usar o valor
        com que o modelo foi treinado ou um valor menor.
    :param figsize: Tamanho da figura em polegadas. Padrão: ``(22, 10)``.
    :return: Par ``(fig, ax)`` do matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=classes,
        filled=True,
        max_depth=max_depth,
        ax=ax,
        fontsize=8,
        impurity=True,
        proportion=True,
        rounded=True,
    )

    depth_label = (
        f"profundidade = {model.get_depth()}"
        if max_depth is None
        else f"exibindo até profundidade {max_depth} de {model.get_depth()}"
    )
    ax.set_title(
        f"Árvore de Decisão — {depth_label}  |  "
        f"{model.get_n_leaves()} folhas  |  "
        f"{model.tree_.n_node_samples[0]:,} amostras na raiz",
        fontsize=10,
    )

    return fig, ax
