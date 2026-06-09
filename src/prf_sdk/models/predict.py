from __future__ import annotations

import pandas as pd


def predict(model, X: pd.DataFrame) -> pd.Series:
    """
    Retorna a classe predita para cada registro de ``X``.

    :param model: Estimador sklearn já treinado.
    :param X: Features no mesmo formato usado durante o treinamento
        (colunas idênticas às de ``X_train`` após encoding).
    :return: Série com as classes preditas, indexada como ``X``.
    """
    return pd.Series(model.predict(X), index=X.index, name="classificacao_predita")


def predict_proba(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna as probabilidades preditas para cada classe.

    :param model: Estimador sklearn já treinado com suporte a
        ``predict_proba``.
    :param X: Features no mesmo formato usado durante o treinamento.
    :return: DataFrame com uma coluna por classe (``model.classes_``) e
        uma linha por registro, indexado como ``X``.
    """
    proba = model.predict_proba(X)
    return pd.DataFrame(proba, index=X.index, columns=model.classes_)
