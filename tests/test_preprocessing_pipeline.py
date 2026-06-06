from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.loader import run_preprocessing_pipeline


@pytest.mark.slow
def test_preprocessing_pipeline(tmp_path):
    """Testa o pipeline de pré-processamento de ponta a ponta.

    Verifica se a base processada possui exatamente 300.086 linhas e 46 colunas,
    se não há dados nulos ou linhas duplicadas inesperadas, e se o resultado final é
    idêntico à base de referência pré-processada.

    :param tmp_path: Fixture do pytest para diretório temporário.
    :type tmp_path: Path
    """
    temp_processed_path = tmp_path / "datatran_processed_temp.csv"

    df_result = run_preprocessing_pipeline(processed_path=temp_processed_path)

    assert df_result.shape == (300086, 46), f"Shape inesperado: {df_result.shape}"

    assert df_result.isnull().sum().sum() == 0, (
        "A base processada contém valores nulos."
    )

    assert df_result.duplicated().sum() == 0, (
        "A base processada contém linhas duplicadas."
    )

    ref_path = Path("data/processed/datatran_2022_2026_processed_v1.csv")

    df_result_loaded = pd.read_csv(temp_processed_path, index_col=0)
    df_ref_loaded = pd.read_csv(ref_path, index_col=0)

    df_result_loaded = df_result_loaded.reindex(
        sorted(df_result_loaded.columns), axis=1
    )
    df_ref_loaded = df_ref_loaded.reindex(sorted(df_ref_loaded.columns), axis=1)

    pd.testing.assert_frame_equal(df_result_loaded, df_ref_loaded)
