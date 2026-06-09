import pandas as pd

from prf_sdk.analysis.temporal import (
    prepare_h2_temporal_features,
    run_h2_proportion_tests,
    summarize_h2_periods,
)


def test_prepare_h2_temporal_features_flags_periods_and_fatal_accidents():
    df = pd.DataFrame(
        {
            "data_hora": [
                "2026-01-03 10:00:00",
                "2026-02-04 10:00:00",
                "2026-07-08 10:00:00",
            ],
            "classificacao_acidente": [
                "Com Vítimas Fatais",
                "Sem Vítimas",
                "Com Vítimas Feridas",
            ],
            "mortos": [1, 0, 0],
            "feriado_nacional": [0, 1, 0],
        }
    )

    result = prepare_h2_temporal_features(df)

    assert result["fim_de_semana"].tolist() == [1, 0, 0]
    assert result["periodo_ferias"].tolist() == [1, 0, 1]
    assert result["periodo_intenso"].tolist() == [1, 1, 1]
    assert result["sinistro_fatal"].tolist() == [1, 0, 0]


def test_summarize_h2_periods_counts_and_rates():
    df = pd.DataFrame(
        {
            "data_hora": [
                "2026-01-03 10:00:00",
                "2026-02-04 10:00:00",
                "2026-02-05 10:00:00",
            ],
            "classificacao_acidente": [
                "Com Vítimas Fatais",
                "Sem Vítimas",
                "Sem Vítimas",
            ],
            "mortos": [1, 0, 0],
            "feriado_nacional": [0, 0, 0],
        }
    )
    prepared = prepare_h2_temporal_features(df)

    summary = summarize_h2_periods(prepared)
    weekend = summary[summary["grupo"] == "Fim de semana"].iloc[0]

    assert weekend["dias_observados"] == 1
    assert weekend["sinistros"] == 1
    assert weekend["sinistros_fatais"] == 1
    assert weekend["proporcao_fatal"] == 1


def test_run_h2_proportion_tests_returns_expected_groups():
    df = pd.DataFrame(
        {
            "data_hora": [
                "2026-01-03 10:00:00",
                "2026-01-04 10:00:00",
                "2026-02-04 10:00:00",
                "2026-02-05 10:00:00",
            ],
            "classificacao_acidente": [
                "Com Vítimas Fatais",
                "Sem Vítimas",
                "Sem Vítimas",
                "Com Vítimas Fatais",
            ],
            "mortos": [1, 0, 0, 1],
            "feriado_nacional": [0, 0, 1, 0],
        }
    )
    prepared = prepare_h2_temporal_features(df)

    tests = run_h2_proportion_tests(prepared)

    assert set(tests["group"]) == {
        "Fim de semana",
        "Feriado nacional",
        "Ferias (jan/jul/dez)",
        "Periodo intenso (fim de semana, feriado ou ferias)",
    }
