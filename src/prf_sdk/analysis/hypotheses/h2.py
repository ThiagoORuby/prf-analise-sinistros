from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from prf_sdk.analysis.temporal import (
    build_monthly_series,
    prepare_h2_temporal_features,
    run_h2_proportion_tests,
    run_stl_decomposition,
    summarize_h2_periods,
)
from prf_sdk.utils.plots import set_plotting_theme


BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_PROCESSED_PATH = BASE_DIR / "data/processed/datatran_2022_2026_processed_v1.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "docs/figures"


def classify_h2_result(summary: pd.DataFrame, tests: pd.DataFrame) -> str:
    """Classifica a hipotese com base em frequencia diaria e proporcao fatal."""
    intense_summary = summary[
        summary["grupo"].isin(
            [
                "Periodo intenso (qualquer)",
                "Demais dias - Periodo intenso (qualquer)",
            ]
        )
    ].set_index("grupo")
    intense_test = tests[
        tests["group"] == "Periodo intenso (fim de semana, feriado ou ferias)"
    ].iloc[0]

    intense = intense_summary.loc["Periodo intenso (qualquer)"]
    reference = intense_summary.loc["Demais dias - Periodo intenso (qualquer)"]
    higher_frequency = intense["sinistros_por_dia"] > reference["sinistros_por_dia"]
    higher_fatality = intense_test["difference"] > 0 and intense_test["p_value"] < 0.05

    if higher_frequency and higher_fatality:
        return "Validada"
    if higher_frequency or higher_fatality:
        return "Parcialmente Confirmada"
    if intense_test["p_value"] >= 0.05:
        return "Inconclusiva"
    return "Refutada"


def _get_h2_plot_data(summary: pd.DataFrame) -> pd.DataFrame:
    """Seleciona os grupos principais usados na figura da H2."""
    plot_df = summary[
        summary["grupo"].isin(
            [
                "Fim de semana",
                "Feriado nacional",
                "Ferias (jan/jul/dez)",
                "Periodo intenso (qualquer)",
                "Demais dias - Periodo intenso (qualquer)",
            ]
        )
    ].copy()
    plot_df["grupo"] = plot_df["grupo"].replace(
        {"Demais dias - Periodo intenso (qualquer)": "Demais dias"}
    )
    plot_df["proporcao_fatal_pct"] = plot_df["proporcao_fatal"] * 100
    return plot_df


def build_h2_figure(summary: pd.DataFrame, output_path: Path) -> Path:
    """Gera figura com frequencia diaria e proporcao fatal por recorte."""
    plot_df = _get_h2_plot_data(summary)

    set_plotting_theme()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(
        data=plot_df,
        x="sinistros_por_dia",
        y="grupo",
        ax=axes[0],
        color="#3b82f6",
    )
    axes[0].bar_label(axes[0].containers[0], fmt="%.1f", padding=4, fontsize=9)
    axes[0].set_title("Sinistros por dia observado")
    axes[0].set_xlabel("Sinistros por dia observado")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, plot_df["sinistros_por_dia"].max() * 1.15)

    sns.barplot(
        data=plot_df,
        x="proporcao_fatal_pct",
        y="grupo",
        ax=axes[1],
        color="#ef4444",
    )
    fatal_labels = [
        f"{row.proporcao_fatal_pct:.2f}% ({row.sinistros_fatais_por_dia:.1f}/dia)"
        for row in plot_df.itertuples()
    ]
    axes[1].bar_label(
        axes[1].containers[0],
        labels=fatal_labels,
        padding=4,
        fontsize=9,
    )
    axes[1].set_title("% de sinistros fatais")
    axes[1].set_xlabel("% de sinistros com vitimas fatais")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, plot_df["proporcao_fatal_pct"].max() * 1.45)

    fig.suptitle("H2 - Periodos de maior circulacao e fatalidade", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_stl_figure(monthly: pd.DataFrame, output_path: Path) -> Path:
    """Gera figura STL de 4 paineis para o volume mensal de sinistros."""
    stl = run_stl_decomposition(monthly["total"])

    set_plotting_theme()
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    components = [
        ("observed", "Observado", "#64748b"),
        ("trend", "Tendência", "#3b82f6"),
        ("seasonal", "Sazonalidade", "#10b981"),
        ("residual", "Resíduo", "#f59e0b"),
    ]
    for ax, (key, label, color) in zip(axes, components):
        s = stl[key]
        ax.plot(s.index, s.values, color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        if key != "observed":
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    pct = stl["pct_variance_seasonal"]
    axes[0].set_title(
        f"Decomposição STL — Sinistros Mensais  "
        f"(sazonalidade explica {pct:.1%} da variância)",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_h2_analysis(
    data_path: Path | str | None = DEFAULT_PROCESSED_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Executa a EDA e o teste estatistico da H2."""
    df = pd.read_csv(data_path, index_col=0, low_memory=False)
    h2_df = prepare_h2_temporal_features(df)
    summary = summarize_h2_periods(h2_df)
    tests = run_h2_proportion_tests(h2_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "h2_resumo_periodos.csv"
    tests_path = output_dir / "h2_testes_proporcao.csv"
    figure_path = output_dir / "h2_periodos_fatalidade.png"
    stl_figure_path = output_dir / "h2_stl_decomposicao.png"

    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    figure_path = build_h2_figure(summary, figure_path)

    monthly = build_monthly_series(df)
    stl_volume = run_stl_decomposition(monthly["total"])
    stl_severity = run_stl_decomposition(monthly["proporcao_fatal"])
    stl_figure_path = build_stl_figure(monthly, stl_figure_path)

    status = classify_h2_result(summary, tests)
    intense = summary[summary["grupo"] == "Periodo intenso (qualquer)"].iloc[0]
    reference = summary[
        summary["grupo"] == "Demais dias - Periodo intenso (qualquer)"
    ].iloc[0]
    intense_test = tests[
        tests["group"] == "Periodo intenso (fim de semana, feriado ou ferias)"
    ].iloc[0]

    evidence = (
        "Periodos intensos registram "
        f"{intense['sinistros_por_dia']:.1f} sinistros/dia contra "
        f"{reference['sinistros_por_dia']:.1f} nos demais dias, com proporcao "
        f"fatal de {intense['proporcao_fatal'] * 100:.2f}% contra "
        f"{reference['proporcao_fatal'] * 100:.2f}%."
    )
    metric = (
        f"Teste z de duas proporcoes: z={intense_test['z_statistic']:.2f}, "
        f"p={intense_test['p_value']:.3g}, "
        f"RR={intense_test['relative_risk']:.3f}."
    )

    return {
        "status": status,
        "evidence": evidence,
        "actionable_insight": (
            "Priorizar reforco operacional em fins de semana, feriados e meses "
            "de ferias, especialmente em acoes preventivas de comportamento de risco."
        ),
        "support_metric": metric,
        "summary": summary,
        "tests": tests,
        "monthly_series": monthly,
        "stl_volume": stl_volume,
        "stl_severity": stl_severity,
        "summary_path": summary_path,
        "tests_path": tests_path,
        "figure_path": figure_path,
        "stl_figure_path": stl_figure_path,
    }


def main() -> None:
    """Executa a H2 via linha de comando."""
    result = run_h2_analysis()
    print(f"Status: {result['status']}")
    print(f"Evidencia: {result['evidence']}")
    print(f"Metrica: {result['support_metric']}")
    print(f"Figura: {result['figure_path']}")
    print(f"Tabelas: {result['summary_path']} | {result['tests_path']}")


if __name__ == "__main__":
    main()
