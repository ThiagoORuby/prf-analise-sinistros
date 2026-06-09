import pandas as pd

from prf_sdk.analysis.hypotheses.h4 import get_h4_metrics


def main():
    print("Carregando os dados...")
    df = pd.read_csv(
        "data/processed/datatran_2022_2026_processed_v1.csv", low_memory=False
    )

    print("Calculando métricas para H4...")
    metrics = get_h4_metrics(df)

    print("\n--- RESULTADOS H4 ---")
    print(f"Total de trechos analisados (faixas de 5km): {metrics['total_trechos']}")
    print(f"Total de sinistros fatais no período: {metrics['total_sinistros_fatais']}")
    print(f"\nConcentração:")
    print(
        f"-> O Top 5% dos trechos mais perigosos concentra {metrics['pct_fatais_no_top_5pct_trechos']:.1%} dos sinistros fatais."
    )
    print(
        f"-> O Top 10% dos trechos mais perigosos concentra {metrics['pct_fatais_no_top_10pct_trechos']:.1%} dos sinistros fatais."
    )
    print(
        f"-> Metade (50%) de todos os sinistros fatais acontece em apenas {metrics['pct_trechos_que_concentram_50pct_fatais']:.1%} da malha (trechos ativos)."
    )

    print("\n--- TOP 10 TRECHOS MAIS FATAIS ---")
    top_10 = metrics["top_10_trechos_fatais"]
    for idx, row in top_10.iterrows():
        print(
            f"{idx + 1}. {row['id_trecho']}: {row['sinistros_fatais']} acidentes fatais (Total de acidentes: {row['total_sinistros']})"
        )


if __name__ == "__main__":
    main()

