import argparse

from prf_sdk.analysis.initial import (
    diagnose_dataset,
    evaluate_best_km_bucket,
    evaluate_target_sensitivity,
    evaluate_variable_inconsistency,
)
from prf_sdk.preprocessing.cleaner import clean_data
from prf_sdk.preprocessing.loader import load_raw_data


def main():
    """Função principal que orquestra a execução da análise inicial via CLI."""
    parser = argparse.ArgumentParser(
        description="Executa as análises estatísticas iniciais e diagnósticos "
        "da base de sinistros da PRF."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help="Caminho alternativo para o arquivo CSV bruto de dados da PRF.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="classificacao_acidente",
        help="Nome da coluna correspondente à variável alvo.",
    )

    args = parser.parse_args()

    print("Carregando base de dados para análise...")
    try:
        df = load_raw_data(args.raw_path)
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return

    # 1. Diagnóstico Estrutural e Resumo de Variáveis
    summary_df = diagnose_dataset(df)
    print("\nResumo das Variáveis:")
    print(summary_df.to_string())
    print("\n" + "=" * 60 + "\n")

    # 2. Sensibilidade da Variável Alvo para dados nulos
    evaluate_target_sensitivity(df, args.target)
    print("=" * 60 + "\n")

    # 3. Avaliação de Inconsistência na variável BR (valor 0)
    evaluate_variable_inconsistency(
        df=df, var="br", invalid_value=0, target=args.target
    )
    print("=" * 60 + "\n")

    df = clean_data(df)

    # 5. Avaliação do suporte de Buckets de KM
    evaluate_best_km_bucket(df)
    print("Análise inicial concluída com sucesso!")


if __name__ == "__main__":
    main()
