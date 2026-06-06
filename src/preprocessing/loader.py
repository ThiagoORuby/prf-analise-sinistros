from pathlib import Path

import pandas as pd

from settings import settings
from src.preprocessing.cleaner import clean_data
from src.preprocessing.features import create_new_features


def load_raw_data(file_path: Path | str | None = None) -> pd.DataFrame:
    """Carrega os dados brutos de sinistros da PRF de um arquivo CSV.

    Caso nenhum caminho seja fornecido, o carregamento padrão é feito a partir de
    'data/raw/datatran_merged_2022_2026.csv' dentro do diretório base do projeto.

    :param file_path: Caminho alternativo para o arquivo bruto de entrada.
    :type file_path: Path | str | None
    :return: DataFrame contendo a base bruta de sinistros.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: Caso o arquivo não seja encontrado no caminho
        especificado.
    """
    if file_path is None:
        file_path = settings.BASE_DIR / "data/raw/datatran_merged_2022_2026.csv"

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados brutos não localizado em: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        encoding="latin1",
        sep=";",
        index_col=0,
        low_memory=False,
    )
    return df


def save_processed_data(
    df: pd.DataFrame, file_path: Path | str | None = None
) -> None:
    """Salva o DataFrame processado em um arquivo CSV.

    Caso nenhum caminho seja fornecido, os dados são salvos em
    'data/processed/datatran_processed.csv' dentro do diretório base do projeto.

    :param df: DataFrame contendo os dados limpos e com engenharia de
        features realizada.
    :type df: pd.DataFrame
    :param file_path: Caminho alternativo para o salvamento dos dados processados.
    :type file_path: Path | str | None
    """
    if file_path is None:
        file_path = settings.BASE_DIR / "data/processed/datatran_processed.csv"

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path)


def load_processed_data(file_path: Path | str | None = None) -> pd.DataFrame:
    """Carrega os dados processados de sinistros da PRF de um arquivo CSV.

    Caso nenhum caminho seja fornecido, o carregamento padrão é feito a partir de
    'data/processed/datatran_processed.csv' dentro do diretório base do projeto.

    :param file_path: Caminho alternativo para o arquivo processado de entrada.
    :type file_path: Path | str | None
    :return: DataFrame contendo os dados processados.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: Caso o arquivo processado não exista no caminho
        indicado.
    """
    if file_path is None:
        file_path = settings.BASE_DIR / "data/processed/datatran_processed.csv"

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados processados não localizado em: {file_path}"
        )

    df = pd.read_csv(file_path, low_memory=False)
    return df


def run_preprocessing_pipeline(
    raw_path: Path | str | None = None,
    processed_path: Path | str | None = None,
) -> pd.DataFrame:
    """Orquestra o pipeline de pré-processamento de ponta a ponta.

    Carrega a base bruta, aplica a limpeza, gera as novas variáveis e salva
    o resultado final no caminho especificado.

    :param raw_path: Caminho alternativo do arquivo CSV bruto.
    :type raw_path: Path | str | None
    :param processed_path: Caminho alternativo de destino do CSV processado.
    :type processed_path: Path | str | None
    :return: DataFrame totalmente limpo e processado.
    :rtype: pd.DataFrame
    """
    print("Iniciando carregamento dos dados brutos...")
    df_raw = load_raw_data(raw_path)

    print("Iniciando etapa de limpeza de dados...")
    df_clean = clean_data(df_raw)

    print("Iniciando engenharia de features...")
    df_processed = create_new_features(df_clean)

    print("Salvando base processada resultante...")
    save_processed_data(df_processed, processed_path)

    print("Pipeline de pré-processamento executado com sucesso!")
    return df_processed
