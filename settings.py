from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Classe de configurações globais do projeto baseada no Pydantic Settings.

    :ivar BASE_DIR: Caminho absoluto para o diretório raiz do projeto.
    :vartype BASE_DIR: Path
    """

    BASE_DIR: Path = Path(__file__).resolve().parent

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
