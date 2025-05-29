from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class DatabaseSettings(BaseSettings):
    DB_USER: str
    DB_PASS: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str

    # .env file is located in the project root directory
    _env_file_path = Path(__file__).parent.parent / ".env"
    model_config = SettingsConfigDict(
        env_file=_env_file_path,
        env_file_encoding="utf-8",
        env_nested_delimiter="__"
    )

    def get_db_uri(self) -> str:
        url = f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        return url