from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Config(BaseSettings):
    DB_USER: str = "postgres"       # Your PostgreSQL username
    DB_PASS: str = "postgres"       # Your PostgreSQL password
    DB_HOST: str = "localhost"      # Since you mapped 5432:5432
    DB_PORT: str = "5432"           # Default PostgreSQL port
    DB_NAME: str = "mimiciv"        
    
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        env_nested_delimiter = "__"
    )
    
    def get_db_uri(self) -> str:
        url =  f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        return url