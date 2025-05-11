from .config import DatabaseSettings
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from typing import Optional, List
import pandas as pd


class SchemaName:
    HOSP = "mimiciv_hosp"
    ICU = "mimiciv_icu"

    @classmethod
    def get_all(cls):
        return [cls.HOSP, cls.ICU]

    @classmethod
    def is_valid(cls, schema_name):
        return schema_name in cls.get_all()


class Database:
    def __init__(self):
        self.settings = DatabaseSettings()
        self.engine = create_engine(self.settings.get_db_uri(), echo=False)
        self.connection = self.engine.connect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Close the connection and dispose of the engine
        self.close()
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        return False

    def show_tables_in_schema(self, schema_name) -> List[str]:
        # Validate schema name
        if schema_name not in SchemaName.get_all():
            raise ValueError("Invalid schema name. Only 'hosp' and 'icu' are allowed.")
        query = text(
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = :schema_name"
        )

        result = self.connection.execute(query, {"schema_name": schema_name})
        return [row[0] for row in result]

    def read_table_to_df(
        self, table_name: str, schema_name: str, limit: Optional[int] = None
    ) -> pd.DataFrame:
        # match table name with schema name if schema name is provided
        if schema_name and not SchemaName.is_valid(schema_name):
            raise ValueError("Invalid schema name. Only 'hosp' and 'icu' are allowed.")
        if schema_name and not self.show_tables_in_schema(schema_name):
            raise ValueError(
                f"Table '{table_name}' does not exist in schema '{schema_name}'."
            )
        if schema_name and table_name not in self.show_tables_in_schema(schema_name):
            raise ValueError(
                f"Table '{table_name}' does not exist in schema '{schema_name}'."
            )

        # Build query
        query = (
            f"SELECT * FROM {schema_name}.{table_name}"
            if schema_name
            else f"SELECT * FROM {table_name}"
        )

        if limit:
            query += f" LIMIT {limit}"
        # Execute with context manager
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

        # df = pd.read_sql(query, self.connection)
        # return df

    def close(self):
        self.connection.close()
        self.engine.dispose()
