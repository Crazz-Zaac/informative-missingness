from sqlalchemy import create_engine
from typing import Optional, List
import pandas as pd
from db_config import Config

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
        self.settings = Config()
        self.engine = create_engine(self.settings.get_db_uri(), echo=True)
        self.connection = self.engine.connect()

    def show_tables_in_schema(self, schema_name) -> List[str]:
        # Validate schema name
        if schema_name not in SchemaName.get_all():
            raise ValueError("Invalid schema name. Only 'hosp' and 'icu' are allowed.")
        result = self.connection.execute(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema_name}';"
        )
        tables = [row[0] for row in result]
        return tables

    def read_table_to_df(
        self, table_name: str, schema_name: Optional[str] = None
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
        # If schema name is provided, use it to query the table
        if schema_name:
            query = f"SELECT * FROM {schema_name}.{table_name}"
        else:
            query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.connection)
        return df

    def close(self):
        self.connection.close()
        self.engine.dispose()