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

    def get_table_info(self, table_name: str, schema_name: str) -> pd.DataFrame:
        if schema_name and not SchemaName.is_valid(schema_name):
            raise ValueError("Invalid schema name. Only 'hosp' and 'icu' are allowed.")
        if schema_name and table_name not in self.show_tables_in_schema(schema_name):
            raise ValueError(
                f"Table '{table_name}' does not exist in schema '{schema_name}'."
            )

        # query to get the size, number of rows, columns, and data types of the table
        query = text(
            f"""
            SELECT 
                table_name, 
                pg_size_pretty(pg_total_relation_size(:schema_name || '.' || :table_name)) AS size,
                (SELECT COUNT(*) FROM {schema_name}.{table_name}) AS num_rows,
                array_agg(column_name || ' ' || data_type) AS columns
            FROM information_schema.columns
            WHERE table_schema = :schema_name AND table_name = :table_name
            GROUP BY table_name;
            """
        )
        result = self.connection.execute(
            query, {"schema_name": schema_name, "table_name": table_name}
        )
        # Convert result to DataFrame
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if df.empty:
            raise ValueError(
                f"No information found for table '{table_name}' in schema '{schema_name}'."
            )
        return df

    def read_table_to_df(
        self,
        table_name: str,
        schema_name: str,
        limit: Optional[int] = None,
        order_by: str = "subject_id"
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
        query = f"""
            SELECT * FROM {schema_name}.{table_name} 
            ORDER BY {order_by}
            LIMIT {limit if limit is not None else 5000}
        """

        # Execute with context manager
        with self.engine.connect() as conn:
            print(
                f"Loading {limit if limit else 5000} data {table_name} from {schema_name} schema"
            )
            df = pd.read_sql(query, conn)
        if df.empty:
            raise ValueError(
                f"No data found in table '{table_name}' in schema '{schema_name}'."
            )
        return df


    def execute_query(self, query: str):
        """
        Execute a raw SQL query and return the result as a pandas DataFrame.
        """
        with self.engine.connect() as conn:
            result = pd.read_sql(query, conn)
        if result.empty:
            raise ValueError("No data returned from the query.")
        return result
    