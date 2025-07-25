from google.cloud import bigquery
from pandas import DataFrame
import os


def open_file(file_path_wth_file_name: str) -> str:
    """Based on provided SQL path reads and returns the sql query.
    Angs:
        file path wth file name(str): SQl query file path
    Returns:
        1I P 10
        str-sq_ query
    """
    with open(file_path_wth_file_name, "r") as file:
        return str(file.read())


def format_sql_query(sql_query: str, params: dict) -> str:
    """Formats the sql script with params.
    Args:
        I
        sql query: Non formatted sql query.
    Returns:
        Str- Formatted sql script.
    """
    return sql_query.format(**params)

def update_sql_query(query_name: str, params: dict) -> str:
    """Reads and updates the sql params and query.
    query name (str) = soL query file name.
    params (dict): params needs to be updated in sql query.
    Returns:
        str - updated sql query.
    SE II SE
    """
    base_path = os.path.join(os.getcwd(), "mortgage", "utils", "sql_queries")
    query_path = os.path.join(base_path, query_name)
    sql_query = open_file(file_path_wth_file_name=query_path)
    sql_query = format_sql_query(sql_query, params)
    return sql_query

def bigquery_client() -> DataFrame:
    """Gets data from BQ table.
    Args:
        table_name(str): Name of the table of interest.
    Returns:
        DataFrame - response from BQ.
    """
    client = bigquery.Client(location='EU')
    table_id = "ltc-reboot25-team-56.mortgage_final.mortgage_dataset"
    query = f"SELECT * FROM `{table_id}`"
    response = client.query(query).to_dataframe()
    return response