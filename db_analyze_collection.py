"""
Analyze a collection
"""

import argparse

from db_doc_loader_backend import get_db_connection
from oraclevs_4_db_loading import OracleVS4DBLoading

from utils import get_console_logger

# handle input for collection_name from command line
logger = get_console_logger()

parser = argparse.ArgumentParser(description="Analyzew a collection.")

parser.add_argument("collection_name", type=str, help="collection name.")

args = parser.parse_args()
collection_name = args.collection_name

with get_db_connection() as conn:
    report = OracleVS4DBLoading.analyze_collection(conn, collection_name)

logger.info("")
print("")
print(report)
logger.info("")
