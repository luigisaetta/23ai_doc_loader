"""
Print the list of documents in the collection
"""

import argparse

from db_doc_loader_backend import get_db_connection
from oraclevs_4_db_loading import OracleVS4DBLoading

from utils import get_console_logger

# handle input for collection_name from command line
logger = get_console_logger()

conn = get_db_connection()

coll_list = OracleVS4DBLoading.list_collections(conn)

logger.info("")
logger.info("List of collections:")
logger.info("")

for coll in coll_list:
    logger.info("* %s", coll)

logger.info("")
