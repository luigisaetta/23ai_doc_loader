"""
Utility to drop an existing collection

beware: check what you're doing
"""

from db_doc_loader_backend import get_db_connection
from oraclevs_4_db_loading import OracleVS4DBLoading

from utils import get_console_logger

logger = get_console_logger()

conn = get_db_connection()

OracleVS4DBLoading.drop_collection(conn, "TEST1")

conn.close()
