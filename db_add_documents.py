"""
Load additional docs in an existing collection
"""

import sys
import os
import argparse
from glob import glob


from db_doc_loader_backend import (
    get_list_collections,
    get_books,
    add_docs_to_23ai,
    get_embed_model,
)
from chunk_index_utils import load_book_and_split
from utils import get_console_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP

# handle input for collection_name from command line
logger = get_console_logger()

parser = argparse.ArgumentParser(description="Document batch loading.")

parser.add_argument("collection_name", type=str, help="collection name to add documents to.")
parser.add_argument("books_dir", type=str, help="Dir with the books to load.")

args = parser.parse_args()
collection_name = args.collection_name
BOOKS_DIR = args.books_dir

# check if collection exist
collection_list = get_list_collections()

if collection_name not in collection_list:
    logger.info("")
    logger.error("Collection %s doesn't exist, exiting!", collection_name)
    logger.info("")

    sys.exit(-1)

# check for existing documents in collection
books_list = get_books(collection_name)

new_books_list = glob(BOOKS_DIR + "/*.pdf")

logger.info("")

docs = []

for book_pathname in new_books_list:
    # check if already loaded

    # strips path
    if os.path.basename(book_pathname) not in books_list:
        logger.info("Loading %s", book_pathname)

        docs += load_book_and_split(book_pathname, CHUNK_SIZE, CHUNK_OVERLAP)

    else:
        logger.info("Document %s already loaded, skipping...", book_pathname)

# embed and save to  DB
if len(docs) > 0:
    embed_model = get_embed_model()

    add_docs_to_23ai(docs, embed_model, collection_name)
