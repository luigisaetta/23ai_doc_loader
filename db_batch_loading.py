"""
Batch loading

Create a new collection and load a set of pdf
Can be used ONLY for a new collection.

sept 2024: refactored to reduce dependencies
"""

import os
import sys
import argparse
from glob import glob

from chunk_index_utils import (
    load_and_split_pdf,
    load_and_split_docx,
)
from db_doc_loader_backend import (
    get_list_collections,
    get_embed_model,
    manage_collection,
)

from utils import get_console_logger, compute_stats
from config import CHUNK_SIZE, CHUNK_OVERLAP


#
# Main
#

# handle input for new_collection_name from command line
parser = argparse.ArgumentParser(description="Document batch loading.")

parser.add_argument("new_collection_name", type=str, help="New collection name.")
parser.add_argument("books_dir", type=str, help="Dir with the books to load.")

args = parser.parse_args()

new_collection_name = args.new_collection_name
BOOKS_DIR = args.books_dir

logger = get_console_logger()

logger.info("")
logger.info("Batch loading books in collection %s ...", new_collection_name)
logger.info("")

# init models
embed_model = get_embed_model()

# check that the collection doesn't exist yet
collection_list = get_list_collections()

if new_collection_name in collection_list:
    logger.info("")
    logger.error("Error: collection %s already exist!", new_collection_name)
    logger.error("Exiting !")
    logger.info("")

    sys.exit(-1)

logger.info("")

# the list of books to be loaded
books_list = glob(BOOKS_DIR + "/*.pdf") + glob(BOOKS_DIR + "/*.docx")

logger.info("These books will be loaded:")
for book in books_list:
    logger.info(book)

logger.info("")

logger.info("Parameters used for chunking:")
logger.info("Chunk size: %s chars", CHUNK_SIZE)
logger.info("Chunk overlap: %s chars", CHUNK_OVERLAP)
logger.info("")

docs = []

for book in books_list:
    logger.info("Chunking: %s", book)
    # get the file extension
    _, file_ext = os.path.splitext(book)

    if file_ext == ".pdf":
        docs += load_and_split_pdf(book, CHUNK_SIZE, CHUNK_OVERLAP)
    if file_ext == ".docx":
        docs += load_and_split_docx(book, CHUNK_SIZE, CHUNK_OVERLAP)

if len(docs) > 0:
    logger.info("")
    logger.info(
        "Embedding and loading documents in collection %s ...", new_collection_name
    )
    manage_collection(docs, embed_model, new_collection_name, is_new_collection=True)

    logger.info("Loading completed.")
    logger.info("")

    mean, stdev, perc_75 = compute_stats(docs)

    logger.info("")
    logger.info("Statistics on the distribution of chunks' lengths:")
    logger.info("Total num. of chunks loaded: %s", len(docs))
    logger.info("Avg. length: %s (chars)", mean)
    logger.info("Std dev: %s (chars)", stdev)
    logger.info("75-perc: %s (chars)", perc_75)
    logger.info("")

else:
    logger.info("No document to load!")
    logger.info("")
