"""
Docs loader backend function

to separate UI logic from backend logic
"""

import os
import tempfile
import oracledb

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import OCIGenAIEmbeddings
from oraclevs_4_db_loading import OracleVS4DBLoading
from translations import translations
from utils import get_console_logger, check_value_in_list

from chunk_index_utils import (
    load_book_and_split,
)

from config_private import DB_USER, DB_PWD, DSN, TNS_ADMIN, WALLET_PWD, COMPARTMENT_ID
from config import ENDPOINT, OCI_EMBED_MODEL, ADB

logger = get_console_logger()


def get_db_connection():
    """
    get a connection to db
    """

    # common params
    conn_parms = {"user": DB_USER, "password": DB_PWD, "dsn": DSN, "retry_count": 3}

    if ADB:
        # connection to ADB, needs wallet
        logger.info("Connecting to ADB database...")

        conn_parms["config_dir"] = TNS_ADMIN
        conn_parms["wallet_location"] = TNS_ADMIN
        conn_parms["wallet_password"] = WALLET_PWD

    logger.info("")
    logger.info("Connecting as USER: %s to DSN: %s", DB_USER, DSN)

    conn = oracledb.connect(**conn_parms)

    return conn


def get_embed_model(model_type="OCI"):
    """
    get the Embeddings Model
    """
    check_value_in_list(model_type, ["OCI"])

    logger.info("")
    logger.info("Using embedding model: %s", OCI_EMBED_MODEL)
    logger.info("")

    embed_model = None

    if model_type == "OCI":
        embed_model = OCIGenAIEmbeddings(
            auth_type="API_KEY",
            model_id=OCI_EMBED_MODEL,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
        )

    return embed_model


# to handle multilingual use the dictionary in translations.py
def translate(text, v_lang):
    """
    to handle labels in different lang
    """
    return translations.get(v_lang, {}).get(text, text)


def get_list_collections():
    """
    return the list of available collections in the DB
    """
    conn = get_db_connection()

    list_collections = OracleVS4DBLoading.list_collections(conn)

    return list_collections


def get_books(collection_name):
    """
    return the list of books in collection
    """
    conn = get_db_connection()

    list_books_in_collection = OracleVS4DBLoading.list_books_in_collection(
        connection=conn, collection_name=collection_name
    )

    return list_books_in_collection


def write_temporary_file(v_tmp_dir_name, v_uploaded_file):
    """
    Write the uploaded file as a temporary file
    """
    temp_file_path = os.path.join(v_tmp_dir_name, v_uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(v_uploaded_file.getbuffer())

    return temp_file_path


def load_uploaded_file_in_vector_store(
    v_uploaded_file, collection_name, chunk_size, chunk_overlap
):
    """
    load the uploaded file in the Vector Store and index

    this handles also the check to see if the file alredy exists
    """
    embed_model = get_embed_model()

    result_status = ""

    # write a temporary file with the content
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        temp_file_path = write_temporary_file(tmp_dir_name, v_uploaded_file)

        # split in docs and prepare for loading
        docs = load_book_and_split(temp_file_path, chunk_size, chunk_overlap)

    # check if collection exists
    if collection_name in get_list_collections():
        # existing collection

        # check that the book has not already been loaded
        if v_uploaded_file.name not in get_books(collection_name):
            # add books to existing
            logger.info(
                "Add book %s to an existing collection...", v_uploaded_file.name
            )

            add_docs_to_23ai(docs, embed_model, collection_name)

            result_status = "OK"
        else:
            logger.info("Book %s already in collection...", v_uploaded_file.name)

            result_status = "KO"
    else:
        # new collection
        # this way it is safe that the collection doesn't exists
        logger.info("Creating the collection and adding documents...")
        logger.info("Add book %s to new collection...", v_uploaded_file.name)

        create_collection_and_add_docs_to_23ai(docs, embed_model, collection_name)

        result_status = "OK"

    return result_status


def delete_documents_in_collection(collection_name, doc_names):
    """
    drop documents in the given collection
    """
    if len(doc_names) > 0:
        conn = get_db_connection()

        logger.info("Delete docs: %s in collection %s", doc_names, collection_name)
        OracleVS4DBLoading.delete_documents(conn, collection_name, doc_names)


def create_collection_and_add_docs_to_23ai(docs, embed_model, collection_name):
    """
    create the collection and load docs in that collection
    To be used only for a NEW collection
    """

    try:
        conn = get_db_connection()

        OracleVS4DBLoading.from_documents(
            docs,
            embed_model,
            client=conn,
            table_name=collection_name,
            distance_strategy=DistanceStrategy.COSINE,
        )

        logger.info("Created collection and documents saved...")

    except oracledb.Error as e:
        err_msg = "An error occurred in create_collection_and_add_docs: " + str(e)
        logger.error(err_msg)


def add_docs_to_23ai(docs, embed_model, collection_name):
    """
    add docs from a book to Oracle vector store
    This is used for an existing collection
    """

    try:

        conn = get_db_connection()

        v_store = OracleVS4DBLoading(
            client=conn,
            table_name=collection_name,
            distance_strategy=DistanceStrategy.COSINE,
            embedding_function=embed_model,
        )

        logger.info("Saving new documents to Vector Store...")

        v_store.add_documents(docs)

        logger.info("Saved new documents to Vector Store !")

    except oracledb.Error as e:
        err_msg = "An error occurred in add_docs_to_23ai: " + str(e)
        logger.error(err_msg)
