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
from custom_rest_embeddings import CustomRESTEmbeddings
from chunk_index_utils import load_and_split_pdf, load_and_split_docx, load_and_split_md

from config_private import DB_USER, DB_PWD, DSN, TNS_ADMIN, WALLET_PWD, COMPARTMENT_ID
from config import ENDPOINT, OCI_EMBED_MODEL, ADB, CHUNK_SIZE, CHUNK_OVERLAP
from config import (
    EMBED_MODEL_TYPE,
    NVIDIA_EMBED_MODEL,
    NVIDIA_EMBED_MODEL_URL,
    AUTH_TYPE,
)

logger = get_console_logger()


def get_db_connection():
    """
    get a connection to db
    """

    # common params
    conn_parms = {"user": DB_USER, "password": DB_PWD, "dsn": DSN, "retry_count": 3}

    if ADB:
        # connection to ADB, needs wallet
        conn_parms.update(
            {
                "config_dir": TNS_ADMIN,
                "wallet_location": TNS_ADMIN,
                "wallet_password": WALLET_PWD,
            }
        )

    logger.info("")
    logger.info("Connecting as USER: %s to DSN: %s", DB_USER, DSN)

    try:
        return oracledb.connect(**conn_parms)
    except oracledb.Error as e:
        logger.error("Database connection failed: %s", str(e))
        raise


def get_embed_model(model_type="OCI"):
    """
    get the Embeddings Model
    """
    check_value_in_list(model_type, ["OCI", "NVIDIA"])

    embed_model = None

    if model_type == "OCI":
        EMBED_MODEL_ID = OCI_EMBED_MODEL

        embed_model = OCIGenAIEmbeddings(
            auth_type=AUTH_TYPE,
            model_id=OCI_EMBED_MODEL,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
        )
    elif model_type == "NVIDIA":
        EMBED_MODEL_ID = NVIDIA_EMBED_MODEL

        embed_model = CustomRESTEmbeddings(
            api_url=NVIDIA_EMBED_MODEL_URL, model=NVIDIA_EMBED_MODEL
        )

    logger.info("")
    logger.info("Using embedding model: %s", EMBED_MODEL_ID)
    logger.info("")

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
    with get_db_connection() as conn:
        list_collections = OracleVS4DBLoading.list_collections(conn)

    return list_collections


def get_books(collection_name):
    """
    return the list of books in collection
    """
    with get_db_connection() as conn:
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
    v_uploaded_file, collection_name, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
):
    """
    load the uploaded file in the Vector Store and index

    this handles also the check to see if the file alredy exists
    """
    embed_model = get_embed_model(EMBED_MODEL_TYPE)

    result_status = ""

    # write a temporary file with the content
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        temp_file_path = write_temporary_file(tmp_dir_name, v_uploaded_file)

        # split in docs and prepare for loading
        _, file_ext = os.path.splitext(temp_file_path)

        docs = []

        if file_ext == ".pdf":
            docs += load_and_split_pdf(temp_file_path, chunk_size, chunk_overlap)
        if file_ext == ".docx":
            docs += load_and_split_docx(temp_file_path, chunk_size, chunk_overlap)

    # check if collection exists
    if collection_name in get_list_collections():
        # existing collection

        # check that the book has not already been loaded
        if v_uploaded_file.name not in get_books(collection_name):
            # add books to existing
            logger.info(
                "Add book %s to an existing collection...", v_uploaded_file.name
            )
            manage_collection(
                docs, embed_model, collection_name, is_new_collection=False
            )

            result_status = "OK"
        else:
            logger.info("Book %s already in collection...", v_uploaded_file.name)

            result_status = "KO"
    else:
        # new collection
        # this way it is safe that the collection doesn't exists
        logger.info("Creating the collection and adding documents...")
        logger.info("Add book %s to new collection...", v_uploaded_file.name)

        manage_collection(docs, embed_model, collection_name, is_new_collection=True)

        result_status = "OK"

    return result_status


def delete_documents_in_collection(collection_name, doc_names):
    """
    drop documents in the given collection
    """
    if len(doc_names) > 0:
        with get_db_connection() as conn:
            logger.info("Delete docs: %s in collection %s", doc_names, collection_name)
            OracleVS4DBLoading.delete_documents(conn, collection_name, doc_names)


def manage_collection(docs, embed_model, collection_name, is_new_collection):
    """
    Create or update a collection in the vector store.
    """
    with get_db_connection() as conn:
        if is_new_collection:
            logger.info(
                "Creating collection '%s' and adding documents...", collection_name
            )
            OracleVS4DBLoading.from_documents(
                docs,
                embed_model,
                client=conn,
                table_name=collection_name,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            logger.info(
                "Updating existing collection '%s' with new documents...",
                collection_name,
            )
            v_store = OracleVS4DBLoading(
                client=conn,
                table_name=collection_name,
                distance_strategy=DistanceStrategy.COSINE,
                embedding_function=embed_model,
            )
            v_store.add_documents(docs)
        logger.info("Operation completed for collection: %s", collection_name)
