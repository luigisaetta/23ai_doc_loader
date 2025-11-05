"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2025-07-15
Python Version: 3.11

Usage: contains the functions to split in chunks and create the index

Update: started to add more "context engineering"
"""

import time
from collections import defaultdict
from tqdm import tqdm
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_unstructured import UnstructuredLoader

from utils import get_console_logger, remove_path_from_ref
from config import (
    VERBOSE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ENABLE_SUMMARY,
    SUMMARY_WINDOW,
    MODEL_4_SUMMARY,
    ENDPOINT,
    AUTH_TYPE,
)
from config_private import COMPARTMENT_ID

logger = get_console_logger()


def generate_summary(text: str) -> str:
    """
    Given a text (set of chunks) generate a summary
    """
    # to avoid to be throttled with on-demand
    if ENABLE_SUMMARY:
        time.sleep(1)

        # 1. Create prompt template that keeps same language
        prompt_template = PromptTemplate(
            input_variables=["text"],
            template=(
                "Read the following text and generate a brief summary in the **same language** of the original text.\n\n"
                "Return ONLY the summary, do not add comments or any other text."
                "Text:\n{text}\n\n"
                "Summary:"
            ),
        )

        # 2. Set up OCI LLM
        llm = ChatOCIGenAI(
            auth_type=AUTH_TYPE,
            model_id=MODEL_4_SUMMARY,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
        )
        # 3. Set up LangChain chain
        summary_chain = prompt_template | llm

        # 4. Run the chain and return result
        return summary_chain.invoke({"text": text}).content

    # summary not enabled
    return ""


def get_recursive_text_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    return a recursive text splitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter


def load_and_split_pdf(book_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    load a single book in pdf format
    """
    text_splitter = get_recursive_text_splitter(chunk_size, chunk_overlap)

    loader = PyPDFLoader(file_path=book_path)

    docs = loader.load_and_split(text_splitter=text_splitter)

    chunk_header = ""

    doc_name = remove_path_from_ref(book_path)
    # split to remove the extension
    doc_title = doc_name.split(".")[0]

    # modified (15/07/2025)
    processed_docs = []
    # summary is built over 2k + 1 chunks
    k = SUMMARY_WINDOW

    for i, doc in tqdm(enumerate(docs), total=len(docs), desc="Processing docs"):
        # to generate a summary taking a window of 2k+1 chunks
        start = max(0, i - k)
        end = min(len(docs), i + k + 1)
        context_docs = docs[start:end]
        context_text = " ".join(d.page_content for d in context_docs)

        # generate the header
        chunk_header = f"# Doc. title: {doc_title}\n"

        if ENABLE_SUMMARY:
            # generate the summary of the window around doc
            summary = generate_summary(context_text)
            # add to header
            chunk_header += f"Summary: {summary}\n\n"

        new_doc = Document(
            page_content=chunk_header + doc.page_content,
            metadata={
                "source": doc_name,
                "page_label": doc.metadata.get("page_label", None),
            },
        )
        if VERBOSE:
            logger.info(new_doc)

        processed_docs.append(new_doc)

    logger.info("Loaded %s chunks...", len(docs))

    return processed_docs


def load_and_split_docx(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    To load docx files
    """
    loader = UnstructuredLoader(file_path)
    docs = loader.load()

    # Raggruppa per numero di pagina (o altro metadato)
    grouped_text = defaultdict(list)

    chunk_header = ""

    if len(docs) > 0:
        doc_name = remove_path_from_ref(file_path)
        # split to remove the extension
        doc_title = doc_name.split(".")[0]
        chunk_header = f"# Doc. title: {doc_title}\n"

    for doc in docs:
        # fallback to 0 if not available
        page = doc.metadata.get("page_number", 0)
        grouped_text[page].append(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    final_chunks = []

    # Per ogni pagina (o gruppo), unisci il testo e splitta
    for page, texts in grouped_text.items():
        full_text = "\n".join(texts)
        splits = splitter.split_text(full_text)

        for chunk in splits:
            final_chunks.append(
                Document(
                    # add more context
                    page_content=chunk_header + chunk,
                    metadata={
                        "source": doc_name,
                        "page_label": str(page),
                    },
                )
            )

    logger.info("Loaded %s chunks...", len(final_chunks))

    return final_chunks


def load_and_split_md(book_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    add a single document in markdown format
    """
    text_splitter = get_recursive_text_splitter(chunk_size, chunk_overlap)

    loader = UnstructuredMarkdownLoader(book_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    chunk_header = ""
    doc_name = remove_path_from_ref(book_path)
    # split to remove the extension
    doc_title = doc_name.split(".")[0]

    # modified (15/07/2025)
    processed_docs = []
    # summary is built over 2k + 1 chunks
    k = SUMMARY_WINDOW

    for i, doc in tqdm(enumerate(docs), total=len(docs), desc="Processing docs"):
        # to generate a summary taking a window of 2k+1 chunks
        start = max(0, i - k)
        end = min(len(docs), i + k + 1)
        context_docs = docs[start:end]
        context_text = " ".join(d.page_content for d in context_docs)

        # generate the header
        chunk_header = f"# Doc. title: {doc_title}\n"

        if ENABLE_SUMMARY:
            # generate the summary of the window around doc
            summary = generate_summary(context_text)
            # add to header
            chunk_header += f"Summary: {summary}\n\n"

        new_doc = Document(
            page_content=chunk_header + doc.page_content,
            metadata={
                "source": doc_name,
                "page_label": doc.metadata.get("page_label", None),
            },
        )
        if VERBOSE:
            logger.info(new_doc)

        processed_docs.append(new_doc)

    logger.info("Loaded %s chunks...", len(docs))

    return processed_docs
