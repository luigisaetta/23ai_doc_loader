"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-09-05
Python Version: 3.11
"""

AUTH_TYPE = "API_KEY"

# current endpoint for OCI GenAI (embed and llm) models
# REGION = "us-chicago-1"
REGION = "eu-frankfurt-1"
ENDPOINT = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com"

# for chunking, in chars
# changed for embed v4
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

# we're taking K docs on each side to create a summary
# to be put in the chunk header
MODEL_4_SUMMARY = "meta.llama-3.3-70b-instruct"
# can be disabled, it can takes time
ENABLE_SUMMARY = False
SUMMARY_WINDOW = 2

VERBOSE = False

LANG_SUPPORTED = ["en", "it", "es", "fr", "de", "el", "nl", "ro"]

# OCI GenAI model used for Embeddings
# to batch embedding with OCI
# with Cohere embeddings max is 96
# value: OCI
EMBED_MODEL_TYPE = "OCI"
OCI_EMBED_MODEL = "cohere.embed-multilingual-v3.0"
# can be OCI or NVIDIA
# now we're using NVIDIA
# EMBED_MODEL_TYPE = "NVIDIA"
NVIDIA_EMBED_MODEL_URL = "http://130.61.225.137:8000/v1/embeddings"
NVIDIA_EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"

# OCI_EMBED_MODEL = "cohere.embed-v4.0"
# Oracle VS
EMBEDDINGS_BITS = 32

# Vector Store
VECTOR_STORE_TYPE = "23AI"

# to enable ADB connection
ADB = True
