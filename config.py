"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-09-05
Python Version: 3.11
"""

# these are parameters probably to check & change
# current endpoint for OCI GenAI (embed and llm) models
ENDPOINT = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"

# for chunking, in chars
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

# end check & change

VERBOSE = True

LANG_SUPPORTED = ["en", "it", "es", "fr", "de", "el", "nl", "ro"]

# OCI GenAI model used for Embeddings
# to batch embedding with OCI
# with Cohere embeddings max is 96
# value: OCI
OCI_EMBED_MODEL = "cohere.embed-multilingual-v3.0"

# Oracle VS
EMBEDDINGS_BITS = 32

# Vector Store
VECTOR_STORE_TYPE = "23AI"

# to enable ADB connection
ADB = True
