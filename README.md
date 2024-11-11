# 23ai Documents Loader
This repo contains the code for utilities to **load documents and embeddings** in 23AI DB Vector Store

## Features
* supports Base Database Service and ADB
* creation of the collection and **First loading** of pdf documents
* addition of documents after the first load

## Scenario
* **Oracle 23AI** is used as Vector Store (ADB, Base DB Service)
* **Cohere** Embeddings multi-lingual as Embeddings Model, from OCI GenAI
* DB table created: default structure from Oracle LangChain integration

## Environment
The code has been tested with **Python 3.11**.

Create a fresh environment with the command and activate it

```
conda create -n <env_name> python==3.11

conda activate <env_name>
```

To install all the required packages, download the file requirements.txt from the repository
and execute:

```
pip install -r requirements.txt
```

## Utilities available
* **test** the connection with test_db_connection.py
* do a **first loading** with db_ai_first_load.sh
* **list** the document loaded in the collection with db_list_documents.py
* add **more documents** with db_add-documents.py
* **drop** a collection with db_drop_collection.py

## References
* [OracleVS](https://python.langchain.com/v0.2/docs/integrations/vectorstores/oracle/)

## Updates
* environment updated to latest: oci, langchain, oracledb (see: requirements.txt)