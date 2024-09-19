# 23ai Documents Loader
This repo contains the code for utilities to **load documents and embeddings** in 23AI DB

## Features
* supports Base Database Service and ADB
* creation of the collection and **firstloading** of pdf documents
* addition of documents after the first load

## Scenario
* **Oracle 23AI** is used as Vector Store (ADB, Base DB Service)
* Cohere Embeddings multi-lingual as Embeddings Model, from OCI GenAI
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

## References
* [OracleVS](https://python.langchain.com/v0.2/docs/integrations/vectorstores/oracle/)
