"""
Custom class to support Embeddings model deployed using NVIDIA E.

License: MIT
"""

from typing import List
from langchain_core.embeddings import Embeddings
import requests

ALLOWED_DIMS = {384, 512, 768, 1024, 2048}


class CustomRESTEmbeddings(Embeddings):
    """
    Custom class to wrap an embedding model with rest interface from NVIDIA NIM

    see:
        https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_2-nv-embedqa-1b-v2-infer
    """

    def __init__(self, api_url: str, model: str, batch_size: int = 10, dimensions=2048):
        """
        Init

        as of now, no security
        args:
            api_url: the endpoint
            model: the model id string
            batch_size
            dimensions: dim of the embedding vector
        """
        self.api_url = api_url
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions

        # Validation at init time (optional)
        if self.dimensions is not None and self.dimensions not in ALLOWED_DIMS:
            raise ValueError(
                f"Invalid dimensions {self.dimensions!r}: must be one of {sorted(ALLOWED_DIMS)}"
            )

    def embed_documents(
        self,
        texts: List[str],
        # must be passage and not document
        input_type: str = "passage",
        truncate: str = "NONE",
    ) -> List[List[float]]:
        """
        Embed a list of documents using batching.
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # process a single batch
            resp = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "input": batch,
                    "input_type": input_type,
                    "truncate": truncate,
                    "dimensions": self.dimensions,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if len(data) != len(batch):
                raise ValueError(f"Expected {len(batch)} embeddings, got {len(data)}")
            all_embeddings.extend(item["embedding"] for item in data)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed the query (a str)
        """
        return self.embed_documents([text], input_type="query")[0]
