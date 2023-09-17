"""Wrapper around Transformers embedding models."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field

from langchain.embeddings.base import Embeddings

import torch
import numpy as np


class TransformerEmbeddings(BaseModel, Embeddings):
    """Wrapper for embedding models from the transformers library that require a separate tokenizer,
    to usee you should have the ''transformers'' python package installed.
    """
    tokenizer_path: Optional[str] = None
    doc_embedding_path: Optional[str] = None
    query_embedding_path: Optional[str] = None
    tokenizer: Optional[Any] = None
    doc_model: Optional[Any] = None
    query_model: Optional[Any] = None
    device: Optional[Any] = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, tokenizer, doc_embedding, query_embedding, **kwargs):
        super().__init__(**kwargs)
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            ) from exc

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer)
        self.doc_model = AutoModel.from_pretrained(pretrained_model_name_or_path=doc_embedding).to(self.device)
        self.query_model = AutoModel.from_pretrained(pretrained_model_name_or_path=query_embedding).to(self.device)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a transformer model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        _gpu_doc_batch_size = 5
        np_embeddings = []
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        for start_index in range(0, len(texts), _gpu_doc_batch_size):
            end_index = min(len(texts), start_index+_gpu_doc_batch_size)
            text_batch = texts[start_index:end_index]
            document_input = self.tokenizer(text_batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
            document_emb = self.doc_model(**document_input).last_hidden_state[:, 0, :]
            document_emb_as_np = document_emb.numpy(force = True)

            np_embeddings.append(document_emb_as_np)

        embeddings = np.concatenate(np_embeddings)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        query_input = self.tokenizer(text, max_length=512, truncation=True, return_tensors='pt').to(self.device)
        query_emb = self.query_model(**query_input).last_hidden_state[:, 0, :]
        query_emb_as_np = query_emb.numpy(force = True)
        #return list(query_emb_as_np.flatten())
        return query_emb_as_np.flatten()