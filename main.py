import os
from contextlib import asynccontextmanager
from typing import List

import tiktoken
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# set Embedding Model path
EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'BAAI/bge-large-zh-v1.5')


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    embeddings = [embedding_model.encode(text) for text in request.input]
    embeddings = [embedding.tolist() for embedding in embeddings]

    def num_tokens_from_string(string: str) -> int:
        """
        Returns the number of tokens in a text string.
        use cl100k_base tokenizer
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    response = {
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": index
            }
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in request.input),  # how many characters in prompt
            "total_tokens": sum(num_tokens_from_string(text) for text in request.input),  # how many tokens (encoding)
        },
    }
    return response


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(EMBEDDING_PATH, device=device)
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
