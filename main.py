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

# 设置文本向量模型
EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'BAAI/bge-large-zh-v1.5')
# 设置服务监听ip
EMBEDDING_HOST = os.environ.get('EMBEDDING_HOST', '0.0.0.0')
# 设置服务监听端口
EMBEDDING_PORT = int(os.environ.get('EMBEDDING_PORT', 8000))
# 设置服务进程数量
EMBEDDING_WORKERS = int(os.environ.get('EMBEDDING_WORKERS', 1))

# 模型全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(EMBEDDING_PATH, device=device)


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


# 提取文本向量接口
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


# 开启服务
def start_server():
    uvicorn.run("main:app", host=EMBEDDING_HOST, port=EMBEDDING_PORT, workers=EMBEDDING_WORKERS)


if __name__ == "__main__":
    start_server()
