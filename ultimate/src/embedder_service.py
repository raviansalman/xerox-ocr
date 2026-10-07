from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

class EmbedRequest(BaseModel):
    texts: List[str]

@app.on_event("startup")
def load_model():
    global model
    logger.info("Initializing centralized embedding model...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu", cache_folder="/app/cache")
    logger.info("Model loaded successfully.")

@app.post("/embed")
def embed(req: EmbedRequest):
    embeddings = model.encode(req.texts, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)
    return {"embeddings": embeddings.tolist()}

@app.get("/health")
async def health():
    return {"status": "ok"}
