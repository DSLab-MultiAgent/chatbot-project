# src/retrievers/hybrid_retriever.py
from typing import List
from src.models import Document
from src.retrievers.vector_retriever import VectorRetriever

class HybridRetriever:
    def __init__(self):
        self.vector = VectorRetriever()

    async def search(self, query: str, sections: List[str], top_k: int = 10) -> List[Document]:
        return await self.vector.search(query=query, sections=sections, top_k=top_k)
