"""
통합 Retriever 모듈
Vector와 Keyword 검색 결과를 통합
"""
from typing import List
from src.models import Document
from src.retrievers.hybrid_retriever import HybridRetriever
from src.config import settings
from src.utils.logger import logger


class Retriever:
    """통합 검색 클래스"""
    
    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.top_k_initial = settings.TOP_K_INITIAL
        self.top_k_secondary = settings.TOP_K_SECONDARY
    
    async def retrieve_initial(self, query: str) -> List[Document]:
        """
        1차 검색 (상위 1~5개)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            상위 문서 리스트
            
        TODO:
        - [ ] Hybrid 검색 실행
        - [ ] 상위 K개 문서 반환
        """
        logger.info(f"1차 검색 시작: top_k={self.top_k_initial}")
        
        documents = await self.hybrid_retriever.search(
            query=query,
            top_k=self.top_k_initial
        )
        
        logger.info(f"1차 검색 완료: {len(documents)}개 문서")
        return documents
    
    async def retrieve_secondary(self, query: str) -> List[Document]:
        """
        2차 검색 (상위 6~10개)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            추가 문서 리스트
            
        TODO:
        - [ ] 더 많은 문서 검색
        - [ ] 1차 검색 결과와 중복 제거
        """
        logger.info(f"2차 검색 시작: top_k={self.top_k_secondary}")
        
        documents = await self.hybrid_retriever.search(
            query=query,
            top_k=self.top_k_secondary
        )
        
        # 상위 5개 제외하고 6~10개만 반환
        secondary_docs = documents[self.top_k_initial:]
        
        logger.info(f"2차 검색 완료: {len(secondary_docs)}개 문서")
        return secondary_docs