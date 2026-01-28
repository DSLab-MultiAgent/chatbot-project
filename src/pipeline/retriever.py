"""
통합 Retriever 모듈
카테고리 기반 하이브리드 검색
"""
from typing import List, Optional
from src.models import Document
from src.retrievers.hybrid_retriever import HybridRetriever
from src.config import settings
from src.utils.logger import logger


class Retriever:
    """통합 검색 클래스"""
    
    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.top_k = 10  # 한번에 10개 검색
    
    async def search(
        self, 
        query: str, 
        category: Optional[str] = None
    ) -> List[Document]:
        """
        카테고리 기반 하이브리드 검색 (Late Interaction)
        
        Args:
            query: 검색 쿼리
            category: 분류된 카테고리 (필터링용)
            
        Returns:
            상위 10개 문서
            
        TODO:
        - [ ] Late Interaction 검색 구현
        - [ ] 카테고리 필터링 구현
        - [ ] 검색 결과 최적화
        """
        logger.info(f"하이브리드 검색 시작: category={category}, top_k={self.top_k}")
        
        # 하이브리드 검색 실행
        documents = await self.hybrid_retriever.search(
            query=query,
            top_k=self.top_k,
            category=category  # 카테고리 필터 전달
        )
        
        logger.info(f"검색 완료: {len(documents)}개 문서")
        return documents
    
    def get_top_n(self, documents: List[Document], n: int) -> List[Document]:
        """
        상위 N개 문서만 추출
        
        Args:
            documents: 전체 문서 리스트
            n: 추출할 개수
            
        Returns:
            상위 N개 문서
        """
        return documents[:n]