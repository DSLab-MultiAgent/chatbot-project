"""
Hybrid Retriever 모듈
Vector 검색과 Keyword 검색 결과를 통합
"""
from typing import List
from src.models import Document
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.keyword_retriever import KeywordRetriever
from src.config import yaml_config
from src.utils.logger import logger


class HybridRetriever:
    """Hybrid 검색 클래스"""
    
    def __init__(self):
        """
        Vector와 Keyword Retriever 초기화
        
        TODO:
        - [ ] 가중치 설정 로드
        - [ ] 두 검색 엔진 초기화
        """
        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = KeywordRetriever()
        
        # 가중치 설정
        retriever_config = yaml_config.get('retriever', {})
        self.vector_weight = retriever_config.get('vector_weight', 0.7)
        self.keyword_weight = retriever_config.get('keyword_weight', 0.3)
        
        logger.info(
            f"HybridRetriever 초기화: "
            f"vector={self.vector_weight}, keyword={self.keyword_weight}"
        )
    
    def _merge_results(
        self, 
        vector_docs: List[Document], 
        keyword_docs: List[Document],
        top_k: int
    ) -> List[Document]:
        """
        두 검색 결과를 통합
        
        Args:
            vector_docs: 벡터 검색 결과
            keyword_docs: 키워드 검색 결과
            top_k: 반환할 문서 수
            
        Returns:
            통합된 문서 리스트
            
        TODO:
        - [ ] 점수 가중 평균 계산
        - [ ] 중복 문서 제거
        - [ ] 점수 순으로 정렬
        - [ ] 상위 K개 반환
        """
        logger.debug("검색 결과 통합 중...")
        
        # TODO: 실제 통합 로직 구현
        # 임시: vector 결과만 반환
        merged = vector_docs[:top_k]
        
        logger.debug(f"통합 완료: {len(merged)}개 문서")
        return merged
    
    async def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Hybrid 검색 실행
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            
        Returns:
            통합된 검색 결과
            
        TODO:
        - [ ] Vector 검색 실행
        - [ ] Keyword 검색 실행
        - [ ] 결과 통합
        """
        logger.info(f"Hybrid 검색 시작: query='{query}', top_k={top_k}")
        
        # 두 검색 동시 실행
        vector_docs = await self.vector_retriever.search(query, top_k=top_k*2)
        keyword_docs = await self.keyword_retriever.search(query, top_k=top_k*2)
        
        # 결과 통합
        merged_docs = self._merge_results(vector_docs, keyword_docs, top_k)
        
        logger.info(f"Hybrid 검색 완료: {len(merged_docs)}개 문서")
        return merged_docs