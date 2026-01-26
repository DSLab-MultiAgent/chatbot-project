"""
Keyword 검색 모듈
BM25 기반 키워드 매칭 검색
"""
from typing import List
from rank_bm25 import BM25Okapi
from src.models import Document
from src.utils.logger import logger


class KeywordRetriever:
    """키워드 검색 클래스"""
    
    def __init__(self):
        """
        BM25 검색 엔진 초기화
        
        TODO:
        - [ ] 문서 코퍼스 로드
        - [ ] BM25 인덱스 생성
        - [ ] 한글 토크나이저 설정 (선택사항)
        """
        self.corpus = []  # 문서 리스트
        self.bm25 = None
        
        logger.info("KeywordRetriever 초기화")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        텍스트 토크나이징
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 리스트
            
        TODO:
        - [ ] 한글 형태소 분석 (KoNLPy 등)
        - [ ] 또는 단순 공백 분리
        - [ ] 불용어 제거
        """
        # 임시: 공백 분리
        return text.split()
    
    def build_index(self, documents: List[str]):
        """
        문서 인덱스 구축
        
        Args:
            documents: 문서 리스트
            
        TODO:
        - [ ] 문서 토크나이징
        - [ ] BM25 인덱스 생성
        """
        logger.info(f"BM25 인덱스 생성: {len(documents)}개 문서")
        
        self.corpus = documents
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("BM25 인덱스 생성 완료")
    
    async def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        키워드 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            
        Returns:
            BM25 점수 높은 문서 리스트
            
        TODO:
        - [ ] 쿼리 토크나이징
        - [ ] BM25 점수 계산
        - [ ] 상위 K개 문서 반환
        - [ ] Document 모델로 변환
        """
        logger.info(f"Keyword 검색: query='{query}', top_k={top_k}")
        
        if self.bm25 is None:
            logger.warning("BM25 인덱스가 없습니다")
            return []
        
        # TODO: 실제 검색 구현
        # 임시 반환
        documents = []
        
        logger.info(f"Keyword 검색 완료: {len(documents)}개")
        return documents