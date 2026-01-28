"""
문서 검증 모듈
검색된 문서가 쿼리와 실제로 관련있는지 개별 검증
"""
from typing import List
from src.models import Document
from src.agents.llm_client import LLMClient
from src.utils.logger import logger


class DocumentValidator:
    """문서 검증 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.relevance_threshold = 0.6  # 관련성 임계값
    
    async def validate_document(
        self, 
        query: str, 
        document: Document
    ) -> float:
        """
        개별 문서의 관련성 점수 계산
        
        Args:
            query: 사용자 질문
            document: 검증할 문서
            
        Returns:
            관련성 점수 (0.0 ~ 1.0)
            
        TODO:
        - [ ] LLM 기반 관련성 판단
        - [ ] 프롬프트 최적화
        - [ ] 점수 계산 로직 개선
        """
        prompt = f"""
다음 문서가 사용자 질문에 답변하는데 도움이 되는지 0.0 ~ 1.0 사이의 점수로 평가해주세요.

사용자 질문: {query}

문서 내용:
{document.content[:500]}...

평가 기준:
- 1.0: 질문에 직접적으로 답변 가능
- 0.7~0.9: 질문과 관련성이 높음
- 0.4~0.6: 부분적으로 관련 있음
- 0.0~0.3: 관련성 없음

숫자만 답변하세요 (예: 0.85):
"""
        
        try:
            response = await self.llm_client.generate(prompt, temperature=0.1)
            score = float(response.strip())
            score = max(0.0, min(1.0, score))  # 0~1 범위로 제한
            
            logger.debug(f"문서 검증 점수: {score:.2f}")
            return score
            
        except Exception as e:
            logger.error(f"문서 검증 실패: {e}")
            # 에러 시 기존 유사도 점수 사용
            return document.score
    
    async def validate_documents(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Document]:
        """
        여러 문서를 검증하고 관련성 순으로 재정렬
        
        Args:
            query: 사용자 질문
            documents: 검증할 문서 리스트
            
        Returns:
            검증 점수로 재정렬된 문서 리스트
            
        TODO:
        - [ ] 병렬 처리로 속도 개선
        - [ ] 배치 처리
        """
        logger.info(f"문서 검증 시작: {len(documents)}개")
        
        validated_docs = []
        
        for doc in documents:
            # 개별 문서 검증
            relevance_score = await self.validate_document(query, doc)
            
            # 기존 Document에 검증 점수 업데이트
            # metadata에 추가 정보 저장
            doc.metadata['validation_score'] = relevance_score
            doc.metadata['original_score'] = doc.score
            
            # 최종 점수는 검증 점수 사용
            doc.score = relevance_score
            
            validated_docs.append(doc)
        
        # 검증 점수로 재정렬
        validated_docs.sort(key=lambda x: x.score, reverse=True)
        
        # 임계값 이상인 문서만 필터링
        valid_docs = [
            doc for doc in validated_docs 
            if doc.score >= self.relevance_threshold
        ]
        
        logger.info(
            f"문서 검증 완료: {len(valid_docs)}/{len(documents)}개 유효"
        )
        return valid_docs