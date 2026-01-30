"""
문서 검증 모듈 (관련성 체크)
검색된 문서가 쿼리와 관련 있는지 YES/NO로 판단
"""
from typing import List, Tuple
from src.models import Document
from src.agents.llm_client import LLMClient
from src.utils.logger import logger
from src.config import yaml_config


class DocumentValidator:
    """문서 검증 클래스 (관련성 이진 분류)"""
        
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_template = yaml_config.get('prompts', {}).get('document_validation', '')  

    async def validate_document(
        self, 
        query: str, 
        document: Document
    ) -> Tuple[bool, str]:
        """
        개별 문서의 관련성 판단 (YES/NO)
        
        Args:
            query: 사용자 질문
            document: 검증할 문서
            
        Returns:
            (is_relevant, reason)
            - is_relevant: True(관련 있음) / False(관련 없음)
            - reason: 판단 이유 (디버깅용)
            
        판단 기준:
        - YES: 질문에 직접적으로 관련됨 OR 관련성이 높음 OR 부분적으로 관련 있음
        - NO: 관련성 낮음
        """
        
        # 프롬프트 생성
        prompt = self.prompt_template.format(
            query=query,
            document=document.content
        )
        
        try:
            response = await self.llm_client.generate(prompt, temperature=0.1)
            response = response.strip().upper()
            
            is_relevant = "YES" in response
            reason = "관련 있음" if is_relevant else "관련 없음"
            
            logger.debug(
                f"문서 관련성: {reason} "
                f"(원본 점수: {document.score:.2f})"
            )
            
            return is_relevant, reason
            
        except Exception as e:
            logger.error(f"문서 검증 실패: {e}")
            # 에러 시 원본 점수로 판단 (0.5 이상이면 관련 있음)
            is_relevant = document.score >= 0.5
            return is_relevant, f"에러 발생 (원본 점수 기준: {document.score:.2f})"
    
    async def get_relevant_documents(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Document]:
        """
        관련성 체크: 관련 있는 문서만 필터링 (YES인 것만)
        
        Args:
            query: 사용자 질문
            documents: 검증할 문서 리스트
            
        Returns:
            관련 있는 문서 리스트 (YES로 판단된 것만)
        """
        logger.info(f"관련성 체크 시작: {len(documents)}개 문서")
        
        relevant_docs = []
        
        for i, doc in enumerate(documents, 1):
            # 개별 문서 관련성 체크 (YES/NO)
            is_relevant, reason = await self.validate_document(query, doc)
            
            # 메타데이터에 검증 결과 저장
            doc.metadata['is_relevant'] = is_relevant
            doc.metadata['validation_reason'] = reason
            doc.metadata['original_score'] = doc.score
            
            # 관련 있는 문서만 추가
            if is_relevant:
                relevant_docs.append(doc)
                logger.debug(f"문서 {i}: ✅ 관련 있음 - {reason}")
            else:
                logger.debug(f"문서 {i}: ❌ 관련 없음 - {reason}")
        
        # 원본 검색 점수 순서 유지 (이미 정렬되어 있음)
        logger.info(
            f"관련성 체크 완료: {len(relevant_docs)}/{len(documents)}개 관련 문서"
        )
        return relevant_docs