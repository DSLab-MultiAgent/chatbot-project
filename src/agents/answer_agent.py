"""
답변 생성 에이전트
검색된 문서를 바탕으로 최종 답변 생성
"""
from typing import List
from src.models import Document, QueryResponse, ResponseType
from src.agents.llm_client import LLMClient
from src.config import yaml_config
from src.utils.logger import logger


class AnswerAgent:
    """답변 생성 에이전트"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_template = yaml_config.get('prompts', {}).get('answer_generation', '')
    
    def _build_context(self, documents: List[Document]) -> str:
        """
        문서들을 컨텍스트로 구성
        
        Args:
            documents: 참조 문서들
            
        Returns:
            포매팅된 컨텍스트 문자열
            
        TODO:
        - [ ] 문서 내용 포매팅
        - [ ] 출처 정보 포함
        - [ ] 토큰 길이 제한 처리
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[문서 {i}]\n{doc.content}\n")
        
        return "\n".join(context_parts)
    
    async def generate(
        self, 
        query: str, 
        documents: List[Document]
    ) -> QueryResponse:
        """
        답변 생성
        
        Args:
            query: 사용자 질문
            documents: 참조 문서
            
        Returns:
            생성된 답변
            
        TODO:
        - [ ] 프롬프트 구성
        - [ ] LLM 호출
        - [ ] 답변 후처리
        - [ ] 신뢰도 계산
        """
        logger.info("답변 생성 시작")
        
        # 컨텍스트 구성
        context = self._build_context(documents)
        
        # 프롬프트 생성
        prompt = self.prompt_template.format(
            query=query,
            context=context
        )
        
        # LLM 호출
        answer = await self.llm_client.generate(prompt)
        
        # TODO: 신뢰도 계산 로직
        confidence = 0.85  # 임시
        
        return QueryResponse(
            answer=answer,
            response_type=ResponseType.ANSWER,
            sources=documents,
            confidence=confidence
        )