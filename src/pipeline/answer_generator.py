"""
답변 생성 모듈
검색된 문서를 기반으로 최종 답변 생성
"""
from typing import List
from src.models import Document, QueryResponse, ResponseType
from src.agents.answer_agent import AnswerAgent
from src.agents.conditional_agent import ConditionalAgent
from src.utils.logger import logger


class AnswerGenerator:
    """답변 생성 클래스"""
    
    def __init__(self):
        self.answer_agent = AnswerAgent()
        self.conditional_agent = ConditionalAgent()
        self.helpfulness_threshold = 0.6  # 답변 가능성 임계값
    
    async def check_helpfulness(
        self, 
        query: str, 
        documents: List[Document]
    ) -> bool:
        """
        검증된 문서로 답변 가능한지 판단
        
        Args:
            query: 사용자 질문
            documents: 검증된 문서들
            
        Returns:
            답변 가능하면 True
            
        TODO:
        - [ ] LLM 기반 답변 가능성 판단
        - [ ] 문서 품질 평가
        """
        logger.info("답변 가능성 확인 중...")
        
        if not documents:
            logger.info("문서 없음 → 답변 불가능")
            return False
        
        # TODO: LLM 기반 판단 구현
        # 현재는 문서 검증 점수의 평균으로 판단
        avg_score = sum(
            doc.metadata.get('validation_score', doc.score) 
            for doc in documents
        ) / len(documents)
        
        is_helpful = avg_score >= self.helpfulness_threshold
        
        logger.info(
            f"답변 가능성: {is_helpful} "
            f"(평균 검증 점수: {avg_score:.2f})"
        )
        return is_helpful
    
    async def generate_answer(
        self, 
        query: str, 
        documents: List[Document]
    ) -> QueryResponse:
        """
        최종 답변 생성
        
        Args:
            query: 사용자 질문
            documents: 참조 문서
            
        Returns:
            생성된 답변
        """
        logger.info("일반 답변 생성 중...")
        
        response = await self.answer_agent.generate(
            query=query,
            documents=documents
        )
        
        logger.info("답변 생성 완료")
        return response
    
    async def generate_conditional_response(
        self, 
        query: str
    ) -> QueryResponse:
        """
        조건부 응답 생성 (답변 불가능 시)
        
        Args:
            query: 사용자 질문
            
        Returns:
            조건부 응답
        """
        logger.info("조건부 응답 생성 중...")
        
        response = await self.conditional_agent.generate(query=query)
        
        logger.info("조건부 응답 생성 완료")
        return response