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
    
    async def check_helpfulness(self, query: str, documents: List[Document]) -> bool:
        """
        검색 결과가 도움이 되는지 판단
        
        Args:
            query: 사용자 질문
            documents: 검색된 문서들
            
        Returns:
            도움이 되면 True, 아니면 False
            
        TODO:
        - [ ] LLM을 사용한 유용성 판단
        - [ ] 유사도 임계값 기반 판단
        - [ ] 답변 가능성 점수 계산
        """
        logger.info("답변 가능성 확인 중...")
        
        if not documents:
            return False
        
        # TODO: 실제 판단 로직 구현
        # 임시로 평균 점수로 판단
        avg_score = sum(doc.score for doc in documents) / len(documents)
        is_helpful = avg_score > 0.7
        
        logger.info(f"답변 가능성: {is_helpful} (평균 점수: {avg_score:.2f})")
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
            
        TODO:
        - [ ] 문서 내용을 컨텍스트로 구성
        - [ ] LLM 프롬프트 생성
        - [ ] 답변 생성 및 후처리
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
            
        TODO:
        - [ ] 교학팀 문의 안내 메시지 생성
        - [ ] 유사 질문 추천 (선택사항)
        """
        logger.info("조건부 응답 생성 중...")
        
        response = await self.conditional_agent.generate(query=query)
        
        logger.info("조건부 응답 생성 완료")
        return response