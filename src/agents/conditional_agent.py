"""
조건부 응답 에이전트
사용자의 추가 정보가 필요한 경우의 답변 생성
"""
from typing import List
from src.models import Document, QueryResponse, ResponseType
from src.agents.llm_client import LLMClient
from src.config import yaml_config
from src.utils.logger import logger
import datetime


class ConditionalAgent:
    """조건부 응답 에이전트"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_template = yaml_config.get('prompts', {}).get('conditional_response', '')
    
    def _build_context(self, documents: List[Document]) -> str:
        """문서들을 컨텍스트로 구성"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[문서 {i}]\n{doc.content}\n")
        return "\n".join(context_parts)
    
    async def generate_with_documents(
        self, 
        query: str,
        documents: List[Document],
    ) -> QueryResponse:
        """
        조건부 응답 생성 (문서 기반)
        
        Args:
            query: 사용자 질문
            documents: 참조 문서
            
        Returns:
            조건부 응답
        """
        logger.info("조건부 응답 생성 시작 (문서 기반)")
        
        # 컨텍스트 구성
        context = self._build_context(documents)
        
        # 프롬프트 생성
        prompt = self.prompt_template.format(
            query=query,
            context=context,
            date=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        
        # LLM 호출
        answer = await self.llm_client.generate(prompt)
        
        # 교학팀 연락처 추가
        final_answer = (
            f"{answer}\n\n"
            f"📞 정확한 답변을 위해 교학팀으로 문의 부탁드립니다.\n"
            f"교학팀 문의: 02-910-4018 또는 business-it@kookmin.ac.kr"
        )
        
        # # 신뢰도 계산
        # avg_score = sum(doc.score for doc in documents) / len(documents) if documents else 0.5
        # confidence = avg_score * 0.7  # 조건부이므로 낮은 신뢰도
        
        # logger.info(f"조건부 응답 생성 완료 (신뢰도: {confidence:.2f})")
        logger.info(f"조건부 응답 생성 완료")
        
        return QueryResponse(
            answer=final_answer,
            response_type=ResponseType.CONDITIONAL,
            sources=documents,
            confidence=0
        )
    
    async def generate_no_documents(self, query: str) -> QueryResponse:
        """
        조건부 응답 생성 (문서 없음 - 교학팀 문의 안내)
        
        Args:
            query: 사용자 질문
            
        Returns:
            교학팀 문의 안내
        """
        logger.info("교학팀 문의 안내 생성")
        
        answer = (
            f"죄송합니다. '{query}'에 대한 정확한 답변을 위해서는 "
            f"귀하의 구체적인 상황을 확인해야 합니다.\n\n"
            f"다음 정보와 함께 교학팀으로 직접 문의해주시기 바랍니다:\n"
            f"• 학번 및 학과\n"
            f"• 현재 학년 및 이수 학점\n"
            f"• 구체적인 상황 설명\n\n"
            f"📞 교학팀 문의\n"
            f"전화: 02-910-4018\n"
            f"이메일: business-it@kookmin.ac.kr\n"
            f"방문: 국제관 2층 교학팀 (평일 09:00-17:00)"
        )
        
        return QueryResponse(
            answer=answer,
            response_type=ResponseType.CONDITIONAL,
            sources=[],
            confidence=0.0
        )