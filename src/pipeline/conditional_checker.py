"""
조건부 응답 여부 체크 모듈
사용자의 개별 조건에 따라 답변이 달라지는지 판단
"""
from typing import List
from src.models import Document
from src.agents.llm_client import LLMClient
from src.utils.logger import logger
from src.config import yaml_config


class ConditionalChecker:
    """조건부 응답 여부 체크 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_template = yaml_config.get('prompts', {}).get('conditional_checker', '')
    
    async def check_conditional(
        self, 
        query: str, 
        documents: List[Document]
    ) -> bool:
        """
        조건부 응답 필요 여부 판단
        
        Args:
            query: 사용자 질문
            documents: 관련 문서들
            
        Returns:
            True: 조건부 응답 필요 (사용자 조건에 따라 답변 달라짐)
            False: 완전 응답 가능 (일반적인 답변)
            
        예시:
        - "수강신청 기간은?" → False (모두에게 동일)
        - "제 경우 휴학 가능한가요?" → True (개인 상황에 따라 다름)
        - "장학금 받을 수 있나요?" → True (성적, 소득 등에 따라 다름)
        
        TODO:
        - [ ] 조건부 판단 기준 명확화
        - [ ] 프롬프트 최적화
        """
        logger.info("조건부 응답 여부 체크 시작")
        
        # 문서 내용 구성
        context = "\n\n".join([
            f"[문서 {i+1}]\n{doc.content[:300]}..."
            for i, doc in enumerate(documents)
        ])
        
        # 프롬프트 구성
        prompt = self.prompt_template.format(query=query, context=context)
        
        try:
            # LLM 호출
            response = await self.llm_client.generate(prompt, temperature=0.2)
            response = response.strip().upper()
            
            logger.debug(f"조건부 체크 응답: {response}")
            
            is_conditional = "CONDITIONAL" in response
            
            if is_conditional:
                logger.info("조건부 응답 필요")
            else:
                logger.info("완전 응답 가능")
            
            return is_conditional
            
        except Exception as e:
            logger.error(f"조건부 체크 중 에러: {e}")
            # 에러 시 안전하게 조건부로 처리
            return True