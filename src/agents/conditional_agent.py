"""
조건부 응답 에이전트
답변 불가능 시 교학팀 문의 안내
"""
from src.models import QueryResponse, ResponseType
from src.agents.llm_client import LLMClient
from src.config import yaml_config
from src.utils.logger import logger


class ConditionalAgent:
    """조건부 응답 에이전트"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_template = yaml_config.get('prompts', {}).get('conditional_response', '')
    
    async def generate(self, query: str) -> QueryResponse:
        """
        조건부 응답 생성
        
        Args:
            query: 사용자 질문
            
        Returns:
            조건부 응답
            
        TODO:
        - [ ] 교학팀 연락처 정보 포함
        - [ ] 친절한 안내 메시지
        - [ ] 유사 질문 추천 (선택사항)
        """
        logger.info("조건부 응답 생성")
        
        # 프롬프트 생성
        prompt = self.prompt_template.format(query=query)
        
        # LLM 호출
        answer = await self.llm_client.generate(prompt)
        
        # 기본 메시지 추가
        final_answer = f"{answer}\n\n교학팀 문의: 02-1234-5678 또는 academic@university.ac.kr"
        
        return QueryResponse(
            answer=final_answer,
            response_type=ResponseType.CONDITIONAL,
            sources=[],
            confidence=0.0
        )