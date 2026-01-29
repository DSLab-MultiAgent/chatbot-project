"""
쿼리 분류 모듈
가비지 쿼리 판별 및 카테고리 분류
"""
from typing import Tuple, Optional
from src.agents.llm_client import LLMClient
from src.utils.logger import logger
from src.config import yaml_config


class QueryClassifier:
    """쿼리 분류 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_template = yaml_config.get('prompts', {}).get('query_classification', '')
        
        # 카테고리 목록 (교학팀 규정 기준)
        self.categories = yaml_config.get("categories", [])
    
    async def classify(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        쿼리 분류 및 가비지 판별
        
        Args:
            query: 사용자 질문
            
        Returns:
            (is_valid, category)
            - is_valid: True면 정상 쿼리, False면 가비지 쿼리
            - category: 분류된 카테고리 (가비지면 None)

        """
        logger.info(f"쿼리 분류 시작: {query}")
        
        # 프롬프트
        prompt = self.prompt_template.format(query=query, category=self.categories)
        
        try:
            # LLM 호출
            response = await self.llm_client.generate(prompt, temperature=0.3)
            response = response.strip()
            
            # 응답 파싱
            if response.startswith("GARBAGE"):
                logger.info("가비지 쿼리로 분류됨")
                return False, None
            
            elif response.startswith("VALID"):
                # 1. "VALID|" 문자열을 모두 제거 (replace)
                # 2. 콤마(,)로 분리하여 리스트화 (split)
                # 3. 각 요소의 앞뒤 공백 제거 (strip)
                categories = [cat.strip() for cat in response.replace("VALID|", "").split(",")]
                
                logger.info(f"정상 쿼리 분류: {categories}")
                return True, categories
            
            else:
                # 예상치 못한 응답
                logger.warning(f"예상치 못한 분류 응답: {response}")
                return True, "기타"
                
        except Exception as e:
            logger.error(f"쿼리 분류 실패: {e}")
            # 에러 시 안전하게 정상 쿼리로 처리
            return True, "기타"
    
    def get_garbage_response(self) -> str:
        """
        가비지 쿼리에 대한 응답 메시지
        
        Returns:
            안내 메시지
        """
        return (
            "죄송합니다. 해당 질문은 교학팀 문의 사항이 아닌 것으로 보입니다.\n\n"
            "교학팀에서는 다음과 같은 문의를 도와드릴 수 있습니다:\n"
            "- 학사운영 관련"
            "- 교육과정 관련"
            "- 졸업시험 및 학위논문 관련"
            "- 장학금 관련"
            "- 증명서 관련"
            "- 입학·모집 안내 관련"
            "- 조교·근로·학사보조 관련"
            "- 시설·IT·생활 안내 관련"
            "- 행사·비교과·교육프로그램 관련"
            "교학팀 관련 질문을 다시 입력해주세요."
        )