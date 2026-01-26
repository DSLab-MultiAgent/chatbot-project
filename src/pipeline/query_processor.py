"""
쿼리 정제 모듈
사용자 질문을 검색에 적합한 형태로 변환
"""
from src.agents.llm_client import LLMClient
from src.config import yaml_config
from src.utils.logger import logger


class QueryProcessor:
    """쿼리 정제 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_template = yaml_config.get('prompts', {}).get('query_refine', '')
    
    async def refine_query(self, query: str) -> str:
        """
        사용자 쿼리를 검색용 키워드로 정제
        
        Args:
            query: 원본 사용자 질문
            
        Returns:
            정제된 검색 쿼리
            
        TODO:
        - [ ] LLM을 사용한 쿼리 정제 구현
        - [ ] 한글 형태소 분석 추가 (선택사항)
        - [ ] 불용어 제거
        - [ ] 동의어 확장
        """
        logger.info(f"쿼리 정제 시작: {query}")
        
        # TODO: 실제 LLM 호출 구현
        # 현재는 임시로 원본 쿼리 반환
        refined = query.strip()
        
        logger.info(f"정제된 쿼리: {refined}")
        return refined
    
    def extract_keywords(self, query: str) -> list:
        """
        쿼리에서 키워드 추출
        
        TODO:
        - [ ] 명사 추출
        - [ ] 중요 키워드 선별
        """
        # 임시 구현
        return query.split()