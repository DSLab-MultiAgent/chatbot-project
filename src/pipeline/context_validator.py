"""
컨텍스트 검증 모듈
쿼리와 문서들로 답변 가능한지 검증
"""
from typing import List, Tuple
from src.models import Document
from src.agents.llm_client import LLMClient
from src.utils.logger import logger


class ContextValidator:
    """컨텍스트 검증 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    async def validate_context(
        self, 
        query: str, 
        documents: List[Document]
    ) -> Tuple[bool, str]:
        """
        컨텍스트 검증: 문서들로 쿼리에 답변 가능한지 판단
        
        Args:
            query: 사용자 질문
            documents: 관련 문서들
            
        Returns:
            (is_valid, reason)
            - is_valid: True면 답변 가능, False면 불가능
            - reason: 판단 이유 (디버깅용)
            
        TODO:
        - [ ] LLM 프롬프트 최적화
        - [ ] 검증 기준 명확화
        """
        logger.info(f"컨텍스트 검증 시작: {len(documents)}개 문서")
        
        if not documents:
            logger.warning("문서 없음 → 검증 실패")
            return False, "관련 문서 없음"
        
        # 문서 내용 구성
        context = "\n\n".join([
            f"[문서 {i+1}]\n{doc.content[:300]}..."
            for i, doc in enumerate(documents)
        ])
        
        # 프롬프트 구성
        prompt = f"""
다음 문서들로 사용자 질문에 답변할 수 있는지 판단해주세요.

사용자 질문: {query}

관련 문서:
{context}

판단 기준:
- YES: 문서에 질문에 대한 명확한 정보가 있어 답변 가능
- NO: 문서가 질문과 관련은 있지만 정보가 부족하거나 불명확함

다음 형식으로만 답변하세요:
"YES" 또는 "NO|이유"

예시:
- "YES"
- "NO|문서에 구체적인 기간 정보가 없음"

답변:
"""
        
        try:
            # LLM 호출
            response = await self.llm_client.generate(prompt, temperature=0.2)
            response = response.strip()
            
            logger.debug(f"컨텍스트 검증 응답: {response}")
            
            # 응답 파싱
            if response.startswith("YES"):
                logger.info("컨텍스트 검증 성공 → 답변 가능")
                return True, "검증 통과"
            
            elif response.startswith("NO"):
                parts = response.split("|", 1)
                reason = parts[1] if len(parts) > 1 else "정보 부족"
                logger.info(f"컨텍스트 검증 실패 → {reason}")
                return False, reason
            
            else:
                # 예상치 못한 응답
                logger.warning(f"예상치 못한 검증 응답: {response}")
                return False, "검증 응답 파싱 실패"
                
        except Exception as e:
            logger.error(f"컨텍스트 검증 중 에러: {e}")
            # 에러 시 False 반환
            return False, f"검증 에러: {str(e)}"