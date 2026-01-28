"""
쿼리 분류 모듈
가비지 쿼리 판별 및 카테고리 분류
"""
from typing import Tuple, Optional
from src.agents.llm_client import LLMClient
from src.utils.logger import logger


class QueryClassifier:
    """쿼리 분류 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        
        # 카테고리 목록 (교학팀 규정 기준)
        self.categories = [
            "수강신청",
            "성적",
            "휴학/복학",
            "장학금",
            "졸업",
            "학적",
            "기타"
        ]
    
    async def classify(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        쿼리 분류 및 가비지 판별
        
        Args:
            query: 사용자 질문
            
        Returns:
            (is_valid, category)
            - is_valid: True면 정상 쿼리, False면 가비지 쿼리
            - category: 분류된 카테고리 (가비지면 None)
            
        TODO:
        - [ ] LLM 프롬프트 최적화
        - [ ] 가비지 쿼리 패턴 정의
        - [ ] 카테고리 분류 정확도 향상
        """
        logger.info(f"쿼리 분류 시작: {query}")
        
        # 프롬프트 구성
        prompt = f"""
다음 질문이 대학교 교학팀 문의 사항인지 판별하고, 맞다면 카테고리를 분류해주세요.

질문: {query}

카테고리 목록:
{', '.join(self.categories)}

다음 형식으로만 답변하세요:
- 가비지 쿼리 (교학팀과 무관)인 경우: "GARBAGE"
- 정상 쿼리인 경우: "VALID|카테고리명"

예시:
- "안녕하세요" → GARBAGE
- "수강신청 기간이 언제인가요?" → VALID|수강신청
- "휴학 신청 방법 알려주세요" → VALID|휴학/복학

답변:
"""
        
        try:
            # LLM 호출
            response = await self.llm_client.generate(prompt, temperature=0.3)
            response = response.strip()
            
            # 응답 파싱
            if response.startswith("GARBAGE"):
                logger.info("가비지 쿼리로 분류됨")
                return False, None
            
            elif response.startswith("VALID"):
                parts = response.split("|")
                if len(parts) == 2:
                    category = parts[1].strip()
                    logger.info(f"정상 쿼리 분류: {category}")
                    return True, category
                else:
                    # 파싱 실패 시 기본값
                    logger.warning("분류 응답 파싱 실패, 기타로 분류")
                    return True, "기타"
            
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
            "- 수강신청 관련\n"
            "- 성적 관련\n"
            "- 휴학/복학 관련\n"
            "- 장학금 관련\n"
            "- 졸업 요건 관련\n"
            "- 학적 관련\n\n"
            "교학팀 관련 질문을 다시 입력해주세요."
        )