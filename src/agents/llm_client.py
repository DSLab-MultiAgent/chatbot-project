"""
LLM API 클라이언트
OpenAI GPT 모델과 통신
"""
from openai import AsyncOpenAI
from src.config import settings
from src.utils.logger import logger
import httpx



class LLMClient:
    """LLM API 클라이언트 클래스"""
    
    def __init__(self):
        """
        OpenAI 클라이언트 초기화
        
        TODO:
        - [ ] API 키 검증
        - [ ] 클라이언트 설정
        """
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            http_client=httpx.AsyncClient(verify=False)
        )
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        
        logger.info(f"LLM 클라이언트 초기화: {self.model}")
    
    async def generate(
        self, 
        prompt: str, 
        system_message: str = None,
        temperature: float = None
    ) -> str:
        """
        LLM 응답 생성
        
        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지
            temperature: 온도 (None이면 기본값 사용)
            
        Returns:
            생성된 텍스트
            
        TODO:
        - [ ] OpenAI Chat Completion API 호출
        - [ ] 에러 처리
        - [ ] 재시도 로직
        """
        logger.debug(f"LLM 호출: model={self.model}")
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content
            logger.debug(f"LLM 응답 수신: {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            raise