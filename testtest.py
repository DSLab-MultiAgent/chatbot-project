import asyncio
from openai import AsyncOpenAI
import httpx

# API 키 보안에 주의하세요!
client = AsyncOpenAI(
    api_key="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",
    http_client=httpx.AsyncClient(verify=False)) 
model = "gpt-3.5-turbo"
messages = [{"role": "user", "content": "hi"}]

async def main():
    try:
        # AsyncOpenAI는 await를 사용해야 합니다.
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        result = response.choices[0].message.content
        print(f'LLM 응답: {result}')

    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {e}")
        # 루프가 없으므로 break는 삭제합니다.

# 비동기 함수 실행
if __name__ == "__main__":
    asyncio.run(main())