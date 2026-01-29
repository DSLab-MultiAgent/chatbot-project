import requests

# API 엔드포인트
url = "http://localhost:8000/query"

# 질문
question = "수강신청 기간이 언제인가요?"

# 요청
response = requests.post(
    url,
    json={"question": question}
)

# 응답 출력
result = response.json()
print(f"질문: {question}")
print(f"\n답변: {result['answer']}")
print(f"\n응답 타입: {result['response_type']}")
print(f"신뢰도: {result['confidence']}")