# 교학팀 문의 챗봇 🤖

멀티에이전트 RAG 기반 교학팀 문의 자동 응답 시스템

## 📋 프로젝트 개요

교학팀 규정 문서를 기반으로 학생들의 질문에 자동으로 답변하는 챗봇 시스템입니다.

### 주요 기능
- 🔍 Hybrid Retriever (Vector + Keyword 검색)
- 🧠 Late-interaction vector 검색
- 🔄 다단계 검색 루프 (최대 2회)
- ✅ 답변 가능성 자동 판단
- 📝 조건부 응답 생성 (답변 불가 시)

## 📋 파이프라인 플로우

### 전체 프로세스

1. **쿼리 분류** (`query_classifier.py`)
   - 가비지 쿼리 판별
   - 카테고리 분류 (수강신청, 성적, 휴학/복학 등)

2. **하이브리드 검색** (`retriever.py`)
   - Late Interaction 방식
   - 카테고리 필터링
   - Top 10개 문서 검색

3. **문서 검증** (`document_validator.py`)
   - LLM 기반 개별 문서 관련성 평가
   - 1차: Top 1~5 검증
   - 2차: Top 6~10 검증 (재시도 시)

4. **답변 가능성 확인** (`answer_generator.py`)
   - 검증된 문서로 답변 가능 여부 판단

5. **답변 생성**
   - 가능: 일반 답변
   - 불가능 (1차): 2차 문서로 재시도
   - 불가능 (2차): 조건부

### 기술 스택
- **Language**: Python 3.10+
- **Framework**: FastAPI, LangChain
- **Vector DB**: ChromaDB
- **LLM**: GPT-4 (OpenAI)
- **Embedding**: Sentence Transformers

## 🚀 시작하기

### 1. Repository Clone
```bash
git clone https://github.com/[organization-name]/chatbot-project.git
cd chatbot-project
```

### 2. uv 설치

#### Windows (PowerShell)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Mac/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. 가상환경 생성 및 의존성 설치
```bash
# 가상환경 생성
uv venv

# 가상환경 활성화
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 의존성 설치 (매우 빠름!)
uv pip install -r requirements.txt
```

### 4. 환경변수 설정
```bash
cp .env.example .env
# .env 파일을 열어서 API 키 입력
# OPENAI_API_KEY=실제_API_키_입력
```

### 5. 실행
```bash
# 개발 서버 실행
python run.py

# 또는 직접 실행
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. API 테스트
브라우저에서 http://localhost:8000/docs 접속하여 Swagger UI로 테스트

## 📁 프로젝트 구조
```
src/
├── pipeline/          # RAG 파이프라인 핵심 로직
│   ├── query_classifier.py      # 쿼리 분류(카테고리 및 가비지)
│   ├── query_processor.py       # 쿼리 정제
│   ├── retriever.py             # 하이브리드 검색
│   ├── document_validator.py     # 관련성 체크 (YES/NO)
│   ├── context_validator.py      # 컨텍스트 검증 (답변 가능 여부)
│   ├── conditional_checker.py    # 조건부 응답 필요 여부
│   └── pipeline.py               # 전체 통합
│
├── retrievers/        # 검색 엔진
│   ├── vector_retriever.py   # 벡터 검색
│   ├── keyword_retriever.py  # 키워드 검색
│   └── hybrid_retriever.py   # Hybrid 통합
│
├── agents/            # LLM 에이전트
│   ├── llm_client.py         # LLM API
│   ├── answer_agent.py       # 완전 응답 생성
│   └── conditional_agent.py  # 조건부 응답 생성
│
└── utils/             # 유틸리티
    ├── logger.py
    └── helpers.py
```

## 🔧 개발 가이드

### Branch 전략
- `main`: 배포용 (안정 버전)
- `develop`: 개발 통합 브랜치
- `feature/모듈명`: 기능 개발 브랜치

### 작업 흐름
```bash
# 1. develop 브랜치에서 시작
git checkout develop
git pull origin develop

# 2. 기능 브랜치 생성
git checkout -b feature/vector-retriever

# 3. 개발 작업...

# 4. Commit & Push
git add .
git commit -m "feat: Vector Retriever 구현"
git push origin feature/vector-retriever

# 5. GitHub에서 Pull Request 생성
```

### 모듈별 담당

| 모듈 | 담당자 | 상태 |
|------|--------|------|
| Query Processor | [이름] | 🔄 진행중 |
| Vector Retriever | [이름] | 📝 예정 |
| Keyword Retriever | [이름] | 📝 예정 |
| Answer Generator | [이름] | 📝 예정 |
| Pipeline Integration | [이름] | 📝 예정 |

## 📝 TODO

- [ ] 벡터 DB 초기 데이터 로딩
- [ ] Vector Retriever 구현
- [ ] Keyword Retriever 구현
- [ ] Hybrid Retriever 통합
- [ ] LLM 답변 생성 로직
- [ ] 조건부 응답 생성
- [ ] 프론트엔드 연동
- [ ] 테스트 코드 작성

## 🤝 팀원

- [이름1] - 팀장, Pipeline 통합
- [이름2] - Retriever 개발
- [이름3] - LLM Agent 개발

## 📄 라이선스

MIT License