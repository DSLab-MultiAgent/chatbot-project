# 대학교 교학팀 문의 챗봇 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

멀티에이전트 RAG 기반 교학팀 문의 자동 응답 시스템

## 🌐 서비스 데모
본 프로젝트에서 제안한 시스템은 배포가 완료되었으며,  
아래 링크를 통해 실제 동작을 확인할 수 있습니다.  

👉 **[대학교 교학팀 문의 챗봇](http://bitchatbot.duckdns.org/)**

※ 본 서비스는 데모 목적의 배포 환경으로,  
서버 상태에 따라 일시적으로 접근이 제한될 수 있습니다.

## 📋 프로젝트 개요

교학팀 규정 문서를 기반으로 학생들의 질문에 자동으로 답변하는 챗봇 시스템입니다.

### 주요 기능
- 🔍 Late Interaction 기반 벡터 검색 (ColBERT)
- 🏷️ 쿼리 분류 및 카테고리 기반 필터링
- 📄 LLM 기반 문서 관련성 검증
- 🔄 다단계 검색 루프 (1차/2차 문서 검증)
- ✅ 답변 가능성 자동 판단
- 📝 조건부/완전 응답 생성
- 🌐 웹 UI 채팅 인터페이스
- ☁️ AWS EC2 배포 지원

## 📋 파이프라인 플로우

### 전체 프로세스

1. **쿼리 분류** (`query_classifier.py`)
   - 가비지 쿼리 판별
   - 카테고리 분류 (학사운영, 교육과정, 장학금 등 10개 카테고리)

2. **하이브리드 검색** (`retriever.py`)
   - ColBERT Late Interaction 방식
   - 카테고리 기반 Pre-filtering
   - PLAID 인덱스 기반 Top-K 문서 검색

3. **문서 검증** (`document_validator.py`)
   - LLM 기반 개별 문서 관련성 평가
   - 1차: Top 1~10 검증
   - 2차: Top 11~20 검증 (재시도 시)

4. **답변 가능성 확인** (`context_validator.py`)
   - 검증된 문서로 답변 가능 여부 판단

5. **조건부 체크** (`conditional_checker.py`)
   - 사용자 상황에 따라 답변이 달라지는지 판단

6. **답변 생성**
   - 완전 응답: 일반 답변 생성 (`answer_agent.py`)
   - 조건부 응답: 상황별 안내 (`conditional_agent.py`)
   - Human Handoff: 교학팀 문의 안내

## 📋 파이프라인 플로우

### 전체 프로세스

1. **쿼리 분류** (`query_classifier.py`)
   - 가비지 쿼리 판별
   - 카테고리 분류 (수강신청, 성적, 휴학/복학 등)

2. **하이브리드 검색** (`retriever.py`)
   - Late Interaction 방식
   - 카테고리 필터링
   - Top 20개 문서 검색

3. **문서 검증** (`document_validator.py`)
   - LLM 기반 개별 문서 관련성 평가
   - 1차: Top 1~10 검증
   - 2차: Top 11~20 검증 (재시도 시)

4. **답변 가능성 확인** (`answer_generator.py`)
   - 검증된 문서로 답변 가능 여부 판단

5. **답변 생성**
   - 가능: 일반 답변
   - 불가능 (1차): 2차 문서로 재시도
   - 불가능 (2차): 조건부

### 기술 스택
- **Language**: Python 3.10.19
- **Framework**: FastAPI, Uvicorn
- **LLM**: OpenAI GPT-3.5-turbo
- **Vector Search**: ColBERT-Matryoshka (dragonkue/colbert-ko-0.1b)
- **Index**: PyLate PLAID
- **Frontend**: HTML/CSS/JavaScript (Single Page)
- **Deploy**: Docker, Nginx, AWS EC2

## 🚀 시작하기

### 1. Repository Clone
```bash
git clone https://github.com/DSLab-MultiAgent/chatbot-project.git
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
# uv 환경 초기화
uv init

# 가상환경 생성
uv venv

# 가상환경 활성화
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 의존성 설치
uv add -r requirements.txt
```

### 4. 환경변수 설정
```bash
cp .env.example .env
# .env 파일을 열어서 API 키 입력
# OPENAI_API_KEY=실제_API_키_입력
# VECTOR_DB_FILE_ID=실제_File_ID_입력
```

### 5. 벡터 DB 다운로드
```bash
python scripts/download_vector.py
```

### 6. 실행
```bash
# 개발 서버 실행
python run.py
```

### 7. 접속
- **웹 UI**: http://localhost:8000


## ☁️ EC2 배포

AWS EC2에 배포하려면 `deploy/DEPLOY_GUIDE.md`를 참고하세요.

```bash
# 배포 스크립트 실행
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

## 📁 프로젝트 구조
```
├── index.html                 # 웹 UI (채팅 인터페이스)
├── config/
│   └── config.yaml            # 카테고리 및 프롬프트 설정
├── deploy/                    # EC2 배포 관련
│   ├── deploy.sh
│   ├── ec2-setup.sh
│   ├── nginx.conf
│   └── chatbot.service
├── scripts/
│   └── download_vector.py     # 벡터 DB 다운로드
├── src/
│   ├── main.py                # FastAPI 앱 엔트리포인트
│   ├── models.py              # Pydantic 모델 정의
│   ├── config.py              # 설정 로드
│   │
│   ├── pipeline/              # RAG 파이프라인
│   │   ├── pipeline.py              # 전체 파이프라인 통합
│   │   ├── query_classifier.py      # 쿼리 분류 (카테고리/가비지)
│   │   ├── retriever.py             # 하이브리드 검색 호출
│   │   ├── document_validator.py    # 문서 관련성 검증
│   │   ├── context_validator.py     # 컨텍스트 검증 (답변 가능 여부)
│   │   └── conditional_checker.py   # 조건부 응답 필요 여부
│   │
│   ├── retrievers/            # 검색 엔진
│   │   ├── vector_retriever.py      # ColBERT 벡터 검색
│   │   ├── hybrid_retriever.py      # 하이브리드 검색 통합
│   │   └── models/                  # ColBERT 모델 파일
│   │
│   ├── agents/                # LLM 에이전트
│   │   ├── llm_client.py            # OpenAI API 클라이언트
│   │   ├── answer_agent.py          # 완전 응답 생성
│   │   └── conditional_agent.py     # 조건부 응답 생성
│   │
│   └── utils/                 # 유틸리티
│       ├── logger.py
│       └── helpers.py
│
├── data/
│   └── vector_db/             # PLAID 벡터 인덱스
│
└── tests/
    └── test_pipeline.py
```

## 🔧 개발 가이드

### Branch 전략
- `main`: 배포용 (안정 버전)
- `develop`: 개발 통합 브랜치
- `feature/모듈명`: 기능 개발 브랜치


## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
