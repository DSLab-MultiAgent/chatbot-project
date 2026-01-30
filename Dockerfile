# Python 3.11 슬림 이미지 사용

FROM python:3.11-slim

# 환경 변수 설정

ENV PYTHONDONTWRITEBYTECODE=1 \

    PYTHONUNBUFFERED=1 \

    PIP_NO_CACHE_DIR=1 \

    PIP_DISABLE_PIP_VERSION_CHECK=1

# 작업 디렉토리 설정

WORKDIR /app

# 시스템 의존성 설치

RUN apt-get update && apt-get install -y --no-install-recommends \

    build-essential \

    curl \

    && rm -rf /var/lib/apt/lists/*

# Python 의존성 먼저 복사 (캐싱 활용)

COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사

COPY . .

# 비root 사용자 생성 및 전환 (보안)

RUN useradd --create-home --shell /bin/bash appuser && \

    chown -R appuser:appuser /app

USER appuser

# 포트 노출

EXPOSE 8000

# 헬스체크

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \

CMD curl -f http://localhost:8000/health || exit 1

# 애플리케이션 실행

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
