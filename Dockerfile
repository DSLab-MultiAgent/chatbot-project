# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 및 uv 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# uv를 PATH에 추가
ENV PATH="/root/.cargo/bin:$PATH"

# Python 의존성 파일 복사 (캐싱 활용)
COPY pyproject.toml uv.lock ./

# uv로 의존성 설치
RUN uv sync --frozen --no-dev

# 애플리케이션 코드 복사
COPY . .

# 비root 사용자 생성 및 전환 (보안)
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# PATH에 uv 가상환경 추가
ENV PATH="/app/.venv/bin:$PATH"

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
