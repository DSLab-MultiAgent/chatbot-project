#!/bin/bash
# =============================================================================
# 배포 스크립트
# EC2 서버에서 실행
# =============================================================================

set -e

# 설정
APP_DIR="/home/$USER/chatbot-project"
REPO_URL="https://github.com/DSLab-MultiAgent/chatbot-project.git"
BRANCH="main"  # 배포할 브랜치

echo "=========================================="
echo "배포를 시작합니다..."
echo "=========================================="

cd $APP_DIR

# 1. 최신 코드 가져오기
echo "[1/4] 최신 코드 가져오는 중..."
if [ -d ".git" ]; then
    git fetch origin
    git checkout $BRANCH
    git pull origin $BRANCH
else
    git clone -b $BRANCH $REPO_URL .
fi

# 2. 환경 변수 확인
echo "[2/4] 환경 변수 확인..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env 파일이 없습니다!"
    echo "cp .env.example .env 후 환경변수를 설정하세요."
    exit 1
fi

# 3. Docker 이미지 빌드 및 컨테이너 시작
echo "[3/4] Docker 컨테이너 빌드 및 시작..."
docker-compose down --remove-orphans || true
docker-compose build --no-cache
docker-compose up -d

# 4. 상태 확인
echo "[4/4] 배포 상태 확인..."
sleep 5  # 컨테이너 시작 대기

if docker-compose ps | grep -q "Up"; then
    echo "=========================================="
    echo "✅ 배포가 완료되었습니다!"
    echo "=========================================="
    echo ""
    echo "서비스 상태:"
    docker-compose ps
    echo ""
    echo "API 확인: curl http://localhost:8000/health"
    echo "로그 확인: docker-compose logs -f"
else
    echo "❌ 배포 실패! 로그를 확인하세요:"
    docker-compose logs
    exit 1
fi
