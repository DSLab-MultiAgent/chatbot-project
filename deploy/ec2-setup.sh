#!/bin/bash
# =============================================================================
# EC2 서버 초기 설정 스크립트
# Ubuntu 22.04 LTS 기준
# =============================================================================

set -e  # 에러 발생 시 스크립트 중단

echo "=========================================="
echo "EC2 서버 초기 설정을 시작합니다..."
echo "=========================================="

# 1. 시스템 업데이트
echo "[1/6] 시스템 패키지 업데이트..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. 필수 패키지 설치
echo "[2/6] 필수 패키지 설치..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    nginx \
    certbot \
    python3-certbot-nginx

# 3. Docker 설치
echo "[3/6] Docker 설치..."
if ! command -v docker &> /dev/null; then
    # Docker GPG 키 추가
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Docker 저장소 추가
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Docker 설치
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

    # 현재 사용자를 docker 그룹에 추가
    sudo usermod -aG docker $USER

    echo "Docker 설치 완료!"
else
    echo "Docker가 이미 설치되어 있습니다."
fi

# 4. Docker Compose 설치 (standalone)
echo "[4/6] Docker Compose 확인..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose 설치 완료!"
else
    echo "Docker Compose가 이미 설치되어 있습니다."
fi

# 5. 애플리케이션 디렉토리 생성
echo "[5/6] 애플리케이션 디렉토리 설정..."
APP_DIR="/home/$USER/chatbot-project"
mkdir -p $APP_DIR
mkdir -p $APP_DIR/data
mkdir -p $APP_DIR/logs

# 6. 방화벽 설정 (UFW)
echo "[6/6] 방화벽 설정..."
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp  # API 포트 (개발용, 프로덕션에서는 제거 권장)
sudo ufw --force enable

echo "=========================================="
echo "초기 설정이 완료되었습니다!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "1. 로그아웃 후 다시 로그인하세요 (docker 그룹 적용)"
echo "2. git clone으로 프로젝트를 가져오세요"
echo "3. .env 파일을 생성하고 환경변수를 설정하세요"
echo "4. docker-compose up -d 로 서비스를 시작하세요"
echo ""
