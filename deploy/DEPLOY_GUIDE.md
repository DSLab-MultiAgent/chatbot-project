# AWS EC2 배포 가이드

이 가이드는 교학팀 챗봇을 AWS EC2에 배포하는 단계별 가이드입니다.

## 목차
1. [사전 준비](#1-사전-준비)
2. [EC2 인스턴스 생성](#2-ec2-인스턴스-생성)
3. [서버 접속 및 초기 설정](#3-서버-접속-및-초기-설정)
4. [애플리케이션 배포](#4-애플리케이션-배포)
5. [도메인 및 HTTPS 설정 (선택)](#5-도메인-및-https-설정-선택)
6. [유지보수](#6-유지보수)

---

## 1. 사전 준비

### 필요한 것
- AWS 계정 (https://aws.amazon.com)
- OpenAI API Key
- SSH 클라이언트 (Windows: PuTTY 또는 WSL, Mac/Linux: 터미널)

### AWS 프리 티어 확인
- EC2 t2.micro: **750시간/월 무료** (1년간)
- 주의: 인스턴스를 계속 켜두면 월 ~720시간 사용

---

## 2. EC2 인스턴스 생성

### Step 1: AWS Console 접속
1. https://console.aws.amazon.com 로그인
2. 상단 검색창에 "EC2" 입력 → EC2 대시보드 이동

### Step 2: 인스턴스 시작
1. **"인스턴스 시작"** 버튼 클릭

2. **이름**: `chatbot-server`

3. **AMI 선택**:
   - `Ubuntu Server 22.04 LTS (HVM), SSD Volume Type`
   - 아키텍처: 64비트 (x86)

4. **인스턴스 유형**:
   - `t2.micro` (프리 티어 사용 가능)
   - 참고: 메모리 1GB - 소규모 테스트에 적합

5. **키 페어 생성**:
   - "새 키 페어 생성" 클릭
   - 이름: `chatbot-key`
   - 유형: RSA
   - 형식: `.pem` (Mac/Linux) 또는 `.ppk` (Windows PuTTY)
   - **⚠️ 다운로드된 키 파일을 안전한 곳에 보관!**

6. **네트워크 설정**:
   - "편집" 클릭
   - 보안 그룹: "보안 그룹 생성"
   - 보안 그룹 이름: `chatbot-sg`
   - 인바운드 규칙 추가:
     | 유형 | 포트 | 소스 | 설명 |
     |------|------|------|------|
     | SSH | 22 | 내 IP | SSH 접속 |
     | HTTP | 80 | 0.0.0.0/0 | 웹 서버 |
     | HTTPS | 443 | 0.0.0.0/0 | HTTPS |
     | 사용자 지정 TCP | 8000 | 0.0.0.0/0 | API (개발용) |

7. **스토리지**:
   - 8GB gp3 (기본값, 프리 티어 30GB까지 무료)

8. **"인스턴스 시작"** 클릭!

### Step 3: 탄력적 IP 할당 (선택, 권장)
고정 IP가 필요하면:
1. EC2 > 탄력적 IP > "탄력적 IP 주소 할당"
2. 할당된 IP 선택 > "작업" > "탄력적 IP 주소 연결"
3. 인스턴스 선택 후 연결

---

## 3. 서버 접속 및 초기 설정

### Step 1: SSH 접속

```bash
# 키 파일 권한 설정 (Mac/Linux)
chmod 400 ~/Downloads/chatbot-key.pem

# SSH 접속
ssh -i ~/Downloads/chatbot-key.pem ubuntu@[EC2-퍼블릭-IP]
```

**Windows (PuTTY) 사용자:**
1. PuTTYgen으로 .pem → .ppk 변환
2. PuTTY > Connection > SSH > Auth > Private key 설정
3. Host: `ubuntu@[EC2-퍼블릭-IP]`

### Step 2: 서버 초기 설정

```bash
# 프로젝트 클론
git clone https://github.com/DSLab-MultiAgent/chatbot-project.git
cd chatbot-project

# 초기 설정 스크립트 실행
chmod +x deploy/ec2-setup.sh
./deploy/ec2-setup.sh

# 로그아웃 후 다시 접속 (docker 그룹 적용)
exit
ssh -i ~/Downloads/chatbot-key.pem ubuntu@[EC2-퍼블릭-IP]
```

---

## 4. 애플리케이션 배포

### Step 1: 환경 변수 설정

```bash
cd ~/chatbot-project

# .env 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env
```

**.env 파일 필수 설정:**
```env
# 필수
OPENAI_API_KEY=sk-your-actual-api-key-here

# 선택 (기본값 사용 가능)
LLM_MODEL=gpt-4
DEBUG=False
LOG_LEVEL=INFO
```

저장: `Ctrl+X` → `Y` → `Enter`

### Step 2: Docker로 배포

```bash
# 방법 1: 배포 스크립트 사용
chmod +x deploy/deploy.sh
./deploy/deploy.sh

# 방법 2: 직접 실행
docker-compose up -d --build
```

### Step 3: 배포 확인

```bash
# 컨테이너 상태 확인
docker-compose ps

# 헬스체크
curl http://localhost:8000/health

# 로그 확인
docker-compose logs -f
```

### Step 4: API 테스트

```bash
# 로컬에서 테스트
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "학사 규정이 뭔가요?"}'

# 외부에서 테스트 (브라우저)
http://[EC2-퍼블릭-IP]:8000/docs
```

---

## 5. 도메인 및 HTTPS 설정 (선택)

### 도메인 연결
1. 도메인 구매 (가비아, Route53 등)
2. DNS A 레코드에 EC2 퍼블릭 IP 설정

### Let's Encrypt SSL 인증서 (무료)

```bash
# Nginx 설정 복사
sudo cp deploy/nginx.conf /etc/nginx/nginx.conf

# 도메인 설정 (nginx.conf에서 server_name 수정)
sudo nano /etc/nginx/nginx.conf
# server_name your-domain.com; 으로 변경

# Nginx 시작
sudo systemctl start nginx
sudo systemctl enable nginx

# SSL 인증서 발급
sudo certbot --nginx -d your-domain.com

# 자동 갱신 확인
sudo certbot renew --dry-run
```

---

## 6. 유지보수

### 자주 사용하는 명령어

```bash
# 서비스 상태 확인
docker-compose ps

# 로그 보기
docker-compose logs -f chatbot

# 서비스 재시작
docker-compose restart

# 서비스 중지
docker-compose down

# 업데이트 배포
git pull origin main
docker-compose up -d --build
```

### 모니터링

```bash
# 리소스 사용량 확인
docker stats

# 디스크 사용량
df -h

# 메모리 사용량
free -h
```

### 백업

```bash
# 데이터 백업 (벡터 DB 등)
tar -czvf backup-$(date +%Y%m%d).tar.gz data/

# S3로 백업 (AWS CLI 설치 필요)
aws s3 cp backup-*.tar.gz s3://your-bucket/backups/
```

---

## 트러블슈팅

### 문제: Docker 명령어 권한 오류
```bash
# docker 그룹에 사용자 추가 후 재접속
sudo usermod -aG docker $USER
exit
# 다시 SSH 접속
```

### 문제: 메모리 부족 (t2.micro)
```bash
# Swap 메모리 추가 (2GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 문제: 포트 접속 불가
1. EC2 보안 그룹 인바운드 규칙 확인
2. UFW 방화벽 상태 확인: `sudo ufw status`
3. 컨테이너 실행 상태 확인: `docker-compose ps`

### 문제: API 응답 느림
- t2.micro 성능 한계 - 더 큰 인스턴스로 업그레이드 고려
- OpenAI API 응답 시간 확인

---

## 비용 관리

### 프리 티어 사용량 모니터링
- AWS Console > Billing > 프리 티어 사용량 확인

### 비용 절감 팁
1. 사용하지 않을 때 인스턴스 중지 (Stop)
2. 탄력적 IP는 사용 중인 인스턴스에만 연결 (미사용 시 요금 부과)
3. CloudWatch 알람 설정으로 비용 알림

### 예상 비용 (프리 티어 이후)
- t2.micro: ~$8-10/월
- 탄력적 IP (미사용): ~$3.6/월
- 데이터 전송: 100GB까지 무료

---

## 다음 단계

배포 완료 후 고려할 사항:
- [ ] 모니터링 설정 (CloudWatch)
- [ ] 자동 배포 파이프라인 (GitHub Actions)
- [ ] 로드밸런서 추가 (트래픽 증가 시)
- [ ] 데이터베이스 분리 (RDS)

질문이나 문제가 있으면 이슈를 등록해주세요!
