import gdown
import os
from pathlib import Path
import zipfile
from dotenv import load_dotenv
import ssl
import urllib3

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
print(f"🔍 .env 파일 경로: {env_path}")
print(f"🔍 .env 파일 존재: {env_path.exists()}")

# .env 로드
print("\n🔄 .env 로드 중...")
load_dotenv(dotenv_path=env_path, override=True)

# 환경변수 확인
file_id = os.getenv("VECTOR_DB_FILE_ID")
print(f"🔍 VECTOR_DB_FILE_ID: {file_id}")

if not file_id:
    raise ValueError("VECTOR_DB_FILE_ID가 .env에 설정되지 않았습니다.")

# 경로 설정
data_dir = Path("data")
vector_db_dir = data_dir / "vector_db"
zip_path = data_dir / "vector_db.zip"
data_dir.mkdir(exist_ok=True)

# URL 생성
url = f"https://drive.google.com/uc?id={file_id}"
print(f"\n📥 최종 다운로드 URL: {url}")

# 다운로드 (SSL 검증 비활성화)
print("\n다운로드 시작...")
print("⚠️ SSL 검증을 비활성화합니다 (네트워크 환경 문제 우회)")

try:
    # verify=False로 SSL 검증 비활성화
    gdown.download(
        url, 
        str(zip_path), 
        quiet=False, 
        fuzzy=True,
        verify=False  # ← SSL 검증 비활성화
    )
    print("✅ 다운로드 완료!")
except Exception as e:
    print(f"❌ 다운로드 실패: {e}")
    raise

# 파일 크기 확인
file_size = zip_path.stat().st_size
print(f"\n📦 다운로드된 파일 크기: {file_size / 1024:.2f} KB ({file_size / 1024 / 1024:.2f} MB)")

if file_size < 1024 * 1024:  # 1MB 미만
    print("⚠️ 파일이 너무 작습니다.")
    raise ValueError("다운로드된 파일이 유효하지 않습니다.")

# 압축 해제
print(f"\n📦 압축 해제 중...")
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print("압축 파일 내용:")
        for name in zip_ref.namelist()[:5]:
            print(f"  - {name}")
        if len(zip_ref.namelist()) > 5:
            print(f"  ... 외 {len(zip_ref.namelist()) - 5}개")
        
        zip_ref.extractall(data_dir)
    
    print(f"✅ 압축 해제 완료: {vector_db_dir}")
    
    # zip 파일 삭제
    zip_path.unlink()
    print("✅ zip 파일 삭제 완료")
    
except zipfile.BadZipFile as e:
    print(f"❌ 압축 해제 실패: {e}")
    raise

print("\n🎉 모든 작업 완료!")