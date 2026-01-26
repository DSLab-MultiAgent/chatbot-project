"""
로깅 설정
"""
from loguru import logger
import sys
from pathlib import Path
from src.config import settings

# 로그 디렉토리 생성
log_dir = Path(settings.LOG_FILE).parent
log_dir.mkdir(parents=True, exist_ok=True)

# 기본 로거 제거
logger.remove()

# 콘솔 로거
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
    colorize=True
)

# 파일 로거
logger.add(
    settings.LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    level=settings.LOG_LEVEL,
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)

logger.info("로거 초기화 완료")