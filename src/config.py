##sungho
"""
설정 관리 모듈
"""
# mssong dd dd
# 23:32 config 변경 해봄

from pydantic_settings import BaseSettings
from typing import Optional
import yaml
from pathlib import Path



# 프로젝트 루트 기준 경로 계산
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """환경 변수 설정"""
    
    # API Keys
    OPENAI_API_KEY: str
    
    # LLM Settings
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # Server Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def load_yaml_config(config_path: Optional[str] = None) -> dict:
    """YAML 설정 파일 로드 (실행 위치와 상관없이 항상 프로젝트 루트 기준으로 로드)"""
    if config_path is None:
        config_file = BASE_DIR / "config" / "config.yaml"
    else:
        config_file = Path(config_path)

    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# 전역 설정 인스턴스
settings = Settings()
yaml_config = load_yaml_config()