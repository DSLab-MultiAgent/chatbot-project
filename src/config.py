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


class Settings(BaseSettings):
    """환경 변수 설정"""
    
    # API Keys
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # LLM Settings
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # Vector DB Settings
    VECTOR_DB_TYPE: str = "chromadb"
    VECTOR_DB_PATH: str = "./data/vector_db"
    VECTOR_DB_FILE_ID: Optional[str] = None
    EMBEDDING_MODEL: str = "src/retrievers/models/dragonkue/colbert-ko-0.1b"
    
    # Retriever Settings
    TOP_K_INITIAL: int = 5
    TOP_K_SECONDARY: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_LOOP_COUNT: int = 1
    
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


def load_yaml_config(config_path: str = "config/config.yaml") -> dict:
    """YAML 설정 파일 로드"""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


# 전역 설정 인스턴스
settings = Settings()
yaml_config = load_yaml_config()