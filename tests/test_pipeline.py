"""
파이프라인 테스트
"""
import pytest
from src.pipeline.pipeline import RAGPipeline


@pytest.mark.asyncio
async def test_pipeline_basic():
    """
    기본 파이프라인 테스트
    
    TODO:
    - [ ] 실제 테스트 케이스 작성
    - [ ] Mock 데이터 준비
    - [ ] 각 모듈별 단위 테스트
    """
    # 임시 테스트
    pipeline = RAGPipeline()
    assert pipeline is not None