"""
헬퍼 함수 모음
"""
from typing import List, Dict, Any


def calculate_average_score(documents: List[Any]) -> float:
    """
    문서들의 평균 점수 계산
    
    Args:
        documents: 점수가 있는 문서 리스트
        
    Returns:
        평균 점수
    """
    if not documents:
        return 0.0
    
    total_score = sum(doc.score for doc in documents if hasattr(doc, 'score'))
    return total_score / len(documents)


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    텍스트를 최대 길이로 자르기
    
    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        
    Returns:
        잘린 텍스트
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    메타데이터를 읽기 쉬운 형식으로 포매팅
    
    Args:
        metadata: 메타데이터 딕셔너리
        
    Returns:
        포매팅된 문자열
    """
    lines = []
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    
    return "\n".join(lines)