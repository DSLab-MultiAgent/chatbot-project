"""
데이터 모델 정의
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class QueryRequest(BaseModel):
    """사용자 쿼리 요청"""
    question: str = Field(..., description="사용자 질문")
    user_id: Optional[str] = Field(None, description="사용자 ID (선택)")


class Document(BaseModel):
    """검색된 문서"""
    content: str = Field(..., description="문서 내용")
    metadata: dict = Field(default_factory=dict, description="메타데이터")
    score: float = Field(..., description="유사도 점수")


class ResponseType(str, Enum):
    """응답 타입"""
    ANSWER = "answer"              # 완전 응답
    CONDITIONAL = "conditional"    # 조건부 응답
    GARBAGE = "garbage"            # 가비지 쿼리
    ERROR = "error"                # 에러


class QueryResponse(BaseModel):
    """쿼리 응답"""
    answer: str = Field(..., description="답변 내용")
    response_type: ResponseType = Field(..., description="응답 타입")
    sources: List[Document] = Field(default_factory=list, description="참조 문서")
    confidence: float = Field(..., description="답변 신뢰도 (0-1)")
    
    class Config:
        use_enum_values = True


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    version: str