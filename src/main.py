"""
FastAPI 메인 애플리케이션
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.models import QueryRequest, QueryResponse, HealthResponse, ResponseType
from src.pipeline.pipeline import RAGPipeline
from src.utils.logger import logger
from src import __version__

# FastAPI 앱 초기화
app = FastAPI(
    title="교학팀 챗봇 API",
    description="멀티에이전트 RAG 기반 교학팀 문의 자동 응답 시스템",
    version=__version__
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 파이프라인 초기화
pipeline = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    global pipeline
    try:
        logger.info("RAG 파이프라인 초기화 중...")
        pipeline = RAGPipeline()
        logger.info("RAG 파이프라인 초기화 완료!")
    except Exception as e:
        logger.error(f"파이프라인 초기화 실패: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """루트 엔드포인트"""
    return {
        "message": "교학팀 챗봇 API 서버",
        "version": __version__,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="healthy",
        version=__version__
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    사용자 쿼리 처리
    
    Args:
        request: 사용자 질문
        
    Returns:
        답변 및 참조 문서
    """
    try:
        logger.info(f"쿼리 수신: {request.question}")
        
        # 파이프라인 실행
        result = await pipeline.process(request.question)
        
        logger.info(f"응답 생성 완료: {result.response_type}")
        return result
        
    except Exception as e:
        logger.error(f"쿼리 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/pipeline-status")
async def pipeline_status():
    """파이프라인 상태 확인 (디버그용)"""
    if pipeline is None:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized",
        "retriever_ready": pipeline.retriever is not None,
        "llm_ready": pipeline.answer_generator is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)