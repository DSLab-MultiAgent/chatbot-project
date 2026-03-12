"""
FastAPI 메인 애플리케이션
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse  # HTML 전송 및 스트리밍을 위해 추가
from src.models import QueryRequest, QueryResponse, HealthResponse, ResponseType
from src.pipeline.pipeline import RAGPipeline
from src.utils.logger import logger
from src import __version__
import httpx
import json
import asyncio

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

# ----------------------------------------------------------------
# 프론트엔드 연결을 위한 엔드포인트 수정
# ----------------------------------------------------------------

@app.get("/")
async def root():
    """
    루트 접속 시 index.html을 반환합니다.
    FileResponse를 사용하여 정적 HTML을 서빙합니다.
    """
    # run.py가 프로젝트 루트에서 실행되므로, index.html이 루트에 있다면 아래와 같이 지정합니다.
    current_path = os.path.dirname(os.path.abspath(__file__)) # src 폴더 위치
    root_path = os.path.dirname(current_path) # 프로젝트 루트 위치
    html_path = os.path.join(root_path, "index.html")
    
    return FileResponse(html_path)

# ----------------------------------------------------------------

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
    """
    try:
        logger.info(f"쿼리 수신: {request.question}")
        
        # 실제 구축된 파이프라인 에이전트 실행
        result = await pipeline.process(request.question)
        
        logger.info(f"응답 생성 완료: {result.response_type}")
        logger.info(f"응답 결과: {result.answer}")
        return result
        
    except Exception as e:
        logger.error(f"쿼리 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def process_query_stream(request: QueryRequest):
    """
    사용자 쿼리 처리 (스트리밍) - 프론트엔드의 SSE 프로그레스 표시 대응
    """
    async def event_stream():
        q = asyncio.Queue()
        
        # 파이프라인 진행 상태를 Queue에 넣는 콜백
        async def progress_callback(msg: str):
            await q.put({"type": "progress", "message": msg})
            
        async def run_pipeline():
            try:
                # 파이프라인 처리 (콜백 전달)
                result = await pipeline.process(request.question, progress_callback=progress_callback)
                
                # Pydantic 호환 객체 딕셔너리 변환
                result_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
                await q.put({"type": "result", "data": result_dict})
            except Exception as e:
                logger.error(f"스트리밍 처리 중 오류: {e}")
                await q.put({"type": "error", "message": "서버 처리 중 오류가 발생했습니다."})

        # 파이프라인 백그라운드 태스크 실행
        asyncio.create_task(run_pipeline())
        
        # Queue에서 이벤트를 꺼내어 프론트엔드에 스트리밍
        while True:
            item = await q.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            
            # 최종 결과나 에러 발생 시 스트리밍 종료
            if item["type"] in ["result", "error"]:
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")

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