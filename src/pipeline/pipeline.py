"""
RAG 파이프라인 통합 모듈
전체 프로세스 플로우 관리
"""
from src.models import QueryResponse
from src.pipeline.query_processor import QueryProcessor
from src.pipeline.retriever import Retriever
from src.pipeline.answer_generator import AnswerGenerator
from src.config import settings
from src.utils.logger import logger


class RAGPipeline:
    """RAG 파이프라인 메인 클래스"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.query_processor = QueryProcessor()
        self.retriever = Retriever()
        self.answer_generator = AnswerGenerator()
        self.max_loop = settings.MAX_LOOP_COUNT
        
        logger.info("RAG 파이프라인 초기화 완료")
    
    async def process(self, query: str) -> QueryResponse:
        """
        전체 파이프라인 실행
        
        Args:
            query: 사용자 질문
            
        Returns:
            최종 답변
            
        플로우:
        1. 쿼리 정제
        2. 1차 검색 (top 1~5)
        3. 답변 가능성 확인
           - 가능 → 답변 생성
           - 불가능 (loop==0) → 2차 검색 (top 6~10)
           - 불가능 (loop==1) → 조건부 응답
        """
        logger.info(f"=== 파이프라인 시작: {query} ===")
        
        # 1. 쿼리 정제
        refined_query = await self.query_processor.refine_query(query)
        
        loop_count = 0
        all_documents = []
        # 2. 검색 실행
        if loop_count == 0:
            # 1차 검색
            documents = await self.retriever.retrieve_initial(refined_query)
        else:
            # 2차 검색
            documents = await self.retriever.retrieve_secondary(refined_query)
        
        all_documents.extend(documents)
        
        # 3. 답변 가능성 확인
        is_helpful = await self.answer_generator.check_helpfulness(
            query=refined_query,
            documents=all_documents
        )
        
        # 4. 답변 생성 또는 다음 루프
        if is_helpful:
            # 답변 생성
            response = await self.answer_generator.generate_answer(
                query=query,
                documents=all_documents
            )
            logger.info("=== 파이프라인 완료: 답변 생성 ===")
            return response
        
        elif loop_count >= self.max_loop:
            # 최대 루프 도달 → 조건부 응답
            response = await self.answer_generator.generate_conditional_response(
                query=query
            )
            logger.info("=== 파이프라인 완료: 조건부 응답 ===")
            return response
        
        loop_count += 1
    
    # 안전장치: 여기 도달하면 조건부 응답
    logger.warning("예상치 못한 종료, 조건부 응답 반환")
    return await self.answer_generator.generate_conditional_response(query=query)