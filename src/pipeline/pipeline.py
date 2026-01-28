"""
RAG 파이프라인 통합 모듈
전체 프로세스 플로우 관리
"""
from src.models import QueryResponse, ResponseType
from src.pipeline.query_classifier import QueryClassifier
from src.pipeline.retriever import Retriever
from src.pipeline.document_validator import DocumentValidator
from src.pipeline.answer_generator import AnswerGenerator
from src.config import settings
from src.utils.logger import logger


class RAGPipeline:
    """RAG 파이프라인 메인 클래스"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.query_classifier = QueryClassifier()
        self.retriever = Retriever()
        self.document_validator = DocumentValidator()
        self.answer_generator = AnswerGenerator()
        self.max_loop = 1  # 최대 1회 재시도
        
        logger.info("RAG 파이프라인 초기화 완료")
    
    async def process(self, query: str) -> QueryResponse:
        """
        전체 파이프라인 실행
        
        Args:
            query: 사용자 질문
            
        Returns:
            최종 답변
            
        플로우:
        1. 쿼리 분류 (가비지 체크)
        2. 하이브리드 검색 (top 1~10, Late Interaction)
        3. 문서 검증 (top 1~5 사용)
        4. 답변 가능성 확인
           - 가능 → 답변 생성
           - 불가능 (loop==0) → 문서 검증으로 back (top 6~10)
           - 불가능 (loop==1) → 조건부 응답
        """
        logger.info(f"=== 파이프라인 시작: {query} ===")
        
        # 1. 쿼리 분류
        is_valid, category = await self.query_classifier.classify(query)
        
        if not is_valid:
            # 가비지 쿼리 → 초기화 응답
            logger.info("가비지 쿼리 감지 → 초기화 응답")
            garbage_msg = self.query_classifier.get_garbage_response()
            return QueryResponse(
                answer=garbage_msg,
                response_type=ResponseType.CONDITIONAL,
                sources=[],
                confidence=0.0
            )
        
        logger.info(f"쿼리 분류 완료: {category}")
        
        # 2. 하이브리드 검색 (top 1~10)
        all_documents = await self.retriever.search(
            query=query,
            category=category
        )
        
        if not all_documents:
            logger.warning("검색 결과 없음 → 조건부 응답")
            return await self.answer_generator.generate_conditional_response(query)
        
        # 루프 시작 (최대 2번: loop 0, 1)
        loop_count = 0
        
        while loop_count <= self.max_loop:
            logger.info(f"--- Loop {loop_count} ---")
            
            # 3. 문서 검증
            if loop_count == 0:
                # 첫 시도: top 1~5 사용
                docs_to_validate = self.retriever.get_top_n(all_documents, 5)
                logger.info("1차 검증: top 1~5 문서")
            else:
                # 재시도: top 6~10 사용
                docs_to_validate = all_documents[5:10]
                logger.info("2차 검증: top 6~10 문서")
            
            # 문서 검증 실행
            validated_docs = await self.document_validator.validate_documents(
                query=query,
                documents=docs_to_validate
            )
            
            # 4. 답변 가능성 확인
            is_helpful = await self.answer_generator.check_helpfulness(
                query=query,
                documents=validated_docs
            )
            
            # 5. 답변 생성 또는 다음 루프
            if is_helpful:
                # 답변 가능 → 생성
                response = await self.answer_generator.generate_answer(
                    query=query,
                    documents=validated_docs
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
            
            # 다음 루프로
            loop_count += 1
        
        # 안전장치: 여기 도달하면 조건부 응답
        logger.warning("예상치 못한 종료 → 조건부 응답")
        return await self.answer_generator.generate_conditional_response(query=query)