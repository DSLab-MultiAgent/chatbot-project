"""
RAG 파이프라인 통합 모듈
전체 프로세스 플로우 관리
"""
from src.models import QueryResponse, ResponseType
from src.pipeline.query_classifier import QueryClassifier
from src.pipeline.retriever import Retriever
from src.pipeline.document_validator import DocumentValidator
from src.pipeline.context_validator import ContextValidator
from src.pipeline.conditional_checker import ConditionalChecker
from src.agents.answer_agent import AnswerAgent
from src.agents.conditional_agent import ConditionalAgent
from src.config import settings
from src.utils.logger import logger


class RAGPipeline:
    """RAG 파이프라인 메인 클래스"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.query_classifier = QueryClassifier()
        self.retriever = Retriever()
        self.document_validator = DocumentValidator()  # 관련성 체크
        self.context_validator = ContextValidator()    # 컨텍스트 검증
        self.conditional_checker = ConditionalChecker()  # 조건부 체크
        self.answer_agent = AnswerAgent()
        self.conditional_agent = ConditionalAgent()
        
        logger.info("RAG 파이프라인 초기화 완료")

    async def process(self, query: str) -> QueryResponse:
        """
        전체 파이프라인 실행
        
        플로우:
        1. 쿼리 분류 (가비지 체크)
        2. 하이브리드 검색 (top 10)
        3. 관련성 체크 (top 1~5)
        4. 컨텍스트 검증
           - YES → 조건부 체크
                ├─ CONDITIONAL → 조건부 응답
                └─ COMPLETE → 완전 응답
           - NO → Exception
        5. (실패 시) 관련성 체크 (top 6~10)
        6. 컨텍스트 검증 (1차 + 2차 문서)
           - YES → 조건부 체크
           - NO → 교학팀 문의
        """
        logger.info(f"=== 파이프라인 시작: {query} ===")
        
        try:
            # 1. 쿼리 분류
            is_valid, chapter = await self.query_classifier.classify(query)
            
            if not is_valid:
                logger.info("가비지 쿼리 감지")
                garbage_msg = self.query_classifier.get_garbage_response()
                return QueryResponse(
                    answer=garbage_msg,
                    response_type=ResponseType.GARBAGE,
                    sources=[],
                    confidence=0.0
                )
            
            logger.info(f"쿼리 분류 완료: {chapter}")
            
            # 쿼리 정제
            # refined_query = await self.query_processor.refine_query(query)
            refined_query = query

            # 2. 하이브리드 검색 (top 10)
            all_documents = await self.retriever.search(
                query=refined_query,
                chapter=chapter
            )
            
            if not all_documents:
                logger.warning("검색 결과 없음")
                return await self.conditional_agent.generate_no_documents(query)
            
            logger.info(f"검색 완료: {len(all_documents)}개 문서")
            
            # === 1차 시도 ===
            logger.info("--- 1차 시도 시작 ---")
            
            # 3. 관련성 체크 (top 1~5)
            top5_docs = all_documents[:5]
            relevant_docs_1st = await self.document_validator.get_relevant_documents(
                query=refined_query,
                documents=top5_docs
            )
            
            if relevant_docs_1st:
                # 4. 컨텍스트 검증
                is_valid_context, reason = await self.context_validator.validate_context(
                    query=refined_query,
                    documents=relevant_docs_1st
                )
                
                if is_valid_context:
                    # 컨텍스트 검증 성공 → 답변 생성
                    logger.info("1차 컨텍스트 검증 성공")
                    
                    # 5. 조건부 응답 여부 체크
                    is_conditional = await self.conditional_checker.check_conditional(
                        query=refined_query,
                        documents=relevant_docs_1st
                    )
                    
                    if is_conditional:
                        # 조건부 응답 생성
                        response = await self.conditional_agent.generate_with_documents(
                            query=query,
                            documents=relevant_docs_1st
                        )
                        logger.info("=== 파이프라인 완료: 조건부 응답 ===")
                        return response
                    else:
                        # 완전 응답 생성
                        response = await self.answer_agent.generate(
                            query=query,
                            documents=relevant_docs_1st
                        )
                        logger.info("=== 파이프라인 완료: 완전 응답 ===")
                        return response
                else:
                    # 컨텍스트 검증 실패 → 2차 시도
                    logger.info(f"1차 컨텍스트 검증 실패: {reason}")
            else:
                logger.info("1차 관련 문서 없음")
            
            # === 2차 시도 ===
            logger.info("--- 2차 시도 시작 ---")
            
            # 6. 관련성 체크 (top 6~10)
            if len(all_documents) > 5:
                bottom5_docs = all_documents[5:]
                relevant_docs_2nd = await self.document_validator.get_relevant_documents(
                    query=refined_query,
                    documents=bottom5_docs
                )
                
                # 1차 + 2차 관련 문서 결합
                all_relevant_docs = relevant_docs_1st + relevant_docs_2nd
                
                if all_relevant_docs:
                    # 7. 컨텍스트 검증 (1차 + 2차)
                    is_valid_context, reason = await self.context_validator.validate_context(
                        query=refined_query,
                        documents=all_relevant_docs
                    )
                    
                    if is_valid_context:
                        # 컨텍스트 검증 성공
                        logger.info("2차 컨텍스트 검증 성공")
                        
                        # 8. 조건부 응답 여부 체크
                        is_conditional = await self.conditional_checker.check_conditional(
                            query=refined_query,
                            documents=all_relevant_docs
                        )
                        
                        if is_conditional:
                            # 조건부 응답 생성
                            response = await self.conditional_agent.generate_with_documents(
                                query=query,
                                documents=all_relevant_docs
                            )
                            logger.info("=== 파이프라인 완료: 조건부 응답 (2차) ===")
                            return response
                        else:
                            # 완전 응답 생성
                            response = await self.answer_agent.generate(
                                query=query,
                                documents=all_relevant_docs
                            )
                            logger.info("=== 파이프라인 완료: 완전 응답 (2차) ===")
                            return response
                    else:
                        # 2차 컨텍스트 검증 실패 → 교학팀 문의
                        logger.info(f"2차 컨텍스트 검증 실패: {reason}")
            
            # 모든 시도 실패 → 교학팀 문의 안내
            logger.info("모든 시도 실패 → 교학팀 문의 안내")
            return await self.conditional_agent.generate_no_documents(query)
            
        except Exception as e:
            # 전체 파이프라인 에러
            logger.error(f"파이프라인 에러: {e}", exc_info=True)
            
            return QueryResponse(
                answer=(
                    f"죄송합니다. 답변 생성 중 오류가 발생했습니다.\n\n"
                    f"📞 교학팀으로 직접 문의해주시기 바랍니다.\n"
                    f"전화: 02-910-4018\n"
                    f"이메일: business-it@kookmin.ac.kr"
                ),
                response_type=ResponseType.ERROR,
                sources=[],
                confidence=0.0
            )