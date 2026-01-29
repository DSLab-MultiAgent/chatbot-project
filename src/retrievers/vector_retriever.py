# src/retrievers/vector_retriever.py
from typing import List, Dict
import json
import os
from collections import defaultdict  # 추가됨

from colbert_matryoshka import MatryoshkaColBERT
from pylate import indexes, retrieve
from src.models import Document
from src.utils.logger import logger

class VectorRetriever:
    def __init__(self):
        # 1) 모델/인덱스 로드 
        logger.info("Retriever 모델 및 인덱스 로드 중...")
        self.model = MatryoshkaColBERT.from_pretrained(
            "./src/retrievers/models/dragonkue/colbert-ko-0.1b", 
            trust_remote_code=True
        )
        self.model.set_active_dim(128)

        self.index = indexes.PLAID(
            index_folder="./data/vector_db/pylate-index", 
            index_name="graduate_regulations"
        )
        self.retriever = retrieve.ColBERT(index=self.index)

        # 2) pylate_data.json 로드 및 맵 구성 (Doc Map & Chapter Map)
        data_path = "./data/vector_db/pylate_data.json"
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.doc_map = {}
        self.chapter_map = defaultdict(list)  # { "Chapter명": [doc_id(int), ...] }

        # 데이터 순회하며 매핑 생성
        for d, m in zip(data["documents"], data["metadatas"]):
            # doc_id 변환 (문자열/정수 관리)
            doc_id_str = str(m["id"])
            doc_id_int = int(m["id"])  # PLAID docids 필터링용 정수 ID

            # 2-1) ID -> (Text, Meta) 맵핑
            self.doc_map[doc_id_str] = {"text": d, "meta": m}
            
            # 2-2) Chapter -> ID 목록 역색인 (Pre-filtering용)
            chapter_name = m.get("chapter")
            if chapter_name:
                self.chapter_map[chapter_name].append(doc_id_int)

        logger.info(f"데이터 로드 완료: 총 문서 {len(self.doc_map)}개")

    async def search(self, query: str, chapter: List[str], top_k: int = 10, top_n: int = 200) -> List[Document]:
        """
        챕터 기반 Pre-Filtering을 적용한 하이브리드 검색
        """
        if hasattr(self.index, 'docids'):
            # 디버깅용: 전체 문서 수 확인
            pass 

        if not chapter:
            raise ValueError("chapter 최소 1개 이상이어야 합니다(필수).")

        # 3) Pre-Filtering: 검색 대상 문서 ID 추출
        target_docids = []
        for ch in chapter:
            ids = self.chapter_map.get(ch)
            if ids:
                target_docids.extend(ids)
        
        # 중복 제거 (혹시 모를 중복 ID 방지)
        target_docids = list(set(target_docids))

        # 해당 챕터에 문서가 하나도 없으면 빠른 리턴
        if not target_docids:
            logger.warning(f"요청한 챕터({chapter})에 해당하는 문서가 없습니다.")
            return []
            
        logger.info(f"검색 범위 제한: 전체 중 {len(target_docids)}개 문서 대상")

        # 4) PLAID 검색 (docids 파라미터로 범위 제한)
        # top_n은 후보군 개수이므로, 실제 대상 문서 수보다 클 수 없음
        k_search = min(top_n, len(target_docids))
        
        q_emb = self.model.encode([query], is_query=True)
        
        # [핵심 변경] docids 인자를 넣어 검색 범위를 물리적으로 제한함
        results = self.retriever.retrieve(
            queries_embeddings=q_emb, 
            k=k_search,
            subset=target_docids 
        )[0]
        
        logger.info(f"리트리버 검색 결과: {len(results)}개")

        # 5) 결과 변환 (Document 객체 생성)
        out: List[Document] = []
        for r in results:
            doc_id = str(r["id"])
            item = self.doc_map.get(doc_id)
            
            if not item:
                continue
            
            # 이미 docids로 필터링했으므로 chapter 재확인 로직 삭제 (속도 향상)

            meta = item["meta"]
            
            # 메타데이터 구성 (요구사항 반영)
            metadata = {
                "id": meta.get("id"),
                "type": meta.get("type", "규정"),
                "doc_no": meta.get("doc_no", meta.get("id", doc_id)),
                "chapter": meta.get("chapter"),
                "section": meta.get("section", "")
            }

            out.append(Document(
                content=item["text"], 
                metadata=metadata, 
                score=float(r["score"])
            ))

            # top_k개 채워지면 중단 (Pre-filtering 덕분에 순서대로 유효한 결과임)
            if len(out) >= top_k:
                break

        return out