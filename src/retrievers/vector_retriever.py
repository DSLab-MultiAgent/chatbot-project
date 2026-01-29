# src/retrievers/vector_retriever.py
from typing import List
import json, os
from colbert_matryoshka import MatryoshkaColBERT
from pylate import indexes, retrieve
from src.models import Document
from src.utils.logger import logger

class VectorRetriever:
    def __init__(self):
        # 1) 모델/인덱스 로드 
        self.model = MatryoshkaColBERT.from_pretrained("./src/retrievers/models/dragonkue/colbert-ko-0.1b", trust_remote_code=True)
        self.model.set_active_dim(128)

        self.index = indexes.PLAID(index_folder="./data/vector_db/pylate-index", index_name="graduate_regulations")
        self.retriever = retrieve.ColBERT(index=self.index)

        # 2) pylate_data.json 로드 및 id->(text, meta) 맵 구성 
        with open("./data/vector_db/pylate_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.doc_map = {
            str(m["id"]): {"text": d, "meta": m}
            for d, m in zip(data["documents"], data["metadatas"])
        }

    async def search(self, query: str, chapter: List[str], top_k: int = 10, top_n: int = 200) -> List[Document]:
        if hasattr(self.index, 'docids'):
            logger.info(f"인덱스 내 문서 수: {len(self.index.docids)}")
        if not chapter:
            raise ValueError("chapter 최소 1개 이상이어야 합니다(필수).")

        sec_set = set(chapter)

        # 3) PLAID에서 우선 top_n 뽑기(전체 코퍼스 기준) 
        q_emb = self.model.encode([query], is_query=True)
        results = self.retriever.retrieve(queries_embeddings=q_emb, k=top_n)[0]
        logger.info(f"리트리버 topk : {results}")

        out: List[Document] = []
        for r in results:
            doc_id = str(r["id"])
            item = self.doc_map.get(doc_id)
            if not item:
                continue

            meta = dict(item["meta"])
            if meta.get("chapter") not in sec_set:
                continue

            # output 요구사항: content, type, doc_no
            # 현재 데이터엔 type이 없으므로 기본값 "규정"
            metadata = {
                "id": meta.get("id"),
                "type": meta.get("type", "규정"),
                "doc_no": meta.get("doc_no", meta.get("id", doc_id)),
                "chapter": meta.get("chapter"),
            }

            # score는 외부엔 필요 없어도 Document 모델상 필수라 채움 :contentReference[oaicite:9]{index=9}
            out.append(Document(content=item["text"], metadata=metadata, score=float(r["score"])))

            # 섹션 필터 후 10개 채우면 종료 (10개 미만이면 끝까지 가도 out은 있는 만큼)
            if len(out) >= top_k:
                break

        return out
