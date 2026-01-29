import asyncio
from src.retrievers.vector_retriever import VectorRetriever


async def test_vector_retriever():
    # 1) Retriever 초기화
    retriever = VectorRetriever()

    # 2) 테스트용 query / sections
    query = "수강 신청 기간은 언제인가요?"
    
    # pylate_data.json 안에 실제 존재하는 section 값 중 하나로 맞춰야 함
    # 예시: ["수강신청"], ["휴학"], ["성적"], 등
    sections = ["수강신청"]

    # 3) 검색 실행
    results = await retriever.search(
        query=query,
        sections=sections,
        top_k=10,
        top_n=200
    )

    # 4) 결과 출력
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Sections filter: {sections}")
    print(f"Retrieved {len(results)} documents")
    print("=" * 80)

    for i, doc in enumerate(results, start=1):
        print(f"[{i}]")
        print(f"  doc_no : {doc.metadata.get('doc_no')}")
        print(f"  type   : {doc.metadata.get('type')}")
        print(f"  section: {doc.metadata.get('section')}")
        print(f"  score  : {doc.score:.4f}")
        print("  content:")
        print(f"  {doc.content[:300]}")  # 너무 길면 앞부분만
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(test_vector_retriever())
