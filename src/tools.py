"""
Tools module for LangChain Agent
Arxiv 논문 검색 도구 정의
LangChain ArxivLoader 활용
"""

from langchain.tools import tool
from langchain_community.document_loaders import ArxivLoader
from typing import Dict


def create_arxiv_tool(config: Dict):
    """
    Arxiv 검색 도구를 생성합니다 (LangChain ArxivLoader 활용).

    Args:
        config: arxiv 설정이 담긴 딕셔너리

    Returns:
        tool: LangChain tool 객체
    """

    @tool
    def get_source(query: str) -> str:
        """
        Arxiv에서 학술 논문을 검색합니다.

        이 도구는 다음과 같은 경우에 사용하세요:
        - 특정 주제에 대한 최신 연구 논문을 찾을 때
        - 저자원 언어 처리, 벤치마크, NLP 관련 연구를 검색할 때
        - 학술적 근거가 필요한 답변을 작성할 때

        Args:
            query (str): 검색할 키워드나 주제 (영어로 입력하는 것이 좋습니다)

        Returns:
            str: 상위 N개 논문의 제목, 저자, 요약 정보
        """
        try:
            # LangChain ArxivLoader 사용
            max_docs = config.get('max_results', 5)
            summary_max_chars = config.get('summary_max_chars', 500)

            loader = ArxivLoader(
                query=query,
                load_max_docs=max_docs,
                doc_content_chars_max=summary_max_chars
            )

            # 요약만 로드 (전체 PDF 텍스트 대신 요약 사용)
            docs = loader.get_summaries_as_docs()

            if not docs:
                return "검색 결과가 없습니다. 다른 키워드로 시도해보세요."

            results = []
            for doc in docs:
                metadata = doc.metadata

                # 저자 정보 추출 (리스트 형태로 제공됨)
                authors = metadata.get('Authors', '')
                if isinstance(authors, list):
                    authors = ', '.join(authors)

                # entry_id가 없으면 다른 필드 시도 (Entry ID, arxiv_id, url 등)
                entry_id = metadata.get('entry_id') or metadata.get('Entry ID') or metadata.get('arxiv_id') or metadata.get('url', 'N/A')

                paper_info = f"""
제목: {metadata.get('Title', 'N/A')}
저자: {authors}
게시일: {metadata.get('Published', 'N/A')}
요약: {doc.page_content[:summary_max_chars]}{'...' if len(doc.page_content) > summary_max_chars else ''}
링크: {entry_id}
---
"""
                results.append(paper_info)

            return "\n".join(results)

        except Exception as e:
            return f"논문 검색 중 오류 발생: {str(e)}\n쿼리를 다시 확인하거나 네트워크 연결을 확인해주세요."

    return get_source
