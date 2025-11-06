"""
Tools module for LangChain Agent
Arxiv 논문 검색 도구 정의
"""

from langchain.tools import tool
import arxiv
from typing import Dict


def create_arxiv_tool(config: Dict):
    """
    Arxiv 검색 도구를 생성합니다.

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
            str: 상위 5개 논문의 제목, 저자, 요약 정보
        """
        try:
            # arxiv 네이티브 라이브러리 사용
            client = arxiv.Client(
                page_size=config.get('page_size', 5),
                delay_seconds=config.get('delay_seconds', 3),
                num_retries=config.get('num_retries', 3)
            )

            search = arxiv.Search(
                query=query,
                max_results=config.get('max_results', 5),
                sort_by=getattr(arxiv.SortCriterion, config.get('sort_by', 'Relevance'))
            )

            results = []
            summary_max_chars = config.get('summary_max_chars', 500)

            for result in client.results(search):
                paper_info = f"""
제목: {result.title}
저자: {', '.join([author.name for author in result.authors])}
게시일: {result.published.strftime('%Y-%m-%d')}
요약: {result.summary[:summary_max_chars]}...
링크: {result.entry_id}
---
"""
                results.append(paper_info)

            if results:
                return "\n".join(results)
            else:
                return "검색 결과가 없습니다. 다른 키워드로 시도해보세요."

        except Exception as e:
            return f"논문 검색 중 오류 발생: {str(e)}\n쿼리를 다시 확인하거나 네트워크 연결을 확인해주세요."

    return get_source
