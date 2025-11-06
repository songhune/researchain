"""
Citation Chain Analyzer
PDF에서 인용 정보를 추출하고 체인 분석
마크다운 및 BibTeX 출력 지원
재귀적 citation chain 분석 지원
LangChain ArxivLoader 활용
"""

import os
import re
from typing import Dict, List, Tuple, Set
from datetime import datetime
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import ArxivLoader
from pypdf import PdfReader


class CitationChainAnalyzer:
    """PDF의 citation을 분석하고 체인을 구축하는 클래스"""

    def __init__(self, config: Dict):
        """
        초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.llm = self._initialize_llm()

        # 모델 provider 확인 (tool calling 지원 여부)
        model_provider = self.config['llm'].get('model_provider', '').lower()
        self.use_tools = model_provider not in ['perplexity']

    def _initialize_llm(self):
        """LLM 초기화"""
        llm_config = self.config['llm']
        model_provider = llm_config['model_provider'].lower()

        # Perplexity는 init_chat_model이 API 키를 제대로 인식하지 못하므로 직접 초기화
        if model_provider == 'perplexity':
            from langchain_perplexity import ChatPerplexity

            # PPLX_API_KEY 환경 변수에서 자동 로드
            return ChatPerplexity(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                request_timeout=llm_config['timeout'],
                max_tokens=llm_config['max_tokens']
            )
        else:
            return init_chat_model(
                model=llm_config['model'],
                model_provider=llm_config['model_provider'],
                temperature=llm_config['temperature'],
                timeout=llm_config['timeout'],
                max_tokens=llm_config['max_tokens']
            )

    def extract_citations_from_pdf(self, pdf_path: str) -> List[str]:
        """
        PDF에서 인용 정보 추출

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            인용 논문 제목 리스트
        """
        try:
            # PyPDF 사용하여 PDF 읽기
            reader = PdfReader(pdf_path)

            # 모든 페이지 텍스트 합치기
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

            # References 섹션 찾기
            ref_pattern = r'(?:References|REFERENCES|Bibliography)(.*?)(?:\n\n\n|$)'
            ref_match = re.search(ref_pattern, full_text, re.DOTALL | re.IGNORECASE)

            if not ref_match:
                print(f"References 섹션을 찾을 수 없습니다: {pdf_path}")
                return []

            references_text = ref_match.group(1)

            # 인용 패턴 매칭 (다양한 포맷 지원)
            # [1] Author et al., "Title", ...
            # Author, A. (Year). Title. ...
            citations = []

            # 줄 단위로 분리
            lines = references_text.split('\n')
            current_citation = ""

            for line in lines:
                line = line.strip()
                if not line:
                    if current_citation:
                        citations.append(current_citation)
                        current_citation = ""
                    continue

                # 새로운 인용 시작 (번호나 저자명으로 시작)
                if re.match(r'^\[?\d+\]?\.?\s', line) or re.match(r'^[A-Z][a-z]+,', line):
                    if current_citation:
                        citations.append(current_citation)
                    current_citation = line
                else:
                    current_citation += " " + line

            if current_citation:
                citations.append(current_citation)

            print(f"추출된 인용 수: {len(citations)}")
            return citations[:50]  # 최대 50개

        except Exception as e:
            print(f"PDF 처리 오류 ({pdf_path}): {str(e)}")
            return []

    def analyze_citation_chain(self, pdf_path: str) -> str:
        """
        PDF의 citation chain을 분석

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            분석 결과 텍스트
        """
        print(f"\nCitation Chain 분석 시작: {pdf_path}")

        # PDF에서 인용 추출
        citations = self.extract_citations_from_pdf(pdf_path)

        if not citations:
            return "인용 정보를 추출할 수 없습니다."

        # LLM을 사용하여 인용 분석
        system_prompt = """당신은 학술 논문의 인용 관계를 분석하는 전문가입니다.
주어진 인용 목록을 분석하여 다음을 수행하세요:

1. 주요 인용 논문 식별 (가장 중요해 보이는 5-10개)
2. 인용 논문의 주제 분류
3. 시간적 흐름 분석 (연도별 주요 연구)
4. 핵심 연구자 식별

결과를 마크다운 형식으로 구조화하여 제공하세요."""

        # Tool calling 미지원 모델인 경우
        if not self.use_tools:
            print("[Tool calling 미지원] 직접 LLM 호출 모드")
            citations_text = "\n".join(citations)
            full_prompt = f"""{system_prompt}

인용 목록:
{citations_text}

다음 PDF의 인용 정보를 분석해주세요: {os.path.basename(pdf_path)}"""

            response = self.llm.invoke(full_prompt)
            return response.content

        # Tool calling 지원 모델인 경우
        @tool
        def get_citations() -> str:
            """인용 목록을 반환합니다."""
            return "\n".join(citations)

        try:
            agent = create_react_agent(
                self.llm,
                [get_citations]
            )

            # 에이전트 실행
            query = f"다음 PDF의 인용 정보를 분석해주세요: {os.path.basename(pdf_path)}"

            # system prompt를 메시지에 포함
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]

            results = []
            for chunk in agent.stream(
                {"messages": messages},
                stream_mode="values"
            ):
                if "messages" in chunk and chunk["messages"]:
                    last_message = chunk["messages"][-1]
                    if hasattr(last_message, "content"):
                        results.append(last_message.content)

            return results[-1] if results else "분석 실패"
        except NotImplementedError:
            # bind_tools 미지원 시 폴백
            print("[Tool calling 실패] 직접 LLM 호출로 폴백")
            citations_text = "\n".join(citations)
            full_prompt = f"""{system_prompt}

인용 목록:
{citations_text}

다음 PDF의 인용 정보를 분석해주세요: {os.path.basename(pdf_path)}"""

            response = self.llm.invoke(full_prompt)
            return response.content

    def extract_citation_metadata(self, citations: List[str]) -> List[Dict]:
        """
        인용 문자열에서 메타데이터 추출

        Args:
            citations: 인용 문자열 리스트

        Returns:
            메타데이터 딕셔너리 리스트
        """
        papers = []

        for idx, citation in enumerate(citations, 1):
            paper = {}

            # 연도 추출 (다양한 형식 지원)
            year_match = re.search(r'\((\d{4})\)|,\s*(\d{4})[,\.]', citation)
            if year_match:
                paper['year'] = year_match.group(1) or year_match.group(2)

            # 저자 추출 (첫 저자)
            # "Smith, J." 또는 "Smith et al."
            author_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', citation)
            if author_match:
                paper['author'] = author_match.group(1)
            else:
                # "[1] Author" 형식
                author_match = re.search(r'^\[?\d+\]?\s*([A-Z][a-z]+)', citation)
                if author_match:
                    paper['author'] = author_match.group(1)

            # 제목 추출 (따옴표 안 또는 이탤릭)
            title_match = re.search(r'["""](.+?)["""]', citation)
            if not title_match:
                title_match = re.search(r'\.\s+([A-Z][^.]+?\.)(?:\s+In|\s+Journal|\s+Proceedings|$)', citation)

            if title_match:
                paper['title'] = title_match.group(1).strip()

            # 기본값 설정
            if not paper.get('author'):
                paper['author'] = f"Unknown{idx}"
            if not paper.get('year'):
                paper['year'] = "n.d."
            if not paper.get('title'):
                # 인용 전체를 제목으로 (최대 100자)
                paper['title'] = citation[:100] + "..." if len(citation) > 100 else citation

            paper['full_citation'] = citation
            paper['index'] = idx

            papers.append(paper)

        return papers

    def generate_citation_bibtex(self, papers: List[Dict], pdf_name: str, output_path: str) -> str:
        """
        Citation chain을 BibTeX로 저장

        Args:
            papers: 논문 메타데이터 리스트
            pdf_name: 원본 PDF 이름
            output_path: 저장 경로

        Returns:
            생성된 BibTeX 파일 경로
        """
        # PDF 이름에서 키워드 추출
        base_name = os.path.splitext(pdf_name)[0]
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', base_name.lower())[:10]

        bib_path = output_path.replace('.md', '.bib')

        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write(f"% Citation Chain from: {pdf_name}\n")
            f.write(f"% Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"% Total citations: {len(papers)}\n\n")

            for paper in papers:
                # BibTeX 키: pdf이름-cite-번호
                bib_key = f"{clean_name}-cite-{paper['index']}"

                # 저자 처리
                author = paper.get('author', 'Unknown')
                if 'et al' in author.lower():
                    author = author.replace(' et al', ' and others')

                f.write(f"@misc{{{bib_key},\n")
                f.write(f"  author = {{{author}}},\n")

                if paper.get('title'):
                    title = paper['title'].replace('{', '\\{').replace('}', '\\}')
                    f.write(f"  title = {{{title}}},\n")

                if paper.get('year'):
                    f.write(f"  year = {{{paper['year']}}},\n")

                # 전체 인용 정보를 note에 저장
                full_cite = paper['full_citation'].replace('{', '\\{').replace('}', '\\}')
                f.write(f"  note = {{{full_cite}}},\n")

                f.write(f"  howpublished = {{Cited in: {pdf_name}}}\n")
                f.write("}\n\n")

        print(f"Citation BibTeX 저장 완료: {bib_path}")
        return bib_path

    def batch_analyze(self, pdf_paths: List[str], output_dir: str) -> Tuple[str, str]:
        """
        여러 PDF의 citation chain을 일괄 분석

        Args:
            pdf_paths: PDF 파일 경로 리스트
            output_dir: 출력 디렉토리

        Returns:
            생성된 마크다운 및 BibTeX 파일 경로 튜플
        """
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f"citation_analysis_{timestamp}.md")
        bibtex_path = os.path.join(output_dir, f"citation_analysis_{timestamp}.bib")

        # BibTeX 파일 헤더
        with open(bibtex_path, 'w', encoding='utf-8') as bib_f:
            bib_f.write(f"% Citation Chain Analysis\n")
            bib_f.write(f"% Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            bib_f.write(f"% Total PDFs analyzed: {len(pdf_paths)}\n\n")

        # 마크다운 파일 작성
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Citation Chain Analysis\n\n")
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"분석 대상 PDF: {len(pdf_paths)}개\n\n")
            f.write("---\n\n")

            for idx, pdf_path in enumerate(pdf_paths, 1):
                print(f"\n[{idx}/{len(pdf_paths)}] 분석 중: {os.path.basename(pdf_path)}")
                pdf_name = os.path.basename(pdf_path)

                f.write(f"## {idx}. {pdf_name}\n\n")

                # LLM 분석 수행
                analysis = self.analyze_citation_chain(pdf_path)
                f.write(analysis)
                f.write("\n\n")

                # 인용 정보 추출 및 BibTeX 생성
                citations = self.extract_citations_from_pdf(pdf_path)
                if citations:
                    papers = self.extract_citation_metadata(citations)

                    # BibTeX 키 생성용 PDF 이름 정리
                    base_name = os.path.splitext(pdf_name)[0]
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '', base_name.lower())[:10]

                    # BibTeX 파일에 추가
                    with open(bibtex_path, 'a', encoding='utf-8') as bib_f:
                        bib_f.write(f"% Citations from: {pdf_name}\n")
                        bib_f.write(f"% Total citations: {len(papers)}\n\n")

                        for paper in papers:
                            bib_key = f"{clean_name}-cite-{paper['index']}"
                            author = paper.get('author', 'Unknown')
                            if 'et al' in author.lower():
                                author = author.replace(' et al', ' and others')

                            bib_f.write(f"@misc{{{bib_key},\n")
                            bib_f.write(f"  author = {{{author}}},\n")

                            if paper.get('title'):
                                title = paper['title'].replace('{', '\\{').replace('}', '\\}')
                                bib_f.write(f"  title = {{{title}}},\n")

                            if paper.get('year'):
                                bib_f.write(f"  year = {{{paper['year']}}},\n")

                            full_cite = paper['full_citation'].replace('{', '\\{').replace('}', '\\}')
                            bib_f.write(f"  note = {{{full_cite}}},\n")
                            bib_f.write(f"  howpublished = {{Cited in: {pdf_name}}}\n")
                            bib_f.write("}\n\n")

                    f.write(f"**추출된 인용 수:** {len(papers)}개\n\n")

                f.write("---\n\n")

        print(f"\nCitation Chain 분석 완료:")
        print(f"  - 마크다운: {output_path}")
        print(f"  - BibTeX: {bibtex_path}")
        return output_path, bibtex_path

    def extract_paper_titles_from_citations(self, citations: List[str]) -> List[str]:
        """
        인용 문자열에서 논문 제목 추출

        Args:
            citations: 인용 문자열 리스트

        Returns:
            논문 제목 리스트
        """
        titles = []

        for citation in citations:
            # 따옴표 안의 제목 추출
            title_match = re.search(r'["""](.+?)["""]', citation)
            if title_match:
                title = title_match.group(1).strip()
                # 너무 짧은 제목은 제외 (오탐 방지)
                if len(title) > 10:
                    titles.append(title)
                continue

            # 마침표로 구분된 제목 추출 (저자명 다음)
            # "Author, A. (Year). Title. Journal..."
            title_match = re.search(r'\.\s+([A-Z][^.]{10,}?)\.(?:\s+In|\s+Journal|\s+Proceedings|\s+arXiv|$)', citation)
            if title_match:
                title = title_match.group(1).strip()
                titles.append(title)

        return titles

    def search_arxiv_by_title(self, title: str, max_results: int = 1) -> List[Dict]:
        """
        Arxiv에서 제목으로 논문 검색 (LangChain ArxivLoader 활용)

        Args:
            title: 논문 제목
            max_results: 최대 결과 수

        Returns:
            논문 메타데이터 딕셔너리 리스트
        """
        try:
            # 제목에서 특수문자 제거하고 검색 쿼리 생성
            clean_title = re.sub(r'[^\w\s]', ' ', title)
            search_query = f'ti:"{clean_title}"'

            # LangChain ArxivLoader 사용
            loader = ArxivLoader(
                query=search_query,
                load_max_docs=max_results
            )

            # 요약만 로드
            docs = loader.get_summaries_as_docs()

            if not docs:
                return []

            # Document 객체를 딕셔너리로 변환
            results = []
            for doc in docs:
                result = {
                    'metadata': doc.metadata,
                    'page_content': doc.page_content,
                    'title': doc.metadata.get('Title', ''),
                    'authors': doc.metadata.get('Authors', []),
                    'published': doc.metadata.get('Published', ''),
                    'entry_id': doc.metadata.get('entry_id', ''),
                    'summary': doc.page_content
                }
                results.append(result)

            return results

        except Exception as e:
            print(f"Arxiv 검색 오류 ({title[:50]}...): {str(e)}")
            return []

    def download_pdf_from_arxiv(self, paper_dict: Dict, output_dir: str, filename: str) -> str:
        """
        Arxiv에서 PDF 다운로드

        Args:
            paper_dict: 논문 정보 딕셔너리 (search_arxiv_by_title 결과)
            output_dir: 출력 디렉토리
            filename: 저장할 파일명

        Returns:
            다운로드된 PDF 파일 경로
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, filename)

            # 이미 존재하면 스킵
            if os.path.exists(pdf_path):
                print(f"이미 존재: {filename}")
                return pdf_path

            # ArxivLoader를 사용하여 전체 문서(PDF 내용) 로드
            entry_id = paper_dict.get('entry_id', '')
            if not entry_id:
                print(f"PDF 다운로드 오류: entry_id가 없습니다")
                return None

            # Arxiv ID 추출 (URL에서)
            arxiv_id = entry_id.split('/')[-1]

            # ArxivLoader로 PDF 내용 로드
            loader = ArxivLoader(query=arxiv_id, load_max_docs=1)
            docs = loader.load()  # 전체 PDF 텍스트 로드

            if not docs:
                print(f"PDF 다운로드 실패: {filename}")
                return None

            # PDF 저장 (ArxivLoader는 텍스트만 제공하므로 별도 다운로드 필요)
            # arxiv 라이브러리 직접 사용
            import arxiv as arxiv_lib
            paper = next(arxiv_lib.Client().results(arxiv_lib.Search(id_list=[arxiv_id])))
            paper.download_pdf(dirpath=output_dir, filename=filename)

            print(f"다운로드 완료: {filename}")
            return pdf_path

        except Exception as e:
            print(f"PDF 다운로드 오류 ({filename}): {str(e)}")
            return None

    def recursive_citation_analysis(
        self,
        initial_pdf_paths: List[str],
        output_dir: str,
        depth: int = 2,
        max_papers_per_level: int = 10,
        max_total_papers: int = 50
    ) -> Tuple[str, str, List[str]]:
        """
        재귀적으로 citation chain을 분석하고 논문 수집

        Args:
            initial_pdf_paths: 시작 PDF 파일 경로 리스트
            output_dir: 출력 디렉토리
            depth: 재귀 깊이 (1 = 1차 인용만, 2 = 2차 인용까지)
            max_papers_per_level: 레벨당 최대 논문 수
            max_total_papers: 전체 최대 논문 수

        Returns:
            (마크다운 파일 경로, BibTeX 파일 경로, 다운로드된 PDF 경로 리스트)
        """
        print("\n" + "=" * 80)
        print(f"재귀적 Citation Chain 분석 시작")
        print(f"  - 초기 PDF: {len(initial_pdf_paths)}개")
        print(f"  - 재귀 깊이: {depth}")
        print(f"  - 레벨당 최대 논문 수: {max_papers_per_level}")
        print(f"  - 전체 최대 논문 수: {max_total_papers}")
        print("=" * 80)

        # PDF 저장 디렉토리
        pdf_output_dir = os.path.join(output_dir, "pdf")
        os.makedirs(pdf_output_dir, exist_ok=True)

        # 타임스탬프
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_path = os.path.join(output_dir, f"recursive_citation_{timestamp}.md")
        bib_path = os.path.join(output_dir, f"recursive_citation_{timestamp}.bib")

        # 처리 상태 추적
        processed_titles: Set[str] = set()  # 이미 처리한 논문 제목
        all_downloaded_pdfs: List[str] = list(initial_pdf_paths)
        all_papers_metadata: List[Dict] = []

        # 레벨별 처리할 PDF 큐
        current_level_pdfs = initial_pdf_paths

        # 마크다운 파일 초기화
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Recursive Citation Chain Analysis\n\n")
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"초기 PDF: {len(initial_pdf_paths)}개\n")
            f.write(f"재귀 깊이: {depth}\n")
            f.write(f"레벨당 최대 논문 수: {max_papers_per_level}\n\n")
            f.write("---\n\n")

        # BibTeX 파일 초기화
        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write(f"% Recursive Citation Chain Analysis\n")
            f.write(f"% Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"% Initial PDFs: {len(initial_pdf_paths)}\n")
            f.write(f"% Recursion depth: {depth}\n\n")

        # 레벨별 재귀 분석
        for current_depth in range(depth):
            print(f"\n{'=' * 80}")
            print(f"레벨 {current_depth + 1}/{depth} 분석 중...")
            print(f"{'=' * 80}")

            if not current_level_pdfs:
                print("더 이상 처리할 PDF가 없습니다.")
                break

            if len(all_downloaded_pdfs) >= max_total_papers:
                print(f"전체 최대 논문 수({max_total_papers})에 도달했습니다.")
                break

            next_level_pdfs = []

            # 마크다운에 레벨 헤더 추가
            with open(md_path, 'a', encoding='utf-8') as f:
                f.write(f"## 레벨 {current_depth + 1}\n\n")

            # 현재 레벨의 각 PDF 처리
            for idx, pdf_path in enumerate(current_level_pdfs, 1):
                if len(all_downloaded_pdfs) >= max_total_papers:
                    break

                pdf_name = os.path.basename(pdf_path)
                print(f"\n[{idx}/{len(current_level_pdfs)}] 분석: {pdf_name}")

                # 마크다운에 PDF 정보 추가
                with open(md_path, 'a', encoding='utf-8') as f:
                    f.write(f"### {idx}. {pdf_name}\n\n")

                # 인용 정보 추출
                citations = self.extract_citations_from_pdf(pdf_path)

                if not citations:
                    with open(md_path, 'a', encoding='utf-8') as f:
                        f.write("인용 정보를 추출할 수 없습니다.\n\n")
                    continue

                # 논문 제목 추출
                titles = self.extract_paper_titles_from_citations(citations)
                print(f"추출된 제목 수: {len(titles)}")

                # 마크다운에 추출된 제목 개수 기록
                with open(md_path, 'a', encoding='utf-8') as f:
                    f.write(f"**추출된 인용 제목:** {len(titles)}개\n\n")

                # Arxiv에서 검색 및 다운로드
                papers_found = 0

                for title in titles[:max_papers_per_level]:
                    if len(all_downloaded_pdfs) >= max_total_papers:
                        break

                    # 이미 처리한 제목인지 확인
                    title_normalized = title.lower().strip()
                    if title_normalized in processed_titles:
                        continue

                    processed_titles.add(title_normalized)

                    # Arxiv 검색
                    print(f"  검색 중: {title[:60]}...")
                    arxiv_results = self.search_arxiv_by_title(title, max_results=1)

                    if not arxiv_results:
                        print(f"  → Arxiv에서 찾을 수 없음")
                        continue

                    paper = arxiv_results[0]  # 딕셔너리 형태
                    papers_found += 1

                    # PDF 다운로드
                    arxiv_id = paper['entry_id'].split('/')[-1].replace('.', '_')
                    filename = f"cited_{current_depth + 1}_{arxiv_id}.pdf"

                    downloaded_path = self.download_pdf_from_arxiv(paper, pdf_output_dir, filename)

                    if downloaded_path:
                        next_level_pdfs.append(downloaded_path)
                        all_downloaded_pdfs.append(downloaded_path)

                        # 저자 정보 처리
                        authors = paper.get('authors', [])
                        if isinstance(authors, list):
                            authors_str = ', '.join(authors) if authors else 'Unknown'
                        else:
                            authors_str = str(authors)

                        # 연도 추출 (Published 필드에서)
                        published = paper.get('published', '')
                        year = published.split('-')[0] if published else 'N/A'

                        # 메타데이터 저장
                        paper_metadata = {
                            'title': paper.get('title', 'Unknown'),
                            'authors': authors_str,
                            'year': year,
                            'arxiv_id': paper['entry_id'].split('/')[-1],
                            'url': paper['entry_id'],
                            'pdf_path': downloaded_path,
                            'cited_by': pdf_name,
                            'depth': current_depth + 1
                        }
                        all_papers_metadata.append(paper_metadata)

                        # 저자 목록 (처음 3명)
                        authors_list = authors if isinstance(authors, list) else [authors]
                        authors_display = ', '.join(authors_list[:3])
                        if len(authors_list) > 3:
                            authors_display += '...'

                        # 마크다운에 논문 정보 추가
                        with open(md_path, 'a', encoding='utf-8') as f:
                            f.write(f"- **{paper.get('title', 'Unknown')}**\n")
                            f.write(f"  - 저자: {authors_display}\n")
                            f.write(f"  - 연도: {year}\n")
                            f.write(f"  - Arxiv ID: {paper['entry_id'].split('/')[-1]}\n")
                            f.write(f"  - PDF: {filename}\n\n")

                        # BibTeX에 추가
                        with open(bib_path, 'a', encoding='utf-8') as f:
                            bib_key = f"cited-{current_depth + 1}-{arxiv_id.replace('.', '-')}"
                            f.write(f"@article{{{bib_key},\n")
                            f.write(f"  title = {{{paper.get('title', 'Unknown')}}},\n")
                            f.write(f"  author = {{{authors_str}}},\n")
                            f.write(f"  year = {{{year}}},\n")
                            f.write(f"  eprint = {{{paper['entry_id'].split('/')[-1]}}},\n")
                            f.write(f"  archivePrefix = {{arXiv}},\n")
                            f.write(f"  url = {{{paper['entry_id']}}},\n")
                            f.write(f"  note = {{Cited by: {pdf_name}, Depth: {current_depth + 1}}}\n")
                            f.write("}\n\n")

                print(f"  → Arxiv에서 찾은 논문: {papers_found}개")

                # 마크다운에 발견된 논문 개수 기록
                with open(md_path, 'a', encoding='utf-8') as f:
                    f.write(f"**Arxiv에서 찾은 논문:** {papers_found}개\n\n")
                    f.write("---\n\n")

            # 다음 레벨 준비
            current_level_pdfs = next_level_pdfs[:max_papers_per_level]
            print(f"\n레벨 {current_depth + 1} 완료: {len(next_level_pdfs)}개의 새 논문 발견")

        # 최종 요약
        with open(md_path, 'a', encoding='utf-8') as f:
            f.write("\n## 최종 요약\n\n")
            f.write(f"- 총 수집된 논문 수: {len(all_papers_metadata)}개\n")
            f.write(f"- 전체 PDF 파일 수: {len(all_downloaded_pdfs)}개\n")
            f.write(f"- 분석 완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 레벨별 통계
            f.write("### 레벨별 논문 수\n\n")
            for d in range(1, depth + 1):
                count = len([p for p in all_papers_metadata if p['depth'] == d])
                f.write(f"- 레벨 {d}: {count}개\n")

        print("\n" + "=" * 80)
        print("재귀적 Citation Chain 분석 완료")
        print(f"  - 총 수집 논문: {len(all_papers_metadata)}개")
        print(f"  - 마크다운: {md_path}")
        print(f"  - BibTeX: {bib_path}")
        print(f"  - PDF 디렉토리: {pdf_output_dir}")
        print("=" * 80)

        return md_path, bib_path, all_downloaded_pdfs
