"""
Output Writer module
에이전트 결과를 마크다운 및 BibTeX 파일로 저장
PDF 다운로드 및 citation chain 분석
"""

import os
import re
from datetime import datetime
from typing import Dict, List
import arxiv


class MarkdownWriter:
    """마크다운 형식으로 결과를 저장하는 클래스"""

    def __init__(self, config: Dict):
        """
        초기화

        Args:
            config: 출력 설정 딕셔너리
        """
        self.config = config
        self.output_dir = config['output']['directory']
        self.timestamp_format = config['output']['timestamp_format']
        self.include_metadata = config['output']['include_metadata']

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

    def write(self, query: str, results: List[Dict], metadata: Dict = None) -> str:
        """
        결과를 마크다운 파일로 저장

        Args:
            query: 사용자 질문
            results: 에이전트 실행 결과
            metadata: 추가 메타데이터

        Returns:
            생성된 파일 경로
        """
        timestamp = datetime.now().strftime(self.timestamp_format)
        filename = f"result_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            # 제목
            f.write("# Research Agent Result\n\n")

            # 메타데이터
            if self.include_metadata and metadata:
                f.write("## Metadata\n\n")
                for key, value in metadata.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")

            # 질문
            f.write("## Query\n\n")
            f.write(f"{query}\n\n")

            # 구분선
            f.write("---\n\n")

            # 결과
            f.write("## Results\n\n")

            for idx, result in enumerate(results, 1):
                result_type = result.get('type', 'Unknown')
                content = result.get('content', '')

                f.write(f"### Step {idx}: {result_type}\n\n")
                # 개행 문제 해결: 내용에 이미 개행이 있으면 그대로, 없으면 추가
                content_with_newlines = content.replace('\n', '\n\n') if '\n' in content else content
                f.write(f"{content_with_newlines}\n\n")

            # 푸터
            f.write("---\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"결과 저장 완료: {filepath}")
        return filepath

    def write_simple(self, query: str, response: str, metadata: Dict = None) -> str:
        """
        간단한 형식으로 저장 (최종 응답만)

        Args:
            query: 사용자 질문
            response: 최종 응답
            metadata: 추가 메타데이터

        Returns:
            생성된 파일 경로
        """
        timestamp = datetime.now().strftime(self.timestamp_format)
        filename = f"result_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            # 제목
            f.write("# Research Agent Result\n\n")

            # 메타데이터
            if self.include_metadata and metadata:
                f.write("## Metadata\n\n")
                for key, value in metadata.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")

            # 질문
            f.write("## Query\n\n")
            f.write(f"{query}\n\n")

            # 구분선
            f.write("---\n\n")

            # 응답
            f.write("## Response\n\n")
            # 개행 문제 해결
            response_with_newlines = response.replace('\n', '\n\n') if '\n' in response else response
            f.write(f"{response_with_newlines}\n\n")

            # 푸터
            f.write("---\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"결과 저장 완료: {filepath}")
        return filepath

class BibTeXWriter:
    """BibTeX 형식으로 논문 정보를 저장하는 클래스"""

    def __init__(self, config: Dict):
        """
        초기화

        Args:
            config: 출력 설정 딕셔너리
        """
        self.config = config
        self.output_dir = config['output']['directory']
        self.timestamp_format = config['output']['timestamp_format']

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_papers_from_text(self, text: str) -> List[Dict]:
        """
        텍스트에서 논문 정보를 추출합니다.

        Args:
            text: 논문 정보가 포함된 텍스트

        Returns:
            논문 정보 딕셔너리 리스트
        """
        papers = []

        # 논문 구분자로 분리
        sections = text.split('---')

        for section in sections:
            if '제목:' not in section:
                continue

            paper = {}

            # 제목 추출
            title_match = re.search(r'제목:\s*(.+?)(?:\n|$)', section)
            if title_match:
                paper['title'] = title_match.group(1).strip()

            # 저자 추출
            author_match = re.search(r'저자:\s*(.+?)(?:\n|$)', section)
            if author_match:
                paper['authors'] = author_match.group(1).strip()

            # 게시일 추출
            date_match = re.search(r'게시일:\s*(\d{4}-\d{2}-\d{2})', section)
            if date_match:
                paper['year'] = date_match.group(1).split('-')[0]
                paper['date'] = date_match.group(1)

            # 링크 추출 (arxiv ID)
            link_match = re.search(r'링크:\s*(https?://arxiv\.org/abs/(.+?)(?:\n|$))', section)
            if link_match:
                paper['url'] = link_match.group(1).strip()
                paper['arxiv_id'] = link_match.group(2).strip()

            # 요약 추출
            summary_match = re.search(r'요약:\s*(.+?)(?:\.\.\.|링크:|$)', section, re.DOTALL)
            if summary_match:
                paper['abstract'] = summary_match.group(1).strip()

            if paper.get('title') and paper.get('arxiv_id'):
                papers.append(paper)

        return papers

    def generate_bibtex_key(self, paper: Dict, keyword: str = "", index: int = 1) -> str:
        """
        BibTeX 키를 생성합니다 (키워드-숫자 형식).

        Args:
            paper: 논문 정보 딕셔너리
            keyword: 검색 키워드
            index: 논문 번호

        Returns:
            BibTeX 키 문자열 (예: lowresource-1, nlp-2)
        """
        if keyword:
            # 키워드를 소문자로 변환하고 공백을 제거
            clean_keyword = keyword.lower().replace(' ', '').replace('-', '')
            # 영문자와 숫자만 남기기
            clean_keyword = re.sub(r'[^a-z0-9]', '', clean_keyword)
            # 너무 길면 앞 15자만 사용
            clean_keyword = clean_keyword[:15]
            return f"{clean_keyword}-{index}"
        else:
            # 키워드가 없으면 arxiv ID 기반
            arxiv_id = paper.get('arxiv_id', '0000')
            arxiv_suffix = arxiv_id.replace('.', '').replace('/', '')[-6:]
            return f"arxiv-{arxiv_suffix}"

    def write(self, results: List[Dict], query: str = "") -> str:
        """
        결과에서 논문 정보를 추출하여 BibTeX 파일로 저장

        Args:
            results: 에이전트 실행 결과
            query: 검색 쿼리 (선택사항)

        Returns:
            생성된 파일 경로
        """
        timestamp = datetime.now().strftime(self.timestamp_format)
        filename = f"papers_{timestamp}.bib"
        filepath = os.path.join(self.output_dir, filename)

        # 모든 결과에서 텍스트 추출
        full_text = ""
        for result in results:
            content = result.get('content', '')
            full_text += content + "\n"

        # 논문 정보 추출
        papers = self.extract_papers_from_text(full_text)

        if not papers:
            print("추출된 논문이 없습니다.")
            return None

        # BibTeX 파일 작성
        with open(filepath, 'w', encoding='utf-8') as f:
            if query:
                f.write(f"% Generated from query: {query}\n")
            f.write(f"% Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"% Total papers: {len(papers)}\n\n")

            # 검색어에서 키워드 추출
            keyword = query.split()[0] if query else ""

            for idx, paper in enumerate(papers, 1):
                bibtex_key = self.generate_bibtex_key(paper, keyword, idx)

                f.write(f"@article{{{bibtex_key},\n")
                f.write(f"  title = {{{paper.get('title', 'Untitled')}}},\n")
                f.write(f"  author = {{{paper.get('authors', 'Unknown')}}},\n")
                f.write(f"  year = {{{paper.get('year', datetime.now().year)}}},\n")

                if paper.get('abstract'):
                    # 중괄호로 감싸서 BibTeX 형식 유지
                    abstract = paper['abstract'].replace('{', '\\{').replace('}', '\\}')
                    f.write(f"  abstract = {{{abstract}}},\n")

                f.write(f"  eprint = {{{paper.get('arxiv_id', '')}}},\n")
                f.write(f"  archivePrefix = {{arXiv}},\n")

                if paper.get('url'):
                    f.write(f"  url = {{{paper.get('url')}}},\n")

                if paper.get('date'):
                    f.write(f"  note = {{Published: {paper.get('date')}}}\n")

                f.write("}\n\n")

        print(f"BibTeX 파일 저장 완료: {filepath} ({len(papers)}개 논문)")
        return filepath

    def write_from_response(self, response: str, query: str = "") -> str:
        """
        최종 응답 문자열에서 BibTeX 파일 생성

        Args:
            response: 최종 응답 문자열
            query: 검색 쿼리

        Returns:
            생성된 파일 경로
        """
        results = [{"content": response}]
        return self.write(results, query)


class PDFDownloader:
    """Arxiv PDF 다운로드 클래스"""

    def __init__(self, config: Dict):
        """
        초기화

        Args:
            config: 출력 설정 딕셔너리
        """
        self.config = config
        self.output_dir = config['output']['directory']
        self.pdf_dir = os.path.join(self.output_dir, 'pdf')

        # PDF 디렉토리 생성
        os.makedirs(self.pdf_dir, exist_ok=True)

    def download_paper(self, arxiv_id: str, filename: str = None) -> str:
        """
        단일 논문 PDF 다운로드

        Args:
            arxiv_id: Arxiv ID (예: 2411.1234)
            filename: 저장할 파일명 (기본값: arxiv_id.pdf)

        Returns:
            다운로드된 PDF 파일 경로
        """
        try:
            # Arxiv client 생성
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])

            paper = next(client.results(search))

            # 파일명 설정
            if not filename:
                filename = f"{arxiv_id.replace('/', '_').replace('.', '_')}.pdf"

            filepath = os.path.join(self.pdf_dir, filename)

            # PDF 다운로드
            paper.download_pdf(dirpath=self.pdf_dir, filename=filename)

            print(f"PDF 다운로드 완료: {filepath}")
            return filepath

        except Exception as e:
            print(f"PDF 다운로드 실패 ({arxiv_id}): {str(e)}")
            return None

    def download_from_results(self, results: List[Dict], keyword: str = "") -> List[str]:
        """
        에이전트 결과에서 논문 ID를 추출하여 PDF 다운로드

        Args:
            results: 에이전트 실행 결과
            keyword: 파일명에 사용할 키워드

        Returns:
            다운로드된 PDF 파일 경로 리스트
        """
        # 모든 결과에서 텍스트 추출
        full_text = ""
        for result in results:
            content = result.get('content', '')
            full_text += content + "\n"

        # Arxiv ID 추출
        arxiv_pattern = r'https?://arxiv\.org/abs/([^\s\)]+)'
        arxiv_ids = re.findall(arxiv_pattern, full_text)

        # 중복 제거
        arxiv_ids = list(set(arxiv_ids))

        print(f"\n총 {len(arxiv_ids)}개 논문 PDF 다운로드 시작...")

        downloaded_files = []
        clean_keyword = keyword.lower().replace(' ', '_')[:15] if keyword else "paper"

        for idx, arxiv_id in enumerate(arxiv_ids, 1):
            filename = f"{clean_keyword}-{idx}.pdf"
            filepath = self.download_paper(arxiv_id, filename)
            if filepath:
                downloaded_files.append(filepath)

        print(f"\n다운로드 완료: {len(downloaded_files)}/{len(arxiv_ids)}개")
        return downloaded_files
