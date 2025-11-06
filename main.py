"""
Main entry point for Research Agent
학술 논문 검색 에이전트 메인 실행 파일
"""

import os
import yaml
import argparse
from dotenv import load_dotenv
from src.tools import create_arxiv_tool
from src.agent import ResearchAgent
from src.output_writer import MarkdownWriter, BibTeXWriter, PDFDownloader
from src.citation_analyzer import CitationChainAnalyzer


def load_config(config_path: str = "config/config.yaml") -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_env():
    """환경 변수 로드 및 확인"""
    load_dotenv()

    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY")
    }

    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"[OK] {key_name} 로드 완료")
        else:
            print(f"[WARN] {key_name}를 찾을 수 없습니다")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Research Agent - 학술 논문 검색 에이전트")
    parser.add_argument("-q", "--query", type=str, help="검색 질문")
    parser.add_argument("-s", "--sample", type=int, choices=[1, 2, 3],
                        help="샘플 쿼리 선택 (1: 저자원 언어, 2: 한문 NLP, 3: 벤치마크 방법론)")
    parser.add_argument("-c", "--config", type=str, default="config/config.yaml",
                        help="설정 파일 경로 (기본값: config/config.yaml)")
    parser.add_argument("-o", "--output", action="store_true",
                        help="결과를 마크다운 파일로 저장")
    parser.add_argument("-b", "--bibtex", action="store_true",
                        help="논문 정보를 BibTeX 파일로 저장")
    parser.add_argument("-p", "--pdf", action="store_true",
                        help="논문 PDF 다운로드")
    parser.add_argument("--cite", type=str,
                        help="PDF 파일의 citation chain 분석 (파일 경로 또는 디렉토리)")
    parser.add_argument("--cite-chain", type=str,
                        help="재귀적 citation chain 분석 (파일 경로 또는 디렉토리)")
    parser.add_argument("--depth", type=int, default=2,
                        help="재귀 깊이 (기본값: 2)")
    parser.add_argument("--max-papers-per-level", type=int, default=10,
                        help="레벨당 최대 논문 수 (기본값: 10)")
    parser.add_argument("--max-total-papers", type=int, default=50,
                        help="전체 최대 논문 수 (기본값: 50)")

    args = parser.parse_args()

    # 재귀적 Citation Chain 분석 모드 (별도 실행)
    if args.cite_chain:
        # 환경 변수 및 설정 로드
        load_dotenv()
        config = load_config(args.config)

        analyzer = CitationChainAnalyzer(config)

        # PDF 파일 목록 수집
        pdf_files = []
        if os.path.isfile(args.cite_chain):
            pdf_files = [args.cite_chain]
        elif os.path.isdir(args.cite_chain):
            pdf_files = [os.path.join(args.cite_chain, f) for f in os.listdir(args.cite_chain) if f.endswith('.pdf')]
        else:
            print("유효한 파일 또는 디렉토리 경로가 아닙니다.")
            return

        if not pdf_files:
            print("PDF 파일을 찾을 수 없습니다.")
            return

        print(f"재귀적 Citation Chain 분석 시작: {len(pdf_files)}개 PDF")
        md_file, bib_file, all_pdfs = analyzer.recursive_citation_analysis(
            initial_pdf_paths=pdf_files,
            output_dir=config['output']['directory'],
            depth=args.depth,
            max_papers_per_level=args.max_papers_per_level,
            max_total_papers=args.max_total_papers
        )

        print(f"\n분석 결과:")
        print(f"  - 마크다운: {md_file}")
        print(f"  - BibTeX: {bib_file}")
        print(f"  - 총 PDF 수: {len(all_pdfs)}개")

        return

    # Citation 분석 모드 (별도 실행)
    if args.cite:
        # 환경 변수 및 설정 로드
        load_dotenv()
        config = load_config(args.config)

        analyzer = CitationChainAnalyzer(config)

        if os.path.isfile(args.cite):
            # 단일 PDF 분석
            print(f"Citation Chain 분석: {args.cite}")
            result = analyzer.analyze_citation_chain(args.cite)
            print(result)
        elif os.path.isdir(args.cite):
            # 디렉토리 내 모든 PDF 분석
            pdf_files = [os.path.join(args.cite, f) for f in os.listdir(args.cite) if f.endswith('.pdf')]
            if pdf_files:
                print(f"{len(pdf_files)}개 PDF 파일 일괄 분석")
                md_file, bib_file = analyzer.batch_analyze(pdf_files, config['output']['directory'])
                print(f"\n분석 결과 저장:")
                print(f"  - 마크다운: {md_file}")
                print(f"  - BibTeX: {bib_file}")
            else:
                print("PDF 파일을 찾을 수 없습니다.")
        else:
            print("유효한 파일 또는 디렉토리 경로가 아닙니다.")

        return

    # 환경 변수 로드
    print("=" * 80)
    print("환경 변수 로드 중...")
    load_env()
    print()

    # 설정 파일 로드
    print("설정 파일 로드 중...")
    config = load_config(args.config)
    print(f"[OK] 설정 파일 로드 완료: {args.config}")
    print()

    # 도구 생성
    print("도구 초기화 중...")
    arxiv_tool = create_arxiv_tool(config['arxiv'])
    print("[OK] Arxiv 도구 생성 완료")
    print()

    # 에이전트 생성
    print("에이전트 초기화 중...")
    agent = ResearchAgent(config, tools=[arxiv_tool])
    print()

    # 질문 결정
    if args.sample:
        sample_queries = config['sample_queries']
        query = sample_queries[args.sample - 1]['query']
        query_name = sample_queries[args.sample - 1]['name']
        print(f"샘플 쿼리 사용: {query_name}")
    elif args.query:
        query = args.query
    else:
        print("질문을 입력하세요 (또는 -s 옵션으로 샘플 쿼리 선택):")
        query = input("> ")

    # 실행
    print("=" * 80)
    print(f"질문: {query}")
    print("=" * 80)
    print()

    print("에이전트 실행 중...")
    print()

    if args.output or args.bibtex or args.pdf:
        # 상세 결과 저장
        results = agent.run(query)

        # 결과 출력
        for result in results:
            print(f"[{result['type']}]")
            print(result['content'])
            print()

        # 메타데이터
        metadata = {
            "model": config['llm']['model'],
            "temperature": config['llm']['temperature'],
            "tools": "get_source"
        }

        # 마크다운 저장
        if args.output:
            writer = MarkdownWriter(config)
            md_filepath = writer.write(query, results, metadata)
            print()
            print(f"마크다운 파일 저장: {md_filepath}")

        # BibTeX 저장
        if args.bibtex:
            bib_writer = BibTeXWriter(config)
            bib_filepath = bib_writer.write(results, query)
            if bib_filepath:
                print(f"BibTeX 파일 저장: {bib_filepath}")

        # PDF 다운로드
        if args.pdf:
            pdf_downloader = PDFDownloader(config)
            keyword = query.split()[0] if query else ""
            pdf_files = pdf_downloader.download_from_results(results, keyword)

            # Citation chain 분석 옵션
            if pdf_files and config['output'].get('citation_analysis', False):
                print("\nCitation Chain 자동 분석 시작...")
                analyzer = CitationChainAnalyzer(config)
                md_file, bib_file = analyzer.batch_analyze(pdf_files, config['output']['directory'])
                print(f"\nCitation 분석 결과:")
                print(f"  - 마크다운: {md_file}")
                print(f"  - BibTeX: {bib_file}")
    else:
        # 최종 응답만 출력
        response = agent.run_sync(query)
        print(response)
        print()

    print("=" * 80)
    print("완료")


if __name__ == "__main__":
    main()
