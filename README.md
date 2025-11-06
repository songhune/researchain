# Research Agent

학술 논문 검색을 위한 LangChain 기반 AI 에이전트

## 프로젝트 개요

KLSBench 연구를 위한 학술 논문 검색 에이전트입니다. Arxiv API를 활용하여 저자원 언어 처리, 한문 NLP, 벤치마크 방법론 관련 연구를 검색하고 분석합니다.

## 주요 기능

- Arxiv 학술 논문 검색
- LangChain Agent 기반 자동 추론 및 도구 활용
- 마크다운 형식 결과 저장 (개행 문제 해결)
- BibTeX 형식 논문 정보 내보내기 (키워드-숫자 형식)
- PDF 자동 다운로드
- PDF Citation Chain 분석 (LLM 기반)
- YAML 기반 설정 관리
- 다양한 LLM 모델 지원 (OpenAI, Anthropic, Perplexity)

## 프로젝트 구조

```
LangChain/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 의존성 패키지
├── .env                    # API 키 (git에 포함되지 않음)
├── .gitignore             # Git 제외 파일 목록
├── README.md              # 프로젝트 문서
├── langchain.ipynb        # Jupyter 노트북 (개발/테스트용)
├── config/
│   └── config.yaml        # 설정 파일
├── src/
│   ├── tools.py              # Arxiv 검색 도구
│   ├── agent.py              # Agent 로직
│   ├── output_writer.py      # 마크다운/BibTeX/PDF 출력
│   └── citation_analyzer.py  # Citation chain 분석
└── output/                   # 결과 파일 저장 디렉토리
    └── pdf/                  # PDF 파일 저장
```

## 설치 방법

### 1. 저장소 클론

```bash
cd /Users/songhune/Workspace/LangChain
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일에 API 키 설정:

```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
PERPLEXITY_API_KEY=your-perplexity-api-key
```

API 키 발급:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/
- Perplexity: https://www.perplexity.ai/settings/api

## 사용 방법

### 기본 실행

```bash
python main.py -q "low-resource language benchmark"
```

### 샘플 쿼리 사용

```bash
# 샘플 1: 저자원 언어 벤치마크
python main.py -s 1

# 샘플 2: 한문 NLP
python main.py -s 2

# 샘플 3: 벤치마크 방법론
python main.py -s 3
```

### 결과를 마크다운으로 저장

```bash
python main.py -s 1 -o
```

### BibTeX 파일로 내보내기

```bash
python main.py -s 1 -b
```

### 마크다운과 BibTeX 동시 저장

```bash
python main.py -s 1 -o -b
```

### PDF 다운로드

```bash
python main.py -s 1 -p
```

### 모든 기능 사용 (마크다운 + BibTeX + PDF)

```bash
python main.py -s 1 -o -b -p
```

### Citation Chain 분석

```bash
# 단일 PDF 파일 분석
python main.py --cite output/pdf/lowresource-1.pdf

# 디렉토리 내 모든 PDF 분석
python main.py --cite output/pdf/
```

### 대화형 모드

```bash
python main.py
```

### 커스텀 설정 파일 사용

```bash
python main.py -c custom_config.yaml -q "transformer architecture"
```

## 설정 파일 (config.yaml)

### LLM 모델 설정

```yaml
llm:
  model: "gpt-4o"           # 모델명
  model_provider: "openai"  # 제공자
  temperature: 0.7          # 창의성 (0.0-1.0)
  timeout: 30               # 타임아웃 (초)
  max_tokens: 2000          # 최대 토큰
```

### Arxiv 검색 설정

```yaml
arxiv:
  page_size: 5              # 페이지당 결과 수
  delay_seconds: 3          # API 요청 간격 (초)
  num_retries: 3            # 재시도 횟수
  max_results: 5            # 최대 결과 수
  summary_max_chars: 500    # 요약 최대 길이
```

### 출력 설정

```yaml
output:
  directory: "output"                  # 출력 디렉토리
  format: "markdown"                   # 출력 형식
  timestamp_format: "%Y%m%d_%H%M%S"   # 파일명 타임스탬프
  include_metadata: true               # 메타데이터 포함 여부
```

## CLI 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `-q, --query` | 검색 질문 지정 | `-q "NLP benchmark"` |
| `-s, --sample` | 샘플 쿼리 선택 (1-3) | `-s 1` |
| `-c, --config` | 설정 파일 경로 | `-c custom.yaml` |
| `-o, --output` | 마크다운 파일로 저장 | `-o` |
| `-b, --bibtex` | BibTeX 파일로 저장 | `-b` |
| `-p, --pdf` | PDF 파일 다운로드 | `-p` |
| `--cite` | PDF Citation Chain 분석 | `--cite output/pdf/` |
| `-h, --help` | 도움말 표시 | `-h` |

## 개발 환경

### Jupyter 노트북 사용

```bash
jupyter notebook langchain.ipynb
```

노트북에서 셀을 순서대로 실행:
1. 환경 변수 로드
2. System Prompt 정의
3. LLM 초기화
4. Tools 정의
5. Agent 생성
6. 쿼리 실행

## 예제 출력

### 콘솔 출력
```
질문: 저자원 언어 처리를 위한 벤치마크 구축에 대한 최신 연구를 찾아줘
================================================================================

에이전트 실행 중...

[AIMessage]
Arxiv에서 관련 논문을 검색하겠습니다...

[ToolMessage]
제목: Low-Resource Language Benchmarking...
저자: John Doe, Jane Smith
게시일: 2024-11-15
...
```

### 마크다운 파일 (output/result_20250106_143022.md)
```markdown
# Research Agent Result

## Metadata
- model: gpt-4o
- temperature: 0.7
- tools: get_source

## Query
저자원 언어 처리를 위한 벤치마크 구축에 대한 최신 연구를 찾아줘

---

## Results

### Step 1: AIMessage
Arxiv에서 관련 논문을 검색하겠습니다...

### Step 2: ToolMessage
제목: Low-Resource Language Benchmarking...
...
```

### BibTeX 파일 (output/papers_20250106_143022.bib)
```bibtex
% Generated from query: 저자원 언어 처리를 위한 벤치마크 구축에 대한 최신 연구를 찾아줘
% Generated at: 2025-01-06 14:30:22
% Total papers: 5

@article{lowresource-1,
  title = {Low-Resource Language Benchmarking: A Comprehensive Study},
  author = {John Doe, Jane Smith},
  year = {2024},
  abstract = {This paper presents a comprehensive benchmark...},
  eprint = {2411.1234},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2411.1234},
  note = {Published: 2024-11-15}
}

@article{lowresource-2,
  title = {Benchmark Construction for Low-Resource Languages},
  ...
}
```

### PDF 다운로드 결과
```
output/pdf/
├── lowresource-1.pdf
├── lowresource-2.pdf
├── lowresource-3.pdf
├── lowresource-4.pdf
└── lowresource-5.pdf
```

### Citation Chain 분석 결과

#### 마크다운 (output/citation_analysis_YYYYMMDD_HHMMSS.md)

```markdown
# Citation Chain Analysis

## 1. lowresource-1.pdf

### 주요 인용 논문
1. Smith et al. (2023) - "Multilingual Benchmarking Framework"
2. Lee et al. (2022) - "Cross-lingual Transfer Learning"
...

### 인용 논문 주제 분류
- 저자원 언어 처리: 15개
- 벤치마크 방법론: 8개
- 전이 학습: 6개
...

**추출된 인용 수:** 50개
```

#### BibTeX (output/citation_analysis_YYYYMMDD_HHMMSS.bib)

```bibtex
% Citation Chain Analysis
% Generated at: 2025-01-06 14:30:22
% Total PDFs analyzed: 5

% Citations from: lowresource-1.pdf
% Total citations: 50

@misc{lowresourc-cite-1,
  author = {Smith and others},
  title = {Multilingual Benchmarking Framework},
  year = {2023},
  note = {Smith et al. (2023). "Multilingual Benchmarking Framework". In Proceedings of ACL 2023.},
  howpublished = {Cited in: lowresource-1.pdf}
}

@misc{lowresourc-cite-2,
  author = {Lee and others},
  title = {Cross-lingual Transfer Learning},
  year = {2022},
  note = {Lee et al. (2022). "Cross-lingual Transfer Learning". arXiv preprint.},
  howpublished = {Cited in: lowresource-1.pdf}
}
...
```

## 문제 해결

### ImportError: No module named 'langchain'
```bash
pip install -r requirements.txt
```

### API 키 에러
`.env` 파일에 올바른 API 키가 설정되어 있는지 확인

### Arxiv API 타임아웃
`config.yaml`에서 `delay_seconds` 값을 늘리기

## 기술 스택

- Python 3.8+
- LangChain 0.3+
- OpenAI API / Anthropic API
- Arxiv API (arxiv.py)
- PyYAML
- python-dotenv

## 라이선스

이 프로젝트는 연구 목적으로 사용됩니다.

## 참고 문서

- [LangChain 공식 문서](https://docs.langchain.com/)
- [OpenAI API 문서](https://platform.openai.com/docs)
- [Arxiv API 문서](https://info.arxiv.org/help/api/index.html)

## 기여

문제를 발견하거나 개선 사항이 있다면 이슈를 등록해주세요.

## 연락처

KLSBench 연구팀
