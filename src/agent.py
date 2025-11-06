"""
Agent module for LangChain
에이전트 생성 및 실행 로직
"""

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from typing import Dict, List, Any


class ResearchAgent:
    """학술 논문 검색을 위한 에이전트 클래스"""

    def __init__(self, config: Dict, tools: List[Any]):
        """
        에이전트 초기화

        Args:
            config: 설정 딕셔너리
            tools: 사용할 도구 리스트
        """
        self.config = config
        self.tools = tools
        self.llm = None
        self.agent = None

        self._initialize_llm()
        self._initialize_agent()

    def _initialize_llm(self):
        """LLM 모델 초기화"""
        llm_config = self.config['llm']
        model_provider = llm_config['model_provider'].lower()

        # Perplexity는 init_chat_model이 API 키를 제대로 인식하지 못하므로 직접 초기화
        if model_provider == 'perplexity':
            from langchain_perplexity import ChatPerplexity

            # PPLX_API_KEY 환경 변수에서 자동 로드
            self.llm = ChatPerplexity(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                request_timeout=llm_config['timeout'],
                max_tokens=llm_config['max_tokens']
            )
        else:
            self.llm = init_chat_model(
                model=llm_config['model'],
                model_provider=llm_config['model_provider'],
                temperature=llm_config['temperature'],
                timeout=llm_config['timeout'],
                max_tokens=llm_config['max_tokens']
            )
        print("LLM 초기화 완료")

    def _initialize_agent(self):
        """에이전트 초기화"""
        self.system_prompt = self._build_system_prompt()

        # 모델 provider 확인
        model_provider = self.config['llm'].get('model_provider', '').lower()

        # Perplexity 등 tool calling 미지원 모델 체크
        if model_provider in ['perplexity']:
            print(f"Tool calling 미지원 모델({model_provider}) 감지: 단순 LLM 모드로 실행")
            self.agent = None  # Agent 없이 직접 LLM 사용
            self.use_tools = False
        else:
            # LangGraph의 create_react_agent (tool calling 지원 모델)
            try:
                self.agent = create_react_agent(
                    self.llm,
                    self.tools
                )
                self.use_tools = True
                print("Agent 생성 완료 (tool calling 활성화)")
            except NotImplementedError:
                print("Tool calling 미지원: 단순 LLM 모드로 실행")
                self.agent = None
                self.use_tools = False

    def _build_system_prompt(self) -> str:
        """설정 파일로부터 시스템 프롬프트 구축"""
        sp_config = self.config['system_prompt']

        prompt = f"""당신은 {sp_config['role']}입니다. {sp_config['description']}

당신의 답변은 다음 기준을 충족해야 합니다:
1. 정확성: {sp_config['criteria']['accuracy']}
2. 주제: {sp_config['criteria']['focus']}

도구 사용 가이드:
"""
        for tool_name, guide in sp_config['tool_usage_guide'].items():
            prompt += f"- {tool_name}: {guide}\n"

        return prompt

    def run(self, query: str, stream_mode: str = "values") -> List[Dict]:
        """
        에이전트 실행

        Args:
            query: 사용자 질문
            stream_mode: 스트리밍 모드 (기본값: "values")

        Returns:
            실행 결과 리스트
        """
        results = []

        # Tool calling 미지원 모델인 경우 직접 LLM 호출
        if not self.use_tools or self.agent is None:
            # Arxiv 도구 먼저 실행
            print("[Tool calling 미지원] Arxiv 검색 실행 중...")
            tool_result = ""
            if self.tools:
                for tool in self.tools:
                    try:
                        # 쿼리에서 키워드 추출 (간단한 방식)
                        tool_result = tool.invoke(query)
                        print(f"[Tool 실행 완료] {len(tool_result)} chars")
                    except Exception as e:
                        print(f"[Tool 실행 오류] {str(e)}")
                        tool_result = ""

            # System prompt + Tool 결과 + 쿼리를 합쳐서 LLM에 전달
            full_prompt = f"""{self.system_prompt}

도구 실행 결과:
{tool_result}

사용자 질문: {query}

위 정보를 바탕으로 답변해주세요."""

            # Perplexity는 메시지 형식 필요
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=full_prompt)])

            # 결과 구성: 사용자 질문 추가
            results.append({
                "type": "HumanMessage",
                "content": query
            })

            # IMPORTANT: Arxiv 원본 데이터를 ToolMessage로 명시적 저장
            # BibTeX/PDF 추출기가 이 데이터를 우선적으로 사용함
            if tool_result:
                results.append({
                    "type": "ToolMessage",
                    "content": tool_result,
                    "source": "arxiv_raw"  # 원본 데이터임을 표시
                })

            # Perplexity의 종합 응답 추가
            results.append({
                "type": "AIMessage",
                "content": response.content
            })
            return results

        # Tool calling 지원 모델인 경우 Agent 사용
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]

        for chunk in self.agent.stream(
            {"messages": messages},
            stream_mode=stream_mode
        ):
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]
                if hasattr(last_message, "content"):
                    results.append({
                        "type": last_message.__class__.__name__,
                        "content": last_message.content
                    })

        return results

    def run_sync(self, query: str) -> str:
        """
        에이전트 동기 실행 (최종 결과만 반환)

        Args:
            query: 사용자 질문

        Returns:
            최종 응답 문자열
        """
        results = self.run(query)
        if results:
            return results[-1]["content"]
        return ""
