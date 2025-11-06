"""
Agent module for LangChain
에이전트 생성 및 실행 로직
"""

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
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
        system_prompt = self._build_system_prompt()

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
        print("Agent 생성 완료")

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

        for chunk in self.agent.stream(
            {"messages": [("user", query)]},
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
