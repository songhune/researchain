"""
Perplexity sonar-pro의 tool calling 지원 여부 테스트
"""

from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# 환경 변수 로드
load_dotenv()

# 간단한 테스트 도구 정의
@tool
def get_weather(city: str) -> str:
    """
    특정 도시의 날씨를 가져옵니다.

    Args:
        city: 도시 이름

    Returns:
        날씨 정보
    """
    return f"{city}의 날씨: 맑음, 기온 20도"


def test_tool_calling():
    """sonar-pro의 tool calling 테스트"""

    print("=" * 80)
    print("Perplexity sonar-pro Tool Calling 테스트")
    print("=" * 80)
    print()

    # 1. ChatPerplexity 초기화
    print("1. ChatPerplexity 모델 초기화 중...")
    try:
        llm = ChatPerplexity(
            model="sonar-pro",
            temperature=0.3,
            request_timeout=30,
            max_tokens=500
        )
        print("   ✓ 모델 초기화 성공")
    except Exception as e:
        print(f"   ✗ 모델 초기화 실패: {e}")
        return

    print()

    # 2. bind_tools 메서드 존재 여부 확인
    print("2. bind_tools 메서드 확인 중...")
    if hasattr(llm, 'bind_tools'):
        print("   ✓ bind_tools 메서드 존재함")
    else:
        print("   ✗ bind_tools 메서드가 없음")
        return

    print()

    # 3. bind_tools 호출 테스트
    print("3. bind_tools 호출 테스트 중...")
    try:
        llm_with_tools = llm.bind_tools([get_weather])
        print("   ✓ bind_tools 호출 성공")
    except Exception as e:
        print(f"   ✗ bind_tools 호출 실패: {e}")
        import traceback
        print("   상세 에러:")
        traceback.print_exc()
        return

    print()

    # 4. LangGraph Agent 생성 테스트
    print("4. LangGraph create_react_agent 테스트 중...")
    try:
        agent = create_react_agent(llm, [get_weather])
        print("   ✓ Agent 생성 성공")
    except NotImplementedError as e:
        print(f"   ✗ Agent 생성 실패 (NotImplementedError): {e}")
        print("   → sonar-pro는 tool calling을 지원하지 않습니다")
        return
    except Exception as e:
        print(f"   ✗ Agent 생성 실패: {e}")
        return

    print()

    # 5. 실제 tool calling 테스트
    print("5. 실제 tool calling 실행 테스트 중...")
    print("   질문: 'What is the weather in Seoul?'")
    print()

    try:
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="What is the weather in Seoul?")]

        # Agent 실행
        result = agent.invoke({"messages": messages})

        print("   ✓ Agent 실행 성공!")
        print()
        print("   결과:")
        print("   " + "-" * 76)

        if "messages" in result:
            for msg in result["messages"]:
                msg_type = msg.__class__.__name__
                print(f"   [{msg_type}]")

                # Tool call 확인
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"   Tool Calls: {msg.tool_calls}")

                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content)[:200]
                    print(f"   {content}")
                print()

        print("   " + "-" * 76)
        print()
        print("=" * 80)
        print("결론: sonar-pro는 tool calling을 지원합니다! ✓")
        print("=" * 80)

    except Exception as e:
        print(f"   ✗ Agent 실행 실패: {e}")
        print()
        print("=" * 80)
        print("결론: sonar-pro는 tool calling을 지원하지 않습니다 ✗")
        print("=" * 80)


if __name__ == "__main__":
    test_tool_calling()
