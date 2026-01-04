"""스크린샷 분석기 메인 그래프 구현.

그래프 구조:
    START → initialize → classification_phase → insight_phase → final_report → END

각 Phase는 Agentic 서브그래프로 구현되어 supervisor ↔ tools 반복 구조를 가짐.
"""

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from screenshot_analyzer.configuration import Configuration
from screenshot_analyzer.state import (
    ClassificationOutputState,
    ClassificationState,
    InputState,
    InsightOutputState,
    InsightState,
    ScreenshotAnalyzerState,
)

# ============================================================
# 설정 가능한 모델 초기화
# ============================================================

configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


# ============================================================
# 메인 그래프 노드들
# ============================================================

async def initialize(state: InputState, config: RunnableConfig) -> dict:
    """그래프 초기화 노드.
    
    입력 State를 받아 메인 State로 변환하고 초기값을 설정합니다.
    
    Args:
        state: 외부에서 전달받은 입력 (images, existing_categories)
        config: 런타임 설정
        
    Returns:
        초기화된 State 필드들
    """
    images = state.get("images", [])
    existing_categories = state.get("existing_categories", None)
    
    return {
        "images": images,
        "existing_categories": existing_categories,
        "vision_results": {},
        "classifications": {},
        "categories": existing_categories or [],
        "category_insights": {},
        "final_report": "",
    }


async def generate_final_report(state: ScreenshotAnalyzerState, config: RunnableConfig) -> dict:
    """최종 보고서 생성 노드.
    
    Phase 1, 2의 결과를 종합하여 최종 보고서를 생성합니다.
    
    TODO: Step 5-4에서 구현 예정
    """
    # 임시 구현 - Step 5-4에서 완성
    return {
        "final_report": "보고서 생성 예정..."
    }


# ============================================================
# 서브그래프 Placeholder (Step 5-2, 5-3에서 구현)
# ============================================================

# Phase 1: Classification 서브그래프
# TODO: Step 5-2에서 구현
async def classification_phase_placeholder(state: ScreenshotAnalyzerState, config: RunnableConfig) -> dict:
    """Classification Phase placeholder.
    
    Step 5-2에서 실제 서브그래프로 교체됩니다.
    """
    return {
        "vision_results": {},
        "classifications": {},
        "categories": [],
    }


# Phase 2: Insight 서브그래프
# TODO: Step 5-3에서 구현
async def insight_phase_placeholder(state: ScreenshotAnalyzerState, config: RunnableConfig) -> dict:
    """Insight Phase placeholder.
    
    Step 5-3에서 실제 서브그래프로 교체됩니다.
    """
    return {
        "category_insights": {},
    }


# ============================================================
# 메인 그래프 구성
# ============================================================

def create_graph():
    """메인 그래프를 생성합니다.
    
    Returns:
        컴파일된 StateGraph
    """
    # 그래프 빌더 생성
    builder = StateGraph(
        ScreenshotAnalyzerState,
        input=InputState,
        config_schema=Configuration,
    )
    
    # 노드 추가
    builder.add_node("initialize", initialize)
    builder.add_node("classification_phase", classification_phase_placeholder)  # Step 5-2에서 교체
    builder.add_node("insight_phase", insight_phase_placeholder)  # Step 5-3에서 교체
    builder.add_node("final_report", generate_final_report)
    
    # 엣지 연결 (순차 실행)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "classification_phase")
    builder.add_edge("classification_phase", "insight_phase")
    builder.add_edge("insight_phase", "final_report")
    builder.add_edge("final_report", END)
    
    return builder.compile()


# 그래프 인스턴스 생성 (langgraph.json에서 참조)
graph = create_graph()
