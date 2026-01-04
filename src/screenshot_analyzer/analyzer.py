"""스크린샷 분석기 메인 그래프 구현.

그래프 구조:
    START → initialize → classification_phase → insight_phase → final_report → END

각 Phase는 Agentic 서브그래프로 구현되어 supervisor ↔ tools 반복 구조를 가짐.
"""

import asyncio
import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from screenshot_analyzer.configuration import Configuration
from screenshot_analyzer.prompts import (
    CLASSIFICATION_HUMAN_PROMPT,
    CLASSIFICATION_PROMPT,
    CLASSIFICATION_SUPERVISOR_SYSTEM_PROMPT,
)
from screenshot_analyzer.state import (
    ClassificationComplete,
    ClassificationOutputState,
    ClassificationState,
    ConductClassification,
    ConductVisionAnalysis,
    InputState,
    InsightOutputState,
    InsightState,
    ScreenshotAnalyzerState,
)
from screenshot_analyzer.utils import (
    analyze_image,
    get_api_key_for_model,
    parse_json_response,
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
    """그래프 초기화 노드."""
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
    
    TODO: Step 5-4에서 구현 예정
    """
    return {
        "final_report": "보고서 생성 예정..."
    }


# ============================================================
# Phase 1: Classification 서브그래프
# ============================================================

async def classification_supervisor(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["classification_tools"]]:
    """Classification Phase의 Supervisor 노드.
    
    현재 상태를 분석하고 다음 작업을 결정합니다:
    - ConductVisionAnalysis: 이미지 분석 지시
    - ConductClassification: 분류 지시
    - ClassificationComplete: Phase 완료
    """
    configuration = Configuration.from_runnable_config(config)
    
    # 현재 상태 추출
    images = state.get("images", [])
    analyzed_images = state.get("analyzed_images", [])
    vision_results = state.get("vision_results", {})
    classifications = state.get("classifications", {})
    existing_categories = state.get("existing_categories", [])
    iteration_count = state.get("iteration_count", 0)
    
    # 미분석 이미지 계산
    pending_images = [img for img in images if img not in analyzed_images]
    
    # 시스템 프롬프트 구성
    system_prompt = CLASSIFICATION_SUPERVISOR_SYSTEM_PROMPT.format(
        total_images=len(images),
        analyzed_count=len(analyzed_images),
        pending_count=len(pending_images),
        iteration_count=iteration_count,
        max_iterations=configuration.max_analysis_iterations,
        vision_results_summary=json.dumps(vision_results, ensure_ascii=False, indent=2) if vision_results else "없음",
        existing_categories=", ".join(existing_categories) if existing_categories else "없음",
    )
    
    # Human 프롬프트 구성
    human_prompt = CLASSIFICATION_HUMAN_PROMPT.format(
        pending_images=", ".join(pending_images) if pending_images else "없음 (모두 분석 완료)",
        current_classifications=json.dumps(classifications, ensure_ascii=False, indent=2) if classifications else "없음",
    )
    
    # 모델 설정
    model_config = {
        "model": configuration.analysis_model,
        "max_tokens": configuration.max_tokens,
        "api_key": get_api_key_for_model(configuration.analysis_model, config),
    }
    
    # 도구 바인딩
    tools = [ConductVisionAnalysis, ConductClassification, ClassificationComplete]
    model_with_tools = configurable_model.bind_tools(tools).with_config(model_config)
    
    # 메시지 구성
    messages = state.get("messages", [])
    if not messages:
        messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=human_prompt))
    
    # LLM 호출
    response = await model_with_tools.ainvoke(messages)
    
    return Command(
        goto="classification_tools",
        update={
            "messages": [response],
            "iteration_count": iteration_count + 1,
        }
    )


async def classification_tools(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["classification_supervisor", "__end__"]]:
    """Classification Phase의 도구 실행 노드.
    
    Supervisor가 호출한 도구를 실행합니다:
    - ConductVisionAnalysis: Vision API로 이미지 분석
    - ConductClassification: LLM으로 분류 수행
    - ClassificationComplete: Phase 종료
    """
    configuration = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent_message = messages[-1] if messages else None
    
    # 도구 호출이 없으면 종료
    if not most_recent_message or not most_recent_message.tool_calls:
        return Command(
            goto=END,
            update={
                "vision_results": state.get("vision_results", {}),
                "classifications": state.get("classifications", {}),
                "categories": state.get("categories", []),
            }
        )
    
    # 반복 횟수 체크
    iteration_count = state.get("iteration_count", 0)
    if iteration_count > configuration.max_analysis_iterations:
        return Command(
            goto=END,
            update={
                "vision_results": state.get("vision_results", {}),
                "classifications": state.get("classifications", {}),
                "categories": state.get("categories", []),
            }
        )
    
    tool_messages = []
    update_payload = {}
    should_end = False
    
    for tool_call in most_recent_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "ConductVisionAnalysis":
            # Vision API로 이미지 분석
            targets = tool_args.get("targets", [])
            images = state.get("images", [])
            
            # "all"이면 미분석 이미지 전체
            if targets == ["all"] or "all" in targets:
                analyzed = state.get("analyzed_images", [])
                targets = [img for img in images if img not in analyzed]
            
            # 병렬로 이미지 분석
            if targets:
                tasks = [analyze_image(img, config) for img in targets]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                new_vision_results = {}
                new_analyzed = []
                for img, result in zip(targets, results):
                    if not isinstance(result, Exception):
                        new_vision_results[img] = result.model_dump()
                        new_analyzed.append(img)
                
                # State 업데이트
                update_payload["vision_results"] = new_vision_results
                update_payload["analyzed_images"] = state.get("analyzed_images", []) + new_analyzed
                
                tool_messages.append(ToolMessage(
                    content=f"Vision 분석 완료: {len(new_analyzed)}장 분석됨\n결과: {json.dumps(new_vision_results, ensure_ascii=False, indent=2)}",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
            else:
                tool_messages.append(ToolMessage(
                    content="분석할 이미지가 없습니다.",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
        
        elif tool_name == "ConductClassification":
            # LLM으로 분류 수행
            vision_results = {**state.get("vision_results", {}), **update_payload.get("vision_results", {})}
            existing_categories = state.get("existing_categories", [])
            
            if vision_results:
                # 분류 프롬프트 구성
                classification_prompt = CLASSIFICATION_PROMPT.format(
                    vision_results=json.dumps(vision_results, ensure_ascii=False, indent=2),
                    existing_categories=", ".join(existing_categories) if existing_categories else "없음",
                )
                
                # 분류 모델 호출
                model_config = {
                    "model": configuration.analysis_model,
                    "max_tokens": configuration.max_tokens,
                    "api_key": get_api_key_for_model(configuration.analysis_model, config),
                }
                classification_model = configurable_model.with_config(model_config)
                
                response = await classification_model.ainvoke([
                    HumanMessage(content=classification_prompt)
                ])
                
                # JSON 파싱
                result_dict = parse_json_response(response.content)
                classifications = result_dict.get("classifications", {})
                categories = result_dict.get("categories", [])
                
                update_payload["classifications"] = classifications
                update_payload["categories"] = categories
                
                tool_messages.append(ToolMessage(
                    content=f"분류 완료: {len(classifications)}장 분류됨\n카테고리: {categories}",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
            else:
                tool_messages.append(ToolMessage(
                    content="분류할 Vision 분석 결과가 없습니다. 먼저 Vision 분석을 수행하세요.",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
        
        elif tool_name == "ClassificationComplete":
            # Phase 완료
            should_end = True
            summary = tool_args.get("summary", "분류 완료")
            categories_found = tool_args.get("categories_found", update_payload.get("categories", []))
            
            tool_messages.append(ToolMessage(
                content=f"Classification Phase 완료: {summary}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
            update_payload["categories"] = categories_found
    
    # 메시지 업데이트
    update_payload["messages"] = tool_messages
    
    if should_end:
        # 최종 결과와 함께 종료
        return Command(
            goto=END,
            update={
                "vision_results": {**state.get("vision_results", {}), **update_payload.get("vision_results", {})},
                "classifications": {**state.get("classifications", {}), **update_payload.get("classifications", {})},
                "categories": update_payload.get("categories", state.get("categories", [])),
            }
        )
    
    return Command(
        goto="classification_supervisor",
        update=update_payload,
    )


# Classification 서브그래프 빌더
def create_classification_subgraph():
    """Classification Phase 서브그래프를 생성합니다."""
    builder = StateGraph(
        ClassificationState,
        output=ClassificationOutputState,
        config_schema=Configuration,
    )
    
    builder.add_node("classification_supervisor", classification_supervisor)
    builder.add_node("classification_tools", classification_tools)
    
    builder.add_edge(START, "classification_supervisor")
    # classification_supervisor → classification_tools는 Command로 처리
    # classification_tools → classification_supervisor 또는 END는 Command로 처리
    
    return builder.compile()


# Classification 서브그래프 인스턴스
classification_subgraph = create_classification_subgraph()


# ============================================================
# Phase 2: Insight 서브그래프 Placeholder
# ============================================================

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
    """메인 그래프를 생성합니다."""
    builder = StateGraph(
        ScreenshotAnalyzerState,
        input=InputState,
        config_schema=Configuration,
    )
    
    # 노드 추가
    builder.add_node("initialize", initialize)
    builder.add_node("classification_phase", classification_subgraph)  # 서브그래프로 교체!
    builder.add_node("insight_phase", insight_phase_placeholder)  # Step 5-3에서 교체
    builder.add_node("final_report", generate_final_report)
    
    # 엣지 연결
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "classification_phase")
    builder.add_edge("classification_phase", "insight_phase")
    builder.add_edge("insight_phase", "final_report")
    builder.add_edge("final_report", END)
    
    return builder.compile()


# 그래프 인스턴스 생성
graph = create_graph()
