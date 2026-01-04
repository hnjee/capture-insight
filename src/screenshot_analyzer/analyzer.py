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
    CATEGORY_MERGE_PROMPT,
    CLASSIFICATION_HUMAN_PROMPT,
    CLASSIFICATION_PROMPT,
    CLASSIFICATION_SUPERVISOR_SYSTEM_PROMPT,
    FINAL_REPORT_PROMPT,
    INSIGHT_HUMAN_PROMPT,
    INSIGHT_SUPERVISOR_SYSTEM_PROMPT,
    SEARCH_INSIGHT_PROMPT,
)
from screenshot_analyzer.state import (
    ClassificationComplete,
    ClassificationOutputState,
    ClassificationState,
    ConductCategoryMerge,
    ConductClassification,
    ConductSearch,
    ConductVisionAnalysis,
    InputState,
    InsightComplete,
    InsightOutputState,
    InsightState,
    ScreenshotAnalyzerState,
)
from screenshot_analyzer.utils import (
    analyze_image,
    get_api_key_for_model,
    parse_json_response,
    search_category_insights,
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
    
    분류 결과와 인사이트를 종합하여 마크다운 형식의 보고서를 생성합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    
    # 상태에서 데이터 추출
    images = state.get("images", [])
    classifications = state.get("classifications", {})
    category_insights = state.get("category_insights", {})
    
    # 분석 일시
    from datetime import datetime
    analysis_date = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    
    # 분류 결과 정리
    classifications_str = json.dumps(classifications, ensure_ascii=False, indent=2)
    
    # 인사이트 정리
    insights_str = json.dumps(category_insights, ensure_ascii=False, indent=2)
    
    # 프롬프트 구성
    report_prompt = FINAL_REPORT_PROMPT.format(
        total_images=len(images),
        analysis_date=analysis_date,
        classifications=classifications_str,
        category_insights=insights_str,
    )
    
    # 모델 설정
    model_config = {
        "model": configuration.report_model,
        "max_tokens": configuration.report_max_tokens,
        "api_key": get_api_key_for_model(configuration.report_model, config),
    }
    report_model = configurable_model.with_config(model_config)
    
    # LLM 호출
    response = await report_model.ainvoke([
        HumanMessage(content=report_prompt)
    ])
    
    return {
        "final_report": response.content
    }


# ============================================================
# Phase 1: Classification 서브그래프
# ============================================================

async def classification_supervisor(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["classification_tools"]]:
    """Classification Phase의 Supervisor 노드."""
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
    tools = [ConductVisionAnalysis, ConductClassification, ConductCategoryMerge, ClassificationComplete]
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
    """Classification Phase의 도구 실행 노드."""
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
            
            if targets == ["all"] or "all" in targets:
                analyzed = state.get("analyzed_images", [])
                targets = [img for img in images if img not in analyzed]
            
            if targets:
                # 순차 처리 + Rate Limit 재시도 로직
                new_vision_results = {}
                new_analyzed = []
                
                for img in targets:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            result = await analyze_image(img, config)
                            new_vision_results[img] = result.model_dump()
                            new_analyzed.append(img)
                            break  # 성공하면 다음 이미지로
                        except Exception as e:
                            error_msg = str(e)
                            if "429" in error_msg or "rate_limit" in error_msg.lower():
                                # Rate Limit - 잠시 대기 후 재시도
                                wait_time = (attempt + 1) * 5  # 5초, 10초, 15초
                                await asyncio.sleep(wait_time)
                                if attempt == max_retries - 1:
                                    # 마지막 시도도 실패하면 건너뛰기
                                    pass
                            else:
                                # 다른 에러는 건너뛰기
                                break
                    
                    # 이미지 간 딜레이 (Rate Limit 방지)
                    await asyncio.sleep(1)
                
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
            vision_results = {**state.get("vision_results", {}), **update_payload.get("vision_results", {})}
            existing_categories = state.get("existing_categories", [])
            
            if vision_results:
                classification_prompt = CLASSIFICATION_PROMPT.format(
                    vision_results=json.dumps(vision_results, ensure_ascii=False, indent=2),
                    existing_categories=", ".join(existing_categories) if existing_categories else "없음",
                )
                
                model_config = {
                    "model": configuration.analysis_model,
                    "max_tokens": configuration.max_tokens,
                    "api_key": get_api_key_for_model(configuration.analysis_model, config),
                }
                classification_model = configurable_model.with_config(model_config)
                
                response = await classification_model.ainvoke([
                    HumanMessage(content=classification_prompt)
                ])
                
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
                    content="분류할 Vision 분석 결과가 없습니다.",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
        
        elif tool_name == "ConductCategoryMerge":
            # 카테고리 통합/정제 수행
            current_classifications = {**state.get("classifications", {}), **update_payload.get("classifications", {})}
            current_categories = update_payload.get("categories", state.get("categories", []))
            
            if current_classifications:
                merge_prompt = CATEGORY_MERGE_PROMPT.format(
                    current_classifications=json.dumps(current_classifications, ensure_ascii=False, indent=2),
                    current_categories=", ".join(current_categories) if current_categories else "없음",
                )
                
                model_config = {
                    "model": configuration.analysis_model,
                    "max_tokens": configuration.max_tokens,
                    "api_key": get_api_key_for_model(configuration.analysis_model, config),
                }
                merge_model = configurable_model.with_config(model_config)
                
                response = await merge_model.ainvoke([
                    HumanMessage(content=merge_prompt)
                ])
                
                result_dict = parse_json_response(response.content)
                merged_classifications = result_dict.get("merged_classifications", current_classifications)
                final_categories = result_dict.get("final_categories", current_categories)
                merge_summary = result_dict.get("merge_summary", {})
                
                # 병합된 결과로 업데이트 (완전 교체)
                update_payload["classifications"] = {"type": "override", "value": merged_classifications}
                update_payload["categories"] = final_categories
                
                tool_messages.append(ToolMessage(
                    content=f"카테고리 통합 완료\n병합 요약: {json.dumps(merge_summary, ensure_ascii=False, indent=2)}\n최종 카테고리: {final_categories}",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
            else:
                tool_messages.append(ToolMessage(
                    content="통합할 분류 결과가 없습니다. 먼저 ConductClassification을 실행하세요.",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
        
        elif tool_name == "ClassificationComplete":
            should_end = True
            summary = tool_args.get("summary", "분류 완료")
            categories_found = tool_args.get("categories_found", update_payload.get("categories", []))
            
            tool_messages.append(ToolMessage(
                content=f"Classification Phase 완료: {summary}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
            update_payload["categories"] = categories_found
    
    update_payload["messages"] = tool_messages
    
    if should_end:
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
    
    return builder.compile()


classification_subgraph = create_classification_subgraph()


# ============================================================
# Phase 2: Insight 서브그래프
# ============================================================

async def insight_supervisor(
    state: InsightState, 
    config: RunnableConfig
) -> Command[Literal["insight_tools"]]:
    """Insight Phase의 Supervisor 노드.
    
    현재 상태를 분석하고 다음 작업을 결정합니다:
    - ConductSearch: 웹 검색 지시
    - InsightComplete: Phase 완료
    """
    configuration = Configuration.from_runnable_config(config)
    
    # 현재 상태 추출
    categories = state.get("categories", [])
    searched_categories = state.get("searched_categories", [])
    category_insights = state.get("category_insights", {})
    classifications = state.get("classifications", {})
    iteration_count = state.get("iteration_count", 0)
    
    # 미검색 카테고리 계산
    pending_categories = [cat for cat in categories if cat not in searched_categories]
    
    # 카테고리별 이미지 수 계산
    category_image_counts = {}
    for img_path, classification in classifications.items():
        cat = classification.get("category", "기타") if isinstance(classification, dict) else "기타"
        category_image_counts[cat] = category_image_counts.get(cat, 0) + 1
    
    # 시스템 프롬프트 구성
    system_prompt = INSIGHT_SUPERVISOR_SYSTEM_PROMPT.format(
        total_categories=len(categories),
        searched_count=len(searched_categories),
        pending_count=len(pending_categories),
        iteration_count=iteration_count,
        max_iterations=configuration.max_analysis_iterations,
        categories_list=", ".join(categories) if categories else "없음",
        insights_summary=json.dumps(category_insights, ensure_ascii=False, indent=2) if category_insights else "없음",
    )
    
    # Human 프롬프트 구성
    human_prompt = INSIGHT_HUMAN_PROMPT.format(
        pending_categories=", ".join(pending_categories) if pending_categories else "없음 (모두 검색 완료)",
        category_image_counts=json.dumps(category_image_counts, ensure_ascii=False, indent=2),
    )
    
    # 모델 설정
    model_config = {
        "model": configuration.analysis_model,
        "max_tokens": configuration.max_tokens,
        "api_key": get_api_key_for_model(configuration.analysis_model, config),
    }
    
    # 도구 바인딩
    tools = [ConductSearch, InsightComplete]
    model_with_tools = configurable_model.bind_tools(tools).with_config(model_config)
    
    # 메시지 구성
    messages = state.get("messages", [])
    if not messages:
        messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=human_prompt))
    
    # LLM 호출
    response = await model_with_tools.ainvoke(messages)
    
    return Command(
        goto="insight_tools",
        update={
            "messages": [response],
            "iteration_count": iteration_count + 1,
        }
    )


async def insight_tools(
    state: InsightState, 
    config: RunnableConfig
) -> Command[Literal["insight_supervisor", "__end__"]]:
    """Insight Phase의 도구 실행 노드.
    
    Supervisor가 호출한 도구를 실행합니다:
    - ConductSearch: Tavily로 웹 검색
    - InsightComplete: Phase 종료
    """
    configuration = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent_message = messages[-1] if messages else None
    
    # 도구 호출이 없으면 종료
    if not most_recent_message or not most_recent_message.tool_calls:
        return Command(
            goto=END,
            update={
                "category_insights": state.get("category_insights", {}),
            }
        )
    
    # 반복 횟수 체크
    iteration_count = state.get("iteration_count", 0)
    if iteration_count > configuration.max_analysis_iterations:
        return Command(
            goto=END,
            update={
                "category_insights": state.get("category_insights", {}),
            }
        )
    
    tool_messages = []
    update_payload = {}
    should_end = False
    
    for tool_call in most_recent_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "ConductSearch":
            # 웹 검색 수행
            category = tool_args.get("category", "")
            keywords = tool_args.get("keywords", [])
            
            if category:
                # 검색 수행
                search_result = await search_category_insights(
                    category=category,
                    keywords=keywords,
                    config=config,
                )
                
                # 검색 결과로 인사이트 생성
                classifications = state.get("classifications", {})
                category_images = [
                    img for img, cls in classifications.items()
                    if (cls.get("category") if isinstance(cls, dict) else "") == category
                ]
                
                # 인사이트 정리 (LLM 호출)
                if search_result.get("sources"):
                    insight_prompt = SEARCH_INSIGHT_PROMPT.format(
                        category=category,
                        image_count=len(category_images),
                        sub_categories=", ".join(set(
                            cls.get("sub_category", "") 
                            for cls in classifications.values() 
                            if isinstance(cls, dict) and cls.get("category") == category
                        )),
                        search_results=json.dumps(search_result.get("sources", []), ensure_ascii=False, indent=2),
                    )
                    
                    model_config = {
                        "model": configuration.analysis_model,
                        "max_tokens": configuration.max_tokens,
                        "api_key": get_api_key_for_model(configuration.analysis_model, config),
                    }
                    insight_model = configurable_model.with_config(model_config)
                    
                    response = await insight_model.ainvoke([
                        HumanMessage(content=insight_prompt)
                    ])
                    
                    insight_dict = parse_json_response(response.content)
                    
                    # State 업데이트
                    new_insights = {category: insight_dict}
                    update_payload["category_insights"] = new_insights
                    update_payload["searched_categories"] = state.get("searched_categories", []) + [category]
                    
                    tool_messages.append(ToolMessage(
                        content=f"'{category}' 카테고리 인사이트 수집 완료\n{json.dumps(insight_dict, ensure_ascii=False, indent=2)}",
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    ))
                else:
                    tool_messages.append(ToolMessage(
                        content=f"'{category}' 카테고리 검색 결과가 없습니다.",
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    ))
            else:
                tool_messages.append(ToolMessage(
                    content="검색할 카테고리가 지정되지 않았습니다.",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
        
        elif tool_name == "InsightComplete":
            # Phase 완료
            should_end = True
            summary = tool_args.get("summary", "인사이트 수집 완료")
            
            tool_messages.append(ToolMessage(
                content=f"Insight Phase 완료: {summary}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
    
    update_payload["messages"] = tool_messages
    
    if should_end:
        return Command(
            goto=END,
            update={
                "category_insights": {**state.get("category_insights", {}), **update_payload.get("category_insights", {})},
            }
        )
    
    return Command(
        goto="insight_supervisor",
        update=update_payload,
    )


def create_insight_subgraph():
    """Insight Phase 서브그래프를 생성합니다."""
    builder = StateGraph(
        InsightState,
        output=InsightOutputState,
        config_schema=Configuration,
    )
    
    builder.add_node("insight_supervisor", insight_supervisor)
    builder.add_node("insight_tools", insight_tools)
    builder.add_edge(START, "insight_supervisor")
    
    return builder.compile()


insight_subgraph = create_insight_subgraph()


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
    builder.add_node("classification_phase", classification_subgraph)
    builder.add_node("insight_phase", insight_subgraph)  # 서브그래프로 교체!
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
