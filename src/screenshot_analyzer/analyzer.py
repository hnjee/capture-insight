"""스크린샷 분석기 메인 그래프 구현."""

import asyncio
import json
import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from screenshot_analyzer.configuration import Configuration
from screenshot_analyzer.prompts import (
    CLASSIFIER_HUMAN_PROMPT,
    CLASSIFIER_SYSTEM_PROMPT,
    STRATEGIST_HUMAN_PROMPT,
    STRATEGIST_SYSTEM_PROMPT,
)
from screenshot_analyzer.state import (
    ClassificationComplete,
    ClassificationOutputState,
    ClassificationState,
    ClassifyImages,
    DesignFolderStructure,
    InputState,
    ReportAmbiguity,
    RequestRefinement,
    ReviseStructure,
    ScreenshotAnalyzerState,
    StrategyComplete,
)
from screenshot_analyzer.utils import (
    batch_ingestion,
    execute_vision_analysis_safely,
    get_api_key_for_model,
)

# 로깅 설정
logger = logging.getLogger(__name__)

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
        # Phase 0: Ingestion
        "image_metadatas": {},  # Phase 0에서 채워짐
        # Phase 1: Classification (Strategist-Classifier)
        "vision_results": {},
        "classifications": {},
        "categories": existing_categories or [],
    }


# ============================================================
# Phase 0: Ingestion 노드 (Workflow)
# ============================================================

async def run_ingestion(state: ScreenshotAnalyzerState, config: RunnableConfig) -> dict:
    """모든 이미지를 경량 VLM으로 일괄 메타데이터 추출."""
    images = state.get("images", [])
    
    if not images:
        return {"image_metadatas": {}}
    
    # 배치 Ingestion 실행
    metadata_dict = await batch_ingestion(images, config)
    
    # IngestionMetadata를 dict로 변환하여 State에 저장
    serialized_metadatas = {
        path: metadata.model_dump()
        for path, metadata in metadata_dict.items()
    }
    
    return {
        "image_metadatas": serialized_metadatas
    }


# ============================================================
# Phase 1: Strategist-Classifier 서브그래프
# ============================================================

def _summarize_metadata(image_metadatas: dict) -> str:
    """메타데이터를 요약 문자열로 변환."""
    if not image_metadatas:
        return "메타데이터 없음"
    
    summaries = []
    for path, meta in list(image_metadatas.items())[:20]:  # 최대 20개만 표시
        if isinstance(meta, dict):
            desc = meta.get("description", "설명 없음")
            ocr = meta.get("ocr_text", "")[:50]
            conf = meta.get("confidence_score", 0)
            needs_refine = meta.get("needs_visual_refinement", False)
            summaries.append(f"- {path}: {desc} (OCR: {ocr}..., 신뢰도: {conf}, VLM필요: {needs_refine})")
    
    if len(image_metadatas) > 20:
        summaries.append(f"... 외 {len(image_metadatas) - 20}장")
    
    return "\n".join(summaries)


def _get_suggested_categories_distribution(image_metadatas: dict) -> str:
    """추천 카테고리 분포를 계산."""
    from collections import Counter
    all_categories = []
    for meta in image_metadatas.values():
        if isinstance(meta, dict):
            all_categories.extend(meta.get("suggested_categories", []))
    
    counter = Counter(all_categories)
    return json.dumps(dict(counter.most_common(15)), ensure_ascii=False, indent=2)


async def strategist_agent(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["strategist_tools"]]:
    """Strategist Agent: 폴더 구조 설계.
    
    메타데이터를 조망하여 최적의 폴더 트리를 설계합니다.
    Classifier로부터 피드백이 오면 구조를 수정합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    
    # 현재 상태 추출
    images = state.get("images", [])
    image_metadatas = state.get("image_metadatas", {})
    existing_categories = state.get("existing_categories", [])
    strategy_iteration = state.get("strategy_iteration", 0)
    classification_feedback = state.get("classification_feedback", [])
    current_folder_tree = state.get("current_folder_tree", {})
    
    # 시스템 프롬프트 구성
    system_prompt = STRATEGIST_SYSTEM_PROMPT.format(
        total_images=len(images),
        strategy_iteration=strategy_iteration,
        max_iterations=configuration.max_analysis_iterations,
        has_feedback="있음" if classification_feedback else "없음",
        existing_categories=", ".join(existing_categories) if existing_categories else "없음",
    )
    
    # Human 프롬프트 구성
    human_prompt = STRATEGIST_HUMAN_PROMPT.format(
        metadata_summary=_summarize_metadata(image_metadatas),
        suggested_categories_distribution=_get_suggested_categories_distribution(image_metadatas),
        classification_feedback="\n".join(classification_feedback) if classification_feedback else "없음",
        current_folder_tree=json.dumps(current_folder_tree, ensure_ascii=False, indent=2) if current_folder_tree else "없음",
    )
    
    # 모델 설정
    model_config = {
        "model": configuration.analysis_model,
        "max_tokens": configuration.max_tokens,
        "api_key": get_api_key_for_model(configuration.analysis_model, config),
    }
    
    # 도구 바인딩 + with_retry() 추가 (open_deep_research 방식)
    tools = [DesignFolderStructure, ReviseStructure, StrategyComplete]
    model_with_tools = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configuration.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # 메시지 구성
    messages = state.get("messages", [])
    if not messages or state.get("current_phase") != "strategist":
        messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=human_prompt))
    
    # LLM 호출
    response = await model_with_tools.ainvoke(messages)
    
    return Command(
        goto="strategist_tools",
        update={
            "messages": [response],
            "strategy_iteration": strategy_iteration + 1,
            "current_phase": "strategist",
        }
    )


async def strategist_tools(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["strategist_agent", "classifier_agent", "__end__"]]:
    """Strategist의 도구 실행."""
    configuration = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent_message = messages[-1] if messages else None
    
    # 도구 호출이 없으면 Classifier로 전환
    if not most_recent_message or not most_recent_message.tool_calls:
        return Command(
            goto="classifier_agent",
            update={"current_phase": "classifier"}
        )
    
    # 반복 횟수 체크
    strategy_iteration = state.get("strategy_iteration", 0)
    if strategy_iteration > configuration.max_analysis_iterations:
        # 강제 종료
        return Command(
            goto=END,
            update={
                "classifications": state.get("assignments", {}),
                "categories": list(state.get("current_folder_tree", {}).keys()),
                "vision_results": state.get("refinement_results", {}),
            }
        )
    
    tool_messages = []
    update_payload = {}
    goto_classifier = False
    should_end = False
    
    for tool_call in most_recent_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "DesignFolderStructure":
            folder_tree = tool_args.get("folder_tree", {})
            folder_descriptions = tool_args.get("folder_descriptions", {})
            reasoning = tool_args.get("reasoning", "")
            
            update_payload["current_folder_tree"] = folder_tree
            update_payload["folder_descriptions"] = folder_descriptions
            update_payload["previous_folder_tree"] = state.get("current_folder_tree")
            
            # 미분류 이미지 설정
            images = state.get("images", [])
            update_payload["pending_images"] = images
            
            tool_messages.append(ToolMessage(
                content=f"폴더 구조 설계 완료\n구조: {json.dumps(folder_tree, ensure_ascii=False)}\n이유: {reasoning}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
        elif tool_name == "ReviseStructure":
            new_folder_tree = tool_args.get("new_folder_tree", {})
            changes = tool_args.get("changes", [])
            reasoning = tool_args.get("reasoning", "")
            
            update_payload["previous_folder_tree"] = state.get("current_folder_tree")
            update_payload["current_folder_tree"] = new_folder_tree
            update_payload["classification_feedback"] = []  # 피드백 처리 완료
            
            tool_messages.append(ToolMessage(
                content=f"폴더 구조 수정 완료\n변경: {json.dumps(changes, ensure_ascii=False)}\n이유: {reasoning}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
        elif tool_name == "StrategyComplete":
            goto_classifier = True
            final_folder_tree = tool_args.get("final_folder_tree", state.get("current_folder_tree", {}))
            summary = tool_args.get("summary", "")
            
            update_payload["current_folder_tree"] = final_folder_tree
            update_payload["is_converged"] = False  # Classifier가 확인할 때까지
            
            tool_messages.append(ToolMessage(
                content=f"Strategist 완료: {summary}\nClassifier로 전환합니다.",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
    
    update_payload["messages"] = tool_messages
    
    if goto_classifier:
        return Command(
            goto="classifier_agent",
            update={**update_payload, "current_phase": "classifier", "classify_iteration": 0}
        )
    
    return Command(
        goto="strategist_agent",
        update=update_payload,
    )


async def classifier_agent(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["classifier_tools"]]:
    """Classifier Agent: 이미지 분류.
    
    Strategist가 설계한 폴더 구조에 따라 이미지를 분류합니다.
    모호한 경우 피드백을 생성하거나 VLM 정밀분석을 요청합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    
    # 현재 상태 추출
    images = state.get("images", [])
    image_metadatas = state.get("image_metadatas", {})
    current_folder_tree = state.get("current_folder_tree", {})
    folder_descriptions = state.get("folder_descriptions", {})
    assignments = state.get("assignments", {})
    pending_images = state.get("pending_images", images)
    refinement_results = state.get("refinement_results", {})
    classify_iteration = state.get("classify_iteration", 0)
    
    # 미분류 이미지의 메타데이터
    pending_metadata = {
        path: image_metadatas.get(path, {})
        for path in pending_images[:30]  # 한 번에 30개까지만 처리
    }
    
    # 시스템 프롬프트 구성
    system_prompt = CLASSIFIER_SYSTEM_PROMPT.format(
        total_images=len(images),
        classified_count=len(assignments),
        pending_count=len(pending_images),
        classify_iteration=classify_iteration,
        max_iterations=configuration.max_analysis_iterations,
        folder_tree=json.dumps(current_folder_tree, ensure_ascii=False, indent=2),
        folder_descriptions=json.dumps(folder_descriptions, ensure_ascii=False, indent=2),
    )
    
    # Human 프롬프트 구성
    human_prompt = CLASSIFIER_HUMAN_PROMPT.format(
        pending_metadata=json.dumps(pending_metadata, ensure_ascii=False, indent=2),
        refinement_results=json.dumps(refinement_results, ensure_ascii=False, indent=2) if refinement_results else "없음",
        current_assignments=json.dumps(assignments, ensure_ascii=False, indent=2) if assignments else "없음",
    )
    
    # 모델 설정
    model_config = {
        "model": configuration.analysis_model,
        "max_tokens": configuration.max_tokens,
        "api_key": get_api_key_for_model(configuration.analysis_model, config),
    }
    
    # 도구 바인딩 + with_retry() 추가 (open_deep_research 방식)
    tools = [ClassifyImages, RequestRefinement, ReportAmbiguity, ClassificationComplete]
    model_with_tools = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configuration.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # 메시지 구성
    messages = state.get("messages", [])
    if state.get("current_phase") != "classifier":
        messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=human_prompt))
    
    # LLM 호출
    response = await model_with_tools.ainvoke(messages)
    
    return Command(
        goto="classifier_tools",
        update={
            "messages": [response],
            "classify_iteration": classify_iteration + 1,
            "current_phase": "classifier",
        }
    )


async def classifier_tools(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["classifier_agent", "strategist_agent", "vision_refiner", "__end__"]]:
    """Classifier의 도구 실행."""
    configuration = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent_message = messages[-1] if messages else None
    
    # 도구 호출이 없으면 종료
    if not most_recent_message or not most_recent_message.tool_calls:
        return Command(
            goto=END,
            update={
                "classifications": state.get("assignments", {}),
                "categories": list(state.get("current_folder_tree", {}).keys()),
                "vision_results": state.get("refinement_results", {}),
            }
        )
    
    # 반복 횟수 체크
    classify_iteration = state.get("classify_iteration", 0)
    if classify_iteration > configuration.max_analysis_iterations:
        return Command(
            goto=END,
            update={
                "classifications": state.get("assignments", {}),
                "categories": list(state.get("current_folder_tree", {}).keys()),
                "vision_results": state.get("refinement_results", {}),
            }
        )
    
    tool_messages = []
    update_payload = {}
    goto_strategist = False
    goto_refiner = False
    should_end = False
    
    for tool_call in most_recent_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "ClassifyImages":
            new_assignments = tool_args.get("assignments", {})
            confidence_scores = tool_args.get("confidence_scores", {})
            reasoning = tool_args.get("reasoning", "")
            
            # 기존 assignments에 병합
            current_assignments = state.get("assignments", {})
            merged_assignments = {**current_assignments, **new_assignments}
            update_payload["assignments"] = merged_assignments
            
            # pending_images 업데이트
            pending = state.get("pending_images", [])
            new_pending = [p for p in pending if p not in new_assignments]
            update_payload["pending_images"] = new_pending
            
            tool_messages.append(ToolMessage(
                content=f"분류 완료: {len(new_assignments)}장\n남은 이미지: {len(new_pending)}장\n이유: {reasoning}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
        elif tool_name == "RequestRefinement":
            image_paths = tool_args.get("image_paths", [])
            questions = tool_args.get("questions", {})
            reason = tool_args.get("reason", "")
            
            # VLM 정밀분석 요청 저장
            update_payload["refinement_requests"] = {
                "image_paths": image_paths,
                "questions": questions,
            }
            goto_refiner = True
            
            tool_messages.append(ToolMessage(
                content=f"VLM 정밀분석 요청: {len(image_paths)}장\n이유: {reason}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
        elif tool_name == "ReportAmbiguity":
            issue_type = tool_args.get("issue_type", "")
            affected_folders = tool_args.get("affected_folders", [])
            affected_images = tool_args.get("affected_images", [])
            suggestion = tool_args.get("suggestion", "")
            
            # 피드백 저장
            feedback = f"[{issue_type}] 폴더: {affected_folders}, 이미지: {len(affected_images)}장, 제안: {suggestion}"
            current_feedback = state.get("classification_feedback", [])
            update_payload["classification_feedback"] = current_feedback + [feedback]
            update_payload["needs_strategy_revision"] = True
            goto_strategist = True
            
            tool_messages.append(ToolMessage(
                content=f"Strategist에게 피드백 전달: {feedback}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
        elif tool_name == "ClassificationComplete":
            should_end = True
            summary = tool_args.get("summary", "분류 완료")
            total_classified = tool_args.get("total_classified", len(state.get("assignments", {})))
            categories_found = tool_args.get("categories_found", list(state.get("current_folder_tree", {}).keys()))
            
            tool_messages.append(ToolMessage(
                content=f"Classification 완료: {summary}\n총 {total_classified}장 분류됨",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
    
    update_payload["messages"] = tool_messages
    
    if should_end:
        # 최종 결과 반환
        final_assignments = {**state.get("assignments", {}), **update_payload.get("assignments", {})}
        return Command(
            goto=END,
            update={
                "classifications": final_assignments,
                "categories": list(state.get("current_folder_tree", {}).keys()),
                "vision_results": state.get("refinement_results", {}),
            }
        )
    
    if goto_refiner:
        return Command(
            goto="vision_refiner",
            update={**update_payload, "current_phase": "refiner"}
        )
    
    if goto_strategist:
        return Command(
            goto="strategist_agent",
            update={**update_payload, "current_phase": "strategist"}
        )
    
    return Command(
        goto="classifier_agent",
        update=update_payload,
    )


async def vision_refiner(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["classifier_agent"]]:
    """Vision Refiner: VLM 정밀분석.
    
    Classifier가 요청한 이미지들을 고성능 VLM으로 분석합니다.
    needs_visual_refinement=True이거나 Classifier가 모호하다고 판단한 이미지들.
    """
    configuration = Configuration.from_runnable_config(config)
    
    refinement_requests = state.get("refinement_requests", {})
    image_paths = refinement_requests.get("image_paths", [])
    questions = refinement_requests.get("questions", {})
    
    if not image_paths:
        # 요청이 없으면 바로 Classifier로 복귀
        return Command(
            goto="classifier_agent",
            update={"current_phase": "classifier"}
        )
    
    # 동시성 제한 (gpt-4o TPM 제한 고려)
    # open_deep_research 방식: 동시성 제한 + with_retry 자동 재시도
    semaphore = asyncio.Semaphore(configuration.refinement_concurrency)
    
    async def process_with_semaphore(img_path: str) -> tuple[str, dict]:
        async with semaphore:
            # Rate Limit 방지를 위한 약간의 딜레이 (with_retry가 대부분 처리)
            await asyncio.sleep(0.5)
            
            # 안전한 실행 함수 사용 (에러 발생 시에도 계속 진행)
            result = await execute_vision_analysis_safely(img_path, config)
            
            return (img_path, {
                "analysis": result.model_dump(),
                "answer": questions.get(img_path, ""),
                "recommended_folder": result.suggested_category,
            })
    
    # 병렬 처리 (동시성 제한 적용, 에러는 안전한 실행 함수에서 처리)
    tasks = [process_with_semaphore(img_path) for img_path in image_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 정리 (에러는 execute_vision_analysis_safely에서 이미 처리됨)
    new_results = {}
    for result in results:
        if isinstance(result, Exception):
            # 예상치 못한 에러 (실행 함수에서 잡지 못한 경우)
            logger.error(f"Vision refinement 예상치 못한 에러: {result}")
            # 에러 정보를 결과에 포함 (부분 성공 허용)
            continue
        img_path, result_dict = result
        new_results[img_path] = result_dict
    
    # 기존 결과에 병합
    current_results = state.get("refinement_results", {})
    merged_results = {**current_results, **new_results}
    
    return Command(
        goto="classifier_agent",
        update={
            "refinement_results": merged_results,
            "refinement_requests": {},  # 요청 처리 완료
            "current_phase": "classifier",
        }
    )


def create_classification_subgraph():
    """Classification Phase 서브그래프를 생성합니다."""
    builder = StateGraph(
        ClassificationState,
        output=ClassificationOutputState,
        config_schema=Configuration,
    )
    
    # 노드 추가
    builder.add_node("strategist_agent", strategist_agent)
    builder.add_node("strategist_tools", strategist_tools)
    builder.add_node("classifier_agent", classifier_agent)
    builder.add_node("classifier_tools", classifier_tools)
    builder.add_node("vision_refiner", vision_refiner)
    
    # 시작점: Strategist
    builder.add_edge(START, "strategist_agent")
    
    return builder.compile()


classification_subgraph = create_classification_subgraph()


# ============================================================
# 메인 그래프 구성
# ============================================================

def create_graph():
    """메인 그래프를 생성합니다.

    그래프 구조:
        START → initialize → ingestion → classification_strategist → END
    
    Phase 0 (Ingestion):
        - 모든 이미지를 경량 VLM으로 메타데이터 추출
        - 비용 60~80% 절감
    
    Phase 1 (Classification):
        - Strategist: 폴더 구조 설계
        - Classifier: 이미지 분류 + 피드백 루프
        - Vision Refiner: 필요 시 VLM 정밀분석
    
    출력:
        - classifications: {image_path: folder_name}
        - categories: 최종 폴더 목록
    """
    builder = StateGraph(
        ScreenshotAnalyzerState,
        input=InputState,
        config_schema=Configuration,
    )
    
    # 노드 추가
    builder.add_node("initialize", initialize)
    builder.add_node("ingestion", run_ingestion)
    builder.add_node("classification_strategist", classification_subgraph)
    
    # 엣지 연결
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "ingestion")
    builder.add_edge("ingestion", "classification_strategist")
    builder.add_edge("classification_strategist", END)
    
    return builder.compile()


# 그래프 인스턴스 생성
graph = create_graph()
