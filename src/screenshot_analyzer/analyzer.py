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
    # 테스트 모드: 입력에 이미 메타데이터가 있으면 보존
    existing_metadatas = state.get("image_metadatas", {})
    
    return {
        "images": images,
        "existing_categories": existing_categories,
        # Phase 0: Ingestion
        "image_metadatas": existing_metadatas if existing_metadatas else {},  # 입력에 있으면 보존, 없으면 빈 dict
        # Phase 1: Classification (Strategist-Classifier)
        "vision_results": {},
        "classifications": {},
        "categories": existing_categories or [],
    }


# ============================================================
# Phase 0: Ingestion 노드 (Workflow)
# ============================================================

async def run_ingestion(state: ScreenshotAnalyzerState, config: RunnableConfig) -> dict:
    """모든 이미지를 경량 VLM으로 일괄 메타데이터 추출.
    
    테스트 모드: 이미 image_metadatas가 있으면 스킵 (VLM 호출 안 함)
    """
    images = state.get("images", [])
    existing_metadatas = state.get("image_metadatas", {})
    
    # 테스트 모드: 이미 메타데이터가 있으면 스킵
    if existing_metadatas:
        logger.info(f"🧪 테스트 모드: 기존 메타데이터 사용 ({len(existing_metadatas)}개). VLM 호출 스킵.")
        return {}  # 상태 변경 없음
    
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


async def strategist(
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
    current_folders = state.get("current_folders", [])
    
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
        current_folders=json.dumps(current_folders, ensure_ascii=False, indent=2) if current_folders else "없음",
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
) -> Command[Literal["strategist", "classifier", "__end__"]]:
    """Strategist의 도구 실행."""
    configuration = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent_message = messages[-1] if messages else None
    
    # 도구 호출이 없으면 Classifier로 전환
    if not most_recent_message or not most_recent_message.tool_calls:
        logger.warning("Strategist: 도구 호출이 없습니다. Classifier로 전환합니다.")
        current_folders = state.get("current_folders", [])
        if not current_folders:
            # 기본값 설정
            logger.error("Strategist: folders가 비어있습니다. 기본값을 설정합니다.")
            current_folders = ["기타"]
        return Command(
            goto="classifier",
            update={
                "current_phase": "classifier",
                "current_folders": current_folders,
            }
        )
    
    # 반복 횟수 체크 (무한 루프 방지)
    strategy_iteration = state.get("strategy_iteration", 0)
    if strategy_iteration > configuration.max_analysis_iterations:
        logger.warning(f"Strategist: 최대 반복 횟수({configuration.max_analysis_iterations}) 초과. 강제 종료합니다.")
        current_folders = state.get("current_folders", [])
        if not current_folders:
            current_folders = ["기타"]
        # 강제 종료
        return Command(
            goto=END,
            update={
                "classifications": state.get("assignments", {}),
                "categories": current_folders,
                "vision_results": state.get("refinement_results", {}),
            }
        )
    
    tool_messages = []
    update_payload = {}
    goto_classifier = False
    should_end = False
    
    # 도구들 처리
    for tool_call in most_recent_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        
        # args 비어있을 때 에러 로그 + 기본값 처리
        if not tool_args:
            logger.error(f"Strategist: {tool_name} 도구의 args가 비어있습니다. 기본값을 사용합니다.")
            tool_args = {}
        
        if tool_name == "DesignFolderStructure":
            folders = tool_args.get("folders", [])
            folder_descriptions = tool_args.get("folder_descriptions", {})
            reasoning = tool_args.get("reasoning", "")
            
            # folders 유효성 검사
            if not folders or not isinstance(folders, list):
                logger.error(f"Strategist: folders가 유효하지 않습니다. 기본값을 사용합니다. args: {tool_args}")
                folders = ["기타"]
            
            # 무한 루프 방지: 이전과 동일한 구조인지 확인
            previous_folders = state.get("previous_folders")
            if previous_folders and set(folders) == set(previous_folders):
                logger.warning(f"Strategist: 이전과 동일한 폴더 구조입니다. 수렴으로 간주합니다.")
                update_payload["is_converged"] = True
                goto_classifier = True
            else:
                update_payload["current_folders"] = folders
                update_payload["folder_descriptions"] = folder_descriptions
                update_payload["previous_folders"] = state.get("current_folders", [])
            
            # 미분류 이미지 설정
            images = state.get("images", [])
            update_payload["pending_images"] = images
            
            tool_messages.append(ToolMessage(
                content=f"폴더 구조 설계 완료\n폴더: {json.dumps(folders, ensure_ascii=False)}\n이유: {reasoning}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
        elif tool_name == "ReviseStructure":
            new_folders = tool_args.get("new_folders", [])
            changes = tool_args.get("changes", [])
            reasoning = tool_args.get("reasoning", "")
            
            # new_folders 유효성 검사
            if not new_folders or not isinstance(new_folders, list):
                logger.error(f"Strategist: new_folders가 유효하지 않습니다. 기본값을 사용합니다. args: {tool_args}")
                new_folders = state.get("current_folders", ["기타"])
            
            # 무한 루프 방지: 이전과 동일한 구조인지 확인
            previous_folders = state.get("previous_folders")
            if previous_folders and set(new_folders) == set(previous_folders):
                logger.warning(f"Strategist: 수정 후에도 이전과 동일한 폴더 구조입니다. 수렴으로 간주합니다.")
                update_payload["is_converged"] = True
                goto_classifier = True
            else:
                update_payload["previous_folders"] = state.get("current_folders", [])
                update_payload["current_folders"] = new_folders
                update_payload["classification_feedback"] = []  # 피드백 처리 완료
            
            tool_messages.append(ToolMessage(
                content=f"폴더 구조 수정 완료\n변경: {json.dumps(changes, ensure_ascii=False)}\n새 폴더: {json.dumps(new_folders, ensure_ascii=False)}\n이유: {reasoning}",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
            
        elif tool_name == "StrategyComplete":
            goto_classifier = True
            final_folders = tool_args.get("final_folders", state.get("current_folders", []))
            summary = tool_args.get("summary", "")
            
            # final_folders 유효성 검사
            if not final_folders or not isinstance(final_folders, list):
                logger.error(f"Strategist: final_folders가 유효하지 않습니다. 기본값을 사용합니다. args: {tool_args}")
                final_folders = state.get("current_folders", ["기타"])
            
            update_payload["current_folders"] = final_folders
            update_payload["is_converged"] = False  # Classifier가 확인할 때까지
            
            tool_messages.append(ToolMessage(
                content=f"Strategist 완료: {summary}\nClassifier로 전환합니다.",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
        
        else:
            # 알 수 없는 도구 호출 - 메시지 체인 유지를 위해 ToolMessage 추가
            logger.warning(f"Strategist: 알 수 없는 도구 호출: {tool_name}. 메시지 체인 유지를 위해 응답을 추가합니다.")
            tool_messages.append(ToolMessage(
                content=f"경고: 알 수 없는 도구 '{tool_name}'가 호출되었습니다. 사용 가능한 도구: DesignFolderStructure, ReviseStructure, StrategyComplete",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
    
    update_payload["messages"] = tool_messages
    
    if goto_classifier:
        return Command(
            goto="classifier",
            update={**update_payload, "current_phase": "classifier", "classify_iteration": 0}
        )
    
    return Command(
        goto="strategist",
        update=update_payload,
    )


async def classifier(
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
    current_folders = state.get("current_folders", [])
    folder_descriptions = state.get("folder_descriptions", {})
    assignments = state.get("assignments", {})
    pending_images = state.get("pending_images", images)
    refinement_results = state.get("refinement_results", {})
    classify_iteration = state.get("classify_iteration", 0)
    
    # 디버깅: refinement_results 확인
    if refinement_results:
        logger.info(f"📋 Classifier: refinement_results {len(refinement_results)}개 발견")
        for img_path, result in refinement_results.items():
            recommended = result.get("recommended_folder", "없음")
            logger.info(f"   - {img_path}: recommended_folder={recommended}")
    else:
        logger.info("📋 Classifier: refinement_results 없음")
    
    # current_folders 유효성 검사
    if not current_folders:
        logger.error("Classifier: current_folders가 비어있습니다. 기본값을 사용합니다.")
        current_folders = ["기타"]
    
    # 미분류 이미지의 메타데이터 (더 명확하게)
    pending_metadata_list = []
    for path in pending_images[:30]:  # 한 번에 30개까지만 처리
        meta = image_metadatas.get(path, {})
        # refinement_results에 있는 이미지는 recommended_folder 포함
        refinement_info = refinement_results.get(path, {})
        recommended_folder = refinement_info.get("recommended_folder", "")
        
        pending_metadata_list.append({
            "image_path": path,  # 🔥 명확히 표시
            "description": meta.get("description", ""),
            "ocr_text": meta.get("ocr_text", ""),
            "confidence_score": meta.get("confidence_score", 0),
            "suggested_categories": meta.get("suggested_categories", []),
            "needs_visual_refinement": meta.get("needs_visual_refinement", False),
            "vlm_recommended_folder": recommended_folder if recommended_folder else None,  # 🔥 VLM 추천 폴더
        })
    
    # 시스템 프롬프트 구성
    system_prompt = CLASSIFIER_SYSTEM_PROMPT.format(
        total_images=len(images),
        classified_count=len(assignments),
        pending_count=len(pending_images),
        classify_iteration=classify_iteration,
        max_iterations=configuration.max_analysis_iterations,
        folders=json.dumps(current_folders, ensure_ascii=False, indent=2),
        folder_descriptions=json.dumps(folder_descriptions, ensure_ascii=False, indent=2),
    )
    
    # Human 프롬프트 구성
    human_prompt = CLASSIFIER_HUMAN_PROMPT.format(
        folders=json.dumps(current_folders, ensure_ascii=False, indent=2),
        folder_descriptions=json.dumps(folder_descriptions, ensure_ascii=False, indent=2),
        pending_metadata=json.dumps(pending_metadata_list, ensure_ascii=False, indent=2),  # 🔥 개선
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
    
    # Phase 전환 시 또는 메시지가 없을 때 초기화
    if not messages or state.get("current_phase") != "classifier":
        messages = [SystemMessage(content=system_prompt)]
    else:
        # 메시지 체인 검증: 마지막 assistant message의 tool_calls가 모두 응답되었는지 확인
        if messages:
            last_message = messages[-1]
            # 마지막 메시지가 AIMessage이고 tool_calls가 있는 경우
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # tool_calls에 대한 ToolMessage가 모두 있는지 확인
                tool_call_ids = set()
                for tc in last_message.tool_calls:
                    if isinstance(tc, dict):
                        tool_call_ids.add(tc.get("id"))
                    elif hasattr(tc, 'get'):
                        tool_call_ids.add(tc.get("id"))
                    elif hasattr(tc, 'id'):
                        tool_call_ids.add(tc.id)
                
                tool_message_ids = set()
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        if hasattr(msg, 'tool_call_id'):
                            tool_message_ids.add(msg.tool_call_id)
                
                missing_ids = tool_call_ids - tool_message_ids
                if missing_ids:
                    logger.warning(f"Classifier: 누락된 tool_call 응답이 있습니다. 메시지를 초기화합니다. missing_ids: {missing_ids}")
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
) -> Command[Literal["classifier", "strategist", "vision_refiner", "__end__"]]:
    """Classifier의 도구 실행."""
    configuration = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent_message = messages[-1] if messages else None
    
    # 도구 호출이 없으면 종료
    if not most_recent_message or not most_recent_message.tool_calls:
        logger.warning("Classifier: 도구 호출이 없습니다. 종료합니다.")
        current_folders = state.get("current_folders", [])
        if not current_folders:
            current_folders = ["기타"]
        return Command(
            goto=END,
            update={
                "classifications": state.get("assignments", {}),
                "categories": current_folders,
                "vision_results": state.get("refinement_results", {}),
            }
        )
    
    # 반복 횟수 체크 (무한 루프 방지)
    classify_iteration = state.get("classify_iteration", 0)
    if classify_iteration > configuration.max_analysis_iterations:
        logger.warning(f"Classifier: 최대 반복 횟수({configuration.max_analysis_iterations}) 초과. 강제 종료합니다.")
        current_folders = state.get("current_folders", [])
        if not current_folders:
            current_folders = ["기타"]
        return Command(
            goto=END,
            update={
                "classifications": state.get("assignments", {}),
                "categories": current_folders,
                "vision_results": state.get("refinement_results", {}),
            }
        )
    
    tool_messages = []
    update_payload = {}
    goto_strategist = False
    goto_refiner = False
    should_end = False
    
    # 도구들 처리
    for tool_call in most_recent_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        
        # args 비어있을 때 에러 로그 + 기본값 처리
        if not tool_args:
            logger.error(f"Classifier: {tool_name} 도구의 args가 비어있습니다. 기본값을 사용합니다.")
            tool_args = {}
        
        if tool_name == "ClassifyImages":
            new_assignments = tool_args.get("assignments", {})
            confidence_scores = tool_args.get("confidence_scores", {})
            reasoning = tool_args.get("reasoning", "")
            
            # assignments 유효성 검사 및 단순 폴더명으로 변환
            if not new_assignments:
                logger.error(f"❌ ClassifyImages의 assignments가 비어있습니다!")
                logger.error(f"받은 args: {tool_args}")
                
                # 🔥 LLM에게 명확한 피드백
                error_message = """ERROR: assignments가 비어있습니다!

ClassifyImages를 호출할 때는 반드시 다음 형식을 따라주세요:

{{
    "assignments": {{
        "이미지경로1": "폴더명1",
        "이미지경로2": "폴더명2"
    }},
    "confidence_scores": {{
        "이미지경로1": 0.95,
        "이미지경로2": 0.88
    }},
    "reasoning": "분류 이유..."
}}

예시:
{{
    "assignments": {{
        "examples/screenshots/IMG_5779.PNG": "건강",
        "examples/screenshots/IMG_6677.PNG": "패션"
    }},
    "confidence_scores": {{
        "examples/screenshots/IMG_5779.PNG": 0.95,
        "examples/screenshots/IMG_6677.PNG": 0.88
    }},
    "reasoning": "IMG_5779는 영양제 상품으로 건강 폴더에 분류"
}}

**중요**: 
- assignments는 반드시 {{이미지경로: 폴더명}} 형태의 딕셔너리여야 합니다
- 이미지 경로는 "분류 대상 이미지 메타데이터"에 나온 "image_path"를 정확히 사용하세요
- 폴더명은 "사용 가능한 폴더" 목록에 있는 것만 사용하세요

다시 시도해주세요."""

                tool_messages.append(ToolMessage(
                    content=error_message,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
                continue  # 다음 tool_call로
            else:
                # assignments의 값이 단순 폴더명인지 확인 (중첩 구조 제거)
                validated_assignments = {}
                current_folders = state.get("current_folders", [])
                for img_path, folder_name in new_assignments.items():
                    # folder_name이 리스트나 딕셔너리인 경우 첫 번째 값 추출
                    if isinstance(folder_name, list):
                        folder_name = folder_name[0] if folder_name else "기타"
                        logger.warning(f"Classifier: {img_path}의 폴더명이 리스트입니다. 첫 번째 값 사용: {folder_name}")
                    elif isinstance(folder_name, dict):
                        folder_name = list(folder_name.keys())[0] if folder_name else "기타"
                        logger.warning(f"Classifier: {img_path}의 폴더명이 딕셔너리입니다. 첫 번째 키 사용: {folder_name}")
                    
                    # 폴더명이 current_folders에 없으면 경고
                    if folder_name not in current_folders:
                        logger.warning(f"Classifier: {img_path}의 폴더명 '{folder_name}'이 current_folders에 없습니다. 그대로 사용합니다.")
                    
                    validated_assignments[img_path] = folder_name
                
                # 기존 assignments에 병합
                current_assignments = state.get("assignments", {})
                merged_assignments = {**current_assignments, **validated_assignments}
                update_payload["assignments"] = merged_assignments
                
                # pending_images 업데이트
                pending = state.get("pending_images", [])
                new_pending = [p for p in pending if p not in validated_assignments]
                update_payload["pending_images"] = new_pending
                
                tool_messages.append(ToolMessage(
                    content=f"분류 완료: {len(validated_assignments)}장\n남은 이미지: {len(new_pending)}장\n이유: {reasoning}",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
            
        elif tool_name == "RequestRefinement":
            image_paths = tool_args.get("image_paths", [])
            questions = tool_args.get("questions", {})
            reason = tool_args.get("reason", "")
            
            # 유효성 검사
            if not image_paths:
                logger.warning("Classifier: RequestRefinement의 image_paths가 비어있습니다. LLM에게 피드백을 전달합니다.")
                # 메시지 체인 유지를 위해 ToolMessage 추가
                tool_messages.append(ToolMessage(
                    content=f"경고: image_paths가 비어있습니다. 정밀분석이 필요한 이미지 경로를 포함해주세요. 이유: {reason or '없음'}",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                ))
            else:
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
            categories_found = tool_args.get("categories_found", state.get("current_folders", []))
            
            # categories_found 유효성 검사
            if not categories_found or not isinstance(categories_found, list):
                logger.warning("Classifier: categories_found가 유효하지 않습니다. current_folders를 사용합니다.")
                categories_found = state.get("current_folders", ["기타"])
            
            tool_messages.append(ToolMessage(
                content=f"Classification 완료: {summary}\n총 {total_classified}장 분류됨",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
        
        else:
            # 알 수 없는 도구 호출 - 메시지 체인 유지를 위해 ToolMessage 추가
            logger.warning(f"Classifier: 알 수 없는 도구 호출: {tool_name}. 메시지 체인 유지를 위해 응답을 추가합니다.")
            tool_messages.append(ToolMessage(
                content=f"경고: 알 수 없는 도구 '{tool_name}'가 호출되었습니다. 사용 가능한 도구: ClassifyImages, RequestRefinement, ReportAmbiguity, ClassificationComplete",
                name=tool_name,
                tool_call_id=tool_call["id"],
            ))
    
    update_payload["messages"] = tool_messages
    
    # 우선순위: refinement > strategist > end > classifier
    # RequestRefinement가 호출되면 반드시 vision_refiner로 이동해야 함
    if goto_refiner:
        # refinement_requests를 저장하고 vision_refiner로 이동
        logger.info(f"🔍 Vision Refiner로 이동: {update_payload.get('refinement_requests', {}).get('image_paths', [])}")
        return Command(
            goto="vision_refiner",
            update={**update_payload, "current_phase": "refiner"}
        )
    
    if goto_strategist:
        return Command(
            goto="strategist",
            update={**update_payload, "current_phase": "strategist"}
        )
    
    if should_end:
        # 최종 결과 반환
        final_assignments = {**state.get("assignments", {}), **update_payload.get("assignments", {})}
        current_folders = state.get("current_folders", [])
        if not current_folders:
            current_folders = ["기타"]
        return Command(
            goto=END,
            update={
                "classifications": final_assignments,
                "categories": current_folders,
                "vision_results": state.get("refinement_results", {}),
            }
        )
    
    return Command(
        goto="classifier",
        update=update_payload,
    )


async def vision_refiner(
    state: ClassificationState, 
    config: RunnableConfig
) -> Command[Literal["classifier"]]:
    """Vision Refiner: VLM 정밀분석.
    
    Classifier가 요청한 이미지들을 고성능 VLM으로 분석합니다.
    needs_visual_refinement=True이거나 Classifier가 모호하다고 판단한 이미지들.
    """
    configuration = Configuration.from_runnable_config(config)
    
    refinement_requests = state.get("refinement_requests", {})
    image_paths = refinement_requests.get("image_paths", []) if refinement_requests else []
    questions = refinement_requests.get("questions", {}) if refinement_requests else {}
    
    logger.info(f"🔍 Vision Refiner 시작: refinement_requests={refinement_requests}")
    logger.info(f"   이미지 경로: {image_paths} (총 {len(image_paths)}장)")
    
    if not image_paths:
        # 요청이 없으면 바로 Classifier로 복귀
        logger.warning("⚠️ Vision Refiner: 분석 요청된 이미지가 없습니다.")
        return Command(
            goto="classifier",
            update={"current_phase": "classifier"}
        )
    
    # 동시성 제한 (gpt-4o TPM 제한 고려)
    # open_deep_research 방식: 동시성 제한 + with_retry 자동 재시도
    semaphore = asyncio.Semaphore(configuration.refinement_concurrency)
    
    async def process_with_semaphore(img_path: str) -> tuple[str, dict]:
        async with semaphore:
            logger.info(f"🖼️ VLM 분석 시작: {img_path}")
            # Rate Limit 방지를 위한 약간의 딜레이 (with_retry가 대부분 처리)
            await asyncio.sleep(0.5)
            
            # 안전한 실행 함수 사용 (에러 발생 시에도 계속 진행)
            result = await execute_vision_analysis_safely(img_path, config)
            
            logger.info(f"✅ VLM 분석 완료: {img_path} → {result.suggested_category} (신뢰도: {result.confidence})")
            
            return (img_path, {
                "analysis": result.model_dump(),
                "answer": questions.get(img_path, ""),
                "recommended_folder": result.suggested_category,
            })
    
    # 병렬 처리 (동시성 제한 적용, 에러는 안전한 실행 함수에서 처리)
    logger.info(f"🚀 {len(image_paths)}장의 이미지를 VLM으로 분석 시작...")
    tasks = [process_with_semaphore(img_path) for img_path in image_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 정리 (에러는 execute_vision_analysis_safely에서 이미 처리됨)
    new_results = {}
    for result in results:
        if isinstance(result, Exception):
            # 예상치 못한 에러 (실행 함수에서 잡지 못한 경우)
            logger.error(f"❌ Vision refinement 예상치 못한 에러: {result}")
            # 에러 정보를 결과에 포함 (부분 성공 허용)
            continue
        img_path, result_dict = result
        new_results[img_path] = result_dict
    
    logger.info(f"📊 Vision Refiner 완료: {len(new_results)}/{len(image_paths)}장 분석 성공")
    
    # 기존 결과에 병합
    current_results = state.get("refinement_results", {})
    merged_results = {**current_results, **new_results}
    
    logger.info(f"💾 refinement_results 저장: 총 {len(merged_results)}개 결과")
    
    # refinement_results에 recommended_folder가 있는 이미지들을 자동으로 분류
    auto_assignments = {}
    current_folders = state.get("current_folders", [])
    pending_images = state.get("pending_images", [])
    
    for img_path, result_dict in new_results.items():
        recommended_folder = result_dict.get("recommended_folder", "")
        if recommended_folder and recommended_folder in current_folders:
            auto_assignments[img_path] = recommended_folder
            logger.info(f"✅ Vision Refiner 결과로 자동 분류: {img_path} → {recommended_folder}")
    
    # 자동 분류된 이미지들을 pending에서 제거
    new_pending = [p for p in pending_images if p not in auto_assignments]
    
    # 기존 assignments에 병합
    current_assignments = state.get("assignments", {})
    merged_assignments = {**current_assignments, **auto_assignments}
    
    return Command(
        goto="classifier",
        update={
            "refinement_results": merged_results,
            "refinement_requests": {},  # 요청 처리 완료
            "assignments": merged_assignments,  # 자동 분류 결과 반영
            "pending_images": new_pending,  # pending에서 제거
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
    builder.add_node("strategist", strategist)
    builder.add_node("strategist_tools", strategist_tools)
    builder.add_node("classifier", classifier)
    builder.add_node("classifier_tools", classifier_tools)
    builder.add_node("vision_refiner", vision_refiner)
    
    # 시작점: Strategist
    builder.add_edge(START, "strategist")
    
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
