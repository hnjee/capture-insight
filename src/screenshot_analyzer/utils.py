"""스크린샷 분석기의 유틸리티 함수들."""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from screenshot_analyzer.configuration import Configuration
from screenshot_analyzer.prompts import INGESTION_PROMPT, VISION_ANALYSIS_PROMPT
from screenshot_analyzer.state import IngestionMetadata, RefinementResult

# 로깅 설정
logger = logging.getLogger(__name__)

# ============================================================
# 전역 모델 인스턴스 (재사용)
# ============================================================

configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


# ============================================================
# Phase 0: Ingestion 함수 (경량 VLM)
# ============================================================

async def ingest_image(
    image_path: str,
    config: Optional[RunnableConfig] = None
) -> IngestionMetadata:
    """경량 VLM으로 단일 이미지의 메타데이터 추출.
    
    비용 효율적인 경량 모델(gpt-4o-mini 등)을 사용하여
    이미지를 텍스트 메타데이터로 변환합니다.
    
    Args:
        image_path: 분석할 이미지 경로
        config: LangGraph 런타임 설정
        
    Returns:
        IngestionMetadata 객체
    """
    configuration = Configuration.from_runnable_config(config)
    
    # 이미지를 base64로 인코딩
    try:
        image_base64 = load_image_as_base64(image_path)
        media_type = get_image_media_type(image_path)
    except FileNotFoundError as e:
        return IngestionMetadata(
            image_path=image_path,
            description="파일을 찾을 수 없음",
            ocr_text="",
            confidence_score=0.0,
            needs_visual_refinement=True,
            suggested_categories=[],
            ingestion_error=str(e)
        )
    
    # 경량 Vision 모델 설정 (with_retry로 자동 재시도)
    api_key = get_api_key_for_model(configuration.ingestion_model, config)
    model_config = {
        "model": configuration.ingestion_model,
        "max_tokens": 1024,  # 메타데이터 추출이므로 작은 토큰으로 충분
        "api_key": api_key,
    }
    # ✅ with_retry() 추가: rate limit 자동 처리
    model = (
        configurable_model
        .with_config(model_config)
        .with_retry(stop_after_attempt=configuration.max_structured_output_retries)
    )
    
    # 프롬프트 구성 (refinement_threshold 주입)
    prompt = INGESTION_PROMPT.format(
        refinement_threshold=configuration.refinement_threshold
    )
    
    # Vision API 호출 (rate limit은 with_retry가 자동 처리)
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_base64}"
                }
            }
        ]
    )
    
    try:
        response = await model.ainvoke([message])
        
        # JSON 응답 파싱
        result_dict = parse_json_response(response.content)
        
        confidence = result_dict.get("confidence_score", 0.5)
        
        return IngestionMetadata(
            image_path=image_path,
            description=result_dict.get("description", ""),
            ocr_text=result_dict.get("ocr_text", ""),
            confidence_score=confidence,
            needs_visual_refinement=result_dict.get(
                "needs_visual_refinement", 
                confidence < configuration.refinement_threshold
            ),
            suggested_categories=result_dict.get("suggested_categories", []),
            ingestion_error=None
        )
        
    except Exception as e:
        # with_retry가 모든 재시도를 시도했지만 실패한 경우
        logger.error(f"Ingestion 실패 ({image_path}): {e}")
        return IngestionMetadata(
            image_path=image_path,
            description="분석 실패",
            ocr_text="",
            confidence_score=0.0,
            needs_visual_refinement=True,
            suggested_categories=[],
            ingestion_error=str(e)
        )


async def execute_ingestion_safely(
    img_path: str, 
    config: Optional[RunnableConfig] = None
) -> tuple[str, IngestionMetadata]:
    """안전하게 Ingestion을 실행 (에러 발생 시에도 계속 진행).
    
    open_deep_research의 execute_tool_safely 패턴을 따름.
    에러 발생 시 기본값을 반환하여 프로세스가 중단되지 않도록 함.
    
    Args:
        img_path: 이미지 경로
        config: LangGraph 런타임 설정
        
    Returns:
        (image_path, IngestionMetadata) 튜플
    """
    try:
        metadata = await ingest_image(img_path, config)
        return (img_path, metadata)
    except Exception as e:
        logger.warning(f"Ingestion 에러 ({img_path}): {e}")
        # 에러 발생 시 기본값 반환 (프로세스 중단 없음)
        return (img_path, IngestionMetadata(
            image_path=img_path,
            description="분석 실패",
            ocr_text="",
            confidence_score=0.0,
            needs_visual_refinement=True,
            suggested_categories=[],
            ingestion_error=str(e)
        ))


async def batch_ingestion(
    images: List[str],
    config: Optional[RunnableConfig] = None
) -> dict[str, IngestionMetadata]:
    """모든 이미지를 일괄 Ingestion하여 메타데이터 추출.
    
    Args:
        images: 이미지 경로 리스트
        config: LangGraph 런타임 설정
        
    Returns:
        {image_path: IngestionMetadata} 딕셔너리
    """
    configuration = Configuration.from_runnable_config(config)
    
    # 동시 처리 수 제한 (Rate Limit 발생 빈도 감소)
    semaphore = asyncio.Semaphore(configuration.ingestion_concurrency)
    
    async def process_with_semaphore(img_path: str) -> tuple[str, IngestionMetadata]:
        async with semaphore:
            # Rate Limit 방지를 위한 약간의 딜레이 (with_retry가 대부분 처리하지만 여전히 필요)
            await asyncio.sleep(0.2)
            # 안전한 실행 함수 사용 (에러 발생 시에도 계속 진행)
            return await execute_ingestion_safely(img_path, config)
    
    # 병렬 처리 (에러는 return_exceptions=True로 처리)
    tasks = [process_with_semaphore(img) for img in images]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 정리 (에러는 execute_ingestion_safely에서 이미 처리됨)
    metadata_dict: dict[str, IngestionMetadata] = {}
    for result in results:
        if isinstance(result, Exception):
            # 예상치 못한 에러 (실행 함수에서 잡지 못한 경우)
            logger.error(f"Batch ingestion 예상치 못한 에러: {result}")
            continue
        img_path, metadata = result
        metadata_dict[img_path] = metadata
    
    # 통계 로깅
    total = len(images)
    success = len(metadata_dict)
    need_refinement = sum(1 for m in metadata_dict.values() if m.needs_visual_refinement)
    
    logger.info(
        f"Ingestion 완료: {success}/{total}장 성공, "
        f"{need_refinement}장 정밀분석 필요"
    )
    
    return metadata_dict

# ============================================================
# 안전한 실행 함수 (open_deep_research 패턴)
# ============================================================

async def execute_ingestion_safely(
    img_path: str, 
    config: Optional[RunnableConfig] = None
) -> tuple[str, IngestionMetadata]:
    """안전하게 Ingestion을 실행 (에러 발생 시에도 계속 진행).
    
    open_deep_research의 execute_tool_safely 패턴을 따름.
    에러 발생 시 기본값을 반환하여 프로세스가 중단되지 않도록 함.
    
    Args:
        img_path: 이미지 경로
        config: LangGraph 런타임 설정
        
    Returns:
        (image_path, IngestionMetadata) 튜플
    """
    try:
        metadata = await ingest_image(img_path, config)
        return (img_path, metadata)
    except Exception as e:
        logger.warning(f"Ingestion 에러 ({img_path}): {e}")
        # 에러 발생 시 기본값 반환 (프로세스 중단 없음)
        return (img_path, IngestionMetadata(
            image_path=img_path,
            description="분석 실패",
            ocr_text="",
            confidence_score=0.0,
            needs_visual_refinement=True,
            suggested_categories=[],
            ingestion_error=str(e)
        ))


async def execute_vision_analysis_safely(
    img_path: str,
    config: Optional[RunnableConfig] = None
) -> RefinementResult:
    """안전하게 Vision 분석을 실행 (에러 발생 시에도 계속 진행).
    
    open_deep_research의 execute_tool_safely 패턴을 따름.
    에러 발생 시 기본값을 반환하여 프로세스가 중단되지 않도록 함.
    
    Args:
        img_path: 이미지 경로
        config: LangGraph 런타임 설정
        
    Returns:
        RefinementResult 객체
    """
    try:
        return await analyze_image(img_path, config)
    except Exception as e:
        logger.warning(f"Vision 분석 에러 ({img_path}): {e}")
        # 에러 발생 시 기본값 반환 (프로세스 중단 없음)
        return RefinementResult(
            image_path=img_path,
            objects=[],
            scene="분석 실패",
            extracted_text="",
            suggested_category="기타",
            confidence=0.0
        )


# ============================================================
# 이미지 처리 유틸
# ============================================================

def load_image_as_base64(image_path: str) -> str:
    """이미지 파일을 base64 문자열로 인코딩.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        base64로 인코딩된 이미지 문자열
        
    Raises:
        FileNotFoundError: 이미지 파일이 없을 경우
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    with open(path, "rb") as f:
        image_data = f.read()
    
    return base64.b64encode(image_data).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """이미지 파일의 MIME 타입 반환.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        MIME 타입 문자열 (예: "image/png")
    """
    extension = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(extension, "image/png")


async def analyze_image(
    image_path: str,
    config: Optional[RunnableConfig] = None
) -> RefinementResult:
    """Vision Refiner: 고성능 VLM으로 이미지 정밀 분석.
    
    needs_visual_refinement=True인 이미지에 대해 상세 분석을 수행합니다.
    객체, 장면, 텍스트, 카테고리 등을 추출합니다.
    
    open_deep_research 방식: with_retry()로 자동 재시도 처리
    
    Args:
        image_path: 분석할 이미지 경로
        config: LangGraph 런타임 설정
        
    Returns:
        RefinementResult 객체
    """
    configuration = Configuration.from_runnable_config(config)
    
    # 이미지를 base64로 인코딩
    try:
        image_base64 = load_image_as_base64(image_path)
        media_type = get_image_media_type(image_path)
    except FileNotFoundError as e:
        return RefinementResult(
            image_path=image_path,
            objects=[],
            scene="파일을 찾을 수 없음",
            extracted_text="",
            suggested_category="기타",
            confidence=0.0
        )
    
    # Vision 모델 설정 (with_retry로 자동 재시도)
    api_key = get_api_key_for_model(configuration.vision_model, config)
    model_config = {
        "model": configuration.vision_model,
        "max_tokens": configuration.max_tokens,
        "api_key": api_key,
    }
    # with_retry() 추가: rate limit 자동 처리
    model = (
        configurable_model
        .with_config(model_config)
        .with_retry(stop_after_attempt=configuration.max_structured_output_retries)
    )
    
    # Vision API 호출 (rate limit은 with_retry가 자동 처리)
    message = HumanMessage(
        content=[
            {"type": "text", "text": VISION_ANALYSIS_PROMPT},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_base64}"
                }
            }
        ]
    )
    
    try:
        response = await model.ainvoke([message])
        
        # JSON 응답 파싱
        result_dict = parse_json_response(response.content)
        
        return RefinementResult(
            image_path=image_path,
            objects=result_dict.get("objects", []),
            scene=result_dict.get("scene", ""),
            extracted_text=result_dict.get("extracted_text", ""),
            suggested_category=result_dict.get("suggested_category", "기타"),
            confidence=result_dict.get("confidence", 0.5)
        )
        
    except Exception as e:
        # with_retry가 모든 재시도를 시도했지만 실패한 경우
        logger.error(f"이미지 분석 실패 ({image_path}): {e}")
        return RefinementResult(
            image_path=image_path,
            objects=[],
            scene="분석 실패",
            extracted_text="",
            suggested_category="기타",
            confidence=0.0
        )


def parse_json_response(content: str) -> dict:
    """LLM 응답에서 JSON 추출.
    
    응답이 JSON 코드 블록으로 감싸져 있을 수 있으므로 처리합니다.
    
    Args:
        content: LLM 응답 문자열
        
    Returns:
        파싱된 dict
    """
    # JSON 코드 블록 제거
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        logger.warning(f"JSON 파싱 실패: {content[:100]}...")
        return {}


# ============================================================
# 모델 및 API 키 관련 유틸
# ============================================================

def get_api_key_for_model(model_name: str, config: Optional[RunnableConfig] = None) -> Optional[str]:
    """모델에 맞는 API 키 반환.
    
    Args:
        model_name: 모델 이름 (예: "gpt-4o", "openai:gpt-4o")
        config: 런타임 설정
        
    Returns:
        API 키 문자열 또는 None
    """
    model_lower = model_name.lower()
    
    # config에서 API 키 확인
    if config:
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if api_keys:
            if "openai" in model_lower or "gpt" in model_lower:
                return api_keys.get("OPENAI_API_KEY")
            elif "anthropic" in model_lower or "claude" in model_lower:
                return api_keys.get("ANTHROPIC_API_KEY")
            elif "google" in model_lower or "gemini" in model_lower:
                return api_keys.get("GOOGLE_API_KEY")
    
    # 환경변수에서 API 키 확인
    if "openai" in model_lower or "gpt" in model_lower:
        return os.getenv("OPENAI_API_KEY")
    elif "anthropic" in model_lower or "claude" in model_lower:
        return os.getenv("ANTHROPIC_API_KEY")
    elif "google" in model_lower or "gemini" in model_lower:
        return os.getenv("GOOGLE_API_KEY")
    
    return None