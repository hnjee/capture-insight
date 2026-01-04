"""스크린샷 분석기 그래프의 State 정의."""

import operator
from typing import Annotated, Literal

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ============================================================
# Reducer 함수들
# ============================================================

def override_reducer(existing: dict | None, new: dict | None) -> dict:
    """기존 dict를 새 dict로 병합/덮어쓰기하는 reducer.
    
    LangGraph에서 State 필드 업데이트 시 사용.
    새 값이 들어오면 기존 값과 병합하고, 중복 키는 덮어씀.
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    return {**existing, **new}


# ============================================================
# 메인 State
# ============================================================

class ScreenshotAnalyzerState(TypedDict):
    """스크린샷 분석기의 메인 State.
    
    LangGraph 노드들이 공유하는 상태 정의.
    Annotated로 각 필드의 reducer를 지정함.
    """
    
    # 메시지 히스토리 (add_messages reducer로 누적)
    messages: Annotated[list, add_messages]
    
    # 입력: 분석할 이미지 경로 목록
    images: list[str]
    
    # Vision API 분석 결과 {이미지경로: 분석결과}
    vision_results: Annotated[dict, override_reducer]
    
    # 최종 분류 결과 {이미지경로: 분류정보}
    classifications: Annotated[dict, override_reducer]
    
    # 현재 Phase ("classification" 또는 "insight")
    current_phase: str
    
    # 카테고리별 인사이트 {카테고리: 인사이트}
    category_insights: Annotated[dict, override_reducer]
    
    # 최종 보고서
    final_report: str


# ============================================================
# Structured Outputs (Pydantic 모델)
# supervisor가 사용하는 도구 스키마
# ============================================================

class ConductAnalysis(BaseModel):
    """supervisor가 researcher에게 분석 작업을 지시할 때 사용."""
    
    task_type: Literal["vision", "classify", "web_search"] = Field(
        description="수행할 작업 유형: vision(이미지 분석), classify(분류), web_search(웹 검색)"
    )
    target: str = Field(
        description="작업 대상 (이미지 경로 또는 검색 키워드)"
    )
    reason: str = Field(
        description="이 작업을 수행하는 이유"
    )


class AnalysisComplete(BaseModel):
    """supervisor가 현재 Phase의 분석 완료를 알릴 때 사용."""
    
    summary: str = Field(
        description="완료된 분석의 요약"
    )


class ImageClassification(BaseModel):
    """이미지 분류 결과 구조."""
    
    category: str = Field(
        description="메인 카테고리 (예: 쇼핑, 뉴스, SNS, 업무)"
    )
    sub_category: str = Field(
        description="세부 카테고리 (예: 의류, 전자제품, 스포츠뉴스)"
    )
    confidence: float = Field(
        description="분류 신뢰도 (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="분류 근거 설명"
    )


class VisionAnalysisResult(BaseModel):
    """Vision API 분석 결과 구조."""
    
    objects: list[str] = Field(
        description="이미지에서 발견된 주요 객체들"
    )
    scene: str = Field(
        description="이미지의 전체적인 장면/컨텍스트"
    )
    extracted_text: str = Field(
        description="이미지에서 추출된 텍스트 (OCR)"
    )
    suggested_category: str = Field(
        description="추천 카테고리"
    )
    description: str = Field(
        description="이미지에 대한 상세 설명"
    )
