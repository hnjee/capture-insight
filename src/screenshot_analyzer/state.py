"""스크린샷 분석기의 State 정의.

그래프 구조:
- 메인 그래프: initialize → ingestion → classification_phase → insight_phase → final_report
- Phase 0 (Ingestion): 경량 VLM으로 메타데이터 추출 (Workflow)
- Phase 1 (Classification): Agentic 서브그래프 (supervisor ↔ tools 반복)
- Phase 2 (Insight): Agentic 서브그래프 (supervisor ↔ tools 반복)
"""

import operator
from typing import Annotated, Dict, List, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ============================================================
# Phase 0: Ingestion - 메타데이터 모델
# ============================================================

class ImageMetadata(BaseModel):
    """경량 VLM으로 추출한 이미지 메타데이터.
    
    Ingestion 단계에서 모든 이미지를 저비용으로 텍스트화하여
    이후 단계에서 이미지 없이도 분류가 가능하도록 함.
    """
    
    image_path: str = Field(description="이미지 파일 경로")
    description: str = Field(description="이미지에 대한 한 문장 요약")
    ocr_text: str = Field(description="이미지에서 추출된 주요 텍스트 (분류 힌트)")
    confidence_score: float = Field(
        description="Ingestion 분석 신뢰도 (0.0~1.0)",
        ge=0.0, 
        le=1.0
    )
    needs_visual_refinement: bool = Field(
        default=False,
        description="텍스트만으로 판단 불가 시 True (VLM 정밀분석 필요)"
    )
    suggested_categories: List[str] = Field(
        default_factory=list,
        description="추천 카테고리 힌트 (Strategist용)"
    )
    ingestion_error: Optional[str] = Field(
        default=None,
        description="Ingestion 실패 시 에러 메시지"
    )


# ============================================================
# Reducer 함수
# ============================================================

def override_reducer(current_value, new_value):
    """기존 값을 새 값으로 덮어쓰기하는 reducer.
    
    - {"type": "override", "value": ...} 형태면 완전 교체
    - 둘 다 dict면 병합
    - 그 외엔 새 값으로 교체
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    if isinstance(current_value, dict) and isinstance(new_value, dict):
        return {**current_value, **new_value}
    return new_value if new_value is not None else current_value


# ============================================================
# 데이터 모델 (분석 결과 구조)
# ============================================================

class ImageAnalysisResult(BaseModel):
    """Vision API 이미지 분석 결과."""
    
    image_path: str = Field(description="분석한 이미지 경로")
    objects: list[str] = Field(description="발견된 주요 객체들")
    scene: str = Field(description="이미지의 장면/컨텍스트")
    extracted_text: str = Field(description="추출된 텍스트 (OCR)")
    suggested_category: str = Field(description="추천 카테고리")
    confidence: float = Field(description="신뢰도 (0.0~1.0)", ge=0.0, le=1.0)


class ImageClassification(BaseModel):
    """이미지 분류 결과."""
    
    category: str = Field(description="메인 카테고리")
    sub_category: str = Field(description="세부 카테고리")
    confidence: float = Field(description="신뢰도", ge=0.0, le=1.0)
    reasoning: str = Field(description="분류 근거")


# ============================================================
# Phase 1: Classification - Structured Outputs (도구)
# ============================================================

class ConductVisionAnalysis(BaseModel):
    """Vision 분석 지시 (Supervisor → Worker).
    
    이미지들을 Vision API로 분석하도록 지시.
    """
    
    targets: list[str] = Field(
        description="분석할 이미지 경로들. 'all'이면 모든 미분석 이미지"
    )
    reason: str = Field(
        description="이 분석을 수행하는 이유"
    )


class ConductClassification(BaseModel):
    """분류 지시 (Supervisor → Worker).
    
    Vision 분석 결과를 바탕으로 이미지들을 분류하도록 지시.
    """
    
    use_existing_categories: bool = Field(
        default=False,
        description="기존 카테고리 체계를 사용할지 여부"
    )
    reason: str = Field(
        description="분류를 수행하는 이유"
    )


class ConductCategoryMerge(BaseModel):
    """카테고리 통합/정제 지시 (Supervisor → Worker).
    
    초벌 분류 후 유사한 카테고리들을 통합하고 정제하도록 지시.
    예: "패션 스타일링", "의류", "패션 광고" → "패션"으로 통합
    """
    
    reason: str = Field(
        description="카테고리 통합을 수행하는 이유"
    )


class ClassificationComplete(BaseModel):
    """Classification Phase 완료 신호.
    
    모든 이미지 분석과 분류가 완료되었을 때 호출.
    """
    
    summary: str = Field(
        description="분류 결과 요약 (총 이미지 수, 카테고리 분포 등)"
    )
    categories_found: list[str] = Field(
        description="발견된 카테고리 목록"
    )


# ============================================================
# Phase 2: Insight - Structured Outputs (도구)
# ============================================================

class ConductSearch(BaseModel):
    """웹 검색 지시 (Supervisor → Worker).
    
    특정 카테고리에 대한 인사이트를 얻기 위해 웹 검색 수행.
    """
    
    category: str = Field(
        description="검색할 카테고리"
    )
    keywords: list[str] = Field(
        description="검색에 사용할 키워드들"
    )
    reason: str = Field(
        description="이 검색을 수행하는 이유"
    )


class InsightComplete(BaseModel):
    """Insight Phase 완료 신호.
    
    모든 카테고리의 인사이트 수집이 완료되었을 때 호출.
    """
    
    summary: str = Field(
        description="인사이트 수집 결과 요약"
    )
    categories_covered: list[str] = Field(
        description="인사이트를 수집한 카테고리 목록"
    )


# ============================================================
# 메인 State (전체 워크플로우)
# ============================================================

class InputState(TypedDict):
    """외부 입력 State.
    
    사용자가 그래프 실행 시 전달하는 입력.
    """
    
    images: list[str]  # 분석할 이미지 경로들
    existing_categories: Optional[list[str]]  # 기존 카테고리 (있으면)


class ScreenshotAnalyzerState(TypedDict):
    """메인 그래프 State.
    
    전체 워크플로우에서 공유되는 상태.
    Phase 0 → Phase 1 → Phase 2 → Report 순으로 데이터가 채워짐.
    """
    
    # 입력 데이터
    images: list[str]
    existing_categories: Optional[list[str]]
    
    # Phase 0 결과: Ingestion (경량 VLM 메타데이터)
    image_metadatas: Annotated[dict, override_reducer]  # {image_path: ImageMetadata.dict()}
    
    # Phase 1 결과: Vision 분석 (Refiner용, 필요한 경우만)
    vision_results: Annotated[dict, override_reducer]  # {image_path: ImageAnalysisResult.dict()}
    
    # Phase 1 결과: 분류
    classifications: Annotated[dict, override_reducer]  # {image_path: ImageClassification.dict()}
    categories: list[str]  # 최종 카테고리 목록
    
    # Phase 2 결과: 인사이트
    category_insights: Annotated[dict, override_reducer]  # {category: insight_dict}
    
    # 최종 결과
    final_report: str


# ============================================================
# Phase 1: Classification State (서브그래프용)
# ============================================================

class ClassificationState(TypedDict):
    """Classification Phase 서브그래프 State.
    
    Supervisor ↔ Tools 반복 구조에서 사용.
    """
    
    # Supervisor 통신용 메시지
    messages: Annotated[list[MessageLikeRepresentation], operator.add]
    
    # 입력 (메인 State에서 전달받음)
    images: list[str]
    existing_categories: Optional[list[str]]
    
    # 작업 진행 상태
    analyzed_images: list[str]  # 분석 완료된 이미지들
    iteration_count: int  # 반복 횟수
    
    # 결과 (메인 State로 반환)
    vision_results: Annotated[dict, override_reducer]
    classifications: Annotated[dict, override_reducer]
    categories: list[str]


class ClassificationOutputState(TypedDict):
    """Classification Phase 출력 State.
    
    서브그래프 완료 시 메인 그래프로 반환하는 데이터.
    """
    
    vision_results: dict
    classifications: dict
    categories: list[str]


# ============================================================
# Phase 2: Insight State (서브그래프용)
# ============================================================

class InsightState(TypedDict):
    """Insight Phase 서브그래프 State.
    
    Supervisor ↔ Tools 반복 구조에서 사용.
    """
    
    # Supervisor 통신용 메시지
    messages: Annotated[list[MessageLikeRepresentation], operator.add]
    
    # 입력 (메인 State에서 전달받음)
    categories: list[str]  # 검색할 카테고리들
    classifications: dict  # 분류 결과 (참고용)
    
    # 작업 진행 상태
    searched_categories: list[str]  # 검색 완료된 카테고리들
    iteration_count: int  # 반복 횟수
    
    # 결과 (메인 State로 반환)
    category_insights: Annotated[dict, override_reducer]


class InsightOutputState(TypedDict):
    """Insight Phase 출력 State.
    
    서브그래프 완료 시 메인 그래프로 반환하는 데이터.
    """
    
    category_insights: dict
