"""스크린샷 분석기의 State 정의."""

import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState, add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


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
# Structured Outputs (Pydantic)
# Supervisor가 Analyzer에게 작업 지시할 때 사용
# ============================================================

class ConductAnalysis(BaseModel):
    """분석 작업 지시 (Supervisor → Analyzer)."""
    
    task_type: Literal["vision", "classify", "web_search"] = Field(
        description="수행할 작업 유형"
    )
    target: str = Field(
        description="작업 대상 (이미지 경로 또는 검색 키워드)"
    )
    reason: str = Field(
        description="이 작업을 수행하는 이유"
    )


class AnalysisComplete(BaseModel):
    """현재 Phase 분석 완료 신호."""
    
    summary: str = Field(
        description="완료된 분석의 요약"
    )


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
# State Definitions
# ============================================================

# 1. 입력 State (외부에서 받는 필드만)
class AnalyzerInputState(MessagesState):
    """외부 입력용 State."""
    
    images: list[str]  # 분석할 이미지 경로들


# 2. 메인 Agent State
class AnalyzerState(MessagesState):
    """메인 Agent 상태 (전체 워크플로우 데이터 관리)."""
    
    # Supervisor 통신용 메시지
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    
    # 입력 데이터
    images: list[str]
    
    # 현재 Phase 추적
    current_phase: str  # "classification" | "insight"
    
    # Phase 1: 분류 결과
    vision_results: Annotated[dict, override_reducer]     # {image_path: ImageAnalysisResult}
    classifications: Annotated[dict, override_reducer]    # {image_path: ImageClassification}
    
    # Phase 2: 인사이트 결과
    category_insights: Annotated[dict, override_reducer]  # {category: insight_text}
    
    # 최종 결과
    final_report: str


# 3. Supervisor State (서브그래프 전용)
class SupervisorState(TypedDict):
    """Supervisor 서브그래프 상태.
    
    분석 전략을 수립하고 Analyzer에게 작업을 지시.
    """
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    current_phase: str  # "classification" | "insight"
    analysis_iterations: int  # 현재까지 반복 횟수
    
    # Analyzer 결과 수집
    vision_results: Annotated[dict, override_reducer]
    classifications: Annotated[dict, override_reducer]


# 4. Analyzer Worker State (서브그래프 전용)
class AnalyzerWorkerState(TypedDict):
    """개별 Analyzer 작업 상태.
    
    실제 Vision API 호출, 웹 검색 등을 수행.
    """
    
    analyzer_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    task_type: str      # "vision" | "classify" | "web_search"
    target: str         # 이미지 경로 또는 검색 키워드
    
    # 결과
    result: Optional[str]  # 압축된 텍스트 결과
    raw_data: Annotated[dict, override_reducer]  # 구조화된 원본 데이터
    
    # 반복 제한
    tool_call_iterations: int


# 5. Analyzer Output State (서브그래프 출력)
class AnalyzerOutputState(BaseModel):
    """Analyzer 서브그래프 출력."""
    
    result: str = Field(description="압축된 분석 결과")
    raw_data: dict = Field(description="구조화된 원본 데이터")
