"""스크린샷 분석기의 State 정의.

그래프 구조:
- 메인 그래프: initialize → ingestion → classification_phase → END
- Phase 0 (Ingestion): 경량 VLM으로 메타데이터 추출 (Workflow)
- Phase 1 (Classification): Strategist-Classifier 자율 에이전트 루프
  - Strategist: 폴더 구조 설계 및 수정
  - Classifier: 이미지 분류 및 피드백
  - Vision Refiner: 선택적 VLM 정밀분석

출력: classifications (분류 결과), categories (폴더 목록)
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
# Phase 1: Strategist - Structured Outputs (도구)
# ============================================================

class DesignFolderStructure(BaseModel):
    """폴더 구조 설계 도구.
    
    Strategist가 전체 메타데이터를 조망하여 최적의 폴더 트리를 설계합니다.
    처음부터 중복 없는 깔끔한 구조를 만들어 CategoryMerge 불필요.
    """
    
    folder_tree: Dict[str, List[str]] = Field(
        description="폴더 구조. {메인폴더: [서브폴더1, 서브폴더2]} 형태"
    )
    folder_descriptions: Dict[str, str] = Field(
        description="각 폴더의 분류 기준 설명. {폴더명: 설명}"
    )
    reasoning: str = Field(
        description="이 구조를 선택한 이유"
    )


class ReviseStructure(BaseModel):
    """폴더 구조 수정 도구.
    
    Classifier로부터 피드백을 받아 폴더 구조를 수정합니다.
    예: "A와 B 폴더 구분이 모호해요" → 병합 또는 기준 명확화
    """
    
    changes: List[Dict[str, str]] = Field(
        description="변경 사항 리스트. [{action: 'merge'|'split'|'rename', from: ..., to: ...}]"
    )
    new_folder_tree: Dict[str, List[str]] = Field(
        description="수정된 폴더 구조"
    )
    reasoning: str = Field(
        description="수정 이유"
    )


class StrategyComplete(BaseModel):
    """Strategist 작업 완료 신호.
    
    폴더 구조 설계가 완료되어 Classifier로 넘어갈 준비가 됨.
    """
    
    final_folder_tree: Dict[str, List[str]] = Field(
        description="최종 확정된 폴더 구조"
    )
    summary: str = Field(
        description="설계 요약"
    )


# ============================================================
# Phase 1: Classifier - Structured Outputs (도구)
# ============================================================

class ClassifyImages(BaseModel):
    """이미지 분류 도구.
    
    Classifier가 메타데이터를 보고 이미지를 폴더에 배정합니다.
    확신이 높은 이미지들만 분류하고, 모호한 경우는 피드백을 남깁니다.
    """
    
    assignments: Dict[str, str] = Field(
        description="분류 결과. {이미지경로: 폴더명}"
    )
    confidence_scores: Dict[str, float] = Field(
        description="각 분류의 확신도. {이미지경로: 0.0~1.0}"
    )
    reasoning: str = Field(
        description="분류 근거 요약"
    )


class RequestRefinement(BaseModel):
    """VLM 정밀분석 요청 도구.
    
    텍스트 메타데이터만으로는 판단이 불가능할 때 VLM 정밀분석을 요청합니다.
    needs_visual_refinement=True인 이미지 또는 Classifier가 모호하다고 판단한 이미지.
    """
    
    image_paths: List[str] = Field(
        description="정밀분석이 필요한 이미지 경로들"
    )
    questions: Dict[str, str] = Field(
        description="각 이미지에 대해 VLM에게 물어볼 질문. {이미지경로: 질문}"
    )
    reason: str = Field(
        description="정밀분석이 필요한 이유"
    )


class ReportAmbiguity(BaseModel):
    """폴더 구조 피드백 도구.
    
    Classifier가 분류 중 구조적 문제를 발견했을 때 Strategist에게 피드백합니다.
    예: "A 폴더와 B 폴더 기준이 겹쳐요", "C 카테고리가 추가로 필요해요"
    """
    
    issue_type: str = Field(
        description="문제 유형: 'overlap'(겹침), 'missing'(누락), 'unclear'(불명확)"
    )
    affected_folders: List[str] = Field(
        description="문제가 있는 폴더들"
    )
    affected_images: List[str] = Field(
        description="분류가 어려운 이미지 경로들"
    )
    suggestion: str = Field(
        description="해결 제안 (예: 'A와 B를 합쳐주세요', 'C 폴더를 추가해주세요')"
    )


class ClassificationComplete(BaseModel):
    """Classification Phase 완료 신호.
    
    모든 이미지 분류가 완료되었을 때 호출.
    """
    
    summary: str = Field(
        description="분류 결과 요약 (총 이미지 수, 폴더별 분포 등)"
    )
    total_classified: int = Field(
        description="분류 완료된 이미지 수"
    )
    categories_found: List[str] = Field(
        description="최종 폴더/카테고리 목록"
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
    Phase 0 (Ingestion) → Phase 1 (Classification) → END
    """
    
    # 입력 데이터
    images: list[str]
    existing_categories: Optional[list[str]]
    
    # Phase 0 결과: Ingestion (경량 VLM 메타데이터)
    image_metadatas: Annotated[dict, override_reducer]  # {image_path: ImageMetadata.dict()}
    
    # Phase 1 결과: Vision 분석 (Refiner용, 필요한 경우만)
    vision_results: Annotated[dict, override_reducer]  # {image_path: ImageAnalysisResult.dict()}
    
    # Phase 1 결과: 분류
    classifications: Annotated[dict, override_reducer]  # {image_path: folder_name}
    categories: list[str]  # 최종 폴더/카테고리 목록


# ============================================================
# Phase 1: Classification State (통합 - Strategist + Classifier)
# ============================================================

class ClassificationState(TypedDict):
    """Classification Phase 통합 State.
    
    Strategist와 Classifier가 공유하는 단일 State.
    - Strategist: 폴더 구조 설계 및 수정
    - Classifier: 이미지 분류 및 피드백
    
    LangGraph의 Partial Update 특성을 활용하여
    각 노드는 자신이 담당하는 필드만 업데이트합니다.
    """
    
    # === Agent 통신용 메시지 ===
    messages: Annotated[list[MessageLikeRepresentation], operator.add]
    
    # === Workflow 데이터 (Ingestion에서 전달받음) ===
    images: list[str]
    image_metadatas: dict  # {image_path: ImageMetadata.dict()}
    existing_categories: Optional[list[str]]
    
    # === Strategist 관리 데이터 ===
    current_folder_tree: Dict[str, List[str]]  # {메인폴더: [서브폴더들]}
    folder_descriptions: Dict[str, str]  # {폴더명: 분류 기준}
    
    # === Classifier 관리 데이터 ===
    assignments: Annotated[dict, override_reducer]  # {image_path: folder_name}
    pending_images: list[str]  # 아직 분류 안 된 이미지들
    refinement_results: Annotated[dict, override_reducer]  # VLM 정밀분석 결과
    
    # === 피드백 루프 데이터 ===
    classification_feedback: list[str]  # Classifier → Strategist 피드백
    
    # === 안정성 및 제어 ===
    strategy_iteration: int  # Strategist 반복 횟수 (무한루프 방지)
    classify_iteration: int  # Classifier 반복 횟수
    previous_folder_tree: Optional[Dict]  # 수렴 판단용
    
    # === 제어 신호 ===
    is_converged: bool  # 수렴 완료 여부
    current_phase: str  # "strategist" | "classifier" | "refiner" | "done"
    needs_strategy_revision: bool  # Classifier가 구조 수정 요청했는지


class ClassificationOutputState(TypedDict):
    """Classification Phase 출력 State.
    
    서브그래프 완료 시 메인 그래프로 반환하는 데이터.
    """
    
    classifications: dict  # assignments를 변환하여 반환
    categories: list[str]  # folder_tree의 키들
    vision_results: dict  # refinement_results


