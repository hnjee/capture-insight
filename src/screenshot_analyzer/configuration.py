"""스크린샷 분석기 시스템의 설정 관리 모듈."""

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """스크린샷 분석 Agent의 메인 설정 클래스."""
    
    # ========== Ingestion 설정 (Phase 0) ==========
    ingestion_model: str = Field(
        default="gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gpt-4o-mini",
                "description": "Ingestion용 경량 Vision 모델 (비용 절감)"
            }
        }
    )
    refinement_threshold: float = Field(
        default=0.6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.6,
                "min": 0.3,
                "max": 0.9,
                "step": 0.1,
                "description": "이 신뢰도 미만이면 VLM 정밀분석 필요"
            }
        }
    )
    ingestion_concurrency: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Ingestion 동시 처리 수 (Rate Limit 고려)"
            }
        }
    )
    
    # ========== 모델 설정 (기존) ==========
    vision_model: str = Field(
        default="gpt-4o",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gpt-4o",
                "description": "Vision Refiner용 고성능 모델 (정밀 분석)"
            }
        }
    )
    analysis_model: str = Field(
        default="gpt-4o",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gpt-4o",
                "description": "분석 및 추론용 모델 (Strategist, Classifier)"
            }
        }
    )
    max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "min": 1000,
                "max": 32000,
                "description": "일반 응답의 최대 출력 토큰 수"
            }
        }
    )
    # Agent 설정
    max_analysis_iterations: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "supervisor의 최대 분석 반복 횟수"
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """RunnableConfig에서 Configuration 인스턴스를 생성한다."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        
        values: dict[str, Any] = {}
        for field_name in field_names:
            # configurable 먼저 확인, 그 다음 환경변수 확인
            if field_name in configurable and configurable[field_name] is not None:
                values[field_name] = configurable[field_name]
            elif os.environ.get(field_name.upper()):
                env_value = os.environ.get(field_name.upper())
                if field_name in ["max_tokens", "max_analysis_iterations", 
                                   "ingestion_concurrency"]:
                    values[field_name] = int(env_value)  # type: ignore
                elif field_name == "refinement_threshold":
                    values[field_name] = float(env_value)  # type: ignore
                else:
                    values[field_name] = env_value
        
        return cls(**values)

    class Config:
        """Pydantic 설정."""
        
        arbitrary_types_allowed = True
