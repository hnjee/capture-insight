# capture-insight

스크린샷을 자동으로 분류하고 인사이트를 제공하는 AI 에이전트 (LangGraph 기반)

## 개요

스크린샷 이미지들을 분석하여 자동으로 카테고리를 분류하고, 각 카테고리에 대한 웹 기반 인사이트를 수집하여 종합 보고서를 생성합니다.

## 기술 스택

- **LangGraph**: 멀티 에이전트 워크플로우 오케스트레이션
- **OpenAI GPT-4o**: Vision API 기반 이미지 분석 및 OCR
- **Tavily**: 실시간 웹 검색

## 그래프 구조

```
START
  ↓
initialize (상태 초기화)
  ↓
classification_phase (서브그래프)
  ├─ classification_supervisor ←─┐
  └─ classification_tools ────────┘ (agentic loop)
  ↓
insight_phase (서브그래프)
  ├─ insight_supervisor ←─┐
  └─ insight_tools ────────┘ (agentic loop)
  ↓
final_report (보고서 생성)
  ↓
END
```

## 설치

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 환경 변수 설정
cp env.example .env
# .env 파일을 열고 API 키 입력
```

## 환경 변수

| 변수명 | 필수 | 설명 |
|--------|------|------|
| `OPENAI_API_KEY` | ✅ | OpenAI API 키 (Vision + 분석) |
| `TAVILY_API_KEY` | ✅ | Tavily 웹 검색 API 키 |
| `VISION_MODEL` | ❌ | Vision 모델 (기본: gpt-4o) |
| `ANALYSIS_MODEL` | ❌ | 분석 모델 (기본: gpt-4o) |
| `MAX_ANALYSIS_ITERATIONS` | ❌ | 최대 반복 횟수 (기본: 10) |

## 사용법

### LangGraph Studio에서 실행

```bash
langgraph dev
```

### 코드에서 실행

```python
import asyncio
from screenshot_analyzer.analyzer import graph

async def main():
    result = await graph.ainvoke({
        "images": [
            "path/to/screenshot1.png",
            "path/to/screenshot2.jpg",
        ],
        "existing_categories": None  # 또는 기존 카테고리 리스트
    })
    
    print(result["final_report"])

asyncio.run(main())
```

## 입력/출력

### 입력 (InputState)

```python
{
    "images": ["이미지 경로 리스트"],
    "existing_categories": ["기존 카테고리"] | None
}
```

### 출력 (ScreenshotAnalyzerState)

```python
{
    "classifications": {
        "이미지경로": {
            "category": "쇼핑",
            "sub_category": "의류",
            "confidence": 0.95,
            "reasoning": "상품 이미지와 가격 정보가 표시됨"
        }
    },
    "category_insights": {
        "쇼핑": {
            "trends": ["트렌드1", "트렌드2"],
            "recommendations": ["추천1", "추천2"]
        }
    },
    "final_report": "마크다운 형식의 종합 보고서"
}
```

## 라이선스

MIT
