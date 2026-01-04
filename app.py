"""Capture Insight - 스크린샷 분석 웹앱.

Streamlit 기반 웹앱으로 스크린샷을 분석하고 분류 결과를 시각화합니다.
"""

import asyncio
import base64
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

# .env 로드 (로컬 개발용)
load_dotenv(project_root / ".env")

# Streamlit Cloud에서는 secrets 사용
if hasattr(st, "secrets"):
    for key in ["OPENAI_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
    # LangSmith 설정
    if "LANGSMITH_API_KEY" in st.secrets:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = st.secrets.get("LANGSMITH_PROJECT", "capture-insight")

from screenshot_analyzer.analyzer import graph

# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="📸 Capture Insight",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 스타일
# ============================================================

st.markdown("""
<style>
    /* 메인 컨테이너 */
    .main > div {
        padding-top: 2rem;
    }
    
    /* 헤더 스타일 */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    .main-header h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #6b7280;
        font-size: 1.1rem;
    }
    
    /* 이미지 그리드 */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 0.75rem;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 12px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .image-item {
        aspect-ratio: 1;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .image-item:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .image-item img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    /* 결과 카드 */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    
    /* 카테고리 뱃지 */
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* 폴더 구조 스타일 */
    .folder-tree {
        font-family: 'SF Mono', 'Monaco', monospace;
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .folder-name {
        color: #569cd6;
    }
    
    .file-name {
        color: #ce9178;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 유틸리티 함수
# ============================================================

def find_images(folder: str) -> list[str]:
    """폴더에서 이미지 파일들을 찾습니다."""
    extensions = [
        "*.png", "*.PNG", "*.jpg", "*.JPG", 
        "*.jpeg", "*.JPEG", "*.gif", "*.GIF",
        "*.webp", "*.WEBP",
    ]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(set(images))


def get_image_base64(image_path: str) -> str:
    """이미지를 base64로 인코딩합니다."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_langsmith_trace_url(run_id: str) -> str:
    """LangSmith 트레이스 URL을 생성합니다."""
    project = os.environ.get("LANGSMITH_PROJECT", "capture-insight")
    return f"https://smith.langchain.com/public/{run_id}/r"


def build_folder_tree(classifications: dict) -> str:
    """분류 결과를 폴더 트리 형태로 변환합니다."""
    # 카테고리별로 그룹화
    tree = {}
    for img_path, cls in classifications.items():
        if not isinstance(cls, dict):
            continue
        category = cls.get("category", "기타")
        sub_category = cls.get("sub_category", "")
        
        if category not in tree:
            tree[category] = {}
        if sub_category:
            if sub_category not in tree[category]:
                tree[category][sub_category] = []
            tree[category][sub_category].append(Path(img_path).name)
        else:
            if "_files" not in tree[category]:
                tree[category]["_files"] = []
            tree[category]["_files"].append(Path(img_path).name)
    
    # 트리 문자열 생성
    lines = ["📁 분류된 스크린샷/"]
    for category in sorted(tree.keys()):
        lines.append(f"├── 📂 {category}/")
        sub_items = tree[category]
        sub_keys = sorted([k for k in sub_items.keys() if k != "_files"])
        
        for i, sub_cat in enumerate(sub_keys):
            is_last_sub = (i == len(sub_keys) - 1) and "_files" not in sub_items
            prefix = "│   └──" if is_last_sub else "│   ├──"
            lines.append(f"{prefix} 📂 {sub_cat}/")
            
            files = sub_items[sub_cat]
            for j, file in enumerate(files[:3]):  # 최대 3개만 표시
                file_prefix = "│   │   └──" if j == min(2, len(files)-1) else "│   │   ├──"
                if is_last_sub:
                    file_prefix = file_prefix.replace("│   │", "    │")
                lines.append(f"{file_prefix} 🖼️ {file}")
            if len(files) > 3:
                lines.append(f"│   │       ... 외 {len(files)-3}개")
        
        if "_files" in sub_items:
            for j, file in enumerate(sub_items["_files"][:3]):
                file_prefix = "│   └──" if j == min(2, len(sub_items["_files"])-1) else "│   ├──"
                lines.append(f"{file_prefix} 🖼️ {file}")
    
    return "\n".join(lines)


# ============================================================
# 메인 앱
# ============================================================

def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📸 Capture Insight</h1>
        <p>스크린샷을 AI가 자동으로 분류하고 인사이트를 제공합니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    screenshots_folder = project_root / "examples" / "screenshots"
    
    # 이미지 목록 가져오기
    if screenshots_folder.exists():
        images = find_images(str(screenshots_folder))
    else:
        images = []
    
    # 세션 상태 초기화
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "trace_id" not in st.session_state:
        st.session_state.trace_id = None
    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False
    
    # ============================================================
    # 이미지 갤러리
    # ============================================================
    
    st.markdown("### 📷 분석할 스크린샷")
    
    if not images:
        st.warning("⚠️ `examples/screenshots/` 폴더에 이미지를 추가해주세요!")
        st.info("""
        ```bash
        mkdir -p examples/screenshots
        # 스크린샷 이미지들을 해당 폴더에 복사
        ```
        """)
        return
    
    st.markdown(f"**{len(images)}장**의 스크린샷이 준비되어 있습니다.")
    
    # 이미지 그리드 표시
    cols = st.columns(8)
    for i, img_path in enumerate(images[:24]):  # 최대 24개 미리보기
        with cols[i % 8]:
            try:
                st.image(img_path, use_container_width=True)
            except Exception:
                st.markdown("🖼️")
    
    if len(images) > 24:
        st.caption(f"... 외 {len(images) - 24}장")
    
    # ============================================================
    # 분석 버튼
    # ============================================================
    
    st.markdown("---")
    
    # 분석 중일 때 LangSmith 링크 표시
    if st.session_state.is_analyzing:
        langsmith_project = os.environ.get("LANGSMITH_PROJECT", "capture-insight")
        langsmith_url = f"https://smith.langchain.com/o/default/projects/p/{langsmith_project}"
        
        if os.environ.get("LANGSMITH_API_KEY"):
            st.markdown(f"""
            <div style="background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <strong>🔗 실시간 트레이스 확인:</strong> 
                <a href="{langsmith_url}" target="_blank" style="color: #3b82f6; text-decoration: none; font-weight: 600;">
                    LangSmith에서 분석 과정 보기 (새 창)
                </a>
                <br>
                <small style="color: #6b7280;">분석이 진행되는 동안 실시간으로 에이전트 실행 과정을 확인할 수 있습니다</small>
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 분석 시작" if not st.session_state.is_analyzing else "⏳ 분석 중...",
            disabled=st.session_state.is_analyzing,
            use_container_width=True,
        )
    
    # ============================================================
    # 분석 실행
    # ============================================================
    
    if analyze_button and not st.session_state.is_analyzing:
        st.session_state.is_analyzing = True
        st.session_state.analysis_result = None
        
        # LangSmith 프로젝트 링크 (분석 시작 시 즉시 표시)
        langsmith_project = os.environ.get("LANGSMITH_PROJECT", "capture-insight")
        langsmith_url = f"https://smith.langchain.com/o/default/projects/p/{langsmith_project}"
        
        # 진행 상황 표시
        progress_container = st.container()
        with progress_container:
            st.markdown("### ⏳ 분석 진행 중...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 이미지 분석 준비 중...")
            progress_bar.progress(10)
            
            try:
                # LangGraph 실행
                status_text.text(f"🤖 {len(images)}장의 이미지 분석 중... (약 1-2분 소요)")
                progress_bar.progress(30)
                
                # 비동기 실행
                async def run_graph():
                    result = await graph.ainvoke({
                        "images": images,
                        "existing_categories": None,
                    })
                    return result
                
                result = asyncio.run(run_graph())
                
                progress_bar.progress(90)
                status_text.text("📊 결과 정리 중...")
                
                # 결과 저장
                st.session_state.analysis_result = result
                
                # LangSmith trace ID 추출 (환경변수에서)
                # 실제로는 callback에서 가져와야 하지만, 여기서는 프로젝트 링크 사용
                st.session_state.trace_id = os.environ.get("LANGSMITH_PROJECT", "capture-insight")
                
                progress_bar.progress(100)
                status_text.text("✅ 분석 완료!")
                
            except Exception as e:
                st.error(f"❌ 분석 중 오류 발생: {str(e)}")
                st.session_state.is_analyzing = False
                return
            
            st.session_state.is_analyzing = False
            st.rerun()
    
    # ============================================================
    # 결과 표시
    # ============================================================
    
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        classifications = result.get("classifications", {})
        insights = result.get("category_insights", {})
        report = result.get("final_report", "")
        
        st.markdown("---")
        st.markdown("## 📊 분석 결과")
        
        # 탭으로 결과 구분
        tab1, tab2, tab3 = st.tabs(["📁 폴더 분류", "💡 인사이트", "📄 보고서"])
        
        # 탭 1: 폴더 분류 결과
        with tab1:
            st.markdown("### 📂 카테고리별 분류 결과")
            
            # 폴더 트리 표시
            folder_tree = build_folder_tree(classifications)
            st.code(folder_tree, language=None)
            
            # 카테고리별 통계
            category_counts = {}
            for img, cls in classifications.items():
                if isinstance(cls, dict):
                    cat = cls.get("category", "기타")
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            st.markdown("#### 📈 카테고리별 통계")
            cols = st.columns(min(len(category_counts), 4))
            for i, (cat, count) in enumerate(sorted(category_counts.items(), key=lambda x: -x[1])):
                with cols[i % 4]:
                    st.metric(cat, f"{count}장")
        
        # 탭 2: 인사이트
        with tab2:
            st.markdown("### 💡 카테고리별 인사이트")
            
            if insights:
                for category, insight_data in insights.items():
                    with st.expander(f"📂 {category}", expanded=True):
                        if isinstance(insight_data, dict):
                            # 트렌드
                            if "trends" in insight_data:
                                st.markdown("**🔥 트렌드**")
                                for trend in insight_data.get("trends", []):
                                    st.markdown(f"- {trend}")
                            
                            # 추천
                            if "recommendations" in insight_data:
                                st.markdown("**💡 추천**")
                                for rec in insight_data.get("recommendations", []):
                                    st.markdown(f"- {rec}")
                            
                            # 요약
                            if "summary" in insight_data:
                                st.markdown("**📝 요약**")
                                st.markdown(insight_data.get("summary", ""))
                        else:
                            st.write(insight_data)
            else:
                st.info("인사이트 정보가 없습니다.")
        
        # 탭 3: 보고서
        with tab3:
            st.markdown("### 📄 분석 보고서")
            if report:
                st.markdown(report)
            else:
                st.info("보고서가 생성되지 않았습니다.")
        
        # ============================================================
        # LangSmith 링크
        # ============================================================
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # LangSmith 프로젝트 페이지 URL
            langsmith_project = os.environ.get("LANGSMITH_PROJECT", "capture-insight")
            langsmith_url = f"https://smith.langchain.com/o/default/projects/p/{langsmith_project}"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <a href="{langsmith_url}" target="_blank" style="
                    display: inline-block;
                    padding: 0.75rem 2rem;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    font-weight: 600;
                    font-size: 1.1rem;
                    transition: transform 0.2s;
                ">
                    🔗 LangSmith에서 자세히 보기
                </a>
                <p style="color: #6b7280; margin-top: 0.5rem; font-size: 0.9rem;">
                    최신 분석 트레이스 확인 (LangSmith 로그인 필요)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # 다시 분석 버튼
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 다시 분석하기", use_container_width=True):
                st.session_state.analysis_result = None
                st.session_state.trace_id = None
                st.rerun()


if __name__ == "__main__":
    main()

