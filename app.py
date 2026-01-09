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

# LangSmith 공개 링크 생성용
try:
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

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
    if "public_trace_url" not in st.session_state:
        st.session_state.public_trace_url = None
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False
    
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
                st.image(img_path, width='stretch')
            except Exception:
                st.markdown("🖼️")
    
    if len(images) > 24:
        st.caption(f"... 외 {len(images) - 24}장")
    
    # ============================================================
    # 분석 버튼
    # ============================================================
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 분석 시작" if not st.session_state.is_analyzing else "⏳ 분석 중...",
            disabled=st.session_state.is_analyzing,
            width='stretch',
        )
    
    # ============================================================
    # 분석 실행
    # ============================================================
    
    if analyze_button and not st.session_state.is_analyzing:
        st.session_state.is_analyzing = True
        st.session_state.analysis_result = None
        st.session_state.public_trace_url = None
        
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
                
                # 비동기 실행 with run_id 캡처
                import uuid
                run_id = str(uuid.uuid4())
                
                async def run_graph():
                    from langchain_core.runnables import RunnableConfig
                    config = RunnableConfig(
                        run_id=run_id,
                        tags=["streamlit", "capture-insight"],
                    )
                    result = await graph.ainvoke(
                        {
                            "images": images,
                            "existing_categories": None,
                        },
                        config=config,
                    )
                    return result
                
                result = asyncio.run(run_graph())
                
                progress_bar.progress(80)
                status_text.text("📊 결과 정리 중...")
                
                # 결과 저장
                st.session_state.analysis_result = result
                st.session_state.trace_id = run_id
                
                # LangSmith 공개 링크 생성 시도
                progress_bar.progress(90)
                status_text.text("🔗 공개 링크 생성 중...")
                
                if LANGSMITH_AVAILABLE and os.environ.get("LANGSMITH_API_KEY"):
                    try:
                        ls_client = LangSmithClient()
                        # 잠시 대기 (트레이스 업로드 완료 대기)
                        import time
                        time.sleep(2)
                        
                        # 공개 링크 생성
                        public_url = ls_client.share_run(run_id)
                        st.session_state.public_trace_url = public_url
                    except Exception as e:
                        # 공개 링크 생성 실패해도 분석 결과는 표시
                        st.warning(f"공개 링크 생성 실패: {e}")
                
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
        
        st.markdown("---")
        st.markdown("## 📊 분석 결과")
        
        # 탭으로 결과 구분
        tab1, tab2 = st.tabs(["📁 폴더구조", "🖼️ 이미지별 폴더 분류된 결과"])
        
        # 탭 1: 폴더구조
        with tab1:
            st.markdown("### 📂 폴더 구조")
            
            # 폴더 트리 표시
            folder_tree = build_folder_tree(classifications)
            st.code(folder_tree, language=None)
        
        # 탭 2: 이미지별 폴더 분류된 결과
        with tab2:
            st.markdown("### 🖼️ 이미지별 폴더 분류 결과")
            
            if classifications:
                # 카테고리별로 그룹화
                category_groups = {}
                for img_path, cls in classifications.items():
                    if isinstance(cls, dict):
                        category = cls.get("category", "기타")
                    elif isinstance(cls, str):
                        category = cls
                    else:
                        category = "기타"
                    
                    if category not in category_groups:
                        category_groups[category] = []
                    category_groups[category].append(img_path)
                
                # 카테고리별로 표시
                for category in sorted(category_groups.keys()):
                    with st.expander(f"📂 {category} ({len(category_groups[category])}장)", expanded=True):
                        # 이미지 그리드
                        images_in_category = category_groups[category]
                        cols = st.columns(4)
                        for i, img_path in enumerate(images_in_category):
                            with cols[i % 4]:
                                try:
                                    st.image(img_path, use_container_width=True)
                                    st.caption(Path(img_path).name)
                                except Exception:
                                    st.markdown(f"🖼️ {Path(img_path).name}")
            else:
                st.info("분류된 이미지가 없습니다.")
        
        # ============================================================
        # LangSmith 공개 링크 (있을 때만 표시)
        # ============================================================
        
        if st.session_state.public_trace_url:
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <a href="{st.session_state.public_trace_url}" target="_blank" style="
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
                        🔗 LangSmith 트레이스 보기 (공개 링크)
                    </a>
                    <p style="color: #10b981; margin-top: 0.5rem; font-size: 0.9rem; font-weight: 600;">
                        ✅ 로그인 없이 누구나 볼 수 있습니다!
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # 다시 분석 버튼
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 다시 분석하기", width='stretch'):
                st.session_state.analysis_result = None
                st.session_state.trace_id = None
                st.rerun()


if __name__ == "__main__":
    main()

