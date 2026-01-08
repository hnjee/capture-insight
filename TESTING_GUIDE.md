# 🧪 테스트 가이드

이 프로젝트를 테스트하는 방법은 두 가지가 있습니다.

## 방법 1: Streamlit 웹앱 (로컬 실행) ⭐ 추천

### 장점
- ✅ 시각적인 UI로 결과 확인 가능
- ✅ 이미지 갤러리 미리보기
- ✅ LangSmith 트레이스 링크 자동 생성
- ✅ 결과를 탭으로 구분해서 보기 편함

### 실행 방법

1. **환경 변수 설정**
```bash
cp env.example .env
# .env 파일을 열어서 API 키 입력
```

2. **의존성 설치** (uv 사용)
```bash
uv sync
```

3. **Streamlit 실행**
```bash
uv run streamlit run app.py
```

또는 (uv 없이)
```bash
pip install -r requirements.txt
streamlit run app.py
```

4. **브라우저에서 확인**
- 자동으로 브라우저가 열립니다 (보통 `http://localhost:8501`)
- `examples/screenshots/` 폴더의 이미지들이 자동으로 로드됩니다
- "🚀 분석 시작" 버튼을 클릭하면 분석이 시작됩니다

---

## 방법 2: CLI 스크립트 (터미널 실행)

### 장점
- ✅ 빠르게 테스트 가능
- ✅ 배치 처리에 적합
- ✅ 폴더 정리 및 보고서 저장 기능

### 실행 방법

1. **환경 변수 설정** (방법 1과 동일)
```bash
cp env.example .env
# .env 파일을 열어서 API 키 입력
```

2. **기본 실행** (분석만)
```bash
uv run python scripts/run_example.py
```

3. **고급 옵션**
```bash
# 분석 + 폴더 정리 + 보고서 저장
uv run python scripts/run_example.py --organize --report

# 특정 폴더 지정
uv run python scripts/run_example.py --folder path/to/screenshots

# 이미지 이동 (복사 대신)
uv run python scripts/run_example.py --organize --move
```

---

## 방법 3: Streamlit Cloud 배포

### 장점
- ✅ 인터넷 어디서나 접근 가능
- ✅ 공유하기 편함
- ✅ 서버 관리 불필요

### 배포 방법

1. **GitHub에 푸시**
```bash
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

2. **Streamlit Cloud에서 배포**
   - https://streamlit.io/cloud 접속
   - GitHub 저장소 연결
   - `app.py` 파일 선택
   - Secrets에 API 키 추가:
     - `OPENAI_API_KEY`
     - `TAVILY_API_KEY`
     - `LANGSMITH_API_KEY` (선택)

---

## 🎯 추천 테스트 순서

1. **먼저 로컬에서 Streamlit 실행** (방법 1)
   - UI를 보면서 결과 확인
   - 문제가 있으면 빠르게 수정 가능

2. **CLI로 빠른 테스트** (방법 2)
   - 여러 이미지로 반복 테스트
   - 폴더 정리 기능 테스트

3. **만족스러우면 Streamlit Cloud 배포** (방법 3)
   - 공유하고 싶을 때

---

## ⚙️ 환경 변수 설정

`.env` 파일에 다음 키들이 필요합니다:

```env
# 필수
OPENAI_API_KEY=sk-your-openai-api-key
TAVILY_API_KEY=tvly-your-tavily-api-key

# 선택 (트레이싱용)
LANGSMITH_API_KEY=lsv2_pt_your-langsmith-api-key
LANGSMITH_PROJECT=capture-insight
LANGSMITH_TRACING=true
```

---

## 🐛 문제 해결

### "ModuleNotFoundError" 발생 시
```bash
# uv 사용
uv sync

# 또는 pip 사용
pip install -r requirements.txt
```

### "API key not found" 오류
- `.env` 파일이 프로젝트 루트에 있는지 확인
- API 키가 올바르게 입력되었는지 확인

### 이미지가 보이지 않을 때
- `examples/screenshots/` 폴더에 이미지 파일이 있는지 확인
- 지원 형식: PNG, JPG, JPEG, GIF, WEBP

