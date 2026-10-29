# QR Phishing Detector - 개발 로드맵 및 체크리스트

## 마일스톤 현황

- [x] **1단계: 환경 및 룰 설정**
  - [x] 프로젝트 디렉토리 구조 생성 (`docs/`, `src/`, `templates/` 등)
  - [x] 에이전트 지침 및 코딩 룰 정의 (`AGENTS.md`)
  - [x] 의존성 패키지 명세서 작성 (`requirements.txt`)
  - [x] 요구사항 명세서 초안 작성 (`docs/requirements.md`)

- [x] **2단계: ML 모델 학습 스크립트 작성 (`train.py`)**
  - [x] 피싱/정상 URL 데이터셋 수집 및 전처리 (내장 더미 60개 샘플 및 CSV 로드 파이프라인 구현)
  - [x] URL 기반 다차원 특징 엔지니어링 (길이, 서브도메인 수, 특수문자, IP 주소 사용 여부 등 17개 특징)
  - [x] Scikit-learn 분류 모델(Random Forest) 학습 및 최적화
  - [x] 모델 평가 지표 확인 (Accuracy 100%, F1-Score 1.0, 혼동 행렬 및 분류 리포트 출력)
  - [x] 학습된 모델 및 메타데이터 직렬화 저장 (`model.pkl`)

- [x] **3단계: 통합 탐지 엔진 및 DB 구현 (`src/detector.py`, `src/database.py`)**
  - [x] OpenCV 기반 QR 코드 감지 및 적응형 디코딩 엔진 구현 (Otsu 전처리 fallback 포함)
  - [x] 규칙 기반 필터링 (IP 직접 사용, HTTP 비보안, 단축 URL, 피싱 키워드, 서브도메인 깊이 검사)
  - [x] ML 모델(`model.pkl`) 예측 파이프라인 연동 및 복합 위험도 스코어링(0~100점) 로직 작성
  - [x] SQLite 데이터베이스(`phishing_history.db`) 스키마 및 CRUD 인터페이스 구현 (검사 이력, 상세 근거 관리)

- [x] **4단계: Flask 웹 UI 및 통합 테스트 구현 (`app.py`, `templates/index.html`)**
  - [x] Flask 웹 서버 라우팅 및 파일 업로드/URL 직접 분석 API 개발 (`uploads/` 안전 저장)
  - [x] 모던 글래스모피즘 웹 대시보드 UI 구현 (드래그 앤 드롭, 실시간 이미지 미리보기, 위험도 게이지 및 판정 배지)
  - [x] 전체 파이프라인 엔드투엔드(E2E) 통합 테스트 및 예외 처리 검증 완료 (GET, POST, API)
  - [x] 최근 검사 이력 조회 및 초기화 기능 연동 완료
