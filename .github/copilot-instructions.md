# Copilot 사용 지침 (간단 요약)

아래 지침은 이 리포지토리에서 AI 코딩 에이전트가 빠르게 생산적으로 작업하도록 돕기 위한 실무 중심 요약입니다.

**프로젝트 개요**
- 이 저장소는 간단한 Streamlit 기반의 주식 내재가치 분석 앱입니다. 진입점는 [app.py](app.py).
- 상태/비밀값은 Streamlit Secrets(`.streamlit/secrets.toml`) 또는 환경변수(`.env`)로 관리되며, 비밀 우선순위는 `st.secrets` → `os.environ` 입니다 ([config.py](config.py)).
- 영구 데이터 저장은 로컬 SQLite를 사용하며 기본 DB 경로는 `data/app.db`입니다 ([app.py](app.py)).

**핵심 파일과 역할**
- [app.py](app.py): Streamlit UI 및 간단한 DB healthcheck. 앱 실행과 Secrets/Env 사용 예시가 포함.
- [config.py](config.py): `.env` 로드(UTF-8/CP949 폴백)와 `get_secret()` 헬퍼. 비밀/환경값 조회 시 이 함수를 사용하세요.
- [skill.md](skill.md): 도메인 로직(내재가치 계산 공식, 데이터 전처리 규칙, 출력 형식). 비즈니스 로직 변경 시 이 문서를 근거로 코드 변경 필요.
- `requirements.txt`: 최소 의존성(`streamlit`).

**개발/실행 요령 (명확한 명령어)**
- 로컬 실행: `streamlit run app.py`
- 로컬 환경 변수 사용: 프로젝트 루트에 `.env` 파일 생성(`KEY=VALUE` 형태). `config.py`가 자동 로드합니다.
- Streamlit Cloud 배포 시에는 `APP_ENV`, `DB_PATH`, 기타 API 키를 Streamlit Secrets에 설정하세요. `get_secret()`가 우선 사용합니다.

**프로젝트 특이 사항 및 규칙(에이전트가 반드시 알아야 할 것)**
- `.env` 파일 인코딩: Windows에서 CP949로 저장된 경우를 고려해 `config._load_dotenv()`에 CP949 폴백이 구현되어 있습니다. 텍스트 인코딩이 다른 파일을 열 때 예외 처리를 유지하세요.
- `get_secret(key, default)` 패턴을 사용해 비밀/환경값을 읽습니다. 직접 `os.getenv` 또는 `st.secrets`를 혼용하면 일관성이 깨집니다. 새 코드에서는 `get_secret()`을 호출하세요.
- DB 경로 처리: `app.py`는 `Path(db_path).parent.mkdir(parents=True, exist_ok=True)`로 부모 폴더를 자동 생성합니다. DB 파일 경로를 변경할 때는 이 행동을 유지하세요.
- 도메인 로직은 문서 기반: `skill.md`에 적힌 공식과 전처리 규칙을 코드에 그대로 반영해야 합니다(예: `(E)` 표기는 무조건 제외, 결산월 처리 등).

**코드 변경 시 구체적 체크리스트 (머신이 자동으로 확인/제안 가능 항목)**
- `config.get_secret()` 대신 직접 `st.secrets`를 사용한 곳이 있는지 확인.
- `.env` 로드 동작을 해치지 않도록 파일 인코딩 예외 처리를 유지.
- `skill.md`의 예시 케이스(연간 vs 분기 포함)를 코드로 변환할 때는 입력 컬럼 명 규칙(YYYY/12, YYYY/03, Q 분기 표기)을 엄격하게 매핑.
- DB 경로를 변경하거나 외부 DB로 전환할 경우 `app.py`의 healthcheck와 `Path(...).parent.mkdir(...)` 호출을 검토.

**예시 변경 유스케이스**
- 새 비밀키 사용 추가: `config.get_secret("NEW_KEY", "default")` 호출을 추가하고, 로컬 개발자를 위해 `README` 또는 `skill.md`에 환경 변수 이름을 문서화.
- 내재가치 계산 개선: `skill.md`의 세부 단계를 따르는 독립 모듈(`intrinsic.py`)을 추가하고, 기존 Streamlit UI에서 해당 모듈을 호출하도록 리팩터링.

**무엇을 수정하지 말 것(주의)**
- `config._load_dotenv()`의 인코딩 폴백과 기본값 로직을 제거하지 마세요. Windows 환경에서 파일을 읽지 못하는 문제가 발생할 수 있습니다.
- `skill.md`에 명시된 데이터 전처리 규칙(`(E)` 표기 제외, 결산월 판단)을 무시하지 마세요.

---
빠른 검토를 원하시면 변경할 파일과 의도(예: "`app.py`에 API 키 읽기 추가")를 알려주세요. 불명확한 규칙(예: 특정 컬럼명 패턴)이 더 필요하면 알려주시면 `skill.md`를 기반으로 상세 규칙을 추가하겠습니다.
