import os
from pathlib import Path
import streamlit as st


def _load_dotenv() -> None:
    """Lightweight .env loader (KEY=VALUE per line)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Windows fallback for local files saved in CP949
        content = env_path.read_text(encoding="cp949")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def get_secret(key: str, default: str | None = None) -> str | None:
    """Prefer Streamlit secrets (Cloud), fall back to env/local .env."""
    if key in st.secrets:
        return str(st.secrets[key])
    return os.getenv(key, default)


APP_ENV = get_secret("APP_ENV", os.getenv("APP_ENV", "local"))