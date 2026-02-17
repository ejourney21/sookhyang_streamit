import ast
import json
from datetime import date, timedelta
from pathlib import Path
import re

import pandas as pd
import requests
import streamlit as st

from intrinsic_value import (
    AnnualPoint,
    QuarterlyPoint,
    compute_intrinsic_value_annual,
    compute_intrinsic_value_with_quarters,
)


st.set_page_config(page_title="내재가치 분석", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
@import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap");

:root {
  --bg: #f5f2ee;
  --panel: #fcfbf9;
  --ink: #1e1c18;
  --muted: #6f675d;
  --accent: #e46b2a;
  --accent-2: #f6b77d;
  --line: #e5ddd2;
  --shadow: 0 18px 50px rgba(35, 23, 15, 0.08);
}

html, body, [class*="css"] {
  color: var(--ink);
  font-family: "Manrope", "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
  font-size: 16px;
  line-height: 1.65;
  letter-spacing: -0.01em;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 20% -10%, #fff6ee 0%, rgba(255, 246, 238, 0) 60%),
    radial-gradient(900px 540px at 90% 0%, #fff2e6 0%, rgba(255, 242, 230, 0) 55%),
    linear-gradient(180deg, #f7f2ed 0%, #f4f1ed 100%);
}

.block-container {
  max-width: 1140px;
  padding-top: 2.6rem;
  padding-bottom: 2.6rem;
}

.warm-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 28px 30px;
  box-shadow: var(--shadow);
}

.section {
  margin-top: 28px;
}

.section-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin: 0 0 10px 0;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.kpi-subtitle {
  margin-top: 14px;
  margin-bottom: 8px;
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.warm-header {
  font-size: 2.3rem;
  font-weight: 800;
  margin: 0 0 0.3rem 0;
  letter-spacing: -0.02em;
}

.warm-sub {
  color: var(--muted);
  margin: 0 0 1.4rem 0;
  font-size: 1.02rem;
}

.accent-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 14px;
  border-radius: 999px;
  background: rgba(228, 107, 42, 0.12);
  color: var(--accent);
  font-size: 0.85rem;
  font-weight: 700;
}

.stMarkdown, .stText, .stCaption, .stTable {
  font-size: 1.01rem;
}

.stTextArea label, .stNumberInput label, .stButton > button {
  font-size: 0.98rem;
  font-weight: 600;
  color: var(--muted);
}

.stButton > button {
  background: var(--accent);
  color: white;
  border-radius: 12px;
  border: none;
  padding: 0.7rem 1.35rem;
  font-weight: 700;
  box-shadow: 0 10px 25px rgba(228, 107, 42, 0.25);
  transition: transform 120ms ease, box-shadow 120ms ease;
}

.stButton > button:hover {
  background: #cf5c20;
  color: white;
  transform: translateY(-1px);
  box-shadow: 0 14px 28px rgba(228, 107, 42, 0.28);
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 16px;
  margin-top: 18px;
}

.kpi-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 10px 26px rgba(28, 18, 10, 0.08);
}

.kpi-card.good {
  border-color: #bfe4c5;
  background: #f3fbf4;
}

.kpi-card.bad {
  border-color: #f2c3ad;
  background: #fff3ed;
}

.kpi-card.neutral {
  border-color: var(--line);
  background: #ffffff;
}

.kpi-title {
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 6px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.kpi-value {
  font-size: 1.5rem;
  font-weight: 800;
  letter-spacing: -0.01em;
}

.kpi-sub {
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.88rem;
}

.stTextArea textarea {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 12px;
  font-size: 1rem;
  line-height: 1.6;
  padding: 12px 14px;
}

.stNumberInput input {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 10px;
  font-size: 1rem;
  padding: 8px 10px;
}

table {
  border-collapse: collapse !important;
}

thead th {
  background: #f3ece3 !important;
  color: var(--ink) !important;
  font-size: 1rem;
  border-bottom: 1px solid var(--line) !important;
}

tbody td {
  background: #ffffff !important;
  font-size: 1rem;
  border-bottom: 1px solid var(--line) !important;
}

.stAlert {
  border-radius: 14px;
  border: 1px solid var(--line);
  box-shadow: 0 8px 22px rgba(20, 12, 6, 0.08);
}

.formula-note {
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.6;
  margin: 0;
}

.detail-spacer {
  height: 14px;
}

@media (max-width: 768px) {
  .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
  }

  .warm-header {
    font-size: 1.9rem;
  }

  .kpi-grid {
    grid-template-columns: 1fr;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="warm-card">
  <span class="accent-pill">Intrinsic Value Studio</span>
  <div class="warm-header">내재가치 분석</div>
  <div class="warm-sub">연간/분기 실적데이터를 기반으로 내재가치를 계산합니다. (E) 추정치는 자동 제외됩니다.</div>
</div>
""",
    unsafe_allow_html=True,
)


_YEAR_PATTERN = re.compile(r"(\d{4})/(\d{1,2})")
STOCK_CODE_PATH = Path(__file__).with_name("stock_code.json")
NAVER_REALTIME_URL = "https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
NAVER_CHART_URL = (
    "https://m.stock.naver.com/front-api/external/chart/domestic/info"
    "?symbol={code}&requestType=1&startTime={start}&endTime={end}&timeframe=week"
)
NAVER_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://m.stock.naver.com/",
    "Origin": "https://m.stock.naver.com",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


def _split_tab_line(line: str) -> list[str]:
    return [cell.strip() for cell in line.split("\t")]


def _is_year_cell(cell: str) -> bool:
    return bool(_YEAR_PATTERN.search(cell))


def _is_ifrs_cell(cell: str) -> bool:
    return "ifrs" in cell.lower()


def _line_has_year_or_ifrs(cells: list[str]) -> bool:
    return any(_is_year_cell(cell) or _is_ifrs_cell(cell) for cell in cells if cell)


def _data_columns(columns: list[str]) -> list[str]:
    if not columns:
        return []
    return columns if _is_year_cell(columns[0]) else columns[1:]


@st.cache_data(ttl=60 * 60 * 24)
def _load_stock_codes() -> list[dict]:
    if not STOCK_CODE_PATH.exists():
        return []
    raw = STOCK_CODE_PATH.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return json.loads(raw.decode(encoding))
        except Exception:
            continue
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return []


def _build_stock_maps(items: list[dict]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    name_to_code: dict[str, str] = {}
    code_to_name: dict[str, str] = {}
    for item in items:
        name = str(item.get("corp_name") or "").strip()
        code = str(item.get("stock_code") or "").strip()
        if not name or not code:
            continue
        if name not in name_to_code:
            name_to_code[name] = code
        if code not in code_to_name:
            code_to_name[code] = name
    names = sorted(name_to_code.keys())
    return name_to_code, code_to_name, names


def _parse_response_payload(text: str | None):
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return None


@st.cache_data(ttl=60 * 5)
def _fetch_json(url: str) -> dict | None:
    try:
        response = requests.get(url, headers=NAVER_HEADERS, timeout=8)
        response.raise_for_status()
        payload = _parse_response_payload(response.text)
        return payload if isinstance(payload, dict) else payload
    except Exception:
        return None


def _fetch_json_with_meta(url: str) -> tuple[dict | None, int | None, str | None, str | None]:
    try:
        response = requests.get(url, headers=NAVER_HEADERS, timeout=8)
        status = response.status_code
        text_preview = response.text[:500] if response.text else None
        response.raise_for_status()
        payload = _parse_response_payload(response.text)
        if payload is None:
            raise ValueError("Empty or unparseable response body")
        return payload, status, None, text_preview
    except Exception as exc:
        status = None
        text_preview = None
        if "response" in locals() and response is not None:
            status = response.status_code
            text_preview = response.text[:500] if response.text else None
        return None, status, str(exc), text_preview


def _find_item_by_code(items: list[dict], code: str) -> dict | None:
    code_keys = ("stockCode", "symbolCode", "itemCode", "code", "symbol", "stock_code")
    for item in items:
        for key in code_keys:
            if str(item.get(key) or "") == code:
                return item
    return None


def _collect_list_of_dicts(payload, lists: list[list[dict]]) -> None:
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            lists.append(payload)
        else:
            for item in payload:
                _collect_list_of_dicts(item, lists)
        return
    if isinstance(payload, dict):
        for value in payload.values():
            _collect_list_of_dicts(value, lists)


def _extract_realtime_item(payload: dict | list | None, code: str) -> dict | None:
    if payload is None:
        return None
    if isinstance(payload, dict) and "result" in payload:
        payload = payload.get("result")
    if isinstance(payload, dict) and "areas" in payload:
        for area in payload.get("areas", []):
            items = area.get("datas") or area.get("data") or area.get("stocks")
            if isinstance(items, list):
                found = _find_item_by_code(items, code)
                if found:
                    return found
        for area in payload.get("areas", []):
            items = area.get("datas") or area.get("data") or area.get("stocks")
            if isinstance(items, list) and items:
                return items[0]
    if isinstance(payload, dict):
        for key in ("datas", "data", "stocks", "items"):
            items = payload.get(key)
            if isinstance(items, list):
                found = _find_item_by_code(items, code)
                if found:
                    return found
                if items:
                    return items[0]
        return payload
    if isinstance(payload, list):
        found = _find_item_by_code(payload, code)
        if found:
            return found
        if payload:
            return payload[0]
    return None


def _get_first_value(item: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in item:
            return item.get(key)
    return None


def _extract_realtime_summary(payload: dict | list | None, code: str) -> dict:
    item = _extract_realtime_item(payload, code) or {}
    name = _get_first_value(
        item,
        ("stockName", "name", "itemName", "corpName", "stock_name", "corp_name"),
    )
    price = _parse_number(
        str(
            _get_first_value(
                item,
                (
                    "closePrice",
                    "tradePrice",
                    "currentPrice",
                    "nowPrice",
                    "stck_prpr",
                ),
            )
            or ""
        )
    )
    change = _parse_number(
        str(
            _get_first_value(
                item,
                (
                    "compareToPreviousClosePrice",
                    "change",
                    "diff",
                    "prdy_vrss",
                ),
            )
            or ""
        )
    )
    change_rate = _parse_number(
        str(
            _get_first_value(
                item,
                (
                    "fluctuationsRatio",
                    "changeRate",
                    "rate",
                    "prdy_ctrt",
                ),
            )
            or ""
        )
    )
    return {
        "name": str(name or "").strip() or None,
        "price": price,
        "change": change,
        "change_rate": change_rate,
    }


def _looks_like_price_point(item: dict) -> bool:
    date_keys = ("localTradedAt", "date", "dt", "time", "x", "localTradedAtDate")
    price_keys = ("closePrice", "close", "price", "stck_prpr", "tradePrice")
    return any(key in item for key in date_keys) and any(key in item for key in price_keys)


def _extract_price_points(payload: dict | list | None) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, dict) and "result" in payload:
        payload = payload.get("result")
    if isinstance(payload, dict):
        for key in ("priceInfos", "prices", "priceInfo", "chartList", "data", "items"):
            items = payload.get(key)
            if isinstance(items, list) and items and isinstance(items[0], dict):
                return items
    candidates: list[list[dict]] = []
    _collect_list_of_dicts(payload, candidates)
    for items in candidates:
        filtered = [item for item in items if _looks_like_price_point(item)]
        if filtered:
            return filtered
    return []


def _parse_chart_date(value) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(int(value))
    value = str(value).strip()
    if value.isdigit() and len(value) == 8:
        return pd.to_datetime(value, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def _build_price_df_from_table(headers: list, rows: list) -> pd.DataFrame:
    header_map = {str(name).strip(): idx for idx, name in enumerate(headers)}
    date_idx = None
    close_idx = None
    for key in ("날짜", "date", "일자", "dt"):
        if key in header_map:
            date_idx = header_map[key]
            break
    for key in ("종가", "close", "price", "현재가"):
        if key in header_map:
            close_idx = header_map[key]
            break
    if date_idx is None or close_idx is None:
        return pd.DataFrame(columns=["date", "price"])
    data_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        if date_idx >= len(row) or close_idx >= len(row):
            continue
        date_ts = _parse_chart_date(row[date_idx])
        price = _parse_number(str(row[close_idx]))
        if price is None or date_ts is None or pd.isna(date_ts):
            continue
        data_rows.append({"date": date_ts, "price": price})
    if not data_rows:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(data_rows).dropna()
    return df.sort_values("date").drop_duplicates(subset=["date"])


def _build_price_df(payload: dict | list | None) -> pd.DataFrame:
    points = _extract_price_points(payload)
    if not points:
        if isinstance(payload, list) and payload and isinstance(payload[0], (list, tuple)):
            return _build_price_df_from_table(payload[0], payload[1:])
    rows: list[dict] = []
    for item in points:
        date_value = _get_first_value(
            item, ("localTradedAt", "date", "dt", "time", "x", "localTradedAtDate")
        )
        price_value = _get_first_value(
            item, ("closePrice", "close", "price", "stck_prpr", "tradePrice")
        )
        price = _parse_number(str(price_value or ""))
        date_ts = _parse_chart_date(date_value)
        if price is None or date_ts is None or pd.isna(date_ts):
            continue
        rows.append({"date": date_ts, "price": price})
    if not rows:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(rows).dropna()
    return df.sort_values("date").drop_duplicates(subset=["date"])


def _normalize_row_key(key: str) -> str:
    return "".join(key.split()).replace("\u00a0", "").strip().lower()


def _find_row_key(
    rows: dict[str, list[str]],
    labels: list[str],
    contains: list[str] | None = None,
) -> str | None:
    for label in labels:
        if label in rows:
            return label
    normalized = {_normalize_row_key(k): k for k in rows.keys()}
    for label in labels:
        norm_label = _normalize_row_key(label)
        if norm_label in normalized:
            return normalized[norm_label]
    if contains:
        for key in rows.keys():
            norm_key = _normalize_row_key(key)
            if all(token in norm_key for token in contains):
                return key
    return None


def _get_row(
    rows: dict[str, list[str]],
    labels: list[str],
    contains: list[str] | None = None,
) -> list[str]:
    found = _find_row_key(rows, labels, contains)
    if not found:
        return []
    return rows.get(found, [])


def _find_header_line(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        cells = _split_tab_line(line)
        year_like = [c for c in cells if _is_year_cell(c)]
        if len(year_like) >= 2:
            return i
    return -1


def _normalize_text(text: str) -> str:
    # Merge header lines like "2024/12\n(IFRS연결)" into one cell.
    return text.replace("\n(IFRS", " (IFRS")


def _clean_header_cells(cells: list[str]) -> list[str]:
    if len(cells) >= 2 and cells[1] in ("연간", "분기"):
        if any(_is_year_cell(c) for c in cells[2:]):
            return [cells[0]] + cells[2:]
    return cells


def _build_stacked_header_columns(lines: list[list[str]]) -> list[str]:
    columns: list[str] = []
    for cells in lines:
        for cell in cells:
            if not cell:
                continue
            if _is_ifrs_cell(cell):
                if columns:
                    columns[-1] = f"{columns[-1]} {cell}"
                continue
            if _is_year_cell(cell):
                columns.append(cell)
    return columns


def _parse_table(text: str) -> dict:
    text = _normalize_text(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {"columns": [], "rows": {}}

    header_idx = _find_header_line(lines)
    if header_idx != -1:
        columns = _clean_header_cells(_split_tab_line(lines[header_idx]))
        rows: dict[str, list[str]] = {}

        for line in lines[header_idx + 1 :]:
            cells = _split_tab_line(line)
            if not cells:
                continue
            key = cells[0]
            if not key:
                continue
            rows[key] = cells[1:]

        return {"columns": columns, "rows": rows}

    first_cells = _split_tab_line(lines[0])
    label_header = first_cells[0] if first_cells and first_cells[0] else "항목"

    start_idx = -1
    for i, line in enumerate(lines):
        cells = _split_tab_line(line)
        if any(_is_year_cell(c) for c in cells):
            start_idx = i
            break
    if start_idx == -1:
        return {"columns": [], "rows": {}}

    header_lines: list[list[str]] = []
    data_start = None
    for i in range(start_idx, len(lines)):
        cells = _split_tab_line(lines[i])
        if _line_has_year_or_ifrs(cells):
            header_lines.append(cells)
            continue
        data_start = i
        break
    if data_start is None:
        data_start = len(lines)

    year_columns = _build_stacked_header_columns(header_lines)
    columns = [label_header] + year_columns

    rows: dict[str, list[str]] = {}
    for line in lines[data_start:]:
        cells = _split_tab_line(line)
        if not cells:
            continue
        key = cells[0]
        if not key:
            continue
        rows[key] = cells[1:]

    return {"columns": columns, "rows": rows}


def _table_to_df(table: dict) -> pd.DataFrame | None:
    columns = table.get("columns") or []
    rows = table.get("rows") or {}
    if not columns or not rows:
        return None

    if _is_year_cell(columns[0]):
        header0 = "항목"
        col_headers = columns
    else:
        header0 = columns[0] if columns[0] else "항목"
        col_headers = columns[1:]
    data = []
    for key, values in rows.items():
        row = {header0: key}
        for i, col in enumerate(col_headers):
            row[col] = values[i] if i < len(values) else None
        data.append(row)

    return pd.DataFrame(data)


def _parse_number(value: str) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    if v in ("-", "N/A", "NA", "n/a"):
        return None
    v = v.replace(",", "")
    is_negative = False
    if v.startswith("(") and v.endswith(")"):
        is_negative = True
        v = v[1:-1].strip()
    v = v.replace("%", "").replace("원", "").replace("배", "")
    if not v:
        return None
    try:
        num = float(v)
    except ValueError:
        return None
    return -num if is_negative else num


def _to_won_if_needed(value: float | None) -> float | None:
    if value is None:
        return None
    # Heuristic: financial statement values are commonly in "억원".
    return value * 100_000_000 if value < 1_000_000 else value


def _real_eps_from_financials(
    net_income: float | None,
    equity: float | None,
    bps: float | None,
) -> float | None:
    if net_income is None or equity is None or bps is None or bps == 0:
        return None
    equity_won = _to_won_if_needed(equity)
    net_income_won = _to_won_if_needed(net_income)
    if equity_won is None or net_income_won is None:
        return None
    float_shares = equity_won / bps
    if float_shares <= 0:
        return None
    return net_income_won / float_shares


def _extract_period(label: str) -> tuple[int, int] | None:
    match = _YEAR_PATTERN.search(label)
    if not match:
        return None
    try:
        year = int(match.group(1))
        month = int(match.group(2))
        return year, month
    except ValueError:
        return None


def _is_estimate(label: str) -> bool:
    label_lower = label.lower()
    return (
        "(e)" in label_lower
        or "estimate" in label_lower
        or "추정" in label_lower
        or "예상" in label_lower
        or "전망" in label_lower
    )


def _build_annual_points(table: dict) -> tuple[list[AnnualPoint], dict[int, float], dict[int, float]]:
    columns = table["columns"]
    rows = table["rows"]
    eps_row = _get_row(
        rows,
        ["EPS(원)", "EPS", "주당순이익", "주당순이익(원)"],
        contains=["eps"],
    )
    net_income_row = _get_row(
        rows,
        ["당기순이익(지배)", "당기순이익", "지배주주순이익"],
        contains=["순이익"],
    )
    equity_row = _get_row(
        rows,
        ["자본총계(지배)", "자본총계", "자기자본"],
        contains=["자본총계"],
    )
    bps_row = _get_row(
        rows,
        ["BPS(원)", "BPS", "주당순자산가치", "주당순자산가치(원)"],
        contains=["bps"],
    )
    pbr_row = _get_row(
        rows,
        ["PBR(배)", "PBR"],
        contains=["pbr"],
    )

    points: list[AnnualPoint] = []
    pbr_by_year: dict[int, float] = {}
    eps_by_year: dict[int, float] = {}

    data_columns = _data_columns(columns)
    for idx, label in enumerate(data_columns):
        period = _extract_period(label)
        if not period:
            continue
        year, _month = period
        if _is_estimate(label):
            continue

        eps_value = _parse_number(eps_row[idx]) if idx < len(eps_row) else None
        net_income_value = (
            _parse_number(net_income_row[idx]) if idx < len(net_income_row) else None
        )
        equity_value = (
            _parse_number(equity_row[idx]) if idx < len(equity_row) else None
        )
        bps = _parse_number(bps_row[idx]) if idx < len(bps_row) else None
        eps = _real_eps_from_financials(net_income_value, equity_value, bps)
        if eps is None:
            eps = eps_value
        pbr = _parse_number(pbr_row[idx]) if idx < len(pbr_row) else None

        if eps is None:
            continue

        points.append(AnnualPoint(year=year, eps=eps, bps=bps, label=label))
        eps_by_year[year] = eps
        if pbr is not None:
            pbr_by_year[year] = pbr

    return points, eps_by_year, pbr_by_year


def _build_quarterly_points(
    table: dict,
) -> tuple[list[QuarterlyPoint], list[float], dict[tuple[int, int], float]]:
    columns = table["columns"]
    rows = table["rows"]
    eps_row = _get_row(
        rows,
        ["EPS(원)", "EPS", "주당순이익", "주당순이익(원)"],
        contains=["eps"],
    )
    net_income_row = _get_row(
        rows,
        ["당기순이익(지배)", "당기순이익", "지배주주순이익"],
        contains=["순이익"],
    )
    equity_row = _get_row(
        rows,
        ["자본총계(지배)", "자본총계", "자기자본"],
        contains=["자본총계"],
    )
    bps_row = _get_row(
        rows,
        ["BPS(원)", "BPS", "주당순자산가치", "주당순자산가치(원)"],
        contains=["bps"],
    )
    pbr_row = _get_row(
        rows,
        ["PBR(배)", "PBR"],
        contains=["pbr"],
    )

    points: list[QuarterlyPoint] = []
    recent_eps: list[float] = []
    pbr_by_period: dict[tuple[int, int], float] = {}

    data_columns = _data_columns(columns)
    for idx, label in enumerate(data_columns):
        period = _extract_period(label)
        if not period:
            continue
        year, month = period
        if _is_estimate(label):
            continue

        quarter = {3: 1, 6: 2, 9: 3, 12: 4}.get(month)
        if not quarter:
            continue

        eps_value = _parse_number(eps_row[idx]) if idx < len(eps_row) else None
        net_income_value = (
            _parse_number(net_income_row[idx]) if idx < len(net_income_row) else None
        )
        equity_value = (
            _parse_number(equity_row[idx]) if idx < len(equity_row) else None
        )
        bps = _parse_number(bps_row[idx]) if idx < len(bps_row) else None
        eps = _real_eps_from_financials(net_income_value, equity_value, bps)
        if eps is None:
            eps = eps_value
        pbr = _parse_number(pbr_row[idx]) if idx < len(pbr_row) else None

        if eps is None:
            continue

        points.append(
            QuarterlyPoint(year=year, quarter=quarter, eps=eps, bps=bps, label=label)
        )
        recent_eps.append(eps)
        if pbr is not None:
            pbr_by_period[(year, quarter)] = pbr

    return points, recent_eps, pbr_by_period


def _format_money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}"


def _format_formula(value: str | None) -> str:
    if not value:
        return "-"
    return value


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def _valuation_pct(current: float | None, target: float | None) -> float | None:
    if current is None or target is None or target == 0:
        return None
    return (current - target) / target * 100.0


def _pct_label(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f}%"


def _kpi_class(value: float | None, threshold: float = 10.0) -> str:
    if value is None:
        return "neutral"
    if abs(value) < threshold:
        return "neutral"
    return "good" if value < 0 else "bad"


WARNING_TEXTS = {
    "annual_est": "연간 헤더에 추정치(E) {count}개가 포함되어 계산에서 제외됩니다.",
    "quarter_est": "분기 헤더에 추정치(E) {count}개가 포함되어 계산에서 제외됩니다.",
    "annual_eps_missing": "연간 표에서 EPS 행을 찾지 못했습니다. 행 이름을 확인하세요.",
    "annual_bps_missing": "연간 표에서 BPS 행을 찾지 못했습니다. 행 이름을 확인하세요.",
    "annual_eps_negative": "최근 3개 연간 EPS 중 음수가 포함되어 가중 EPS가 왜곡될 수 있습니다.",
    "quarter_eps_negative": "최근 4개 분기 EPS 중 음수가 포함되어 추정 연간 EPS가 왜곡될 수 있습니다.",
    "quarter_non_consecutive": "최근 4개 분기 데이터가 연속되지 않아 추정 연간 EPS가 왜곡될 수 있습니다.",
    "roe_low": "ROE가 {value:.2f}%로 낮습니다(기준 {threshold:.2f}%). 기준 시점: {period}.",
    "debt_high": "부채비율이 {value:.2f}%로 높습니다(기준 {threshold:.2f}%). 기준 시점: {period}.",
    "roe_missing": "ROE 행을 찾지 못했습니다. 행 이름을 확인하세요.",
    "debt_missing": "부채비율 행을 찾지 못했습니다. 행 이름을 확인하세요.",
}


NOTICE_TEXTS = {
    "fiscal_year_end": "결산월 {month}월 감지: 연간/분기 계산은 {month}월 결산 기준으로 처리됩니다.",
}


def _warning_text(key: str, **kwargs) -> str:
    template = WARNING_TEXTS.get(key, "")
    if not template:
        return key
    return template.format(**kwargs)


def _notice_text(key: str, **kwargs) -> str:
    template = NOTICE_TEXTS.get(key, "")
    if not template:
        return key
    return template.format(**kwargs)


def _count_estimate_columns(columns: list[str]) -> int:
    data_columns = _data_columns(columns)
    if not data_columns:
        return 0
    return sum(1 for label in data_columns if _is_estimate(label))


def _has_negative(values: list[float]) -> bool:
    return any(value < 0 for value in values)


def _is_consecutive_quarters(points: list[QuarterlyPoint]) -> bool:
    if len(points) < 4:
        return False
    ordered = sorted(points, key=lambda p: (p.year, p.quarter))
    last4 = ordered[-4:]
    for prev, curr in zip(last4, last4[1:]):
        exp_q = prev.quarter + 1
        exp_y = prev.year
        if exp_q == 5:
            exp_q = 1
            exp_y += 1
        if curr.year != exp_y or curr.quarter != exp_q:
            return False
    return True


def _latest_row_value(
    rows: dict[str, list[str]],
    columns: list[str],
    *,
    labels: list[str],
    contains: list[str] | None = None,
) -> tuple[float | None, str | None]:
    row = _get_row(rows, labels, contains)
    if not row:
        return None, None
    data_columns = _data_columns(columns)
    latest_value = None
    latest_label = None
    for idx, label in enumerate(data_columns):
        if _is_estimate(label):
            continue
        if idx >= len(row):
            continue
        value = _parse_number(row[idx])
        if value is None:
            continue
        latest_value = value
        latest_label = label
    return latest_value, latest_label


def _period_key(label: str | None) -> int | None:
    if not label:
        return None
    period = _extract_period(label)
    if not period:
        return None
    year, month = period
    return year * 12 + month


def _latest_metric_from_tables(
    *,
    annual_table: dict,
    quarter_table: dict | None,
    labels: list[str],
    contains: list[str] | None = None,
) -> tuple[float | None, str | None]:
    annual_val, annual_label = _latest_row_value(
        annual_table.get("rows", {}),
        annual_table.get("columns", []),
        labels=labels,
        contains=contains,
    )
    quarter_val, quarter_label = (None, None)
    if quarter_table:
        quarter_val, quarter_label = _latest_row_value(
            quarter_table.get("rows", {}),
            quarter_table.get("columns", []),
            labels=labels,
            contains=contains,
        )

    annual_key = _period_key(annual_label)
    quarter_key = _period_key(quarter_label)
    if quarter_val is not None and (annual_key is None or (quarter_key or 0) > (annual_key or 0)):
        return quarter_val, quarter_label
    if annual_val is not None:
        return annual_val, annual_label
    return quarter_val, quarter_label


def _detect_fiscal_year_end_month(columns: list[str]) -> int | None:
    data_columns = _data_columns(columns)
    months: list[int] = []
    for label in data_columns:
        if _is_estimate(label):
            continue
        period = _extract_period(label)
        if period:
            months.append(period[1])
    if not months:
        return None
    return max(set(months), key=months.count)


def _collect_warnings(
    *,
    annual_table: dict,
    quarter_table: dict | None,
    annual_points: list[AnnualPoint],
    quarterly_points: list[QuarterlyPoint] | None,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    notices: list[str] = []

    annual_est = _count_estimate_columns(annual_table.get("columns", []))
    if annual_est:
        warnings.append(_warning_text("annual_est", count=annual_est))

    if quarter_table:
        quarter_est = _count_estimate_columns(quarter_table.get("columns", []))
        if quarter_est:
            warnings.append(_warning_text("quarter_est", count=quarter_est))

    annual_rows = annual_table.get("rows", {})
    if not _find_row_key(annual_rows, ["EPS(원)", "EPS", "주당순이익"], contains=["eps"]):
        warnings.append(_warning_text("annual_eps_missing"))
    if not _find_row_key(annual_rows, ["BPS(원)", "BPS", "주당순자산가치"], contains=["bps"]):
        warnings.append(_warning_text("annual_bps_missing"))

    if annual_points:
        if _has_negative([p.eps for p in annual_points[-3:]]):
            warnings.append(_warning_text("annual_eps_negative"))

    if quarterly_points:
        if _has_negative([p.eps for p in quarterly_points[-4:]]):
            warnings.append(_warning_text("quarter_eps_negative"))
        if not _is_consecutive_quarters(quarterly_points):
            warnings.append(_warning_text("quarter_non_consecutive"))

    roe_value, roe_period = _latest_metric_from_tables(
        annual_table=annual_table,
        quarter_table=quarter_table,
        labels=["ROE(%)", "ROE", "자기자본이익률", "자기자본이익률(%)"],
        contains=["roe"],
    )
    if roe_value is None:
        warnings.append(_warning_text("roe_missing"))
    else:
        roe_threshold = 5.0
        if roe_value < roe_threshold:
            warnings.append(
                _warning_text(
                    "roe_low",
                    value=roe_value,
                    threshold=roe_threshold,
                    period=roe_period or "N/A",
                )
            )

    debt_value, debt_period = _latest_metric_from_tables(
        annual_table=annual_table,
        quarter_table=quarter_table,
        labels=["부채비율", "부채비율(%)"],
        contains=["부채비율"],
    )
    if debt_value is None:
        warnings.append(_warning_text("debt_missing"))
    else:
        debt_threshold = 200.0
        if debt_value > debt_threshold:
            warnings.append(
                _warning_text(
                    "debt_high",
                    value=debt_value,
                    threshold=debt_threshold,
                    period=debt_period or "N/A",
                )
            )

    fiscal_month = _detect_fiscal_year_end_month(annual_table.get("columns", []))
    if fiscal_month and fiscal_month != 12:
        notices.append(_notice_text("fiscal_year_end", month=fiscal_month))

    return warnings, notices


def _format_period_label(label: str | None, year: int | None = None) -> str:
    if label:
        period = _extract_period(label)
        if period:
            y, m = period
            return f"{y}/{m:02d}"
    if year is not None:
        return f"{year}"
    return "-"


def _ttm_label_from_quarters(points: list[QuarterlyPoint] | None) -> str | None:
    if not points or len(points) < 4:
        return None
    ordered = sorted(points, key=lambda p: (p.year, p.quarter))
    last4 = ordered[-4:]
    start = _format_period_label(last4[0].label, last4[0].year)
    end = _format_period_label(last4[-1].label, last4[-1].year)
    return f"TTM({start}~{end})"


def _period_end_timestamp(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _build_segment_series(
    dates: pd.Index, segments: list[tuple[pd.Timestamp, float]]
) -> pd.Series:
    if dates is None:
        return pd.Series(dtype="float64")
    if not segments:
        return pd.Series(index=dates, dtype="float64")
    ordered = sorted(segments, key=lambda item: item[0])
    values = []
    idx = 0
    current = None
    for dt in dates:
        while idx < len(ordered) and dt >= ordered[idx][0]:
            current = ordered[idx][1]
            idx += 1
        values.append(current)
    return pd.Series(values, index=dates)


def _series_by_year(
    table: dict, labels: list[str], contains: list[str] | None = None
) -> dict[int, float]:
    rows = table.get("rows", {}) if table else {}
    columns = table.get("columns", []) if table else []
    row = _get_row(rows, labels, contains)
    if not row:
        return {}
    data_columns = _data_columns(columns)
    series: dict[int, float] = {}
    for idx, label in enumerate(data_columns):
        if _is_estimate(label):
            continue
        period = _extract_period(label)
        if not period:
            continue
        year, _month = period
        if idx >= len(row):
            continue
        value = _parse_number(row[idx])
        if value is None:
            continue
        series[year] = value
    return series


def _calc_cagr(series: dict[int, float]) -> tuple[float | None, int | None, int | None]:
    if len(series) < 2:
        return None, None, None
    years = sorted(series.keys())
    start_y = years[0]
    end_y = years[-1]
    start = series.get(start_y)
    end = series.get(end_y)
    if start is None or end is None or start <= 0 or end <= 0 or end_y <= start_y:
        return None, start_y, end_y
    span = end_y - start_y
    cagr = (end / start) ** (1 / span) - 1
    return cagr * 100.0, start_y, end_y


def _latest_value(series: dict[int, float]) -> tuple[float | None, int | None]:
    if not series:
        return None, None
    year = max(series.keys())
    return series.get(year), year


def _avg_last_n(series: dict[int, float], n: int) -> tuple[float | None, list[int]]:
    if len(series) < n:
        return None, sorted(series.keys())
    years = sorted(series.keys())[-n:]
    values = [series.get(y) for y in years if series.get(y) is not None]
    if len(values) < n:
        return None, years
    return sum(values) / n, years


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}배"


def _format_pct_compact(value: float | None) -> str:
    if value is None:
        return "-"
    rounded = round(value, 1)
    if float(rounded).is_integer():
        return f"{int(rounded)}%"
    return f"{rounded:.1f}%"


def _format_pct_transition(prev: float | None, curr: float | None) -> str:
    if prev is None or curr is None:
        return "-"
    return f"{_format_pct_compact(prev)} -> {_format_pct_compact(curr)}"


def _format_score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _style_intrinsic_row(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _row_style(row: pd.Series) -> list[str]:
        if str(row.get("항목", "")).strip() == "내재가치(원)":
            return [
                "background-color: #fff3ed; font-weight: 700; color: #a24a1d;"
            ] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)


def _ratio_series(
    numer: dict[int, float], denom: dict[int, float], *, percent: bool = False
) -> dict[int, float]:
    series: dict[int, float] = {}
    for year in set(numer.keys()) & set(denom.keys()):
        d = denom.get(year)
        n = numer.get(year)
        if d is None or n is None or d == 0:
            continue
        value = n / d
        if percent:
            value *= 100.0
        series[year] = value
    return series


def _latest_two(series: dict[int, float]) -> tuple[int, float, int, float] | None:
    if len(series) < 2:
        return None
    years = sorted(series.keys())
    prev_year, last_year = years[-2], years[-1]
    prev_value = series.get(prev_year)
    last_value = series.get(last_year)
    if prev_value is None or last_value is None:
        return None
    return prev_year, prev_value, last_year, last_value


st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">입력</div>', unsafe_allow_html=True)
stock_items = _load_stock_codes()
name_to_code, code_to_name, stock_names = _build_stock_maps(stock_items)
company_options = ["회사명 선택"] + stock_names
default_index = 0

selected_name = st.selectbox(
    "회사명 검색",
    options=company_options,
    index=default_index,
)
selected_code = (
    name_to_code.get(selected_name)
    if selected_name and selected_name != "회사명 선택"
    else None
)

stock_summary = None
price_df = pd.DataFrame(columns=["date", "price"])
chart_meta = {"url": None, "status": None, "error": None, "preview": None}
if selected_code:
    today = date.today()
    start_date = today - timedelta(days=365 * 3)
    realtime_payload = _fetch_json(NAVER_REALTIME_URL.format(code=selected_code))
    stock_summary = _extract_realtime_summary(realtime_payload, selected_code)
    chart_url = NAVER_CHART_URL.format(
        code=selected_code,
        start=start_date.strftime("%Y%m%d"),
        end=today.strftime("%Y%m%d"),
    )
    chart_payload, chart_status, chart_error, chart_preview = _fetch_json_with_meta(
        chart_url
    )
    chart_meta = {
        "url": chart_url,
        "status": chart_status,
        "error": chart_error,
        "preview": chart_preview,
    }
    price_df = _build_price_df(chart_payload)

if selected_code:
    display_name = stock_summary.get("name") if stock_summary else None
    display_name = display_name or selected_name
    price_value = stock_summary.get("price") if stock_summary else None
    change_value = stock_summary.get("change") if stock_summary else None
    change_rate = stock_summary.get("change_rate") if stock_summary else None
    delta_text = None
    if change_value is not None:
        delta_text = f"{change_value:+,.0f}"
        if change_rate is not None:
            delta_text = f"{delta_text} ({change_rate:+.2f}%)"
    info_cols = st.columns(3)
    info_cols[0].metric("회사명", display_name or "-")
    info_cols[1].metric("종목코드", selected_code)
    if delta_text:
        info_cols[2].metric("현재가(원)", _format_money(price_value), delta=delta_text)
    else:
        info_cols[2].metric("현재가(원)", _format_money(price_value))

if "current_price" not in st.session_state:
    st.session_state.current_price = 0
if "current_price_code" not in st.session_state:
    st.session_state.current_price_code = None

# Reset default when company changes, but keep user edits otherwise.
if selected_code != st.session_state.current_price_code:
    if selected_code and stock_summary and stock_summary.get("price") is not None:
        st.session_state.current_price = int(stock_summary["price"])
    else:
        st.session_state.current_price = 0
    st.session_state.current_price_code = selected_code

current_price = st.number_input(
    "현재주가 (원)",
    min_value=0,
    step=10,
    format="%d",
    key="current_price",
)

annual_text = st.text_area(
    "연간 실적 (텍스트 붙여넣기)",
    height=220,
)

quarter_text = st.text_area(
    "분기 실적 (선택, 텍스트 붙여넣기)",
    height=220,
)

run = st.button("분석 실행")
st.markdown("</div>", unsafe_allow_html=True)

annual_table = _parse_table(annual_text) if "annual_text" in locals() else None
quarter_table = (
    _parse_table(quarter_text)
    if "quarter_text" in locals() and quarter_text.strip()
    else None
)

if "run" in locals() and run:

    annual_points, annual_eps_map, annual_pbr_map = _build_annual_points(annual_table)
    if not annual_points:
        st.error("연간 데이터 파싱에 실패했습니다. 붙여넣기 형식을 확인해주세요.")
        st.stop()

    try:
        annual_result = compute_intrinsic_value_annual(annual_points)
    except Exception as exc:
        st.error(f"연간 분석 실패: {exc}")
        st.stop()

    quarter_result = None
    quarter_eps_sum = None
    quarterly_pbr = None
    quarterly_points = None
    latest_quarter = None
    if quarter_table:
        quarterly_points, eps_list, quarterly_pbr_map = _build_quarterly_points(
            quarter_table
        )
        if quarterly_points:
            if len(quarterly_points) >= 4:
                quarter_eps_sum = sum(q.eps for q in quarterly_points[-4:])
            latest_q = max(quarterly_points, key=lambda p: (p.year, p.quarter))
            quarterly_pbr = quarterly_pbr_map.get((latest_q.year, latest_q.quarter))
            latest_quarter = latest_q
            try:
                quarter_result = compute_intrinsic_value_with_quarters(
                    annual_points, quarterly_points
                )
            except Exception as exc:
                st.warning(f"분기 분석 건너뜀: {exc}")

    latest_annual_year = max(p.year for p in annual_points)
    annual_pbr = annual_pbr_map.get(latest_annual_year)
    st.session_state.intrinsic_annual = annual_result.intrinsic_value
    st.session_state.intrinsic_ttm = (
        quarter_result.intrinsic_value if quarter_result else None
    )
    st.session_state.intrinsic_code = selected_code

    warnings, notices = _collect_warnings(
        annual_table=annual_table,
        quarter_table=quarter_table,
        annual_points=annual_points,
        quarterly_points=quarterly_points if quarter_table else None,
    )

    annual_sorted = sorted(annual_points, key=lambda p: p.year)
    last3 = annual_sorted[-3:]
    annual_columns = [_format_period_label(p.label, p.year) for p in last3]
    latest_annual_col = annual_columns[-1] if annual_columns else None
    ttm_label = _ttm_label_from_quarters(quarterly_points) if quarterly_points else None
    annual_index = {p.year: idx for idx, p in enumerate(annual_sorted)}

    def _weighted_by_year(year: int) -> float | None:
        idx = annual_index.get(year)
        if idx is None or idx < 2:
            return None
        p0 = annual_sorted[idx]
        p1 = annual_sorted[idx - 1]
        p2 = annual_sorted[idx - 2]
        raw = (p0.eps * 3.0) + (p1.eps * 2.0) + (p2.eps * 1.0)
        return (raw / 6.0) * 10.0

    def _intrinsic_by_year(year: int) -> float | None:
        idx = annual_index.get(year)
        if idx is None:
            return None
        point = annual_sorted[idx]
        weighted = _weighted_by_year(year)
        if weighted is None or point.bps is None:
            return None
        return (point.bps + weighted) / 2.0

    def add_row(
        label: str,
        values: dict[str, float | None],
        *,
        pct: bool = False,
    ) -> None:
        row = {"항목": label}
        for col in annual_columns:
            value = values.get(col)
            row[col] = _format_pct(value) if pct else _format_money(value)
        if ttm_label:
            value = values.get(ttm_label)
            row[ttm_label] = _format_pct(value) if pct else _format_money(value)
        rows.append(row)

    def _values_by_period(
        labels: list[str], contains: list[str] | None = None
    ) -> dict[int, float]:
        row = _get_row(annual_table.get("rows", {}), labels, contains)
        if not row:
            return {}
        data_columns = _data_columns(annual_table.get("columns", []))
        values: dict[int, float] = {}
        for idx, label in enumerate(data_columns):
            if _is_estimate(label):
                continue
            key = _period_key(label)
            if key is None or idx >= len(row):
                continue
            value = _parse_number(row[idx])
            if value is None:
                continue
            values[key] = value
        return values

    def _real_eps_by_period(table: dict) -> dict[int, float]:
        rows = table.get("rows", {})
        net_income_row = _get_row(
            rows,
            ["당기순이익(지배)", "당기순이익", "지배주주순이익"],
            contains=["순이익"],
        )
        equity_row = _get_row(
            rows,
            ["자본총계(지배)", "자본총계", "자기자본"],
            contains=["자본총계"],
        )
        bps_row = _get_row(
            rows,
            ["BPS(원)", "BPS", "주당순자산가치", "주당순자산가치(원)"],
            contains=["bps"],
        )
        if not net_income_row or not equity_row or not bps_row:
            return {}
        data_columns = _data_columns(table.get("columns", []))
        values: dict[int, float] = {}
        for idx, label in enumerate(data_columns):
            if _is_estimate(label):
                continue
            key = _period_key(label)
            if key is None or idx >= len(net_income_row):
                continue
            net_income_value = _parse_number(net_income_row[idx])
            equity_value = (
                _parse_number(equity_row[idx]) if idx < len(equity_row) else None
            )
            bps_value = _parse_number(bps_row[idx]) if idx < len(bps_row) else None
            real_eps = _real_eps_from_financials(
                net_income_value, equity_value, bps_value
            )
            if real_eps is None:
                continue
            values[key] = real_eps
        return values

    def _values_for_columns(
        values: dict[int, float],
    ) -> dict[str, float | None]:
        mapped: dict[str, float | None] = {}
        for col, point in zip(annual_columns, last3):
            key = _period_key(point.label) or (point.year * 12 + 12)
            mapped[col] = values.get(key)
        return mapped

    def _quarter_values_by_period(
        labels: list[str], contains: list[str] | None = None
    ) -> dict[tuple[int, int], float]:
        if not quarter_table:
            return {}
        row = _get_row(quarter_table.get("rows", {}), labels, contains)
        if not row:
            return {}
        data_columns = _data_columns(quarter_table.get("columns", []))
        values: dict[tuple[int, int], float] = {}
        for idx, label in enumerate(data_columns):
            if _is_estimate(label):
                continue
            period = _extract_period(label)
            if not period or idx >= len(row):
                continue
            year, month = period
            quarter = {3: 1, 6: 2, 9: 3, 12: 4}.get(month)
            if not quarter:
                continue
            value = _parse_number(row[idx])
            if value is None:
                continue
            values[(year, quarter)] = value
        return values

    def _quarter_real_eps_by_period() -> dict[tuple[int, int], float]:
        if not quarter_table:
            return {}
        rows = quarter_table.get("rows", {})
        net_income_row = _get_row(
            rows,
            ["당기순이익(지배)", "당기순이익", "지배주주순이익"],
            contains=["순이익"],
        )
        equity_row = _get_row(
            rows,
            ["자본총계(지배)", "자본총계", "자기자본"],
            contains=["자본총계"],
        )
        bps_row = _get_row(
            rows,
            ["BPS(원)", "BPS", "주당순자산가치", "주당순자산가치(원)"],
            contains=["bps"],
        )
        if not net_income_row or not equity_row or not bps_row:
            return {}
        data_columns = _data_columns(quarter_table.get("columns", []))
        values: dict[tuple[int, int], float] = {}
        for idx, label in enumerate(data_columns):
            if _is_estimate(label):
                continue
            period = _extract_period(label)
            if not period or idx >= len(net_income_row):
                continue
            year, month = period
            quarter = {3: 1, 6: 2, 9: 3, 12: 4}.get(month)
            if not quarter:
                continue
            net_income_value = _parse_number(net_income_row[idx])
            equity_value = (
                _parse_number(equity_row[idx]) if idx < len(equity_row) else None
            )
            bps_value = _parse_number(bps_row[idx]) if idx < len(bps_row) else None
            real_eps = _real_eps_from_financials(
                net_income_value, equity_value, bps_value
            )
            if real_eps is None:
                continue
            values[(year, quarter)] = real_eps
        return values

    def _ttm_sum_from_quarters(
        labels: list[str], contains: list[str] | None = None
    ) -> float | None:
        values = _quarter_values_by_period(labels, contains)
        if len(values) < 4:
            return None
        ordered = sorted(values.items(), key=lambda item: item[0])
        last4 = ordered[-4:]
        for idx in range(len(last4) - 1):
            (y1, q1), _ = last4[idx]
            (y2, q2), _ = last4[idx + 1]
            next_q = q1 + 1
            next_y = y1
            if next_q == 5:
                next_q = 1
                next_y += 1
            if y2 != next_y or q2 != next_q:
                return None
        return sum(value for _, value in last4)

    def _ttm_sum_from_quarter_values(
        values: dict[tuple[int, int], float]
    ) -> float | None:
        if len(values) < 4:
            return None
        ordered = sorted(values.items(), key=lambda item: item[0])
        last4 = ordered[-4:]
        for idx in range(len(last4) - 1):
            (y1, q1), _ = last4[idx]
            (y2, q2), _ = last4[idx + 1]
            next_q = q1 + 1
            next_y = y1
            if next_q == 5:
                next_q = 1
                next_y += 1
            if y2 != next_y or q2 != next_q:
                return None
        return sum(value for _, value in last4)

    def add_row2(
        label: str,
        values: dict[str, float | None],
        *,
        pct: bool = False,
        ratio: bool = False,
        ttm_value: float | None = None,
    ) -> None:
        row = {"항목": label}
        for col in annual_columns:
            value = values.get(col)
            if pct:
                row[col] = _format_pct(value)
            elif ratio:
                row[col] = _format_ratio(value)
            else:
                row[col] = _format_money(value)
        if ttm_label:
            if pct:
                row[ttm_label] = _format_pct(ttm_value)
            elif ratio:
                row[ttm_label] = _format_ratio(ttm_value)
            else:
                row[ttm_label] = _format_money(ttm_value)
        rows2.append(row)

    rows = []
    rows2 = []
    annual_eps_values = _values_for_columns(
        _values_by_period(["EPS(원)", "EPS", "주당순이익"], contains=["eps"])
    )
    annual_bps_values = {col: p.bps for col, p in zip(annual_columns, last3)}
    annual_real_eps_values = _values_for_columns(_real_eps_by_period(annual_table))
    annual_weighted_values = {
        col: _weighted_by_year(p.year) for col, p in zip(annual_columns, last3)
    }
    annual_intrinsic_values = {
        col: _intrinsic_by_year(p.year) for col, p in zip(annual_columns, last3)
    }

    segment_points = []
    for point in last3:
        value = _intrinsic_by_year(point.year)
        if value is None:
            continue
        month = 12
        if point.label:
            period = _extract_period(point.label)
            if period:
                month = period[1]
        end_ts = _period_end_timestamp(point.year, month)
        segment_points.append({"end": end_ts.strftime("%Y-%m-%d"), "value": value})
    st.session_state.intrinsic_segments = segment_points
    if quarter_result and latest_quarter:
        ttm_end = _period_end_timestamp(latest_quarter.year, latest_quarter.quarter * 3)
        st.session_state.intrinsic_ttm_segment = {
            "end": ttm_end.strftime("%Y-%m-%d"),
            "value": quarter_result.intrinsic_value,
        }
    else:
        st.session_state.intrinsic_ttm_segment = None

    reported_eps_ttm = _ttm_sum_from_quarters(
        ["EPS(원)", "EPS", "주당순이익"], contains=["eps"]
    )
    real_eps_ttm = None
    real_quarter_values = _quarter_real_eps_by_period()
    if real_quarter_values:
        real_eps_ttm = _ttm_sum_from_quarter_values(real_quarter_values)

    add_row(
        "EPS(원)",
        {
            **annual_eps_values,
            **({ttm_label: reported_eps_ttm} if ttm_label else {}),
        },
    )
    add_row(
        "BPS(원)",
        {
            **annual_bps_values,
            **({ttm_label: latest_quarter.bps} if ttm_label and latest_quarter else {}),
        },
    )
    add_row(
        "실질EPS(원)",
        {
            **annual_real_eps_values,
            **({ttm_label: real_eps_ttm} if ttm_label else {}),
        },
    )
    add_row(
        "가중 EPS(원)",
        {
            **annual_weighted_values,
            **({ttm_label: quarter_result.weighted_eps} if ttm_label and quarter_result else {}),
        },
    )
    add_row(
        "내재가치(원)",
        {
            **annual_intrinsic_values,
            **({ttm_label: quarter_result.intrinsic_value} if ttm_label and quarter_result else {}),
        },
    )
    add_row(
        "내재가치 대비 괴리율",
        {
            **{col: _valuation_pct(current_price, value) for col, value in annual_intrinsic_values.items()},
            **(
                {ttm_label: _valuation_pct(current_price, quarter_result.intrinsic_value)}
                if ttm_label and quarter_result
                else {}
            ),
        },
        pct=True,
    )

    revenue_values = _values_for_columns(
        _values_by_period(["매출액", "매출", "수익"], contains=["매출"])
    )
    op_profit_values = _values_for_columns(
        _values_by_period(["영업이익", "영업이익(발표기준)"], contains=["영업이익"])
    )
    net_income_values = _values_for_columns(
        _values_by_period(
            ["당기순이익(지배)", "당기순이익", "지배주주순이익"],
            contains=["순이익"],
        )
    )
    eps_values = _values_for_columns(
        _values_by_period(["EPS(원)", "EPS", "주당순이익"], contains=["eps"])
    )
    per_values = _values_for_columns(_values_by_period(["PER(배)", "PER"], contains=["per"]))
    bps_values = _values_for_columns(
        _values_by_period(["BPS(원)", "BPS", "주당순자산가치"], contains=["bps"])
    )
    pbr_values = _values_for_columns(_values_by_period(["PBR(배)", "PBR"], contains=["pbr"]))
    dps_values = _values_for_columns(
        _values_by_period(["현금DPS(원)", "DPS", "현금배당금", "현금배당"], contains=["dps", "배당"])
    )
    div_yield_values = _values_for_columns(
        _values_by_period(["현금배당수익률", "배당수익률"], contains=["배당수익률"])
    )
    payout_values = _values_for_columns(
        _values_by_period(["현금배당성향(%)", "배당성향", "배당성향(%)"], contains=["배당성향"])
    )

    revenue_ttm = _ttm_sum_from_quarters(["매출액", "매출", "수익"], contains=["매출"])
    op_profit_ttm = _ttm_sum_from_quarters(
        ["영업이익", "영업이익(발표기준)"], contains=["영업이익"]
    )
    net_income_ttm = _ttm_sum_from_quarters(
        ["당기순이익(지배)", "당기순이익", "지배주주순이익"], contains=["순이익"]
    )
    eps_ttm = reported_eps_ttm if ttm_label else None
    bps_ttm = latest_quarter.bps if latest_quarter else None
    pbr_ttm = quarterly_pbr if latest_quarter else None

    dps_ttm = _ttm_sum_from_quarters(
        ["현금DPS(원)", "DPS", "현금배당금", "현금배당"],
        contains=["dps", "배당"],
    )
    per_ttm = None
    if eps_ttm is not None and eps_ttm > 0 and current_price > 0:
        per_ttm = current_price / eps_ttm
    div_yield_ttm = None
    if dps_ttm is not None and current_price > 0:
        div_yield_ttm = dps_ttm / current_price * 100.0
    payout_ttm = None
    if dps_ttm is not None and eps_ttm is not None and eps_ttm > 0:
        payout_ttm = dps_ttm / eps_ttm * 100.0

    add_row2("매출액", revenue_values, ttm_value=revenue_ttm)
    add_row2("영업이익", op_profit_values, ttm_value=op_profit_ttm)
    add_row2("당기순이익(지배)", net_income_values, ttm_value=net_income_ttm)
    add_row2("EPS(원)", eps_values, ttm_value=eps_ttm)
    add_row2("PER(배)", per_values, ratio=True, ttm_value=per_ttm)
    add_row2("BPS(원)", bps_values, ttm_value=bps_ttm)
    add_row2("실질EPS(원)", annual_real_eps_values, ttm_value=real_eps_ttm)
    add_row2("PBR(배)", pbr_values, ratio=True, ttm_value=pbr_ttm)
    add_row2("현금DPS(원)", dps_values, ttm_value=dps_ttm)
    add_row2("현금배당수익률", div_yield_values, pct=True, ttm_value=div_yield_ttm)
    add_row2("현금배당성향(%)", payout_values, pct=True, ttm_value=payout_ttm)
    if warnings or notices:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">주의사항</div>', unsafe_allow_html=True)
        for message in warnings:
            st.warning(message)
        for message in notices:
            st.info(message)
        st.markdown("</div>", unsafe_allow_html=True)

    eps_series = _series_by_year(
        annual_table,
        ["EPS(원)", "EPS", "주당순이익", "주당순이익(원)"],
        contains=["eps"],
    )
    dps_series = _series_by_year(
        annual_table,
        ["현금DPS(원)", "DPS", "현금배당금", "현금배당"],
        contains=["dps", "배당"],
    )
    payout_series = _series_by_year(
        annual_table,
        ["현금배당성향(%)", "배당성향", "배당성향(%)"],
        contains=["배당성향"],
    )
    per_series = _series_by_year(
        annual_table,
        ["PER(배)", "PER"],
        contains=["per"],
    )
    pbr_series = _series_by_year(
        annual_table,
        ["PBR(배)", "PBR"],
        contains=["pbr"],
    )
    div_yield_series = _series_by_year(
        annual_table,
        ["현금배당수익률", "배당수익률"],
        contains=["배당수익률"],
    )
    roe_series = _series_by_year(
        annual_table,
        ["ROE(%)", "ROE", "자기자본이익률", "자기자본이익률(%)"],
        contains=["roe"],
    )
    debt_ratio_series = _series_by_year(
        annual_table,
        ["부채비율", "부채비율(%)"],
        contains=["부채비율"],
    )
    fcf_series = _series_by_year(annual_table, ["FCF", "자유현금흐름"], contains=["fcf"])
    share_series = _series_by_year(
        annual_table,
        ["발행주식수(보통주)", "발행주식수", "주식수"],
        contains=["발행주식수", "보통주"],
    )
    debt_total_series = _series_by_year(
        annual_table,
        ["부채총계", "총부채"],
        contains=["부채총계"],
    )
    equity_total_series = _series_by_year(
        annual_table,
        ["자본총계", "자본총계(지배)", "자기자본"],
        contains=["자본총계"],
    )
    revenue_series = _series_by_year(
        annual_table,
        ["매출액", "매출", "수익"],
        contains=["매출"],
    )
    op_profit_series = _series_by_year(
        annual_table,
        ["영업이익", "영업이익(발표기준)"],
        contains=["영업이익"],
    )
    net_income_series = _series_by_year(
        annual_table,
        ["당기순이익", "당기순이익(지배)", "지배주주순이익"],
        contains=["순이익"],
    )
    cfo_series = _series_by_year(
        annual_table,
        ["영업활동현금흐름", "영업활동현금흐름(연결)"],
        contains=["영업활동현금흐름"],
    )
    op_margin_series = _ratio_series(op_profit_series, revenue_series, percent=True)
    net_margin_series = _ratio_series(net_income_series, revenue_series, percent=True)
    cfo_ni_series = _ratio_series(cfo_series, net_income_series, percent=False)

    eps_cagr, eps_start, eps_end = _calc_cagr(eps_series)
    dps_cagr, dps_start, dps_end = _calc_cagr(dps_series)
    eps_period = f"{eps_start}~{eps_end}" if eps_start and eps_end else None
    dps_period = f"{dps_start}~{dps_end}" if dps_start and dps_end else None

    dps_yoy = None
    dps_yoy_period = None
    if len(dps_series) >= 2:
        dps_years = sorted(dps_series.keys())
        prev_year, last_year = dps_years[-2], dps_years[-1]
        prev_value = dps_series.get(prev_year)
        last_value = dps_series.get(last_year)
        if prev_value is not None and prev_value > 0 and last_value is not None:
            dps_yoy = (last_value / prev_value - 1) * 100.0
            dps_yoy_period = f"{prev_year}->{last_year}"

    eps_yoy = None
    eps_yoy_period = None
    if len(eps_series) >= 2:
        eps_years = sorted(eps_series.keys())
        prev_year, last_year = eps_years[-2], eps_years[-1]
        prev_value = eps_series.get(prev_year)
        last_value = eps_series.get(last_year)
        if prev_value is not None and prev_value > 0 and last_value is not None:
            eps_yoy = (last_value / prev_value - 1) * 100.0
            eps_yoy_period = f"{prev_year}->{last_year}"

    triple_yoy = None
    triple_yoy_period = None
    common_years = sorted(
        set(revenue_series.keys())
        & set(op_profit_series.keys())
        & set(net_income_series.keys())
    )
    if len(common_years) >= 2:
        prev_year, last_year = common_years[-2], common_years[-1]
        rev_prev = revenue_series.get(prev_year)
        rev_last = revenue_series.get(last_year)
        op_prev = op_profit_series.get(prev_year)
        op_last = op_profit_series.get(last_year)
        ni_prev = net_income_series.get(prev_year)
        ni_last = net_income_series.get(last_year)
        if (
            rev_prev is not None
            and rev_last is not None
            and op_prev is not None
            and op_last is not None
            and ni_prev is not None
            and ni_last is not None
        ):
            triple_yoy = (
                rev_last > rev_prev
                and op_last > op_prev
                and ni_last > ni_prev
            )
            triple_yoy_period = f"{prev_year}->{last_year}"

    payout_prev = None
    payout_curr = None
    payout_period = None
    if len(payout_series) >= 2:
        payout_years = sorted(payout_series.keys())
        prev_year, last_year = payout_years[-2], payout_years[-1]
        payout_prev = payout_series.get(prev_year)
        payout_curr = payout_series.get(last_year)
        payout_period = f"{prev_year}->{last_year}"

    per_value, per_year = _latest_value(per_series)
    pbr_value, pbr_year = _latest_value(pbr_series)
    div_yield_value, div_yield_year = _latest_value(div_yield_series)

    john_neff_score = None
    if (
        eps_cagr is not None
        and div_yield_value is not None
        and per_value is not None
        and per_value > 0
    ):
        john_neff_score = (eps_cagr + div_yield_value) / per_value

    dps_eps_ratio = None
    if eps_cagr is not None and eps_cagr != 0 and dps_cagr is not None:
        dps_eps_ratio = dps_cagr / eps_cagr

    roe_avg, roe_years = _avg_last_n(roe_series, 3)
    roe_period = f"{roe_years[0]}~{roe_years[-1]}" if len(roe_years) >= 3 else None

    debt_ratio, debt_year = _latest_value(debt_ratio_series)
    if debt_ratio is None:
        common_years = set(debt_total_series.keys()) & set(equity_total_series.keys())
        if common_years:
            latest_year = max(common_years)
            debt_total = debt_total_series.get(latest_year)
            equity_total = equity_total_series.get(latest_year)
            if debt_total is not None and equity_total and equity_total > 0:
                debt_ratio = debt_total / equity_total * 100.0
                debt_year = latest_year
    debt_period = f"{debt_year}" if debt_year else None

    coverage_value = None
    coverage_year = None
    common_years = (
        set(fcf_series.keys())
        & set(dps_series.keys())
        & set(share_series.keys())
    )
    if common_years:
        coverage_year = max(common_years)
        fcf_value = fcf_series.get(coverage_year)
        dps_value = dps_series.get(coverage_year)
        shares_value = share_series.get(coverage_year)
        if fcf_value is not None and dps_value and shares_value and dps_value > 0 and shares_value > 0:
            total_dividend = dps_value * shares_value
            # Heuristic: financial statement values are commonly in "억원".
            if fcf_value < 1_000_000:
                total_dividend = total_dividend / 100_000_000
            if total_dividend > 0:
                coverage_value = fcf_value / total_dividend
    coverage_period = f"{coverage_year}" if coverage_year else None

    intrinsic_gap = _valuation_pct(current_price, annual_result.intrinsic_value)

    kpis = []

    if eps_cagr is None:
        eps_status = "neutral"
        eps_note = "유효 데이터 부족"
    elif eps_cagr >= 10:
        eps_status = "good"
        eps_note = "이익 성장 속도 양호"
    elif eps_cagr >= 0:
        eps_status = "neutral"
        eps_note = "이익 성장 속도 완만"
    else:
        eps_status = "bad"
        eps_note = "이익 성장 둔화/감소"
    kpis.append(
        {
            "title": "EPS 성장성(CAGR)",
            "value": _format_pct(eps_cagr),
            "status": eps_status,
            "note": eps_note,
            "period": eps_period,
            "group": "기타",
        }
    )

    if eps_yoy is None:
        eps_yoy_status = "neutral"
        eps_yoy_note = "전년 비교 데이터 부족"
    elif eps_yoy >= 10:
        eps_yoy_status = "good"
        eps_yoy_note = "이익 전년 성장 양호"
    elif eps_yoy >= 0:
        eps_yoy_status = "neutral"
        eps_yoy_note = "이익 전년 성장 완만"
    else:
        eps_yoy_status = "bad"
        eps_yoy_note = "이익 전년 감소"
    kpis.append(
        {
            "title": "EPS 성장률(전년)",
            "value": _format_pct(eps_yoy),
            "status": eps_yoy_status,
            "note": eps_yoy_note,
            "period": eps_yoy_period,
            "group": "배당성장",
        }
    )

    if dps_cagr is None:
        dps_status = "neutral"
        dps_note = "유효 데이터 부족"
    elif dps_cagr >= 5:
        dps_status = "good"
        dps_note = "배당 성장 양호"
    elif dps_cagr >= 0:
        dps_status = "neutral"
        dps_note = "배당 성장 완만"
    else:
        dps_status = "bad"
        dps_note = "배당 성장 둔화/감소"
    kpis.append(
        {
            "title": "배당 성장성(DPS CAGR)",
            "value": _format_pct(dps_cagr),
            "status": dps_status,
            "note": dps_note,
            "period": dps_period,
            "group": "기타",
        }
    )

    if dps_yoy is None:
        dps_yoy_status = "neutral"
        dps_yoy_note = "전년 비교 데이터 부족"
    elif dps_yoy >= 5:
        dps_yoy_status = "good"
        dps_yoy_note = "배당 전년 성장 양호"
    elif dps_yoy >= 0:
        dps_yoy_status = "neutral"
        dps_yoy_note = "배당 전년 성장 완만"
    else:
        dps_yoy_status = "bad"
        dps_yoy_note = "배당 전년 감소"
    kpis.append(
        {
            "title": "배당 성장률(전년)",
            "value": _format_pct(dps_yoy),
            "status": dps_yoy_status,
            "note": dps_yoy_note,
            "period": dps_yoy_period,
            "group": "배당성장",
        }
    )

    payout_value = _format_pct_transition(payout_prev, payout_curr)
    if payout_prev is None or payout_curr is None:
        payout_status = "neutral"
        payout_note = "배당성향 데이터 부족"
    elif 20 <= payout_prev <= 60 and 20 <= payout_curr <= 60:
        payout_status = "good"
        payout_note = "배당성향 적정 범위"
    elif payout_prev > 70 or payout_curr > 70 or payout_prev < 10 or payout_curr < 10:
        payout_status = "bad"
        payout_note = "배당성향 과도/과소"
    else:
        payout_status = "neutral"
        payout_note = "배당성향 변동"
    kpis.append(
        {
            "title": "배당성향(최근 2년)",
            "value": payout_value,
            "status": payout_status,
            "note": payout_note,
            "period": payout_period,
            "group": "배당성장",
        }
    )

    if triple_yoy is None:
        triple_status = "neutral"
        triple_note = "3대 실적 비교 데이터 부족"
        triple_value = "-"
    elif triple_yoy:
        triple_status = "good"
        triple_note = "매출·영업이익·순이익 동반 증가"
        triple_value = "충족"
    else:
        triple_status = "bad"
        triple_note = "3대 실적 동반 증가 아님"
        triple_value = "미충족"
    kpis.append(
        {
            "title": "3대 실적 동시 증가",
            "value": triple_value,
            "status": triple_status,
            "note": triple_note,
            "period": triple_yoy_period,
            "group": "기타",
        }
    )

    if dps_eps_ratio is None:
        ratio_status = "neutral"
        ratio_note = "EPS CAGR 기준 미확보"
    elif 0.6 <= dps_eps_ratio <= 1.2:
        ratio_status = "good"
        ratio_note = "배당 성장, 이익 성장과 동행"
    elif 0.4 <= dps_eps_ratio < 0.6 or 1.2 < dps_eps_ratio <= 1.5:
        ratio_status = "neutral"
        ratio_note = "배당 성장 속도 차이"
    else:
        ratio_status = "bad"
        ratio_note = "배당 성장과 이익 성장 괴리"
    kpis.append(
        {
            "title": "DPS/EPS 성장비율",
            "value": _format_ratio(dps_eps_ratio),
            "status": ratio_status,
            "note": ratio_note,
            "period": eps_period,
            "group": "배당성장",
        }
    )

    if roe_avg is None:
        roe_status = "neutral"
        roe_note = "최근 3년 데이터 부족"
    elif roe_avg >= 12:
        roe_status = "good"
        roe_note = "자본 효율 높음"
    elif roe_avg >= 8:
        roe_status = "neutral"
        roe_note = "자본 효율 보통"
    else:
        roe_status = "bad"
        roe_note = "자본 효율 낮음"
    kpis.append(
        {
            "title": "ROE 3년 평균",
            "value": _format_pct(roe_avg),
            "status": roe_status,
            "note": roe_note,
            "period": roe_period,
            "group": "기타",
        }
    )

    if debt_ratio is None:
        debt_status = "neutral"
        debt_note = "부채비율 데이터 부족"
    elif debt_ratio < 50:
        debt_status = "good"
        debt_note = "재무 안정성 우수"
    elif debt_ratio <= 100:
        debt_status = "neutral"
        debt_note = "재무 안정성 보통"
    else:
        debt_status = "bad"
        debt_note = "재무 레버리지 부담"
    kpis.append(
        {
            "title": "부채비율(최신)",
            "value": _format_pct(debt_ratio),
            "status": debt_status,
            "note": debt_note,
            "period": debt_period,
            "group": "숙향 판단기준",
        }
    )

    if coverage_value is None:
        coverage_status = "neutral"
        coverage_note = "현금 커버리지 산출 불가"
    elif coverage_value >= 1.5:
        coverage_status = "good"
        coverage_note = "배당 여력 충분"
    elif coverage_value >= 1.0:
        coverage_status = "neutral"
        coverage_note = "배당 여력 보통"
    else:
        coverage_status = "bad"
        coverage_note = "배당 지속성 주의"
    kpis.append(
        {
            "title": "배당 커버리지(FCF)",
            "value": _format_ratio(coverage_value),
            "status": coverage_status,
            "note": coverage_note,
            "period": coverage_period,
            "group": "기타",
        }
    )

    if intrinsic_gap is None:
        gap_status = "neutral"
        gap_note = "내재가치 산출 필요"
    elif intrinsic_gap <= -20:
        gap_status = "good"
        gap_note = "내재가치 대비 충분히 낮음"
    elif intrinsic_gap <= 10:
        gap_status = "neutral"
        gap_note = "내재가치 근접"
    else:
        gap_status = "bad"
        gap_note = "내재가치 대비 높음"
    kpis.append(
        {
            "title": "내재가치 괴리율",
            "value": _pct_label(intrinsic_gap),
            "status": gap_status,
            "note": gap_note,
            "period": "연간 기준",
            "group": "숙향 판단기준",
        }
    )

    if john_neff_score is None:
        neff_status = "neutral"
        neff_note = "존 네프 스코어 산출 불가"
    elif john_neff_score >= 2.0:
        neff_status = "good"
        neff_note = "존 네프 기준 충족"
    else:
        neff_status = "bad"
        neff_note = "존 네프 기준 미달"
    kpis.append(
        {
            "title": "존 네프 스코어",
            "value": _format_score(john_neff_score),
            "status": neff_status,
            "note": neff_note,
            "period": "EPS CAGR + 배당수익률 ÷ PER",
            "group": "기타",
        }
    )

    if pbr_value is None:
        pbr_status = "neutral"
        pbr_note = "PBR 데이터 부족"
    elif pbr_value <= 1:
        pbr_status = "good"
        pbr_note = "PBR 1배 이하"
    else:
        pbr_status = "bad"
        pbr_note = "PBR 1배 초과"
    kpis.append(
        {
            "title": "PBR(최신)",
            "value": _format_ratio(pbr_value),
            "status": pbr_status,
            "note": pbr_note,
            "period": f"{pbr_year}" if pbr_year else None,
            "group": "숙향 판단기준",
        }
    )

    if per_value is None:
        per_status = "neutral"
        per_note = "PER 데이터 부족"
    elif per_value <= 10:
        per_status = "good"
        per_note = "PER 10배 이하"
    else:
        per_status = "bad"
        per_note = "PER 10배 초과"
    kpis.append(
        {
            "title": "PER(최신)",
            "value": _format_ratio(per_value),
            "status": per_status,
            "note": per_note,
            "period": f"{per_year}" if per_year else None,
            "group": "숙향 판단기준",
        }
    )

    if div_yield_value is None:
        div_status = "neutral"
        div_note = "배당수익률 데이터 부족"
    elif div_yield_value >= 3:
        div_status = "good"
        div_note = "배당수익률 3% 이상"
    else:
        div_status = "bad"
        div_note = "배당수익률 3% 미만"
    kpis.append(
        {
            "title": "배당수익률(최신)",
            "value": _format_pct(div_yield_value),
            "status": div_status,
            "note": div_note,
            "period": f"{div_yield_year}" if div_yield_year else None,
            "group": "숙향 판단기준",
        }
    )

    checkpoint_messages = []

    eps_trend = _latest_two(eps_series)
    if eps_trend:
        prev_year, prev_value, last_year, last_value = eps_trend
        if last_value < prev_value:
            checkpoint_messages.append(
                f"EPS가 {prev_year}→{last_year}에 감소했습니다."
            )
    if eps_cagr is not None and eps_cagr < 0:
        checkpoint_messages.append("EPS CAGR이 음수입니다.")

    revenue_trend = _latest_two(revenue_series)
    if revenue_trend:
        prev_year, prev_value, last_year, last_value = revenue_trend
        if last_value < prev_value:
            checkpoint_messages.append(
                f"매출이 {prev_year}→{last_year}에 감소했습니다."
            )

    op_margin_trend = _latest_two(op_margin_series)
    if op_margin_trend:
        prev_year, prev_value, last_year, last_value = op_margin_trend
        if last_value < prev_value:
            checkpoint_messages.append(
                f"영업이익률이 {prev_year}→{last_year}에 하락했습니다."
            )

    net_margin_trend = _latest_two(net_margin_series)
    if net_margin_trend:
        prev_year, prev_value, last_year, last_value = net_margin_trend
        if last_value < prev_value:
            checkpoint_messages.append(
                f"순이익률이 {prev_year}→{last_year}에 하락했습니다."
            )

    if triple_yoy is False:
        checkpoint_messages.append(
            "매출·영업이익·순이익이 같은 기간에 모두 증가하지 않았습니다."
        )

    if roe_avg is not None and roe_avg < 8:
        checkpoint_messages.append("ROE 3년 평균이 8% 미만입니다.")

    if debt_ratio is not None and debt_ratio > 100:
        checkpoint_messages.append("부채비율이 100%를 초과합니다.")

    fcf_trend = _latest_two(fcf_series)
    if fcf_trend:
        _prev_year, _prev_value, last_year, last_value = fcf_trend
        if last_value < 0:
            checkpoint_messages.append(f"{last_year} FCF가 음수입니다.")

    if coverage_value is not None and coverage_value < 1.0:
        checkpoint_messages.append("배당 커버리지(FCF)가 1배 미만입니다.")

    if dps_eps_ratio is not None:
        if dps_eps_ratio > 1.5:
            checkpoint_messages.append("배당 성장 속도가 이익 성장보다 빠릅니다.")
        elif dps_eps_ratio < 0.4:
            checkpoint_messages.append("배당 성장 속도가 이익 성장보다 느립니다.")

    cfo_ni_trend = _latest_two(cfo_ni_series)
    if cfo_ni_trend:
        _prev_year, _prev_value, last_year, last_value = cfo_ni_trend
        if last_value < 0.7:
            checkpoint_messages.append(
                f"{last_year} 영업현금흐름이 순이익을 충분히 뒷받침하지 못합니다."
            )

    if intrinsic_gap is not None and intrinsic_gap > 10:
        checkpoint_messages.append("내재가치 대비 고평가 구간입니다.")

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">요약 KPI</div>', unsafe_allow_html=True)
    group_order = [
        (
            "숙향 판단기준",
            "숙향 판단기준",
            ["PER(최신)", "PBR(최신)", "배당수익률(최신)", "부채비율(최신)", "내재가치 괴리율"],
        ),
        ("배당성장", "배당성장", None),
        ("기타", "기타", None),
    ]
    status_label_map = {
        "good": "좋음",
        "neutral": "보통",
        "bad": "주의",
    }
    sections = []
    for group_key, group_title, order_titles in group_order:
        group_items = [item for item in kpis if item.get("group") == group_key]
        if not group_items:
            continue
        if order_titles:
            title_index = {name: idx for idx, name in enumerate(order_titles)}
            group_items = sorted(
                group_items,
                key=lambda item: title_index.get(item.get("title"), len(title_index)),
            )
        sections.append(f'<div class="kpi-subtitle">{group_title}</div>')
        kpi_cards = []
        for item in group_items:
            period = item.get("period")
            status_label = status_label_map.get(item.get("status"), "보통")
            sub_parts = []
            if period:
                sub_parts.append(period)
            sub_parts.append(f"{status_label} · {item.get('note')}")
            sub_text = " · ".join(sub_parts)
            kpi_cards.append(
                f"""
  <div class="kpi-card {item.get('status')}">
    <div class="kpi-title">{item.get('title')}</div>
    <div class="kpi-value">{item.get('value')}</div>
    <div class="kpi-sub">{sub_text}</div>
  </div>
"""
            )
        sections.append(
            "<div class=\"kpi-grid\">\n" + "\n".join(kpi_cards) + "\n</div>"
        )
    st.markdown("\n".join(sections), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">체크포인트</div>', unsafe_allow_html=True)
    if checkpoint_messages:
        for message in checkpoint_messages:
            st.warning(message)
    else:
        st.info("뚜렷한 리스크 신호가 발견되지 않았습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">상세 테이블</div>', unsafe_allow_html=True)
    detail_df = pd.DataFrame(rows)
    st.dataframe(_style_intrinsic_row(detail_df), use_container_width=True, hide_index=True)
    st.markdown('<div class="detail-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="formula-note">
  계산 근거<br/>
  · 유통주식수 = 자본총계(지배) / BPS (필요 시 억→원 환산)<br/>
  · 실질 EPS = 자본총계/BPS로 유통주식수 역산, 당기순이익(지배)/유통주식수 (필요 시 억→원 환산)<br/>
  · 가중 EPS = (실질 EPS(n)×3 + 실질 EPS(n-1)×2 + 실질 EPS(n-2)×1) / 6 × 10<br/>
  · TTM(추정 연간 EPS) = 최근 4분기 EPS 합산<br/>
  · 내재가치 = (BPS + 가중 EPS) / 2<br/>
  · 내재가치 대비 괴리율 = (현재주가 - 내재가치) / 내재가치 × 100
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="detail-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">상세테이블2</div>', unsafe_allow_html=True)
    detail2_df = pd.DataFrame(rows2)
    st.dataframe(detail2_df, use_container_width=True, hide_index=True)
    st.markdown(
        """
<div class="formula-note">
  계산 근거<br/>
  · TTM = 최근 4분기 합산(연속 4분기 기준)<br/>
  · 유통주식수 = 자본총계(지배) / BPS (필요 시 억→원 환산)<br/>
  · 실질 EPS = 자본총계/BPS로 유통주식수 역산, 당기순이익(지배)/유통주식수 (필요 시 억→원 환산)<br/>
  · TTM PER = 현재주가 / TTM EPS<br/>
  · TTM 배당수익률 = TTM DPS / 현재주가 × 100<br/>
  · TTM 배당성향 = TTM DPS / TTM EPS × 100
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


intrinsic_annual = (
    st.session_state.get("intrinsic_annual")
    if st.session_state.get("intrinsic_code") == selected_code
    else None
)
intrinsic_ttm = (
    st.session_state.get("intrinsic_ttm")
    if st.session_state.get("intrinsic_code") == selected_code
    else None
)
segment_data = (
    st.session_state.get("intrinsic_segments")
    if st.session_state.get("intrinsic_code") == selected_code
    else None
)
ttm_segment = (
    st.session_state.get("intrinsic_ttm_segment")
    if st.session_state.get("intrinsic_code") == selected_code
    else None
)
if selected_code and not price_df.empty:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">최근 3년 주가</div>', unsafe_allow_html=True)
    chart_df = price_df.copy()
    chart_df = chart_df.set_index("date").rename(columns={"price": "주가"})
    used_annual_segment = False
    if segment_data:
        seg_points: list[tuple[pd.Timestamp, float]] = []
        for item in segment_data:
            end = item.get("end") if isinstance(item, dict) else None
            value = item.get("value") if isinstance(item, dict) else None
            if not end or value is None:
                continue
            end_ts = pd.to_datetime(end, errors="coerce")
            if pd.isna(end_ts):
                continue
            seg_points.append((end_ts, value))
        if seg_points:
            chart_df["내재가치(연간-구간)"] = _build_segment_series(
                chart_df.index, seg_points
            )
            used_annual_segment = True
    if not used_annual_segment and intrinsic_annual is not None:
        chart_df["내재가치(연간)"] = intrinsic_annual

    used_ttm_segment = False
    if (
        isinstance(ttm_segment, dict)
        and ttm_segment.get("end")
        and ttm_segment.get("value") is not None
    ):
        ttm_end = pd.to_datetime(ttm_segment["end"], errors="coerce")
        if not pd.isna(ttm_end):
            chart_df["내재가치(TTM-구간)"] = _build_segment_series(
                chart_df.index, [(ttm_end, ttm_segment["value"])]
            )
            used_ttm_segment = True
    if not used_ttm_segment and intrinsic_ttm is not None:
        chart_df["내재가치(TTM)"] = intrinsic_ttm
    st.line_chart(chart_df)
    st.markdown("</div>", unsafe_allow_html=True)
elif selected_code:
    st.info("최근 3년 주가 데이터를 불러오지 못했습니다. 네트워크/요청 제한 또는 응답 형식을 확인해주세요.")
    if st.checkbox("차트 디버그 보기"):
        st.markdown("**요청 정보**")
        st.code(chart_meta.get("url") or "-", language="text")
        st.markdown("**응답 상태**")
        st.write({"status": chart_meta.get("status"), "error": chart_meta.get("error")})
        st.markdown("**응답 미리보기**")
        st.code(chart_meta.get("preview") or "-", language="json")
