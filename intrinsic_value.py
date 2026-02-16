from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional


@dataclass(frozen=True)
class AnnualPoint:
    year: int
    eps: float
    bps: Optional[float] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class QuarterlyPoint:
    year: int
    quarter: int  # 1..4
    eps: float
    bps: Optional[float] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class IntrinsicValueResult:
    method: str
    bps: float
    weighted_eps: float
    intrinsic_value: float
    implied_price: Optional[float] = None
    implied_pbr: Optional[float] = None
    note: Optional[str] = None


def _is_estimate_label(label: Optional[str]) -> bool:
    if not label:
        return False
    return "(e)" in label.lower()


def _weighted_eps(eps_n: float, eps_n1: float, eps_n2: float) -> float:
    raw = (eps_n * 3.0) + (eps_n1 * 2.0) + (eps_n2 * 1.0)
    return (raw / 6.0) * 10.0


def _intrinsic_value(bps: float, weighted_eps: float) -> float:
    return (bps + weighted_eps) / 2.0


def _sort_annual(points: Iterable[AnnualPoint]) -> list[AnnualPoint]:
    return sorted(points, key=lambda p: p.year)


def _sort_quarterly(points: Iterable[QuarterlyPoint]) -> list[QuarterlyPoint]:
    return sorted(points, key=lambda p: (p.year, p.quarter))


def compute_intrinsic_value_annual(
    annual_points: Iterable[AnnualPoint],
    *,
    pbr: Optional[float] = None,
) -> IntrinsicValueResult:
    """
    Annual-only method.
    Uses latest 3 annual EPS points and latest annual BPS.
    """
    filtered = [p for p in annual_points if not _is_estimate_label(p.label)]
    points = _sort_annual(filtered)
    if len(points) < 3:
        raise ValueError("Need at least 3 annual EPS points (excluding estimates).")

    latest = points[-1]
    prev1 = points[-2]
    prev2 = points[-3]

    if latest.bps is None:
        raise ValueError("Latest annual BPS is required.")

    weighted = _weighted_eps(latest.eps, prev1.eps, prev2.eps)
    intrinsic = _intrinsic_value(latest.bps, weighted)

    implied_price = None
    if pbr is not None:
        implied_price = latest.bps * pbr

    return IntrinsicValueResult(
        method="annual",
        bps=latest.bps,
        weighted_eps=weighted,
        intrinsic_value=intrinsic,
        implied_price=implied_price,
        implied_pbr=pbr,
    )


def compute_intrinsic_value_with_quarters(
    annual_points: Iterable[AnnualPoint],
    quarterly_points: Iterable[QuarterlyPoint],
    *,
    pbr: Optional[float] = None,
) -> IntrinsicValueResult:
    """
    Annual + quarterly method.
    Latest 4 quarters sum -> estimated annual EPS for "most recent year".
    Latest quarterly BPS used.
    Then weighted EPS uses (estimated annual EPS, prior annual EPS, two-years-ago EPS).
    """
    annual_filtered = [p for p in annual_points if not _is_estimate_label(p.label)]
    annual = _sort_annual(annual_filtered)
    if len(annual) < 2:
        raise ValueError("Need at least 2 annual EPS points (excluding estimates).")

    quarterly_filtered = [p for p in quarterly_points if not _is_estimate_label(p.label)]
    quarters = _sort_quarterly(quarterly_filtered)
    if len(quarters) < 4:
        raise ValueError("Need at least 4 quarterly EPS points (excluding estimates).")

    last4 = quarters[-4:]
    est_annual_eps = sum(q.eps for q in last4)

    latest_q = quarters[-1]
    if latest_q.bps is None:
        raise ValueError("Latest quarterly BPS is required.")

    prev1 = annual[-1]
    prev2 = annual[-2]

    weighted = _weighted_eps(est_annual_eps, prev1.eps, prev2.eps)
    intrinsic = _intrinsic_value(latest_q.bps, weighted)

    implied_price = None
    if pbr is not None:
        implied_price = latest_q.bps * pbr

    return IntrinsicValueResult(
        method="annual+quarterly",
        bps=latest_q.bps,
        weighted_eps=weighted,
        intrinsic_value=intrinsic,
        implied_price=implied_price,
        implied_pbr=pbr,
        note=f"estimated_annual_eps={est_annual_eps}",
    )


def build_pipeline_summary(
    annual_points: Iterable[AnnualPoint],
    quarterly_points: Optional[Iterable[QuarterlyPoint]] = None,
    *,
    pbr: Optional[float] = None,
) -> dict:
    """
    Pipeline entrypoint.
    Returns a serializable summary dict for UI or JSON output.
    """
    results: list[IntrinsicValueResult] = []
    results.append(compute_intrinsic_value_annual(annual_points, pbr=pbr))

    if quarterly_points:
        results.append(
            compute_intrinsic_value_with_quarters(
                annual_points, quarterly_points, pbr=pbr
            )
        )

    return {
        "as_of": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "results": [
            {
                "method": r.method,
                "bps": r.bps,
                "weighted_eps": r.weighted_eps,
                "intrinsic_value": r.intrinsic_value,
                "implied_price": r.implied_price,
                "implied_pbr": r.implied_pbr,
                "note": r.note,
            }
            for r in results
        ],
    }
