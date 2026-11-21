"""GTA VI trophy earn-rate forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for each trophy, five quantiles of its Exophase earn rate at the
case's target. Standard library only, deterministic, no network.

Model, per trophy, in logit space:
  * x = ln(1 + hours since the game's release, 2026-11-19T00:00Z). Trophy
    progress is roughly linear in log-time: players move through the game
    fast at first, then the tail slows down, and new players joining dilute
    the rates of trophies they have not earned yet. A logit-linear fit in
    log-time captures both the rise of earners and that dilution in one
    slope, and it cannot run away the way a slope in raw hours can over a
    4.5-day horizon.
  * The slope is fit by least squares on the case's reads from the last
    36 hours (all of them when there are fewer), weighted toward recent
    reads, then shrunk toward the median slope of the trophy's tier: a
    single trophy's noisy reads should not drag the forecast on their own.
  * The median is the last read's logit moved by the shrunk slope over
    (x_target - x_last), with the move clamped to [-1.5, +2.5] logit.
  * The spread is honest about what we do not know: a floor for refresh
    jumps Exophase makes on its own schedule, a random-walk term that grows
    with the square root of the horizon in days, the residuals of the fit,
    and the slope's own uncertainty. A trophy whose page has not moved at
    all (the practice cases, frozen at 638 players) still gets a real
    spread, because the next refresh can move every rate at once.
  * The 100% trophy stays at 100%: every tracked player has it, and the
    next tracked players will too.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# How far back the slope fit looks, in hours.
FIT_WINDOW_HOURS = 36.0
# Clamp on the median's logit move from the last read.
DELTA_MIN, DELTA_MAX = -1.5, 2.5
# Spread floor (logit) for refresh jumps, and the random-walk rate per sqrt day.
SPREAD_FLOOR = 0.30
SPREAD_PER_SQRT_DAY = 0.45


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t: datetime) -> float:
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def fit_slope(xs, ys, ws):
    """Weighted least-squares slope of y on x; returns (slope, resid_var, n)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return 0.0, 0.0, len(xs)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-9:
        return 0.0, 0.0, len(xs)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    resid_var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return slope, resid_var, len(xs)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))
    horizon_days = case["horizon_minutes"] / 1440.0

    last = history[-1]
    last_time = parse_time(last["at"])
    x_last = math.log1p(hours_since_release(last_time))
    # The target is always after the checkpoint, but never extrapolate
    # backwards if a case ever says otherwise.
    dx = max(0.0, x_target - x_last)

    # Reads for the fit: the last 36 hours, oldest first.
    cutoff = last_time.timestamp() - FIT_WINDOW_HOURS * 3600.0
    fit_reads = [h for h in history if parse_time(h["at"]).timestamp() >= cutoff]
    if len(fit_reads) < 2:
        fit_reads = history[-2:] if len(history) >= 2 else history

    trophies = case["trophies"]
    n = len(trophies)

    # Per-trophy slopes, then the tier median each slope shrinks toward.
    slopes = []
    resid_vars = []
    last_rates = []
    for i in range(n):
        xs, ys, ws = [], [], []
        for h in fit_reads:
            rate = h["rates"][i]
            if rate is None:
                continue
            t = parse_time(h["at"])
            xs.append(math.log1p(hours_since_release(t)))
            ys.append(logit(rate))
            # Recent reads count double: the game's first days move fast.
            age_h = (last_time - t).total_seconds() / 3600.0
            ws.append(2.0 if age_h <= 12.0 else 1.0)
        slope, resid_var, _ = fit_slope(xs, ys, ws)
        slopes.append(slope)
        resid_vars.append(min(resid_var, 0.25))
        rate = last["rates"][i]
        if rate is None:
            # Fall back to the most recent non-null read for this trophy.
            rate = next(
                (h["rates"][i] for h in reversed(history) if h["rates"][i] is not None),
                0.5,
            )
        last_rates.append(rate)

    tier_slopes = {}
    for tier in {t["tier"] for t in trophies}:
        vals = sorted(slopes[i] for i, t in enumerate(trophies) if t["tier"] == tier)
        tier_slopes[tier] = vals[len(vals) // 2] if vals else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate >= 99.99:
            # Everybody tracked has it; the next players will too.
            rows.append({"id": trophy["id"], "q": [99.95, 99.99, 100, 100, 100]})
            continue
        # Shrink the trophy's slope toward its tier's: half weight each
        # when the fit used few reads, more weight to the trophy as reads grow.
        w_trophy = 0.5 if len(fit_reads) <= 4 else 0.7
        slope = w_trophy * slopes[i] + (1.0 - w_trophy) * tier_slopes[trophy["tier"]]
        delta = max(DELTA_MIN, min(DELTA_MAX, slope * dx))
        median_logit = logit(rate) + delta
        # Spread: floor + random walk over the horizon + fit residuals +
        # uncertainty in the slope over the extrapolation distance.
        slope_se = 0.15 + 0.10 * abs(slope)
        sigma = math.sqrt(
            SPREAD_FLOOR ** 2
            + (SPREAD_PER_SQRT_DAY ** 2) * horizon_days
            + resid_vars[i]
            + (slope_se * dx) ** 2
        )
        qs = [round4(min(99.99, max(0.01, sigmoid(median_logit + z * sigma)))) for z in Z]
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
