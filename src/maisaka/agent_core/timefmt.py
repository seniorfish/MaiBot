"""Human-friendly duration text for wait/sleep tool results."""

from __future__ import annotations

SHORT_WAIT_SECONDS = 1800.0


def format_wait_duration(seconds: float) -> str:
    """Format an elapsed wait; long sleeps switch to day/hour/minute units.

    Short waits keep one decimal so the model can reason about sub-minute
    timing; anything from half an hour up reads better as 2 小时 13 分.
    """

    seconds = max(0.0, float(seconds))
    if seconds < SHORT_WAIT_SECONDS:
        return f"{seconds:.1f} 秒"
    remaining = int(round(seconds))
    days, remaining = divmod(remaining, 86400)
    hours, remaining = divmod(remaining, 3600)
    minutes = remaining // 60
    parts = []
    if days:
        parts.append(f"{days} 天")
    if hours:
        parts.append(f"{hours} 小时")
    # Days and hours already carry enough precision; minutes only matter below a day.
    if minutes and not days:
        parts.append(f"{minutes} 分")
    return " ".join(parts) or "0 秒"
