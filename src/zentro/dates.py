"""Deterministic date maths. The LLM never counts days itself; it calls this."""
from __future__ import annotations
import csv
from datetime import date, timedelta
from pathlib import Path


def load_holidays(data_dir: Path) -> set[date]:
    p = Path(data_dir) / "holidays.csv"
    out: set[date] = set()
    if not p.exists():
        return out
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.add(date.fromisoformat((row.get("date") or "").strip()))
            except ValueError:
                continue
    return out


def fmt(d: date) -> str:
    return f"{d.strftime('%A')}, {d.day} {d.strftime('%B %Y')}"


class DateCalc:
    """Business days run Mon-Sat excluding holidays. Return windows are calendar days."""

    def __init__(self, today: date, holidays: set[date] | None = None):
        self.today = today
        self.holidays = holidays or set()

    def parse(self, s: str) -> date:
        s = (s or "").strip().lower()
        if s in ("today", "now", ""):
            return self.today
        return date.fromisoformat(s[:10])

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 6 and d not in self.holidays

    def add_business_days(self, start: date, n: int) -> date:
        step = 1 if n >= 0 else -1
        d, left = start, abs(n)
        while left:
            d += timedelta(days=step)
            if self.is_business_day(d):
                left -= 1
        return d

    def business_days_between(self, a: date, b: date) -> int:
        """Business days elapsed after `a` up to and including `b` (negative if b < a)."""
        if b < a:
            return -self.business_days_between(b, a)
        d, n = a, 0
        while d < b:
            d += timedelta(days=1)
            if self.is_business_day(d):
                n += 1
        return n

    def run(self, op: str, date_a: str = "today", date_b: str = "today", days: int = 0) -> dict:
        a = self.parse(date_a)
        if op == "add_business_days":
            r = self.add_business_days(a, int(days))
            return {"result": r.isoformat(), "human": fmt(r), "rule": "Mon-Sat, holidays skipped"}
        if op == "add_calendar_days":
            r = a + timedelta(days=int(days))
            return {"result": r.isoformat(), "human": fmt(r), "rule": "calendar days"}
        b = self.parse(date_b)
        if op == "business_days_between":
            return {"result": self.business_days_between(a, b), "rule": "Mon-Sat, holidays skipped, start day excluded"}
        if op == "calendar_days_between":
            return {"result": (b - a).days, "rule": "calendar days (b minus a)"}
        if op == "weekday":
            return {"result": a.strftime("%A"), "is_business_day": self.is_business_day(a), "human": fmt(a)}
        return {"error": f"unknown op {op}"}
