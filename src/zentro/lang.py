"""Tiny language helpers: script detection, Hinglish detection, opening line per language."""
from __future__ import annotations
import re

HINGLISH = {"hai", "hain", "kya", "mujhe", "mera", "meri", "mere", "aap", "aapka", "nahi", "nahin", "kab", "karna", "karein",
            "kar", "karo", "chahiye", "haan", "aur", "ek", "batao", "bhi", "toh", "tak", "wala", "wali", "mein", "pe", "se",
            "ko", "ke", "ki", "ka", "hoon", "hun", "sakta", "sakti", "sakte", "kaise", "kitne", "kitna", "paisa", "wapas",
            "aaya", "aayega", "mil", "milega", "theek", "achha", "abhi", "baare", "madad", "sawaal", "dena", "lena",
            "bhai", "shukriya", "dhanyawad", "namaste", "namaskar", "bol", "bolo", "boliye", "bhejo", "bheja", "paas",
            "liye", "kuch", "kuchh", "kardo", "kar do", "bataye", "batayein", "poochna", "jaanna"}


def has_devanagari(t: str) -> bool:
    return bool(re.search(r"[\u0900-\u097f]", t or ""))


def is_hinglish(t: str) -> bool:
    if has_devanagari(t):
        return False
    words = re.findall(r"[a-z]+", (t or "").lower())
    return sum(w in HINGLISH for w in words) >= 2


def opening_line(user_text: str) -> str:
    if has_devanagari(user_text):
        return "नमस्ते, मैं ZentroMart की एआई सहायक हूँ। "
    if is_hinglish(user_text):
        return "Namaste, main ZentroMart ki AI assistant hoon. "
    return "Hi, I'm ZentroMart's AI assistant. "


AI_WORDS = re.compile(r"\bAI\b|artificial|एआई|कृत्रिम", re.I)
