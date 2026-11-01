from __future__ import annotations
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Config:
    dataset_dir: Path = field(default_factory=lambda: Path(_env("DATASET_DIR", "./Capstone_Project_Dataset_FTTT_2026")))
    today: date = field(default_factory=lambda: date.fromisoformat(_env("TODAY", "2026-10-20")))
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openrouter"))  # openrouter | openai | anthropic
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "anthropic/claude-sonnet-5"))
    llm_fast_model: str = field(default_factory=lambda: _env("LLM_FAST_MODEL", "") or _env("LLM_MODEL", "anthropic/claude-sonnet-5"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", ""))
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "auto"))  # auto | none
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "8")))
    stt_language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "en-IN"))
    stt_mode: str = field(default_factory=lambda: _env("STT_MODE", "itn"))  # itn | boost
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "timbre-v2.5"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "Kaveri"))
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", ".cache")))

    @property
    def kb_dir(self) -> Path:
        return self.dataset_dir / "kb"

    @property
    def data_dir(self) -> Path:
        return self.dataset_dir / "data"

    @property
    def practice_dir(self) -> Path:
        return self.dataset_dir / "practice"
