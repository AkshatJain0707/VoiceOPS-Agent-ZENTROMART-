from __future__ import annotations
import csv
from .agent import SupportAgent
from .config import Config
from .dates import DateCalc, load_holidays
from .kb import KnowledgeBase
from .llm import make_llm
from .speech_norm import infer_id_format
from .tools_adapter import ToolBox


class Runtime:
    """Heavy objects built once (KB index, LLM client, tools); cheap per-conversation agents on top."""

    def __init__(self, cfg: Config | None = None, with_kb: bool = True):
        self.cfg = cfg or Config()
        self.llm = make_llm(self.cfg)
        self.kb = KnowledgeBase(self.cfg.kb_dir, self.cfg.embed_backend, self.cfg.embed_model, self.cfg.cache_dir) if with_kb else None
        self.calc = DateCalc(self.cfg.today, load_holidays(self.cfg.data_dir))
        self.order_fmt = self._order_fmt()
        self.tools = ToolBox(self.cfg, self.calc, self.order_fmt)

    def _order_fmt(self):
        p = self.cfg.data_dir / "orders.csv"
        if not p.exists():
            return None
        with open(p, newline="", encoding="utf-8") as f:
            ids = [r["order_id"] for _, r in zip(range(200), csv.DictReader(f)) if r.get("order_id")]
        return infer_id_format(ids)

    def new_agent(self, mode: str = "full", channel: str = "text") -> SupportAgent:
        return SupportAgent(self.llm, self.kb, self.tools, self.cfg, mode=mode, channel=channel, top_k=self.cfg.top_k)
