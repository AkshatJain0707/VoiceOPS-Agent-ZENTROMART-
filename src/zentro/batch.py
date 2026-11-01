"""Batch runner: folder of customer audio -> reply text + reply audio + retrieved chunks, one JSON per file."""
from __future__ import annotations
import csv
import json
import re
import traceback
from pathlib import Path

from .factory import Runtime
from .voice import GnaniVoice, VoiceAgent, UNCLEAR

AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
TURN_RE = re.compile(r"^(?P<conv>.+?)[_\-.]?(?:t|turn)[_\-]?(?P<n>\d+)$", re.I)


def group_files(files: list[Path]) -> dict[str, list[Path]]:
    """conv01_turn2.wav style names become multi-turn conversations; anything else is its own conversation."""
    groups: dict[str, list[tuple[int, Path]]] = {}
    for f in files:
        m = TURN_RE.match(f.stem)
        key, n = (m.group("conv"), int(m.group("n"))) if m else (f.stem, 1)
        groups.setdefault(key, []).append((n, f))
    return {k: [p for _, p in sorted(v)] for k, v in sorted(groups.items())}


def run_batch(audio_dir: str, out_dir: str, language: str | None = None, rt: Runtime | None = None) -> list[dict]:
    rt = rt or Runtime()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT)
    voice = GnaniVoice(rt.cfg)
    results: list[dict] = []
    for conv, paths in group_files(files).items():
        va = VoiceAgent(rt.new_agent("full", "voice"), voice, rt.cfg)
        for p in paths:
            try:
                rec = va.turn(p, out / f"{p.stem}_reply.wav", language)
            except Exception as e:  # one bad file must never stop the batch
                rec = {"audio": str(p), "transcript": "", "reply": UNCLEAR[0], "error": f"{type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-800:]}
                try:
                    voice.synthesize(UNCLEAR[0], out / f"{p.stem}_reply.wav")
                    rec["reply_audio"] = str(out / f"{p.stem}_reply.wav")
                except Exception:
                    pass
            rec["conversation"] = conv
            rec["file"] = p.name
            (out / f"{p.stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(rec)
            safe_heard = (rec.get('transcript', '') or '')[:60].encode('ascii', 'replace').decode()
            safe_reply = (rec.get('reply', '') or '')[:70].encode('ascii', 'replace').decode()
            print(f"[{conv}] {p.name}: heard={safe_heard!r} -> {safe_reply!r}")
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "conversation", "transcript", "reply_text", "decision", "sections_used", "retrieved_chunks", "reply_audio"])
        for r in results:
            w.writerow([r.get("file"), r.get("conversation"), r.get("transcript"), r.get("reply"), r.get("decision"),
                        ";".join(r.get("used_sections", [])),
                        ";".join(f"{c['doc']}#{c['section_id']}" for c in r.get("retrieved", [])), r.get("reply_audio", "")])
    return results
