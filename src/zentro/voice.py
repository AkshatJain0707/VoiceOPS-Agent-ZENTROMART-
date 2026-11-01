"""Gnani STT/TTS layer: retries, long-clip splitting, ear-friendly text, language-aware voices."""
from __future__ import annotations
import io
import os
import re
import time
import wave
from datetime import date
from pathlib import Path

import requests

from .lang import is_hinglish

STT_URL = "https://api.vachana.ai/stt/v3"
ITN_LANGS = {"en-IN", "hi-IN"}  # ITN is only supported for these (per Gnani docs)
BOOST_WORDS = ["ZentroMart", "refund", "refunds", "cancellation", "cancel", "return", "replacement", "warranty",
               "delivery", "shipped", "dispatched", "invoice", "ticket", "compensation", "pincode", "UPI", "COD",
               "Plus", "voucher", "exchange", "damaged", "defective"]
# TODO(verify on first call): voice names for non-English come from docs.gnani.ai/voice-catalog; override in .env if different.
VOICES = {"hi-en": "Poorvi", "en-IN": "Kaveri", "hi-IN": "Nalini", "ta-IN": "Asmita", "te-IN": "Suhana", "kn-IN": "Saanvi",
          "ml-IN": "Reshma", "mr-IN": "Zahira", "bn-IN": "Kirra", "gu-IN": "Falak", "pa-IN": "Mehuli"}
SCRIPTS = [("hi-IN", "\u0900", "\u097f"), ("bn-IN", "\u0980", "\u09ff"), ("pa-IN", "\u0a00", "\u0a7f"),
           ("gu-IN", "\u0a80", "\u0aff"), ("ta-IN", "\u0b80", "\u0bff"), ("te-IN", "\u0c00", "\u0c7f"),
           ("kn-IN", "\u0c80", "\u0cff"), ("ml-IN", "\u0d00", "\u0d7f")]
MAX_CLIP_S = 55
MAX_TTS_CHARS = 380


# ----------------------------------------------------------------------------- text for the ear
def detect_lang(text: str, prefer: str = "en-IN") -> str:
    counts = {}
    for ch in text:
        for code, lo, hi in SCRIPTS:
            if lo <= ch <= hi:
                counts[code] = counts.get(code, 0) + 1
    if not counts:
        return "hi-en" if is_hinglish(text) else "en-IN"
    best = max(counts, key=counts.get)
    if best == "hi-IN" and prefer == "mr-IN":
        return "mr-IN"
    return best


def _year_for(month: int, today: date) -> int:
    return today.year + 1 if month < today.month - 6 else today.year


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
_MN = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def for_ear(text: str, today: date | None = None) -> str:
    """Make a written reply safe and clear to speak: no markdown, years on dates, grouped long IDs."""
    t = text or ""
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)
    t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", t)
    t = re.sub(r"\[[A-Za-z0-9_.\-]+\]", "", t)          # strip any [SECTION-ID] that leaked
    t = re.sub(r"\(\s*\)", "", t)
    today = today or date.today()

    def add_year(m):
        mon = _MN[m.group("mon")[:3].lower()]
        return f"{m.group(0)} {_year_for(mon, today)}"
    t = re.sub(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?P<mon>{_MONTHS})\b(?!\s*,?\s*\d{{4}})", add_year, t)
    t = re.sub(rf"\b(?P<mon>{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{{4}})", add_year, t)

    t = re.sub(r"\bRs\.?\s*(\d+)", r"\1 rupees", t, flags=re.I)

    # Ticket ids like TKT + 32 hex chars are unreadable aloud: say only the last six characters, spaced out.
    t = re.sub(r"\b(ticket\s*(?:number|id)\s*(?:is\s*)?)TKT([0-9a-fA-F]{8,})\b",
               lambda m: "ticket reference ending " + " ".join(m.group(2)[-6:].upper()), t, flags=re.I)
    t = re.sub(r"\bTKT([0-9a-fA-F]{8,})\b", lambda m: "ending " + " ".join(m.group(1)[-6:].upper()), t)

    def group_id(m):  # 89782837 -> 8978 2837 (STY-3: order IDs in groups of four digits)
        pre, digits = m.group(1), m.group(2)
        return (pre + " " if pre else "") + " ".join(digits[i:i + 4] for i in range(0, len(digits), 4))
    t = re.sub(r"(?<![\d.,])\b([A-Za-z]{0,5})[-_]?(\d{7,})\b(?![.,]\d)", group_id, t)
    return re.sub(r"\s{2,}", " ", t).strip()


def split_sentences(text: str, limit: int = MAX_TTS_CHARS) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out or [text]


# ----------------------------------------------------------------------------- audio utils
def wav_duration(path: str | Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            file_bytes = max(0, Path(path).stat().st_size - 44)
            bytes_per_frame = nchannels * sampwidth
            if bytes_per_frame and framerate:
                actual_frames = file_bytes // bytes_per_frame
                if nframes > actual_frames or nframes >= 2147483647:
                    return actual_frames / float(framerate)
                return nframes / float(framerate)
            return None
    except Exception:
        return None


def split_wav(path: str | Path, seg_s: int = MAX_CLIP_S) -> list[bytes]:
    """Split a long WAV into <=seg_s clips (stdlib only). Returns [] if not a plain WAV."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            per = rate * seg_s
            out = []
            while True:
                frames = w.readframes(per)
                if not frames:
                    break
                buf = io.BytesIO()
                with wave.open(buf, "wb") as o:
                    o.setnchannels(nchannels)
                    o.setsampwidth(sampwidth)
                    o.setframerate(rate)
                    o.writeframes(frames)
                out.append(buf.getvalue())
            return out
    except Exception:
        return []


def concat_wavs(chunks: list[bytes]) -> bytes:
    if len(chunks) == 1:
        return chunks[0]
    buf = io.BytesIO()
    params = None
    with wave.open(buf, "wb") as out:
        for c in chunks:
            with wave.open(io.BytesIO(c), "rb") as w:
                if params is None:
                    params = w.getparams()
                    out.setparams(params)
                out.writeframes(w.readframes(w.getnframes()))
    return buf.getvalue()


# ----------------------------------------------------------------------------- Gnani clients
class GnaniVoice:
    def __init__(self, cfg):
        self.cfg = cfg
        self.key = os.environ.get("GNANI_API_KEY", "")
        self._tts = None

    # --- STT: raw REST so we can use bias_list; same Gnani endpoint the SDK wraps ---
    def _stt_bytes(self, data: bytes, filename: str, language: str, mode: str) -> dict:
        form = {"language_code": language}
        if mode == "boost":
            form["format"] = "verbatim"
            form["bias_list"] = '["' + '","'.join(w for w in BOOST_WORDS if w.isalpha()) + '"]'
            form["bias_score"] = "1.0"
        elif language in ITN_LANGS:
            form["format"] = "transcribe"
        last = ""
        for attempt in range(4):
            try:
                r = requests.post(STT_URL, headers={"X-API-Key-ID": self.key},
                                  files={"audio_file": (filename, data)}, data=form, timeout=60)
                if r.status_code == 200:
                    return r.json()
                last = f"{r.status_code} {r.text[:200]}"
                if r.status_code in (400, 401, 403):
                    break
            except requests.RequestException as e:
                last = str(e)
            time.sleep(1.0 * (2 ** attempt))
        return {"success": False, "transcript": "", "error": last}

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> dict:
        language = language or self.cfg.stt_language
        path = Path(audio_path)
        dur = wav_duration(path)
        clips = split_wav(path) if (dur and dur > MAX_CLIP_S) else [path.read_bytes()]
        texts, errs = [], []
        for i, data in enumerate(clips):
            name = path.name if len(clips) == 1 else f"part{i}.wav"
            res = self._stt_bytes(data, name, language, self.cfg.stt_mode)
            if res.get("transcript"):
                texts.append(res["transcript"].strip())
            elif res.get("error"):
                errs.append(res["error"])
        return {"transcript": " ".join(texts).strip(), "error": "; ".join(errs), "language": language,
                "clips": len(clips), "duration_s": dur}

    # --- TTS via the official SDK ---
    def _client(self):
        if self._tts is None:
            from gnani.tts import GnaniTTSClient
            self._tts = GnaniTTSClient()
        return self._tts

    def synthesize(self, text: str, out_path: str | Path, lang: str = "en-IN", voice: str | None = None,
                   speed: float = 1.0) -> dict:
        voice = voice or (self.cfg.tts_voice if lang == "en-IN" else VOICES.get(lang, self.cfg.tts_voice))
        api_lang = "auto" if lang == "hi-en" else lang
        # try the language-matched voice first, then fall back to the default English voice so a reply is never silent
        chain = [(voice, api_lang)] + ([(self.cfg.tts_voice, "en-IN")] if (voice, api_lang) != (self.cfg.tts_voice, "en-IN") else [])
        pieces, err = [], ""
        for sent in split_sentences(text):
            audio = None
            for v, lg in chain:
                for attempt in range(2):
                    try:
                        audio = self._client().synthesize(sent, voice=v, model=self.cfg.tts_model, language=lg, speed=speed)
                        break
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        time.sleep(1.0 * (2 ** attempt))
                if audio:
                    break
            if audio:
                pieces.append(audio)
        if not pieces:
            return {"ok": False, "error": err}
        try:
            data = concat_wavs(pieces)
        except Exception:
            data = pieces[0]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(data)
        return {"ok": True, "voice": voice, "lang": lang, "bytes": len(data)}


# ----------------------------------------------------------------------------- recovery phrases
UNCLEAR = ["Sorry, I didn't catch that. Could you say it again?",
           "I'm still having trouble hearing you. Could you speak a little slower and closer to the microphone?",
           "I'm sorry, the line is unclear. Please try calling again, or send us a message and we will help."]


def is_unclear(transcript: str) -> bool:
    return len(re.findall(r"\w", transcript or "")) < 3


class VoiceAgent:
    """STT -> agent -> TTS, with recovery from unclear audio. Same brain as the text agent."""

    def __init__(self, agent, voice: GnaniVoice, cfg):
        self.agent, self.voice, self.cfg = agent, voice, cfg
        self.unclear_streak = 0

    def turn(self, audio_path: str | Path, out_wav: str | Path, language: str | None = None) -> dict:
        language = language or self.cfg.stt_language
        stt = self.voice.transcribe(audio_path, language)
        heard = stt["transcript"]
        if is_unclear(heard) and language == "en-IN":  # pure-Hindi speech sometimes needs the Hindi model
            alt = self.voice.transcribe(audio_path, "hi-IN")
            if not is_unclear(alt["transcript"]):
                stt, heard, language = alt, alt["transcript"], "hi-IN"
        rec = {"audio": str(audio_path), "transcript": heard, "stt_error": stt.get("error", ""), "stt_language": language}
        if is_unclear(heard):
            reply = UNCLEAR[min(self.unclear_streak, len(UNCLEAR) - 1)]
            self.unclear_streak += 1
            rec.update(reply=reply, decision="clarify", used_sections=[], retrieved=[], tool_calls=[], unclear=True)
        else:
            self.unclear_streak = 0
            r = self.agent.respond(heard)
            reply = r.text
            rec.update(reply=reply, decision=r.decision, used_sections=r.used_sections, retrieved=r.retrieved,
                       tool_calls=r.tool_calls, query=r.query, ticket_id=r.ticket_id, ticket_created=r.ticket_created)
        spoken = for_ear(reply, self.cfg.today)
        lang = detect_lang(spoken, language)
        tts = self.voice.synthesize(spoken, out_wav, lang=lang)
        rec.update(spoken_text=spoken, tts=tts, reply_audio=str(out_wav) if tts.get("ok") else "")
        return rec
