#!/usr/bin/env python
"""Live demo: browser mic -> Gnani STT -> agent -> Gnani TTS.   pip install -r requirements-demo.txt ; python demo_server.py"""
import base64
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from zentro.factory import Runtime  # noqa: E402
from zentro.voice import GnaniVoice, VoiceAgent, detect_lang, for_ear  # noqa: E402

app = FastAPI()
rt = Runtime()
voice = GnaniVoice(rt.cfg)
sessions: dict[str, VoiceAgent] = {}


def session(sid: str) -> VoiceAgent:
    if sid not in sessions:
        sessions[sid] = VoiceAgent(rt.new_agent("full", "voice"), voice, rt.cfg)
    return sessions[sid]


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")


@app.post("/api/reset")
def reset(sid: str = Form(...)):
    sessions.pop(sid, None)
    return {"ok": True}


@app.post("/api/turn")
async def turn(sid: str = Form(...), language: str = Form("en-IN"), audio: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as d:
        inp, outp = Path(d) / "in.wav", Path(d) / "out.wav"
        inp.write_bytes(await audio.read())
        rec = session(sid).turn(inp, outp, language)
        b64 = base64.b64encode(outp.read_bytes()).decode() if outp.exists() else ""
    rec["audio_b64"] = b64
    return JSONResponse(rec)


@app.post("/api/chat")
async def chat(sid: str = Form(...), message: str = Form(...), language: str = Form("en-IN")):
    with tempfile.TemporaryDirectory() as d:
        outp = Path(d) / "out.wav"
        agent = session(sid).agent
        r = agent.respond(message)
        spoken = for_ear(r.text, rt.cfg.today)
        lang = detect_lang(spoken, language)
        tts = voice.synthesize(spoken, outp, lang=lang)
        b64 = base64.b64encode(outp.read_bytes()).decode() if outp.exists() else ""
    return JSONResponse({
        "transcript": message,
        "reply": r.text,
        "spoken_text": spoken,
        "decision": r.decision,
        "used_sections": r.used_sections,
        "retrieved": r.retrieved,
        "tool_calls": r.tool_calls,
        "query": r.query,
        "ticket_id": r.ticket_id,
        "ticket_created": r.ticket_created,
        "audio_b64": b64
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
