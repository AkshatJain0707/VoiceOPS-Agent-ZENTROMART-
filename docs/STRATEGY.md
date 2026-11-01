# Winning plan: VoiceOps 2026

## Where the marks are (and what covers each)
| Marks | Criterion | What in this repo targets it |
|---|---|---|
| 25 | Grounded answers + honest "I don't know" | Section-aware retrieval that follows `00_about_this_knowledge_base`; superseded sections used only for old dates; uncited answers bounced; `dont_know` replies offer a colleague (the practice set's expected answer); product and pincode tools for two-source questions |
| 15 | Retrieval quality | Hybrid retrieval, English/Hindi/Hinglish vocabulary bridge, LLM query rewrite (translates, resolves "that"), cross-references, pinned guide sections. Keyword-only recall@6 of gold sections is 0.95 on the practice questions |
| 20 | Conversation | Code-held state; unverified lookup first, then one verification value, one retry, then blocked; per-order verification; ESC-1 flags computed from the order record; refuse vs escalate as the practice set expects; tickets only when `created=True` |
| 10 | Own evaluation | `evaluate.py`: baseline vs RAG vs full, with decision match, must_include, recall, tool-call match |
| 10 | Voice end to end | Always speaks, never crashes on a bad file, splits clips over 60 s, retries, Hindi STT retry for unclear audio, TTS voice fallback |
| 10 | Voice experience | Two or three short sentences; dates with month and year; order IDs in groups of four (STY-3); long ticket ids shortened to their last six characters; Hinglish voice for Hinglish replies |
| 10 | Report + demo | Template, live demo with an evidence panel |

## What the dataset taught the design
- Order IDs are 8 digits spoken as "8978 2837"; `lookup_order(order_id)` alone shows nothing private, then `verification_value` (4-digit phone suffix or 6-digit pincode). The practice set expects exactly that two-call pattern, and a retry on a wrong value.
- Practice conversations expect: refuse (no ticket) for compensation demands; escalate with a ticket for ESC-1 cases (human requested, third contact, item over 10,000 rupees, and so on).
- About 40% of practice questions are Hinglish or Hindi, and `asr_text` is lowercase with spoken digits, so the vocabulary bridge, query rewrite and SKU/ID normaliser matter.
- Some answers need `products.csv` (return category, warranty) or `pincodes.csv`, which plain document search cannot supply.

## Verified here (no API keys or network to Gnani/LLMs)
22 offline tests; all 96 practice-conversation turns replayed through the real `tools/tools.py` with the expected tool calls matching (order-ID normalisation, verification, retry cap, tickets); keyword-only retrieval recall 0.95 at k=6 and 0.97 at k=8 on the practice questions; pinned guide sections included.

## Not verified, do these first (about an hour)
1. Put `GNANI_API_KEY` and an LLM key in `.env`. Run `python evaluate.py --limit 8 --judge`, read the replies, then the full run. Compare `--col question` with `--col asr_text`.
2. Record 10 clips (English, Hinglish, Hindi, one with a spoken order ID, one over a minute). Run `run.py`, listen to every reply wav.
3. Check the Hinglish and Hindi voices: `voice.VOICES` names came from my notes of the voice catalog; fix any that error (the code falls back to the English voice rather than going silent).
4. Optional dense retrieval: `pip install fastembed` downloads a multilingual model on first run; confirm it loads on your machine and on the organisers' (otherwise it silently uses keyword retrieval, which is fine).
5. Open `web/index.html` via `demo_server.py` in Chrome and test the recorder.

## 2-minute demo script
0:00 "Hello, my order hasn't arrived, can I cancel?" Agent says it is an AI assistant, asks for the order number.
0:20 Say the order number in groups of four. Evidence panel: `lookup_order` called, not yet verified. Agent asks for last four digits or pincode.
0:40 Give the last four digits. Panel shows verified; agent explains the shipped-order cancellation policy with the section ids used.
1:05 "How long will the refund take?" No repeated questions, exact date from the date tool.
1:25 "I want 500 rupees compensation right now." Polite refusal with what policy allows. Then "Can I talk to a person?": ticket created, reference spoken.
1:45 Ask something the documents do not cover: "Is there a student discount?" It says it has no information and offers a colleague.
1:55 One sentence with your Step 1 vs Step 2 numbers.

## Cheap experiments for the report
`STT_MODE=boost` vs `itn`; `TOP_K` 6 vs 8 vs 10; `EMBED_BACKEND=none` vs `auto`; with and without pinned sections. One table, one sentence of insight each.
