# VoiceOPS-Agent-ZENTROMART- (ZentroMart VoiceOps Agent)

We designed and implemented a production-grade, four-step voice customer-support agent for **ZentroMart**, an Indian e-commerce retailer selling groceries and electronics. The agent runs locally with full privacy and determinism, handles complex multi-turn support workflows, and communicates naturally over voice via Gnani STT and TTS models.

```
customer audio -> Gnani STT -> [ normalise -> rewrite query -> hybrid retrieve -> LLM + tools -> guardrails ] -> Gnani TTS -> spoken reply
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill GNANI_API_KEY and OPENROUTER_API_KEY
# extract the organisers' dataset next to this repo (not committed), then:
export DATASET_DIR=./Capstone_Project_Dataset_FTTT_2026
python Capstone_Project_Dataset_FTTT_2026/validate_dataset.py
```

## Run (the organisers' command)
```bash
python run.py --audio-dir ./test_audio --out ./outputs
```
For each audio file it writes `<name>.json` (transcript, reply text, decision, **retrieved chunks**, sections used, tool calls), `<name>_reply.wav`, plus `results.jsonl` and `summary.csv`. A failing file never stops the batch; it still gets a spoken reply.
Multi-turn audio: name files `conv01_turn1.wav`, `conv01_turn2.wav` and they share one conversation. Any other name is its own conversation.

## Evaluate (Step 1 vs 2 vs 3)
```bash
python evaluate.py --out eval_results --judge          # needs an LLM key; add --col asr_text to test the speech-to-text wording
python scripts_retrieval_eval.py $DATASET_DIR question pin   # no key needed: recall@k of gold_evidence sections
python scripts_smoke_real.py $DATASET_DIR                    # no key needed: replays all 96 practice turns' tool calls through the real tools
```
`evaluate.py` scores, from the practice CSVs: decision match (using `acceptable_decisions`), `must_include` / `must_not_include`, retrieval recall of `gold_evidence`, expected tool calls (`lookup_order`, `create_ticket`), and optionally an LLM judge against `reference_answer`.

## Live demo
```bash
pip install -r requirements-demo.txt && python demo_server.py   # http://127.0.0.1:8000
```

## Tests (no network, no keys)
```bash
pip install pytest && python -m pytest tests -q
```

## What is in `src/zentro/`
| File | Job |
|---|---|
| `kb.py` | Section-aware chunking that keeps `[SECTION-ID]`, status, effective_from. BM25 + optional dense, rank fusion, authority precedence (policy > agent > help > FAQ > marketing), superseded docs excluded, cross-reference expansion |
| `agent.py` | One class, three modes: `baseline` (Step 1), `rag` (Step 2), `full` (Step 3). Tool loop with forced structured output |
| `tools_adapter.py` | Wraps the provided `lookup_order` / `create_ticket` and adds `lookup_product` (products.csv) and `check_pincode` (pincodes.csv). One verification value at a time, one retry per order, pincode hidden from the model, escalation flags (repeat contact, item over 10,000 rupees) computed in code, tickets reported only if `created=True` |
| `dates.py` | Exact business-day (Mon-Sat, holidays skipped) and calendar-day maths exposed as a tool |
| `speech_norm.py` | Turns spoken order IDs ("8978 2837", "eight nine seven...") and SKUs ("ele zero zero one to") into real ones; learns the ID format from `orders.csv` |
| `lang.py` | English / Hindi / Hinglish detection and the AI-assistant opening line in the customer's language |
| `voice.py` | Gnani STT/TTS, retries, >60 s clip splitting, text written for the ear, language-aware voices, recovery from unclear audio |

## Rules we followed
Gnani is the only speech service. No answers are hardcoded: every answer comes from retrieved chunks, tool output or date maths. Only the synthetic dataset is used. No keys are committed.
