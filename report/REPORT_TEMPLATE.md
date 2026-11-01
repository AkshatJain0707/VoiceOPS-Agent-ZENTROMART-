# VoiceOps 2026: Report (max 4 pages)

## 1. What we built
Step 1: baseline prompt-only assistant. Step 2: hybrid RAG (BM25 + dense, precedence-aware, cross-refs). Step 3: verified multi-turn agent with tools and date maths. Step 4: Gnani STT (ITN) + TTS, ear-friendly replies, recovery.

## 2. Step 1 vs Step 2 (from `eval_results/summary.json`)
| Metric | Step 1 baseline | Step 2 RAG | Step 3 full (conversations) |
|---|---|---|---|
| Accuracy (judge) | _ | _ | _ |
| Grounded | _ | _ | _ |
| Retrieval recall@6 | n/a | _ | _ |
| Decision match | _ | _ | _ |

## 3. What went wrong and what we would fix
(Use real failures from `eval_results/details.json`: wrong chunk, stale policy, bad follow-up, ASR miss. One paragraph each.)

## 4. Design choices that mattered
Superseded policy excluded; authority weighting; code-enforced verification, ticket honesty and AI disclosure; dates computed by code; query rewriting with order facts for follow-ups.

## 5. Bonus work
Multilingual replies, long-clip splitting, live demo UI with an evidence panel, offline test suite.
