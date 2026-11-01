# VoiceOps 2026: ZentroMart Voice Support Agent
**Course Capstone Report · Consulting & Analytics Club, IIT Guwahati × Gnani.ai**  
*Team ZentroMart VoiceOps*

---

## 1. What We Built

We designed and implemented a production-grade, four-step voice customer-support agent for **ZentroMart**, an Indian e-commerce retailer selling groceries and electronics. The agent runs locally with full privacy and determinism, handles complex multi-turn support workflows, and communicates naturally over voice via Gnani STT and TTS models.

```
Customer Audio ──► Gnani STT (ITN/Boost) ──► Speech Normalization ──► Query Rewriter
                                                                             │
Spoken Audio ◄── Gnani TTS (timbre-v2.5) ◄── Ear-Friendly Text ◄── Hybrid Retrieval (BM25+Dense)
                                                                             │
                                                                   LLM + Deterministic Tools
                                                                   (Orders, Catalog, Dates)
```

- **Step 1 (Baseline Assistant)**: Prompt-engineered baseline AI assistant that strictly identifies itself as an AI on the first turn, answers politely, but lacks external document access.
- **Step 2 (Grounded RAG System)**: Section-aware knowledge base chunker that preserves section IDs (`[SECTION-ID]`), document authority hierarchy (`current policy (1.0) > agent guide (0.85) > help (0.7) > FAQ (0.6) > marketing (0.3)`), bidirectional cross-reference expansion, bilingual alignment (Hindi FAQ ↔ English Policy), and superseded policy isolation. Achieved **100.0% recall@6** and **100.0% recall@8** on practice test sets.
- **Step 3 (Verified Multi-Turn Conversation)**: Stateful conversation tracker enforcing the strict e-commerce caller verification protocol: unverified order lookup first, single verification value challenge (phone last 4 digits OR delivery pincode), maximum 1 retry on mismatch, and complete lockdown on two failures. Deterministic tools include `lookup_product` (with SKU homophone resolution and fuzzy catalog matching), `check_pincode` (serviceability, COD, express availability), and `date_math` (exact business-day calculations skipping Sundays and gazetted holidays).
- **Step 4 (Voice Integration & Recovery)**: Real-time Gnani STT and TTS pipeline. Includes automated clip splitting for audio exceeding 50s, inverse text normalisation (ITN), domain term boosting, language-matched voices across 10 Indian languages (English, Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi), Hinglish detection (`hi-en`), and graceful recovery prompts when audio is unclear.

---

## 2. Quantitative Evaluation: Step 1 vs Step 2 vs Step 3

| Evaluation Metric | Step 1 (Baseline) | Step 2 (RAG) | Step 3 (Full Agent) | Target / Notes |
|---|:---:|:---:|:---:|---|
| **Retrieval Recall@4** | N/A | **97.4%** | **97.4%** | Practice questions (77 gold sections) |
| **Retrieval Recall@6** | N/A | **100.0%** | **100.0%** | 0 misses across 47 questions |
| **Retrieval Recall@8** | N/A | **100.0%** | **100.0%** | 100% evidence coverage |
| **ASR Text Recall@6** | N/A | **100.0%** | **100.0%** | Tested on noisy speech-to-text inputs |
| **Decision Match Rate** | 63.5% | 88.5% | **96.2%** | Answer vs Clarify vs Refuse vs Escalate |
| **Tool Call Match Rate** | N/A | 84.6% | **100.0%** | 96/96 practice conversation turns |
| **Ticket Handover Accuracy** | 0% (hallucinated) | 0% | **100.0%** | Real tickets created only on ESC-1 rules |
| **Offline Test Suite** | 26 / 26 PASS | 26 / 26 PASS | **26 / 26 PASS** | Deterministic unit & integration tests |

### Key Findings:
1. **Prompt-Only vs RAG**: Baseline hallucinated return timelines (e.g., claiming 30-day windows for electronics instead of 7 days) and invented policy exceptions. Grounding with section-aware retrieval immediately eliminated factual hallucinations.
2. **Decision Alignment**: Baseline tended to over-escalate or promise compensation vouchers. Step 3 accurately refused policy violations (such as arbitrary compensation demands) with allowed alternatives, and escalated solely when ESC-1 criteria were triggered.
3. **Tool Precision**: 96 out of 96 conversation turns in the official practice dataset matched expected tool invocations, order ID parsing, and ticket creations with zero discrepancies.

---

## 3. What Went Wrong and How We Fixed It

### 1. ASR Number & SKU Corruption
*Failure*: When callers spoke "SKU ELE zero zero one two" or "order number eight nine seven eight two eight three seven", standard STT output yielded strings like "ele zero zero one to" or spaced digits "8978 2837". Tool lookups failed because the database required canonical IDs (`ELE-0012`, `89782837`).  
*Fix*: Built a deterministic speech normalizer (`speech_norm.py`) that identifies phonetic homophones ("to" → "2", "for" → "4", "won" → "1", "ate" → "8"), converts spoken multi-words ("double four double zero" → "4400"), and canonicalizes IDs based on the learned order format.

### 2. The 1 August 2026 Superseded Returns Policy Trap
*Failure*: `pol_returns_v3_superseded.md` specifies a 10-day return window for electronics, whereas current `pol_returns_v4.md` specifies a 7-day window. Unfiltered semantic vector search frequently retrieved the superseded passage, causing the agent to state the obsolete 10-day timeline for recent purchases.  
*Fix*: Implemented status and date-aware filtering in `kb.py`. Superseded chunks are down-weighted by 60% unless the query explicitly specifies past purchase dates prior to 1 August 2026 (e.g., "delivered in July").

### 3. Bogus Audio Stream Header Overflow in Modern Python
*Failure*: Gnani TTS streaming responses write WAV headers where `nframes` is flagged as `0x7FFFFFFF` (INT_MAX for indeterminate stream lengths). Python 3.14's `wave` module threw `struct.error: 'L' format requires 0 <= number <= 4294967295` when computing durations, mistaking 3-second clips for 44,739-second files and causing empty STT transcriptions.  
*Fix*: Rewrote `wav_duration` and `split_wav` to calculate exact frame counts from physical file size and sample rates, completely bypassing bogus header headers and ensuring seamless audio processing.

### 4. Bilingual Hindi FAQ Mismatch
*Failure*: Pure Hindi questions (e.g., "खराब या गलत सामान आया तो क्या करें?") retrieved the Hindi FAQ `FAQH-3` but missed the authoritative English policy `DWM-1` (`pol_damaged_wrong_missing.md`), dropping recall.  
*Fix*: Added bidirectional cross-reference links (`POLICY_LINKS`) in `kb.py` mapping Hindi FAQ sections to their corresponding English policies. This raised retrieval recall on Hindi queries from 93.5% to **100.0%**.

---

## 4. Design Choices That Mattered

1. **Precedence Hierarchy Enforcement**: Authority weighting directly in the hybrid reciprocal rank fusion algorithm: `current policy (1.0) > agent guide (0.85) > help (0.7) > FAQ (0.6) > marketing (0.3)`. Marketing materials never override official return or refund policies.
2. **Code-Enforced Verification & Privacy Guardrails**: The LLM is never trusted to perform authentication in its context window. Order lookups enforce caller verification in Python code; secrets like delivery pincodes are scrubbed before reaching the prompt context.
3. **No Hallucinated Actions**: The agent cannot claim it processed a refund, cancelled an order, or created a ticket unless the backend tool returned `created=True`. Claimed actions without backend execution are intercepted and corrected before reaching the caller.
4. **Deterministic Date Math**: The LLM is forbidden from calculating dates mentally. All calendar and business-day operations invoke `date_math` using the official `holidays.csv` calendar.
5. **Speech Formatted for the Ear (`for_ear`)**: Written text is converted into auditory-friendly speech: years are explicitly attached to month names ("3 November 2026"), order numbers are spoken in rhythmically spaced 4-digit groups (STY-3), long ticket IDs are condensed to their last 6 characters, and currency symbols (`₹`, `Rs.`) are verbalized as "rupees".

---

## 5. Bonus Work & Highlights

- **Bilingual & Multilingual Fluency**: Native support for English, Hindi (Devanagari), and Hinglish (Latin script Hindi). Auto-detects caller script and switches dynamically without restart.
- **Interactive Live Demo UI (`web/index.html`)**: Single-file glassmorphic web interface featuring:
  - Real-time AudioContext waveform visualizer during mic recording.
  - Transparent **Agent Reasoning Panel** displaying Decisions, Search Queries, Cited Sections, and Tool Executions.
  - Hybrid input: push-to-talk voice recording plus full text chat fallback with synchronized TTS audio playback.
  - Dark and light theme modes, session reset, and confetti celebration on verified escalation ticket generation.
- **Complete Test & Verification Suite**: 26 offline tests executing in under 1 second without network or API keys, ensuring reproducible verification for course evaluators.
