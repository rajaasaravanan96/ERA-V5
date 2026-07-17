# ERA V5 · Session 3 — Bharat-40B Design Brief

Design brief for a 40B-parameter, Gemma-4-class model that is excellent at **coding,
agentic work and Indic languages**, and is **India-first** — because its corpus and its
annotators are, not because a system prompt says so.

**Report:** [`netlify-deploy/index.html`](netlify-deploy/index.html) — deploy by dragging
the `netlify-deploy/` folder into Netlify (Add new site → Deploy manually).

## Decisions at a glance

| Question | Decision |
|---|---|
| Pre-training budget | ~10T tokens (≈12× Chinchilla), dense 40B, d_model 6144 |
| Pre-training mix | 34% English · 24% code (incl. PRs/issues/diffs) · 16% Indic (12 langs, incl. 0.3T parallel) · 9% math/science · 6% India-first curated (law, NCERT-class, Parliament, judgments) · 3% agentic traces · 8% verified synthetic |
| Post-training | ~2M SFT: 40% repo-level code (diff format) · 25% verified agentic trajectories · 20% natively-authored Indic (incl. Hinglish) · 10% India-first QA · 5% safety |
| RL / alignment | RLVR (code tests, math answers, sandboxed agent tasks) + preference RL with a quota-based pan-India annotator pool and a constitutional-values rubric |
| Cleaning | own LID (code-mix aware) → NFC only, preserve ZWJ/ZWNJ, no NFKC on code → exact+MinHash dedup → **per-language** quality classifiers → license/secret scrub for code → India-tuned PII/toxicity → 13-gram decontamination |
| Evaluation | LiveCodeBench + SWE-bench Verified · τ²-bench/WebArena/Terminal-Bench + error-recovery rate · MILU/IndicXTREME/Flores + native human eval · **BharatEval** (5k natively-authored India-first items, JEE/NEET-style STEM) · everything ablated at 1.5B/8B first |
| Fertility targets | en ≤1.25 · hi ≤1.35 · bn/ta/te/mr ≤1.55 · other Indic ≤1.70 · code ≤0.30 tok/char · LaTeX ≤0.35 tok/char · JSON ≤1.1 tok/unit |
| **Vocab size** | **131,072 (2^17)** — 2·V·d = 1.6B params (4% of 40B); 65k can't hit Indic targets, 262k adds <2% compression for 8% parameter tax. Budget: 46k Latin/code · 42k Indic · 10k math/sci · 32k shared · 256 byte-fallback · ~600 specials |

The tokenizer is built the same way as our Session-2 submission — weighted BPE with a
measure-fertility → adjust-weights → retrain loop — scaled to 12 languages + code + math.
