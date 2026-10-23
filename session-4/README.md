# ERA V5 · Session 4 — data cleaning and deduplication

**Widget (Netlify):** https://era-v5-s4-data-cleaning-768.netlify.app

I cleaned **`lordx64/reasoning-distill-opus-4-7-max-sft`**: 7,823 rows, 29.3 MB, Apache-2.0. It is the dataset behind
the model linked in the assignment. The pipeline follows the Session 4 lesson stage by stage, it is deterministic
(two runs give the same content hash), and every number on the widget is generated from the run.

| file | what it is |
|---|---|
| [`clean.py`](clean.py) | the pipeline: format → normalize → quality → dedup → language → PII → decontamination → manifest, plus the harm test |
| [`make_site.py`](make_site.py) | builds `site/index.html` from `report.json` |
| [`report.json`](report.json) | per-stage counts, real before/after examples, harm test, determinism check |
| [`data_clean/manifest.json`](data_clean/manifest.json) | provenance: source, licence, raw and content hashes, script hash, tokens, languages |
| [`data_clean/clean.jsonl.gz`](data_clean/clean.jsonl.gz) | the cleaned data, one JSON object per line |

To reproduce it (the raw data is not committed; download it first):

```bash
mkdir -p data_raw/evals
curl -L -o data_raw/opus47_train.parquet https://huggingface.co/datasets/lordx64/reasoning-distill-opus-4-7-max-sft/resolve/main/data/train-00000-of-00001.parquet
curl -L -o data_raw/evals/gsm8k_test.parquet https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet
curl -L -o data_raw/evals/humaneval_test.parquet https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet
curl -L -o data_raw/evals/math500_test.jsonl https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
python clean.py --check-determinism && python make_site.py
```

## How many strategies?

**8 cleaning strategies, made of 27 named techniques.** Extraction is taught in Session 3, so it is not counted.

| # | strategy | techniques named in the lesson |
|---|---|---|
| 1 | Normalization | NFC · strip invisible noise · keep ZWJ/ZWNJ · unescape HTML · collapse whitespace |
| 2 | One conversation format (ghost tags) | one canonical format · real special tokens |
| 3 | Quality filtering | average word length · symbol ratio · line-ending punctuation · duplicated lines · stop words · educational-value classifier |
| 4 | Deduplication | exact · shingles + MinHash · LSH banding · global (not local) |
| 5 | Language identification | detect at runtime · validate codes, fail loudly |
| 6 | PII removal | regex layer · ML (name) layer |
| 7 | Decontamination | fingerprint eval sets · scan and remove · canary strings |
| 8 | Reproducibility and manifest | content-based ids · hash after cleaning · provenance manifest with licence gate |

## What happened to the data

| stage | rows in → out | what it did |
|---|---|---|
| format | 7,823 → 7,822 | parsed literal `<\|im_start\|>` markers into `messages[]` and a separate `reasoning` field; 1 row with a broken `<think>` block dropped; 1,179 rows without reasoning kept and labelled |
| normalize | 7,822 → 7,822 | 11 mojibake runs repaired (`chiefâ\x80\x94` → `chief—`, `Â£1m` → `£1m`), 11 HTML entities unescaped (prose only), 2 control characters and 2 word joiners removed, 8 ZWJ kept |
| quality | 7,822 → 7,615 | 91 prompts with no real task, 24 prompts under 15 characters, 49 truncated answers (unclosed code block), 43 with more than 30% duplicated lines |
| dedup | 7,615 → 6,636 | 29 exact duplicates; 950 near-duplicate prompts in 948 clusters (MinHash 128 perms, 16 × 8 LSH, verified Jaccard ≥ 0.70); the fullest answer kept |
| language | 6,636 | 6,014 English, 403 code-heavy or non-English Latin, 219 in other scripts (the card says "en") |
| PII | 6,636 | 20 emails, 16 phone numbers and 22 public IPs masked; documentation domains, private IPs, version strings and toll-free lines kept |
| decontamination | 6,636 | 0 rows share a 13-gram with GSM8K test, HumanEval or MATH-500 |

**Final:** 6,636 rows (84.8%), 25.06M characters, **5.89M tokens** (Qwen2.5 tokenizer), of which 5,518 rows have reasoning.

- Content sha256: `f894c10f2fe4bcae…`
- Determinism check: **PASS** (two runs, identical hash)

## The harm test

I ran the same normalization and PII scrubbing on 307 known-good documents. They were the Session 2 English and Hindi
Wikipedia paragraphs, Telugu, Malayalam, Hindi and Bengali strings with ZWJ/ZWNJ, an emoji ZWJ sequence, and code blocks.

| check | result |
|---|---|
| harmful changes | **0 (0.0%)** |
| documents changed at all | 5 (1.6%), every one a real fix: byte-order marks and HTML entities |
| ZWJ/ZWNJ before → after | **19 → 19** |
| code blocks byte-exact | **yes** |

**PASS.** The first version **failed** at 9.1%, because a "collapse runs of spaces" rule rewrote Markdown tables and double
spaces. The harm test is how I found it, and that rule was removed.

## Other concerns (details on the widget)

- **Licence:** the answers are a commercial model's outputs, and the provider's terms limit training competing models on them. This needs a legal check before V5 uses it.
- **Mixed formats:** 1,118 kept rows have no `<think>`. They are labelled, not mixed in silently.
- **Language label:** 622 rows are not plain English, against a card that says "en".
- **No ML name detection:** only the regex PII layer ran.
- **Limited decontamination:** only 3 benchmarks were scanned, with exact 13-grams.
- **English heuristics:** web rules would have destroyed the code and maths rows, so they were replaced with chat-specific rules.
