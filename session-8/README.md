# ERA V5 · Session 8 — attention, in the order it was launched

**Live app:** https://era-v5-s8-attention-timeline.netlify.app (static HTML + JS, no build step)

The app has four parts:
- **The timeline:** 31 attention mechanisms in launch order, each explained as an answer to the problem that existed
  *at that moment*. Each one lists what it buys, what it costs, and when I would actually pick it, with a small picture of its attention pattern.
- **Filters** by family: exact, position, KV cache, sparse/window, linear/state, memory.
- **A KV-cache calculator** for a 30B-class decoder (48 layers, 64 heads, head dim 128). Drag the context length from 1K
  to 1M and watch which ideas change the slope. At 128K, one sequence needs **192 GiB with MHA**, 24 GiB with GQA-8,
  6.8 GiB with MLA, and 0.65 GiB with block compression plus a 4K window.
- **What the timeline shows** (question 2) and **the mechanisms the lesson did not cover**.

| file | what it is |
|---|---|
| [`index.html`](index.html) | the app |
| [`mechanisms.js`](mechanisms.js) | the data: date, source, problem, mechanism, pros, cons and "when to pick" for every entry |
| [`verify_dates.py`](verify_dates.py) | re-checks every arXiv date against the arXiv API |

## Sources for the dates

This is the part that is easiest to get wrong, so every date was checked rather than remembered.

- **29 dates come from arXiv:** the `published` (v1) date returned by the arXiv API for the paper's id. Run
  `python verify_dates.py`; the output ends `31 mechanisms, 29 arXiv dates checked, 0 mismatches`.
- **NTK-aware RoPE scaling** has no paper. It was a June 2023 post by u/bloc97 on r/LocalLLaMA, and YaRN (arXiv 2309.00071)
  cites it as its origin. I give the month only, because I could not find a citable record of the exact day.
- **DeepSeek Sparse Attention (DSA)** is dated by the DeepSeek-V3.2-Exp release on Hugging Face (repository created
  2025-09-29). The paper came later (arXiv 2512.02556, 2025-12-02).
- **DeepSeek-V4 CSA + HCA** is dated by the DeepSeek-V4-Flash release on Hugging Face (2026-04-22). The report is arXiv
  2606.19348, published 2026-04-26.
- Some mechanisms have a paper date that differs from the date they became mainstream. Sliding-window attention is
  dated by Longformer (2020-04-10). Its decoder adoption in Mistral 7B (2023-10-10) is noted on the card, not used as the date.

The lesson's minimum list, with dates:
- standard attention 2017-06-12, absolute learned positions 2017-05-08 (ConvS2S), sinusoidal 2017-06-12
- RoPE 2021-04-20, ALiBi 2021-08-27
- MQA 2019-11-06, GQA 2023-05-22
- sliding window 2020-04-10, attention sinks 2023-09-29
- NTK-aware June 2023, YaRN 2023-08-31
- linear attention 2020-06-29, delta rule 2021-02-22, Gated DeltaNet 2024-12-09
- MLA 2024-05-07
- sparse 2019-04-23, top-k 2021-06-13
- DeepSeek's compressed and sparse attention: NSA 2025-02-16, DSA 2025-09-29, V4 CSA/HCA 2026-04-22
- DroPE 2025-12-13

## Question 2: what the timeline shows that a list does not

1. **The field changed its mind four times, and the order is the lesson.**
   - **Exactness**, 2014–2017: every token reads every token.
   - **Length**: the Sparse Transformer (2019), Longformer and linear attention (2020) gave up exactness to reach longer sequences.
   - **Memory back**: MQA was published in 2019 and mostly ignored until decoding cost dominated. Then GQA (2023) and MLA (2024)
     cut the KV cache *while keeping exact attention*.
   - **Both at once**: Gated DeltaNet hybrids, learned sparsity (NSA, DSA) and compressed sequences (DeepSeek-V4's CSA/HCA).
2. **FlashAttention (May 2022) is the hinge.** Before it, approximation (Reformer, Performer, linear) was the road to
   length. After it, exact attention was fast enough again, and the work moved from "approximate the scores" to
   "keep exact scores, shrink what we store". Grouped by family, you cannot see that turn.
3. **Ideas come back, renamed, when the reason they failed is removed.**
   - Fixed sparse patterns (2019) → learned block selection (NSA 2025).
   - Linear attention (2020) → the delta rule (2021) → Gated DeltaNet (2024), the cheap layer in every 2025 hybrid.
   - Transformer-XL's segment memory (2019) → V4's Memory Stream.
4. **Position has its own clock.**
   - Learned tables (2017) → rotations (RoPE 2021) → stretched rotations (PI, NTK, YaRN, all in 2023) → removing positions
     after training (DroPE, Dec 2025).
   - Every step asks the same question again: how far past the training length can we go?
5. **What comes next.** The bill still left is the cache that grows with every token in the attention layers. I expect:
   - less stored per token than a latent (V4's HCA already points there);
   - fewer attention layers per hybrid;
   - learning *which tokens deserve a cache entry at all*, rather than selecting them at read time.

### Mechanisms I added that the lesson did not cover

| mechanism | first public | source |
|---|---|---|
| Additive attention (Bahdanau) | 2014-09-01 | arXiv 1409.0473 |
| Relative position representations (Shaw et al.) | 2018-03-06 | arXiv 1803.02155 |
| Transformer-XL segment recurrence | 2019-01-09 | arXiv 1901.02860 |
| Reformer (LSH attention + reversible layers) | 2020-01-13 | arXiv 2001.04451 |
| Performer (FAVOR+ random features) | 2020-09-30 | arXiv 2009.14794 |
| FlashAttention (exact, IO-aware) | 2022-05-27 | arXiv 2205.14135 |
| Position Interpolation | 2023-06-27 | arXiv 2306.15595 |
| Mamba (selective state spaces) | 2023-12-01 | arXiv 2312.00752 |
| Lightning attention hybrid (MiniMax-01) | 2025-01-14 | arXiv 2501.08313 |
| MoBA (mixture of block attention) | 2025-02-18 | arXiv 2502.13189 |

Gated DeltaNet's production uses are noted on its card: Qwen3-Next (2025-09-09) and Kimi Linear (arXiv 2510.26692, 2025-10-30).
