Main run: 14.25T tokens in 3 stages, plus a 0.30T anneal = 14.55T total.

| lane | share of main run | tokens | real unique supply (after hold-back) | epochs of real data | must be generated |
|---|---|---|---|---|---|
| general web | 55.1% | 7,852B | 3,835B | 2.05 | 0B (0%) |
| STEM papers | 3.4% | 488B | 97B | 4.00 | 100B (20%) |
| code | 26.2% | 3,735B | 859B | 4.00 | 299B (8%) |
| maths & reasoning | 5.6% | 795B | 70B | 4.00 | 514B (65%) |
| long context | 1.3% | 180B | 100B | 1.80 | 0B (0%) |
| agentic-shaped | 1.0% | 142B | 4B | 4.00 | 126B (89%) |
| Indic | 7.4% | 1,058B | see the tier table | | |

| Indic tier | share of Indic lane | tokens | real unique supply | epochs | must be generated or translated |
|---|---|---|---|---|---|
| verified | 30% | 317B | 51.2B | 4.00 | 112B |
| unverified | 20% | 212B | 44.9B | 4.00 | 32B |
| translated | 20% | 212B | 5.0B | 4.00 | 192B |
| synthetic | 30% | 317B | 162.0B | 1.96 | 0B |

| stage | tokens | context | general web | STEM papers | code | maths & reasoning | Indic | long context | agentic-shaped |
|---|---|---|---|---|---|---|---|---|---|
| 1 foundation | 6.00T | 4K | 67% | 3% | 18% | 4% | 7.5% | 0% | 0.5% |
| 2 code + reasoning | 6.00T | 8K | 47% | 4% | 33% | 7% | 7.5% | 0% | 1.5% |
| 3 long context | 2.25T | 32K | 45% | 3% | 30% | 6% | 7% | 8% | 1% |
| **whole main run** | 14.25T | | **55.1%** | **3.4%** | **26.2%** | **5.6%** | **7.4%** | **1.3%** | **1.0%** |

| anneal lane | share | tokens |
|---|---|---|
| general web (top slice) | 25% | 75B |
| code (top slice + verified synthetic) | 30% | 90B |
| maths & reasoning | 20% | 60B |
| Indic (held-back verified) | 15% | 45B |
| agentic | 5% | 15B |
| long context | 5% | 15B |

Protected floor = 9.0% of every batch: Indic verified + unverified 5.0%, maths & reasoning (AON) 2.5%, agentic-shaped (commits) 0.5%, benchmark TRAIN splits (format only) 1.0%

| held back from the main run | tokens |
|---|---|
| FineWeb-Edu (5%) | 65.0B |
| The Stack v2 (5%) | 45.0B |
| Sangraha verified (20%) | 12.8B |
| AON (V4) (10%) | 7.8B |
| OpenThoughts2 (100%) | 3.0B |
| OpenMathReasoning (100%) | 2.0B |
| NuminaMath (100%) | 0.5B |
| OpenR1-Math (100%) | 1.6B |
| Agentic trajectories + function calls (100%) | 0.6B |
