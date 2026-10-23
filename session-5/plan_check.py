"""Checks the V5 mixture plan against real supply.

Supply numbers are the course's Session 5 dataset inventory (tokens, unique). The plan is written as
stage mixtures; this script computes the overall share of every lane, the tokens each source must provide,
how many epochs of repetition that means, and how much must be generated. Run: python plan_check.py
It writes plan_tables.md, which the README tables are copied from.
"""
import json

T = 1e12
B = 1e9

# ------------------------------------------------------------------ supply (course inventory, unique tokens)
SUPPLY = {
    # general web & STEM
    'FineWeb-Edu': 1.3 * T, 'DCLM-Baseline': 2.6 * T, 'peS2o': 42 * B, 'proof-pile-2': 55 * B,
    # code
    'The Stack v2': 900 * B, 'CommitPack': 4 * B,
    # reasoning & maths
    'AON (V4)': 78 * B, 'OpenThoughts2': 3 * B, 'OpenMathReasoning': 2 * B, 'NuminaMath': 0.5 * B, 'OpenR1-Math': 1.6 * B,
    # long context
    'Repo-packed code 32K+': 60 * B, 'Book-length packed': 40 * B,
    # Indic
    'Sangraha verified': 64 * B, 'Sangraha unverified': 24 * B, 'IndicCorpV2': 20.9 * B,
    'Sangraha synthetic': 162 * B, 'BPCC parallel': 3 * B, 'Samanantar parallel': 2 * B,
    # agentic (all of it)
    'Agentic trajectories + function calls': 0.627 * B,
}
# held back from the main run for the anneal and for SFT (fraction of each source)
HOLD = {'FineWeb-Edu': 0.05, 'The Stack v2': 0.05, 'Sangraha verified': 0.20, 'AON (V4)': 0.10,
        'OpenThoughts2': 1.0, 'OpenMathReasoning': 1.0, 'NuminaMath': 1.0, 'OpenR1-Math': 1.0,
        'Agentic trajectories + function calls': 1.0}

# ------------------------------------------------------------------ lanes: which sources feed them (main run)
LANES = {
    'general web': {'FineWeb-Edu': 0.40, 'DCLM-Baseline': 0.60},
    'STEM papers': {'peS2o': 0.45, 'proof-pile-2': 0.55},
    'code': {'The Stack v2': 0.99, 'CommitPack': 0.01},
    'maths & reasoning': {'AON (V4)': 1.0},
    'Indic': {'Sangraha verified': 1.0, 'Sangraha unverified': 1.0, 'IndicCorpV2': 1.0,
              'Sangraha synthetic': 1.0, 'BPCC parallel': 1.0, 'Samanantar parallel': 1.0},
    'long context': {'Repo-packed code 32K+': 0.6, 'Book-length packed': 0.4},
    'agentic-shaped': {'CommitPack': 1.0},
}
# Indic tier split of the Indic lane (main run). verified = Sangraha verified; unverified = Sangraha unverified
# + IndicCorpV2; translated = parallel corpora + new translation of English STEM/code text; synthetic = Sangraha synthetic.
INDIC_TIERS = {'verified': 0.30, 'unverified': 0.20, 'translated': 0.20, 'synthetic': 0.30}
INDIC_TIER_SOURCES = {'verified': ['Sangraha verified'], 'unverified': ['Sangraha unverified', 'IndicCorpV2'],
                      'translated': ['BPCC parallel', 'Samanantar parallel'], 'synthetic': ['Sangraha synthetic']}

# ------------------------------------------------------------------ the run: stages and their mixtures (%)
STAGES = [
    ('1 foundation', 6.0 * T, 4096, {'general web': 67, 'STEM papers': 3, 'code': 18, 'maths & reasoning': 4,
                                     'Indic': 7.5, 'long context': 0, 'agentic-shaped': 0.5}),
    ('2 code + reasoning', 6.0 * T, 8192, {'general web': 47, 'STEM papers': 4, 'code': 33, 'maths & reasoning': 7,
                                           'Indic': 7.5, 'long context': 0, 'agentic-shaped': 1.5}),
    ('3 long context', 2.25 * T, 32768, {'general web': 45, 'STEM papers': 3, 'code': 30, 'maths & reasoning': 6,
                                         'Indic': 7, 'long context': 8, 'agentic-shaped': 1}),
]
ANNEAL = ('4 anneal (reserve)', 0.30 * T, 32768,
          {'general web (top slice)': 25, 'code (top slice + verified synthetic)': 30, 'maths & reasoning': 20,
           'Indic (held-back verified)': 15, 'agentic': 5, 'long context': 5})
FLOOR = {'Indic verified + unverified': 5.0, 'maths & reasoning (AON)': 2.5, 'agentic-shaped (commits)': 0.5,
         'benchmark TRAIN splits (format only)': 1.0}


def main():
    total = sum(s[1] for s in STAGES)
    lane_tokens = {l: 0.0 for l in LANES}
    for name, tok, ctx, mix in STAGES:
        assert abs(sum(mix.values()) - 100) < 1e-9, (name, sum(mix.values()))
        for l, p in mix.items(): lane_tokens[l] += tok * p / 100
    out = []
    out.append(f'Main run: {total / T:.2f}T tokens in {len(STAGES)} stages, plus a {ANNEAL[1] / T:.2f}T anneal '
               f'= {(total + ANNEAL[1]) / T:.2f}T total.\n')
    out.append('| lane | share of main run | tokens | real unique supply (after hold-back) | epochs of real data | must be generated |')
    out.append('|---|---|---|---|---|---|')
    rows = {}
    for l, need in lane_tokens.items():
        if l == 'Indic':
            continue
        srcs = LANES[l]
        uniq = sum(SUPPLY[s] * (1 - HOLD.get(s, 0)) * (1 if l != 'agentic-shaped' else 1) for s in srcs) if l != 'agentic-shaped' else SUPPLY['CommitPack']
        cap = {'general web': 4, 'STEM papers': 4, 'code': 4, 'maths & reasoning': 4, 'long context': 4, 'agentic-shaped': 4}[l]
        real = min(need, uniq * cap)
        gen = need - real
        ep = real / uniq
        rows[l] = dict(tokens=need, share=need / total, unique=uniq, epochs=ep, generated=gen)
        out.append(f'| {l} | {100 * need / total:.1f}% | {need / B:,.0f}B | {uniq / B:,.0f}B | {ep:.2f} | '
                   f'{gen / B:,.0f}B ({100 * gen / need:.0f}%) |')
    # Indic, tier by tier
    need = lane_tokens['Indic']
    out.append(f'| Indic | {100 * need / total:.1f}% | {need / B:,.0f}B | see the tier table | | |')
    out.append('')
    out.append('| Indic tier | share of Indic lane | tokens | real unique supply | epochs | must be generated or translated |')
    out.append('|---|---|---|---|---|---|')
    tier_rows = {}
    for tier, p in INDIC_TIERS.items():
        tn = need * p
        uniq = sum(SUPPLY[s] * (1 - HOLD.get(s, 0)) for s in INDIC_TIER_SOURCES[tier])
        real = min(tn, uniq * 4)
        gen = tn - real
        tier_rows[tier] = dict(tokens=tn, unique=uniq, epochs=real / uniq, generated=gen)
        out.append(f'| {tier} | {100 * p:.0f}% | {tn / B:,.0f}B | {uniq / B:,.1f}B | {real / uniq:.2f} | {gen / B:,.0f}B |')
    out.append('')
    out.append('| stage | tokens | context | ' + ' | '.join(LANES) + ' |')
    out.append('|---|---|---|' + '---|' * len(LANES))
    for name, tok, ctx, mix in STAGES:
        out.append(f'| {name} | {tok / T:.2f}T | {ctx // 1024}K | ' + ' | '.join(f'{mix[l]:g}%' for l in LANES) + ' |')
    out.append('| **whole main run** | ' + f'{total / T:.2f}T | | ' + ' | '.join(f'**{100 * lane_tokens[l] / total:.1f}%**' for l in LANES) + ' |')
    out.append('')
    an = ANNEAL[3]; assert abs(sum(an.values()) - 100) < 1e-9
    out.append('| anneal lane | share | tokens |')
    out.append('|---|---|---|')
    for l, p in an.items():
        out.append(f'| {l} | {p}% | {ANNEAL[1] * p / 100 / B:,.0f}B |')
    out.append('')
    out.append(f'Protected floor = {sum(FLOOR.values()):.1f}% of every batch: ' + ', '.join(f'{k} {v}%' for k, v in FLOOR.items()))
    held = {s: SUPPLY[s] * f for s, f in HOLD.items()}
    out.append('')
    out.append('| held back from the main run | tokens |')
    out.append('|---|---|')
    for s, v in held.items(): out.append(f'| {s} ({100 * HOLD[s]:.0f}%) | {v / B:,.1f}B |')
    open('plan_tables.md', 'w').write('\n'.join(out) + '\n')
    json.dump(dict(lanes=rows, indic=tier_rows, total_main=total, anneal=ANNEAL[1]), open('plan_numbers.json', 'w'), indent=1)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
