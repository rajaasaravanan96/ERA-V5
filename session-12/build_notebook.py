"""Builds session12_zero.ipynb.

Run:  python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace session12_zero.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Session 12 — 32 virtual GPUs: data parallel, ZeRO-1, ZeRO-2 and ZeRO-3

The simulator (`zero_sim.py`) gives every virtual GPU its own storage. Ranks only exchange data through
the three collectives, and every collective counts the bytes each rank sends (ring algorithm). The training is real:
each strategy trains the same byte-level transformer on the same data, and at the end we check that all of them
produced **bit-identical weights** to a single-GPU reference. Then we compare what each GPU had to store, send and compute.
""")

code("""
import json, math, time
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from zero_sim import ByteLM, Layout, Cluster, Strategy, memory_ladder, N_GPU

torch.manual_seed(0)
model = ByteLM(d=128, n_layers=4, n_heads=4, ctx=64)
L = Layout(model)
print(f'virtual GPUs: {N_GPU}')
print(f'parameters P = {L.P:,} (padded to {L.Ppad:,} so it splits evenly; shard = {L.shard:,} per GPU)')
for u, n in L.units.items():
    print(f'  unit {u:12s} {n:9,d} weights')
text = open('../session-2/corpus/en.faithful.txt', 'rb').read() + open('../session-2/corpus/hi.faithful.txt', 'rb').read()
data = torch.tensor(list(text), dtype=torch.long)
print(f'data: {len(data):,} bytes of English + Hindi text (byte-level, vocab 256)')
def dark(ax, t, xl, yl):
    ax.set_facecolor('#0F151C'); ax.figure.set_facecolor('#0A0E13')
    for s in ax.spines.values(): s.set_color('#35485E')
    ax.tick_params(colors='white', which='both'); ax.grid(color='#25323F', axis='y')
    ax.set_title(t, color='white'); ax.set_xlabel(xl, color='white'); ax.set_ylabel(yl, color='white')
RESULTS = {}
""")

md("""
## 1. The collectives, checked

A reduce-scatter followed by an all-gather must equal an all-reduce — and cost the same.
""")

code("""
cl = Cluster(N_GPU)
vals = [torch.randn(N_GPU * 4).to(torch.bfloat16) for _ in range(N_GPU)]
ar = cl.all_reduce(vals, torch.float32); sent_ar = cl.ranks[0].sent_bytes
cl2 = Cluster(N_GPU)
rs = cl2.reduce_scatter(vals, torch.float32); sent_rs = cl2.ranks[0].sent_bytes
ag = cl2.all_gather(rs); sent_both = cl2.ranks[0].sent_bytes
expect = torch.stack([v.float() for v in vals]).sum(0)
print('every rank holds the full sum after all-reduce        :', all(torch.equal(a, expect) for a in ar))
print('rank 5 holds only slice 5 after reduce-scatter         :', torch.equal(rs[5], expect[20:24]))
print('reduce-scatter + all-gather == all-reduce              :', all(torch.equal(a, b) for a, b in zip(ag, ar)))
print(f'bytes sent per rank: reduce-scatter {sent_rs}, + all-gather {sent_both - sent_rs}, all-reduce {sent_ar}')
""")

md("""
## 2. Train with every strategy, and prove they compute the same thing

Global batch = 32 micro-batches of 2 × 64 bytes (one micro-batch per GPU). 30 optimizer steps, AdamW.
The single-GPU reference runs all 32 micro-batches itself.
""")

code("""
STEPS, MB, T = 30, 2, 64
def micro_batches(step):
    g = torch.Generator().manual_seed(step)
    ix = torch.randint(0, len(data) - T - 1, (N_GPU, MB), generator=g)
    return [torch.stack([data[i:i + T + 1] for i in row]) for row in ix]
flops_fwd_bwd = 6 * L.P * MB * T            # per micro-batch (6 FLOPs per weight per token)

runs, curves, times = {}, {}, {}
for stage in ['single', 'dp', 'zero1', 'zero2', 'zero3']:
    torch.manual_seed(0); m = ByteLM(d=128, n_layers=4, n_heads=4, ctx=64)
    s = Strategy(stage, m, L); curves[stage] = []
    t0 = time.perf_counter()
    for step in range(STEPS):
        curves[stage].append(s.step(micro_batches(step), flops_fwd_bwd))
    times[stage] = time.perf_counter() - t0
    runs[stage] = s
    print(f'{stage:6s} loss step 1 {curves[stage][0]:.4f} -> step {STEPS} {curves[stage][-1]:.4f}   ({times[stage]:.1f}s simulated wall time)')

ref = runs['single'].full_weights()
print()
for stage in ['dp', 'zero1', 'zero2', 'zero3']:
    w = runs[stage].full_weights()
    same = torch.equal(w, ref)
    print(f'{stage:6s} final weights bit-identical to the single-GPU run: {same}   max |diff| = {(w.float() - ref.float()).abs().max().item()}')
    assert same
# in ZeRO-3 no rank ever holds the full weights between steps: show what rank 7 has
print(f'\\nZeRO-3 rank 7 stores {runs["zero3"].cl.ranks[7].store["weights_bf16"].numel():,} weights (its slice), not {L.Ppad:,}')
RESULTS['loss_curves'] = curves; RESULTS['sim_seconds'] = times
""")

md("""
## 3. Memory per GPU

Measured from the tensors each rank actually stores (`nbytes`), plus the transient buffers each schedule needs
(ZeRO-2: one unit's full gradient before its reduce-scatter; ZeRO-3: the gathered weights of the current unit, a prefetch of
the next, and a gradient bucket).
""")

code("""
rep = {st: runs[st].report() for st in ['dp', 'zero1', 'zero2', 'zero3']}
KB = 1024
parts = ['weights_bf16', 'grads_bf16', 'master_fp32', 'adam_m_fp32', 'adam_v_fp32']
print(f"{'stage':7s} " + ' '.join(f'{p:>13s}' for p in parts) + f" {'transient':>10s} {'peak':>9s} {'bytes/weight':>13s}")
for st, r in rep.items():
    print(f'{st:7s} ' + ' '.join(f"{r['breakdown'][p] / KB:11.0f}KB" for p in parts)
          + f" {r['transient_bytes'] / KB:8.0f}KB {r['peak_bytes'] / KB:7.0f}KB {r['persistent_bytes'] / L.Ppad:13.2f}")
RESULTS['memory'] = rep

fig, ax = plt.subplots(figsize=(9, 4.5))
cols = {'weights_bf16': '#3FA7FF', 'grads_bf16': '#FF9F35', 'master_fp32': '#26D7C8', 'adam_m_fp32': '#B07BFF', 'adam_v_fp32': '#F5D33B'}
x = list(rep)
bottom = [0] * 4
for p in parts:
    h = [rep[s]['breakdown'][p] / 2 ** 20 for s in x]
    ax.bar(x, h, bottom=bottom, color=cols[p], label=p, edgecolor='#0A0E13'); bottom = [b + hh for b, hh in zip(bottom, h)]
h = [rep[s]['transient_bytes'] / 2 ** 20 for s in x]
ax.bar(x, h, bottom=bottom, color='#4A6076', hatch='//', label='transient buffers', edgecolor='#0A0E13')
for i, s in enumerate(x):
    ax.text(i, bottom[i] + h[i] + 0.1, f"{rep[s]['peak_bytes'] / 2**20:.2f} MiB", ha='center', color='white')
dark(ax, f'Memory on one of {N_GPU} GPUs (model of {L.P/1e6:.2f}M weights)', 'strategy', 'MiB per GPU')
ax.legend(facecolor='#141C25', labelcolor='white', edgecolor='#35485E', fontsize=8); fig.tight_layout(); fig.savefig('figures/memory.png', dpi=130)
""")

md("## 4. Communication and computation per GPU per step")

code("""
P2 = L.Ppad * 2   # one bf16 copy of all the weights, in bytes
print(f"{'stage':7s} {'sent/step':>12s} {'in units of P (bf16)':>22s} {'fwd+bwd FLOPs':>15s} {'optimizer FLOPs':>16s} {'sim time':>9s}")
comm = {}
for st in ['dp', 'zero1', 'zero2', 'zero3']:
    s = runs[st]; r0 = s.cl.ranks[0]
    sent = rep[st]['sent_bytes_per_step']
    opt_flops = 12 * (L.Ppad if st == 'dp' else L.shard)
    comm[st] = dict(sent=sent, units=sent / P2, fwd_bwd=flops_fwd_bwd, opt=opt_flops)
    print(f'{st:7s} {sent / 2**20:9.2f}MiB {sent / P2:22.2f} {flops_fwd_bwd:15.3e} {opt_flops:16.3e} {times[st]:8.1f}s')
print('\\n(ring cost: each collective sends (n-1)/n of the buffer, so 2P becomes 2P x 31/32 = 1.94P)')
RESULTS['comm'] = comm

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(list(comm), [c['units'] for c in comm.values()], color=['#4A6076', '#3FA7FF', '#26D7C8', '#FF9F35'])
for i, c in enumerate(comm.values()): axes[0].text(i, c['units'] + 0.05, f"{c['units']:.2f} P", ha='center', color='white')
dark(axes[0], 'Bytes each GPU sends per step (P = one bf16 copy of the weights)', 'strategy', 'multiples of P')
axes[1].bar(list(comm), [c['opt'] for c in comm.values()], color=['#4A6076', '#3FA7FF', '#26D7C8', '#FF9F35'])
axes[1].set_yscale('log')
for i, c in enumerate(comm.values()): axes[1].text(i, c['opt'] * 1.15, f"{c['opt']:.1e}", ha='center', color='white')
dark(axes[1], 'Optimizer FLOPs per GPU per step (fwd+bwd is the same for all)', 'strategy', 'FLOPs (log)')
fig.tight_layout(); fig.savefig('figures/comm_compute.png', dpi=130)
""")

md("""
## 5. How memory falls as GPUs are added

First measured in the simulator at 1 to 32 GPUs (persistent bytes per GPU, no training needed),
then the same formula at the lesson's 30B-parameter model on 80 GB cards (74.5 GiB usable).
""")

code("""
meas = {st: [] for st in ['dp', 'zero1', 'zero2', 'zero3']}
NS = [1, 2, 4, 8, 16, 32]
for n in NS:
    Ln = Layout(model, n_gpu=n)
    for st in meas:
        s = Strategy(st, model, Ln, n=n); meas[st].append(s.cl.ranks[0].persistent_bytes() / Ln.Ppad)
print('bytes per weight held by one GPU (measured):')
print(f"{'GPUs':>5s} " + ' '.join(f'{st:>7s}' for st in meas))
for i, n in enumerate(NS):
    print(f'{n:5d} ' + ' '.join(f'{meas[st][i]:7.2f}' for st in meas))

GP = [8, 16, 32, 64, 128, 256]
lad = memory_ladder(30e9, GP)
print('\\n30B model, GiB of training state per GPU:')
print(f"{'GPUs':>5s} " + ' '.join(f'{st:>8s}' for st in lad))
for i, n in enumerate(GP):
    print(f'{n:5d} ' + ' '.join(f'{lad[st][i]:8.1f}' for st in lad))
fig, ax = plt.subplots(figsize=(8, 4.5))
for st, c, ls in [('dp', '#4A6076', '-'), ('zero1', '#3FA7FF', '--'), ('zero2', '#26D7C8', '-.'), ('zero3', '#FF9F35', ':')]:
    ax.plot(GP, lad[st], color=c, lw=2.5, ls=ls, marker='o', label=st)
ax.axhline(74.5, color='#F5D33B', lw=1.5); ax.text(GP[0], 80, '74.5 GiB usable on an 80 GB card', color='white')
ax.set_xscale('log', base=2); ax.set_yscale('log')
dark(ax, '30B model: training state per GPU', 'number of GPUs', 'GiB per GPU (log)')
ax.legend(facecolor='#141C25', labelcolor='white', edgecolor='#35485E'); fig.tight_layout(); fig.savefig('figures/ladder_30b.png', dpi=130)
RESULTS['bytes_per_weight_measured'] = {st: dict(zip(NS, v)) for st, v in meas.items()}
RESULTS['ladder_30b_gib'] = {st: dict(zip(GP, v)) for st, v in lad.items()}
json.dump(RESULTS, open('results.json', 'w'), indent=1, default=str)
print('saved results.json')
""")

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'session12_zero.ipynb')
print('wrote session12_zero.ipynb with', len(cells), 'cells')
