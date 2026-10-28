"""Builds session13_reversible.ipynb: live checks plus the results of the three 50M-token runs (from logs/)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Session 13 — a 20M LLM, standard against reversible

The three 50M-token runs take hours, so they run from the command line (`./run_all.sh`, which calls `rev_train.py`).
This notebook (1) re-checks the reversible backward pass live, (2) re-measures memory, and (3) loads the logs of the
three full runs.
""")
code("""
import json, sys, numpy as np, torch
sys.argv = ['x']
import rev_train as r
print('device', r.DEVICE, '| d', r.D, 'layers', r.LAYERS, 'heads', r.HEADS, 'context', r.CTX)
for v in ['standard', 'midpoint', 'revnet']:
    print(f'{v:9s} parameters {sum(p.numel() for p in r.LM(10000, v).parameters()):,}')
""")
md("## 1. The reversible backward gives the same gradients as autograd (fp32)")
code("""
train, val = r.load()
torch.manual_seed(0)
m = r.LM(10000, 'midpoint').to(r.DEVICE).float()
xb = r.batch(train, 4, np.random.default_rng(0))
F = torch.nn.functional
loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, 10000), xb[:, 1:].reshape(-1)); loss.backward()
g_rev = torch.cat([p.grad.flatten() for p in m.blocks.parameters()]); m.zero_grad()
x = m.emb(xb[:, :-1]) + m.pos(torch.arange(r.CTX, device=r.DEVICE)); prev, cur = x, x
for b in m.blocks: prev, cur = cur, prev + 2 * r.H * b(cur)            # same rule, ordinary autograd, everything stored
F.cross_entropy(m.head(m.norm(cur)).reshape(-1, 10000), xb[:, 1:].reshape(-1)).backward()
g_ref = torch.cat([p.grad.flatten() for p in m.blocks.parameters()])
print('relative gradient error, reversible vs autograd:', ((g_rev - g_ref).norm() / g_ref.norm()).item())
print('reconstruction error max |p[0] - p[-1]| (both should equal the input):', r.MidpointStack.recon)
""")
md("## 2. Memory: the budget is the standard model at batch 32")
code("""
probe = json.load(open('logs/memory_probe.json'))
print(f"budget {probe['budget_GiB']:.2f} GiB; largest batch inside it: {probe['max_batch']}")
for row in probe['rows']:
    if 'bs' in row: print(f"{row['variant']:9s} batch {row['bs']:4d}  peak {row['peak_GiB']:5.2f} GiB  {'fits' if row['fits'] else 'over'}")
""")
md("## 3. The three 50M-token runs")
code("""
for n in ['standard', 'reversible', 'rev_maxbatch']:
    s = json.load(open(f'logs/{n}_summary.json'))
    print(f"{n:13s} {s['variant']:9s} batch {s['batch']:3d}  steps {s['steps']:5d}  tokens {s['tokens']:,}  "
          f"final train {s['final_train_loss']:.3f}  final val {s['final_val_loss']:.3f}  peak {s['peak_GiB']:.2f} GiB")
bench = json.load(open('logs/bench_idle.json'))
print('\\nspeed measured on an idle machine (bench.py):')
for b in bench: print(f"{b['name']:13s} batch {b['batch']:3d}  {b['tok_s']:,} tok/s  {b['step_s']} s/step  peak {b['peak_GiB']} GiB")
""")
code("""
from IPython.display import Image
Image('figures/results.png')
""")
nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'session13_reversible.ipynb'); print('ok')
