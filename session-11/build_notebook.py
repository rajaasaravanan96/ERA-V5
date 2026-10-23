"""Builds session11_optimizers.ipynb.

Run:  python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace session11_optimizers.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Session 11 — optimizers and learning-rate schedules, measured

1. Adam by hand on one weight and five gradients, checked against PyTorch.
2. Bias correction switched off, first twenty steps.
3. The update-to-weight ratio of every layer during warmup.
4. Cosine against WSD, both stopped at step 200 of a 300-step plan, each with its own tuned learning rate.
5. A learning-rate sweep at widths 256, 512 and 1,024, and a prediction for 4,096.

Model: small pre-norm transformer (`tinylm.py`). Data: my Session 2 tokenizer and English + Hindi corpus.
""")

code("""
import json, math, os, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tinylm import TinyLM, load_corpus

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('torch', torch.__version__, '| device', DEVICE)
tok, ids_per_doc = load_corpus()
V = tok.get_vocab_size() + 1
stream = torch.tensor([t for d in ids_per_doc for t in d])
split = int(0.95 * len(stream)); train_stream, val_stream = stream[:split], stream[split:]
os.makedirs('figures', exist_ok=True); RESULTS = {}

def get_batch(src, bs=16, T=128, gen=None):
    ix = torch.randint(0, len(src) - T - 1, (bs,), generator=gen)
    return torch.stack([src[i:i + T + 1] for i in ix]).to(DEVICE)

g_val = torch.Generator().manual_seed(999)
VAL_BATCHES = [get_batch(val_stream, bs=16, T=128, gen=g_val) for _ in range(8)]
@torch.no_grad()
def val_loss(m):
    m.eval(); tot = sum(F.cross_entropy(m(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1)).item() for b in VAL_BATCHES)
    m.train(); return tot / len(VAL_BATCHES)

def dark(ax, title, xl, yl):
    ax.set_facecolor('#0F151C'); ax.figure.set_facecolor('#0A0E13')
    for s in ax.spines.values(): s.set_color('#35485E')
    ax.tick_params(colors='white', which='both'); ax.grid(color='#25323F')
    ax.set_title(title, color='white'); ax.set_xlabel(xl, color='white'); ax.set_ylabel(yl, color='white')
def legend(ax, **kw): ax.legend(facecolor='#141C25', labelcolor='white', edgecolor='#35485E', **kw)
""")

md("""
## 1. Adam by hand

One weight w₀ = 0.5, five gradients, lr η = 0.1, β₁ = 0.9, β₂ = 0.999, ε = 1e-8, no weight decay.

m_t = β₁ m_{t−1} + (1 − β₁) g_t
v_t = β₂ v_{t−1} + (1 − β₂) g_t²
m̂_t = m_t / (1 − β₁ᵗ),  v̂_t = v_t / (1 − β₂ᵗ)
step_t = η · m̂_t / (√v̂_t + ε),  w_t = w_{t−1} − step_t
""")

code("""
grads = [0.30, -0.10, 0.25, 0.40, -0.05]
lr, b1, b2, eps, w0 = 0.1, 0.9, 0.999, 1e-8, 0.5

# --- by hand, plain Python floats ---
m = v = 0.0; w = w0; hand = []
for t, g in enumerate(grads, 1):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
    step = lr * m_hat / (math.sqrt(v_hat) + eps)
    w = w - step
    hand.append(dict(t=t, g=g, m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=step, w=w))

# --- PyTorch, same weight, same gradients (float64 so we compare maths, not rounding) ---
p = torch.nn.Parameter(torch.tensor([w0], dtype=torch.float64))
opt = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps)
print(f"{'t':>2} {'g':>6} | {'m':>11} {'v':>11} {'m_hat':>11} {'v_hat':>11} {'step':>11} {'w':>11} | max |hand - torch|")
worst = 0.0
for h in hand:
    w_before = p.item()
    p.grad = torch.tensor([h['g']], dtype=torch.float64); opt.step()
    st = opt.state[p]
    t_m, t_v = st['exp_avg'].item(), st['exp_avg_sq'].item()
    t_mh, t_vh = t_m / (1 - b1 ** h['t']), t_v / (1 - b2 ** h['t'])
    t_step, t_w = w_before - p.item(), p.item()
    diff = max(abs(a - b) for a, b in zip([h['m'], h['v'], h['m_hat'], h['v_hat'], h['step'], h['w']],
                                          [t_m, t_v, t_mh, t_vh, t_step, t_w]))
    worst = max(worst, diff)
    print(f"{h['t']:>2} {h['g']:>6.2f} | {h['m']:11.8f} {h['v']:11.8f} {h['m_hat']:11.8f} {h['v_hat']:11.8f} "
          f"{h['step']:11.8f} {h['w']:11.8f} | {diff:.1e}")
assert worst < 1e-12
print(f'[PASS] hand values match torch.optim.Adam to {worst:.1e} (better than 12 decimal places)')
RESULTS['adam_by_hand'] = hand
""")

md("""
## 2. Bias correction switched off

Without correction, m and v both start biased towards 0. The step becomes
η · m / √v = η · m̂(1 − β₁ᵗ) / (√v̂ · √(1 − β₂ᵗ)), so the uncorrected step is the corrected one times
**(1 − β₁ᵗ) / √(1 − β₂ᵗ)**. That factor is the whole story. First on one weight with a steady gradient,
then on the real model.
""")

code("""
def factor(t, b1=0.9, b2=0.999): return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)
T20 = range(1, 21)
print('step  corrected step  uncorrected step   ratio')
for t in [1, 2, 3, 5, 10, 15, 20]:
    print(f'{t:>4}  {lr:14.4f}  {lr * factor(t):16.4f}  {factor(t):6.2f}x')

def first_within(tol, b2):
    t = 1
    while abs(factor(t, b2=b2) - 1) > tol: t += 1
    return t
stops = {b2: {tol: first_within(tol, b2) for tol in (0.10, 0.01)} for b2 in (0.999, 0.95)}
for b2, d in stops.items():
    print(f'beta2={b2}: uncorrected step within 10% of corrected after {d[0.10]} steps, within 1% after {d[0.01]} steps')

# real model: AdamW with and without bias correction, same seed, same batches, 20 steps
class AdamNoBC(torch.optim.Optimizer):
    def __init__(s, params, lr, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            b1, b2 = g['betas']
            for p in g['params']:
                if p.grad is None: continue
                st = s.state[p]
                if not st: st['m'] = torch.zeros_like(p); st['v'] = torch.zeros_like(p)
                st['m'].mul_(b1).add_(p.grad, alpha=1 - b1); st['v'].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.addcdiv_(st['m'], st['v'].sqrt().add_(g['eps']), value=-g['lr'])

def run20(opt_fn, steps=20, lr_=1e-3):
    torch.manual_seed(0); m_ = TinyLM(V, d=192, n_layers=4, n_heads=6, ctx=256).to(DEVICE); o = opt_fn(m_.parameters(), lr_)
    gb = torch.Generator().manual_seed(5); losses, ratios = [], []
    for _ in range(steps):
        xb = get_batch(train_stream, gen=gb)
        l = F.cross_entropy(m_(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
        w = m_.blocks[1].w1.weight; before = w.detach().clone()
        o.zero_grad(); l.backward(); o.step()
        losses.append(l.item()); ratios.append(((w.detach() - before).norm() / before.norm()).item())
    return losses, ratios, val_loss(m_)

l_bc, r_bc, v_bc = run20(lambda p, lr_: torch.optim.Adam(p, lr=lr_))
l_no, r_no, v_no = run20(lambda p, lr_: AdamNoBC(p, lr=lr_))
print(f'\\nmodel, 20 steps at lr 1e-3: val loss with correction {v_bc:.3f}, without {v_no:.3f}')
print('update/weight ratio of blocks[1].w1, steps 1/5/10/20:',
      [f'{r_bc[i]:.2e}' for i in (0, 4, 9, 19)], 'vs', [f'{r_no[i]:.2e}' for i in (0, 4, 9, 19)])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(list(T20), [lr] * 20, color='#3FA7FF', lw=2, label='with bias correction')
axes[0].plot(list(T20), [lr * factor(t) for t in T20], color='#FF9F35', lw=2, ls='--', label='without')
dark(axes[0], 'One weight, steady gradient: step size', 'step', 'step size (lr = 0.1)'); legend(axes[0])
axes[1].plot(range(1, 21), r_bc, color='#3FA7FF', lw=2, label='with'); axes[1].plot(range(1, 21), r_no, color='#FF9F35', lw=2, ls='--', label='without')
axes[1].set_yscale('log'); dark(axes[1], 'Model: update / weight, blocks[1].w1', 'step', 'ratio'); legend(axes[1])
axes[2].plot(range(1, 21), l_bc, color='#3FA7FF', lw=2, label='with'); axes[2].plot(range(1, 21), l_no, color='#FF9F35', lw=2, ls='--', label='without')
dark(axes[2], 'Model: training loss', 'step', 'loss'); legend(axes[2])
fig.tight_layout(); fig.savefig('figures/bias_correction.png', dpi=130)
RESULTS['bias_correction'] = dict(factor_first20=[factor(t) for t in T20], stops={str(k): v for k, v in stops.items()},
                                  val_with=v_bc, val_without=v_no, loss_with=l_bc, loss_without=l_no)
""")

md("""
## 3. The update-to-weight ratio of every layer, through warmup

AdamW, peak lr 2e-3, linear warmup over 100 steps, then constant. After every step I log ‖Δw‖ / ‖w‖ for every
weight matrix. The question: at which step does warmup stop changing it?

Rule: the ratio of each layer is smoothed over 10 steps; warmup has stopped changing it at the first step after which
the smoothed ratio stays within ±25% of its median over steps 150–300 for good.
""")

code("""
torch.manual_seed(0)
m3 = TinyLM(V, d=192, n_layers=4, n_heads=6, ctx=256).to(DEVICE)
o3 = torch.optim.AdamW(m3.parameters(), lr=2e-3, weight_decay=0.1)
WARM = 100
layers = {n: p for n, p in m3.named_parameters() if p.dim() == 2 and not n.startswith('pos')}
ratio_log = {n: [] for n in layers}; lr3 = []
for step in range(300):
    lr_ = 2e-3 * min(1, (step + 1) / WARM)
    for g in o3.param_groups: g['lr'] = lr_
    xb = get_batch(train_stream)
    l = F.cross_entropy(m3(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
    before = {n: p.detach().clone() for n, p in layers.items()}
    o3.zero_grad(); l.backward(); o3.step(); lr3.append(lr_)
    for n, p in layers.items():
        ratio_log[n].append(((p.detach() - before[n]).norm() / before[n].norm()).item())

def smooth(xs, k=10): return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]
settle = {}
for n, xs in ratio_log.items():
    s = smooth(xs); ref = sorted(s[150:])[len(s[150:]) // 2]
    inside = [abs(x - ref) <= 0.25 * ref for x in s]
    t = next((i for i in range(len(s)) if all(inside[i:])), len(s) - 1)
    settle[n] = (t, ref)
print(f'{"layer":28s} {"ratio @ step 1":>14s} {"@ 50":>9s} {"@ 100":>9s} {"plateau":>9s} {"settles at step":>16s}')
for n in layers:
    r = ratio_log[n]
    print(f'{n:28s} {r[0]:14.2e} {r[49]:9.2e} {r[99]:9.2e} {settle[n][1]:9.2e} {settle[n][0]:16d}')
steps_settle = sorted(t for t, _ in settle.values())
median_settle = steps_settle[len(steps_settle) // 2]
print(f'\\nmedian layer settles at step {median_settle} (warmup ends at step {WARM}); range {steps_settle[0]}-{steps_settle[-1]}')

fig, ax = plt.subplots(figsize=(10, 4.5))
pal = ['#3FA7FF', '#FF9F35', '#26D7C8', '#B07BFF', '#F5D33B', '#4A6076']
show = ['emb.weight', 'blocks.0.qkv.weight', 'blocks.0.w1.weight', 'blocks.3.qkv.weight', 'blocks.3.w2.weight', 'blocks.3.o.weight']
for c, n in zip(pal, show):
    ax.plot(smooth(ratio_log[n]), color=c, lw=1.8, label=n)
ax.axvline(WARM, color='white', lw=1, ls=':'); ax.axvline(median_settle, color='#F5D33B', lw=1.5, ls='--')
ax.set_yscale('log'); dark(ax, f'Update / weight per layer (warmup ends at {WARM} dotted; median layer settles at {median_settle} dashed)', 'step', '||dw|| / ||w||')
legend(ax, fontsize=8, ncol=2); fig.tight_layout(); fig.savefig('figures/update_ratio.png', dpi=130)
RESULTS['update_ratio'] = {'warmup_steps': WARM, 'settle_step': {n: t for n, (t, _) in settle.items()},
                           'plateau': {n: r for n, (_, r) in settle.items()}, 'median_settle': median_settle,
                           'ratio_step1': {n: v[0] for n, v in ratio_log.items()}}
json.dump(ratio_log, open('update_ratio_log.json', 'w'))
""")

md("""
## 4. Cosine against WSD, both stopped at step 200 of a 300-step plan

Both schedules are planned for 300 steps with 15 steps of warmup:

- **cosine**: from the peak down to 10% of the peak at step 300.
- **WSD**: flat at the peak until step 240, then a linear decay to 10% of the peak over the last 60 steps (20%).

Both runs are **stopped at step 200**. Tuning both sides: each schedule gets its own 3-point learning-rate sweep, and the
comparison uses each one's best. I also do the thing WSD exists for: take the step-200 WSD checkpoint and give it its own
short 30-step decay.
""")

code("""
TOTAL, WARM4, STOP = 300, 15, 200
def lr_at(sched, step, peak):
    if step < WARM4: return peak * (step + 1) / WARM4
    if sched == 'cosine':
        prog = (step - WARM4) / (TOTAL - WARM4); return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
    if sched == 'wsd':
        d0 = int(0.8 * TOTAL)
        return peak if step < d0 else peak * (1 - 0.9 * (step - d0) / (TOTAL - d0))
    raise ValueError(sched)

def train_sched(sched, peak, stop=STOP, seed=0, state=None, decay_from=None, decay_len=30):
    torch.manual_seed(seed)
    m_ = TinyLM(V, d=192, n_layers=4, n_heads=6, ctx=256).to(DEVICE)
    o_ = torch.optim.AdamW(m_.parameters(), lr=peak, weight_decay=0.1, betas=(0.9, 0.95))
    start = 0
    if state is not None:
        m_.load_state_dict(state['model']); o_.load_state_dict(state['opt']); start = state['step']
    gb = torch.Generator().manual_seed(42 + start)
    curve = []
    for step in range(start, stop):
        if decay_from is not None:
            lr_ = peak * (1 - 0.9 * (step - decay_from + 1) / decay_len)
        else:
            lr_ = lr_at(sched, step, peak)
        for g in o_.param_groups: g['lr'] = lr_
        xb = get_batch(train_stream, gen=gb)
        l = F.cross_entropy(m_(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
        o_.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m_.parameters(), 1.0); o_.step()
        if (step + 1) % 20 == 0: curve.append((step + 1, val_loss(m_)))
    return m_, o_, curve

sweep4 = {}
for sched in ('cosine', 'wsd'):
    for peak in (1e-3, 2e-3, 4e-3):
        m_, o_, curve = train_sched(sched, peak)
        sweep4[(sched, peak)] = dict(val200=curve[-1][1], curve=curve, model=m_, opt=o_)
        print(f'{sched:6s} peak {peak:.0e}: val loss at step 200 = {curve[-1][1]:.4f}')
best = {s: min((k for k in sweep4 if k[0] == s), key=lambda k: sweep4[k]['val200']) for s in ('cosine', 'wsd')}
cos_k, wsd_k = best['cosine'], best['wsd']
print(f"\\nbest cosine: peak {cos_k[1]:.0e}, loss at 200 = {sweep4[cos_k]['val200']:.4f}")
print(f"best WSD   : peak {wsd_k[1]:.0e}, loss at 200 = {sweep4[wsd_k]['val200']:.4f}")

# WSD's real advantage: decay the step-200 checkpoint separately, 30 steps
state = dict(model=sweep4[wsd_k]['model'].state_dict(), opt=sweep4[wsd_k]['opt'].state_dict(), step=STOP)
_, _, c_dec = train_sched('wsd', wsd_k[1], stop=STOP + 30, state=state, decay_from=STOP, decay_len=30)
# cosine given the same 30 extra steps on its own schedule (it keeps decaying towards step 300)
state_c = dict(model=sweep4[cos_k]['model'].state_dict(), opt=sweep4[cos_k]['opt'].state_dict(), step=STOP)
_, _, c_cos_more = train_sched('cosine', cos_k[1], stop=STOP + 30, state=state_c)
print(f'WSD step-200 checkpoint + its own 30-step decay  -> {c_dec[-1][1]:.4f} at step 230')
print(f'cosine continued 30 more steps on its schedule   -> {c_cos_more[-1][1]:.4f} at step 230')

# full 300-step runs at the best peaks, for reference
_, _, full_cos = train_sched('cosine', cos_k[1], stop=TOTAL); _, _, full_wsd = train_sched('wsd', wsd_k[1], stop=TOTAL)
print(f'full 300 steps: cosine {full_cos[-1][1]:.4f}, WSD {full_wsd[-1][1]:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
xs = range(TOTAL)
axes[0].plot(xs, [lr_at('cosine', s, cos_k[1]) for s in xs], color='#3FA7FF', lw=2, label=f'cosine (peak {cos_k[1]:.0e})')
axes[0].plot(xs, [lr_at('wsd', s, wsd_k[1]) for s in xs], color='#FF9F35', lw=2, ls='--', label=f'WSD (peak {wsd_k[1]:.0e})')
axes[0].axvline(STOP, color='#F5D33B', lw=1.5); dark(axes[0], 'Planned schedules (gold: where we stop)', 'step', 'learning rate'); legend(axes[0])
axes[1].plot(*zip(*full_cos), color='#3FA7FF', lw=2, label='cosine, full 300')
axes[1].plot(*zip(*full_wsd), color='#FF9F35', lw=2, ls='--', label='WSD, full 300')
axes[1].plot(*zip(*([(STOP, sweep4[wsd_k]['val200'])] + c_dec)), color='#26D7C8', lw=2.5, label='WSD @200 + own 30-step decay')
axes[1].axvline(STOP, color='#F5D33B', lw=1.5); axes[1].set_xlim(100, 305)
lo = min(v for _, v in full_cos + full_wsd + c_dec); axes[1].set_ylim(lo - 0.1, lo + 1.0)
dark(axes[1], 'Validation loss', 'step', 'loss'); legend(axes[1])
fig.tight_layout(); fig.savefig('figures/cosine_vs_wsd.png', dpi=130)
RESULTS['cosine_vs_wsd'] = dict(sweep={f'{k[0]}@{k[1]}': v['val200'] for k, v in sweep4.items()},
                                best_cosine=[cos_k[1], sweep4[cos_k]['val200']], best_wsd=[wsd_k[1], sweep4[wsd_k]['val200']],
                                wsd_200_plus_decay30=c_dec[-1][1], cosine_230=c_cos_more[-1][1],
                                full300_cosine=full_cos[-1][1], full300_wsd=full_wsd[-1][1])
for v in sweep4.values(): v.pop('model'); v.pop('opt')
""")

md("""
## 5. Learning-rate sweep at widths 256, 512 and 1,024

Standard parameterization (init std 0.02 at every width, one global AdamW lr), 2 layers, head dim 64,
200 steps, batch 16 × 128 tokens, 20 warmup steps then cosine to 10%. Seven learning rates per width, the same
data order for every run. The marked minimum is the vertex of a parabola fitted through the best point and its
two neighbours in log-lr space.
""")

code("""
WIDTHS = [256, 512, 1024]
LRS = [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]
STEPS5 = 200
def sweep_run(width, peak):
    torch.manual_seed(0)
    m_ = TinyLM(V, d=width, n_layers=2, n_heads=width // 64, ctx=256).to(DEVICE)
    o_ = torch.optim.AdamW(m_.parameters(), lr=peak, weight_decay=0.1, betas=(0.9, 0.95))
    gb = torch.Generator().manual_seed(7)
    for step in range(STEPS5):
        lr_ = peak * (step + 1) / 20 if step < 20 else peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 20) / (STEPS5 - 20))))
        for g in o_.param_groups: g['lr'] = lr_
        xb = get_batch(train_stream, gen=gb)
        l = F.cross_entropy(m_(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
        if not torch.isfinite(l): return float('nan')
        o_.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m_.parameters(), 1.0); o_.step()
    return val_loss(m_)

sweep = {}
t0 = time.time()
for w in WIDTHS:
    sweep[w] = []
    for peak in LRS:
        sweep[w].append(sweep_run(w, peak))
    print(f'width {w:5d}: ' + '  '.join(f'{p:.1e}->{v:.3f}' for p, v in zip(LRS, sweep[w])) + f'   ({time.time() - t0:.0f}s)')
    json.dump({str(k): v for k, v in sweep.items()}, open('lr_sweep.json', 'w'))

def fitted_min(lrs, losses):
    i = min(range(len(losses)), key=lambda j: losses[j] if losses[j] == losses[j] else 1e9)
    i = min(max(i, 1), len(lrs) - 2)
    x = [math.log(lrs[j]) for j in (i - 1, i, i + 1)]; y = [losses[j] for j in (i - 1, i, i + 1)]
    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    A = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
    B = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0]) + x[0] ** 2 * (y[1] - y[2])) / denom
    xv = -B / (2 * A) if A > 0 else x[1]
    xv = min(max(xv, x[0]), x[2])
    C = y[1] - A * x[1] ** 2 - B * x[1]
    return math.exp(xv), A * xv ** 2 + B * xv + C
minima = {w: fitted_min(LRS, sweep[w]) for w in WIDTHS}
for w, (lr_star, l_star) in minima.items():
    print(f'width {w:5d}: best lr = {lr_star:.2e}  (fitted loss {l_star:.3f})')

# power law lr* = c * width^slope, least squares in log-log
lx = [math.log(w) for w in WIDTHS]; ly = [math.log(minima[w][0]) for w in WIDTHS]
mx, my = sum(lx) / 3, sum(ly) / 3
slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
icpt = my - slope * mx
pred4096 = math.exp(icpt + slope * math.log(4096))
resid = [b - (icpt + slope * a) for a, b in zip(lx, ly)]
print(f'\\nfit: lr* ~ width^{slope:.2f}  (lesson rule of thumb: width^-1.00)')
print(f'prediction at width 4096: {pred4096:.2e}   (the 1/width rule from width 1024 would give {minima[1024][0] / 4:.2e})')
print('log residuals of the fit:', [f'{r:+.2f}' for r in resid])

fig, ax = plt.subplots(figsize=(8, 4.5))
for c, w in zip(['#3FA7FF', '#FF9F35', '#26D7C8'], WIDTHS):
    ax.plot(LRS, sweep[w], color=c, lw=2, marker='o', label=f'width {w}')
    ax.plot([minima[w][0]], [minima[w][1]], marker='*', ms=18, color=c, markeredgecolor='white')
ax.set_xscale('log'); dark(ax, 'Validation loss after 200 steps against learning rate (stars: fitted minima)', 'peak learning rate', 'validation loss')
legend(ax); fig.tight_layout(); fig.savefig('figures/lr_sweep.png', dpi=130)
RESULTS['lr_sweep'] = dict(lrs=LRS, losses={str(w): sweep[w] for w in WIDTHS}, minima={str(w): minima[w] for w in WIDTHS},
                           slope=slope, pred4096=pred4096, residuals=resid)
json.dump(RESULTS, open('results.json', 'w'), indent=1, default=str)
print('saved results.json')
""")

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'session11_optimizers.ipynb')
print('wrote session11_optimizers.ipynb with', len(cells), 'cells')
