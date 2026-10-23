"""Builds session10_training_loop.ipynb.

Run:  python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace session10_training_loop.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Session 10 — make the training loop tell the truth about itself

Six checks on a small model and a real loop: shapes, a hand-verified gradient, broken gradient accumulation,
the grad norm moving before the loss, an honest MFU, and the number 0.1 in three formats.

Model: 4-layer pre-norm transformer, tied head (`tinylm.py`). Tokenizer and text: my Session 2 BPE tokenizer and
English + Hindi Wikipedia corpus.
""")

code("""
import json, math, os, struct, time
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tinylm import TinyLM, load_corpus

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('torch', torch.__version__, '| device', DEVICE)
tok, ids_per_doc = load_corpus()
PAD_ID = tok.get_vocab_size(); V = PAD_ID + 1
stream = torch.tensor([t for d in ids_per_doc for t in d])
split = int(0.95 * len(stream)); train_stream, val_stream = stream[:split], stream[split:]
print(f'V = {V}, {len(stream):,} tokens')
CFG = dict(d=192, n_layers=4, n_heads=6, ctx=256)
os.makedirs('figures', exist_ok=True)
RESULTS = {}

def get_batch(src, bs=16, T=256, device=DEVICE):
    ix = torch.randint(0, len(src) - T - 1, (bs,))
    return torch.stack([src[i:i + T + 1] for i in ix]).to(device)

def dark(ax, title, xl, yl):
    ax.set_facecolor('#0F151C'); ax.figure.set_facecolor('#0A0E13')
    for s in ax.spines.values(): s.set_color('#35485E')
    ax.tick_params(colors='white'); ax.grid(color='#25323F')
    ax.set_title(title, color='white'); ax.set_xlabel(xl, color='white'); ax.set_ylabel(yl, color='white')
""")

md("## 1. Every tensor shape in one training step")

code("""
torch.manual_seed(0)
model = TinyLM(V, **CFG).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
xb = get_batch(train_stream, bs=4, T=32)
inputs, targets = xb[:, :-1], xb[:, 1:]
trace = []
logits = model(inputs, trace)
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
loss.backward()
opt.step()

rows = [('batch from loader', xb, '(batch, time+1): one extra token so inputs and targets can both be cut from it'),
        ('inputs', inputs, '(batch, time): tokens the model reads'),
        ('targets', targets, '(batch, time): the same tokens shifted left by one')] + trace + [
        ('logits.reshape(-1, V)', logits.reshape(-1, V), '(batch*time, vocab): every prediction as one row'),
        ('targets.reshape(-1)', targets.reshape(-1), '(batch*time,): the correct id for each row'),
        ('loss', loss, '(): scalar mean cross-entropy'),
        ('emb.weight.grad', model.emb.weight.grad, '(vocab, d_model): gradient, same shape as the weight (tied with the head)'),
        ('blocks[0].w1.weight.grad', model.blocks[0].w1.weight.grad, '(ffn_hidden, d_model): gradient of the first FFN matrix'),
        ('Adam exp_avg (w1)', opt.state[model.blocks[0].w1.weight]['exp_avg'], '(ffn_hidden, d_model): first moment m, one per weight'),
        ('Adam exp_avg_sq (w1)', opt.state[model.blocks[0].w1.weight]['exp_avg_sq'], '(ffn_hidden, d_model): second moment v, one per weight')]
for name, t, meaning in rows:
    print(f'{name:26s} {str(tuple(t.shape)):22s} {meaning}')
""")

md("""
## 2. Verify one gradient by hand

Nudge one weight by ±ε, measure the change in loss, and compare with what `backward()` reported:

grad ≈ (L(w + ε) − L(w − ε)) / 2ε

Done in float64 on the CPU so that rounding does not hide a real disagreement.
""")

code("""
torch.manual_seed(0)
m64 = TinyLM(V, **CFG).double()
xb = get_batch(train_stream, bs=2, T=32, device='cpu')
def L():
    return F.cross_entropy(m64(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
m64.zero_grad(); L().backward()
checks = []
for name, param, idx in [('blocks[0].w1.weight', m64.blocks[0].w1.weight, (3, 5)),
                         ('blocks[2].qkv.weight', m64.blocks[2].qkv.weight, (100, 17)),
                         ('emb.weight (tied head)', m64.emb.weight, (int(xb[0, 5]), 9))]:
    auto = param.grad[idx].item()
    eps = 1e-6
    with torch.no_grad():
        w0 = param[idx].item()
        param[idx] = w0 + eps; lp = L().item()
        param[idx] = w0 - eps; lm = L().item()
        param[idx] = w0
    numeric = (lp - lm) / (2 * eps)
    rel = abs(numeric - auto) / max(abs(auto), 1e-12)
    checks.append((name, idx, auto, numeric, rel))
    print(f'{name:24s}{str(idx):12s} backward {auto: .10e}   by hand {numeric: .10e}   rel. diff {rel:.1e}')
assert all(c[4] < 1e-5 for c in checks)
print('[PASS] hand-computed gradients agree with backward() to at least 5 significant digits')
RESULTS['grad_check'] = [{'param': c[0], 'index': list(c[1]), 'backward': c[2], 'numeric': c[3], 'rel_diff': c[4]} for c in checks]
""")

md("""
## 3. Break gradient accumulation on purpose

Each optimizer step gathers 4 micro-batches. The micro-batches hold **different numbers of real tokens**:
two long ones (256 tokens), and two short ones (24 and 8 tokens, the rest padding).

- **Correct:** add up the loss of every real token across all 4 micro-batches, then divide by the total token count.
- **Broken (the pre-2024 bug):** take each micro-batch's mean loss, then average the 4 means. An 8-token micro-batch
  gets the same vote as a 256-token one.

First, the gradient itself: the correct accumulation must equal one big batch exactly, and the broken one must not.
""")

code("""
LENS = [256, 256, 24, 8]
def make_micro(src, lens, T=256):
    out = []
    for n in lens:
        i = torch.randint(0, len(src) - n - 1, (1,)).item()
        seq = torch.full((T + 1,), PAD_ID); seq[:n + 1] = src[i:i + n + 1]
        tgt = seq[1:].clone(); tgt[n:] = -100            # padding targets ignored
        out.append((seq[:-1], tgt))
    return out

def accumulate(model, micro, mode):
    total_tokens = sum(int((t != -100).sum()) for _, t in micro)
    logged = 0.0
    for x, t in micro:
        lg = model(x[None].to(DEVICE))
        tok_loss = F.cross_entropy(lg.reshape(-1, V), t.to(DEVICE), ignore_index=-100, reduction='sum')
        n = int((t != -100).sum())
        if mode == 'correct':
            part = tok_loss / total_tokens                 # every token weighted equally
        else:
            part = (tok_loss / n) / len(micro)             # average of averages
        part.backward(); logged += part.item()
    return logged

torch.manual_seed(3)
micro = make_micro(train_stream, LENS)
base = TinyLM(V, **CFG).to(DEVICE)
def grad_vec(m): return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])

# reference: all real tokens in one big batch (padded), mean over real tokens
base.zero_grad()
X = torch.stack([x for x, _ in micro]).to(DEVICE); Y = torch.stack([t for _, t in micro]).to(DEVICE)
ref_loss = F.cross_entropy(base(X).reshape(-1, V), Y.reshape(-1), ignore_index=-100); ref_loss.backward()
g_ref = grad_vec(base).clone()
base.zero_grad(); l_ok = accumulate(base, micro, 'correct'); g_ok = grad_vec(base).clone()
base.zero_grad(); l_bad = accumulate(base, micro, 'broken'); g_bad = grad_vec(base).clone()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f'tokens per micro-batch       : {LENS}  (total {sum(LENS)})')
print(f'one big batch loss           : {ref_loss.item():.5f}')
print(f'correct accumulation loss    : {l_ok:.5f}   gradient rel. error vs big batch {rel(g_ok, g_ref):.2e}')
print(f'broken accumulation loss     : {l_bad:.5f}   gradient rel. error vs big batch {rel(g_bad, g_ref):.2e}')
print(f'weight of the 8-token micro-batch: correct {8/sum(LENS):.1%}, broken {1/len(LENS):.1%}')
RESULTS['accumulation_grad'] = {'lens': LENS, 'ref_loss': ref_loss.item(), 'correct_loss': l_ok, 'broken_loss': l_bad,
                                'correct_rel_err': rel(g_ok, g_ref), 'broken_rel_err': rel(g_bad, g_ref)}
""")

md("""
Now train two identical models (same seed, same micro-batches in the same order) for 300 steps, one with each rule,
and plot both curves together. The y-axis is the **token-weighted validation loss** for both, so they are judged by the same ruler.
""")

code("""
def train_accum(mode, steps=300, seed=0):
    torch.manual_seed(seed); m = TinyLM(V, **CFG).to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.1)
    g = torch.Generator().manual_seed(123)
    hist = []
    for step in range(steps):
        torch.manual_seed(1000 + step)                     # identical micro-batches for both runs
        micro = make_micro(train_stream, LENS)
        o.zero_grad(); logged = accumulate(m, micro, mode)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
        if step % 10 == 0 or step == steps - 1:
            m.eval()
            with torch.no_grad():
                torch.manual_seed(7)
                vb = get_batch(val_stream, bs=16, T=256)
                v = F.cross_entropy(m(vb[:, :-1]).reshape(-1, V), vb[:, 1:].reshape(-1)).item()
            m.train(); hist.append((step, logged, v))
    return hist

t0 = time.time()
h_ok = train_accum('correct'); h_bad = train_accum('broken')
print(f'trained both in {time.time() - t0:.0f}s')
for (s, lo, vo), (_, lb, vb) in list(zip(h_ok, h_bad))[::5] + [(h_ok[-1], h_bad[-1])]:
    print(f'step {s:3d}  val loss correct {vo:.3f}   broken {vb:.3f}   gap {vb - vo:+.3f}   (logged train loss {lo:.3f} vs {lb:.3f})')
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot([h[0] for h in h_ok], [h[2] for h in h_ok], color='#3FA7FF', lw=2, label='correct: sum over tokens / total tokens')
ax.plot([h[0] for h in h_bad], [h[2] for h in h_bad], color='#FF9F35', lw=2, ls='--', label='broken: average of micro-batch averages')
dark(ax, 'Gradient accumulation, micro-batches of 256, 256, 24 and 8 tokens', 'optimizer step', 'validation loss (token-weighted)')
ax.legend(facecolor='#141C25', labelcolor='white', edgecolor='#35485E'); fig.tight_layout(); fig.savefig('figures/accumulation.png', dpi=130)
RESULTS['accumulation_train'] = {'final_val_correct': h_ok[-1][2], 'final_val_broken': h_bad[-1][2],
                                 'curve_correct': h_ok, 'curve_broken': h_bad}
""")

md("""
## 4. Log the grad norm at every step, and find a step where it moved before the loss

The run trains normally for 200 steps (warmup to 2e-3), then the learning rate grows by 3% every step —
the kind of mistake a broken schedule makes. Clipping is off, and the grad norm is logged at **every** step.

To decide *when* each signal moved, both get the same rule: take the 10-step rolling mean, compare it with its own
spread over a calm baseline window (steps 200–240), and call the first step where it is more than 4 baseline
standard deviations away "moved".
""")

code("""
def roll(xs, k=10): return [sum(xs[i - k + 1:i + 1]) / k if i >= k - 1 else float('nan') for i in range(len(xs))]
def first_move(xs, lo=200, hi=240, z=4.0):
    r = roll(xs); base = r[lo:hi]; mu = sum(base) / len(base)
    sd = (sum((b - mu) ** 2 for b in base) / len(base)) ** 0.5
    for t in range(hi, len(r)):
        if abs(r[t] - mu) > z * sd:
            return t, mu, sd, r[t]
    return None, mu, sd, None

def lr_ramp_run(beta2):
    m = TinyLM(V, **CFG).to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.1, betas=(0.9, beta2))
    G, L, LR = [], [], []
    for step in range(400):
        lr = 2e-3 * min(1, (step + 1) / 30) if step < 200 else 2e-3 * 1.03 ** (step - 200)
        for g in o.param_groups: g['lr'] = lr
        xb = get_batch(train_stream)
        loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
        o.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), float('inf')).item()   # only measures the norm
        o.step(); G.append(gn); L.append(loss.item()); LR.append(lr)
    return G, L, LR

def lr_ramp_seed(seed, beta2=0.95):
    torch.manual_seed(seed)
    return lr_ramp_run(beta2)

runs = {}
for seed in range(5):
    G, L, LR = lr_ramp_seed(seed)
    g_t, g_mu, g_sd, g_val = first_move(G); l_t, l_mu, l_sd, l_val = first_move(L)
    runs[seed] = dict(grad_norm=G, loss=L, lr=LR, gn_moved=g_t, loss_moved=l_t, base=(g_mu, g_sd, l_mu, l_sd))
    lead = l_t - g_t
    print(f'seed {seed}: grad norm moved at step {g_t} ({g_val:.3f} vs {g_mu:.3f} +/- {g_sd:.3f}),  '
          f'loss moved at step {l_t} ({l_val:.3f} vs {l_mu:.3f} +/- {l_sd:.3f})  ->  '
          + (f'grad norm FIRST by {lead} steps' if lead > 0 else (f'loss first by {-lead} steps' if lead < 0 else 'same step')))
best = max(runs, key=lambda k: runs[k]['loss_moved'] - runs[k]['gn_moved'])
r = runs[best]; g_t = r['gn_moved']
print(f'\\nclearest case, seed {best}: at step {g_t} the grad norm (10-step mean {roll(r["grad_norm"])[g_t]:.3f}) had left its band, '
      f'while the loss (10-step mean {roll(r["loss"])[g_t]:.3f}) was still inside {r["base"][2]:.3f} +/- 4x{r["base"][3]:.3f}')
runs = {f'seed {best}': runs[best], **{f'seed {k}': v for k, v in runs.items() if k != best}}
events = {k: (v['gn_moved'], v['loss_moved']) for k, v in runs.items()}
json.dump(runs, open('gradnorm_log.json', 'w'))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (beta2, r) in zip(axes, list(runs.items())[:2]):
    ax2 = ax.twinx()
    ax.plot(r['grad_norm'], color='#3FA7FF', lw=0.8, alpha=0.7); ax.plot(roll(r['grad_norm']), color='#3FA7FF', lw=2.5, label='grad norm (10-step mean)')
    ax2.plot(r['loss'], color='#FF9F35', lw=0.8, alpha=0.7); ax2.plot(roll(r['loss']), color='#FF9F35', lw=2.5, ls='--', label='loss (10-step mean)')
    ax.axvline(r['gn_moved'], color='#3FA7FF', lw=1.5, ls=':'); ax.axvline(r['loss_moved'], color='#FF9F35', lw=1.5, ls=':')
    ax.axvline(200, color='#F5D33B', lw=1)
    ax.set_ylim(0, 2.5); ax2.set_ylim(5.5, 9.5); ax.set_xlim(150, 340)
    dark(ax, f'{beta2}: grad norm moved at {r["gn_moved"]}, loss at {r["loss_moved"]}', 'step  (LR grows 3%/step after the gold line)', 'grad norm')
    ax2.tick_params(colors='white'); ax2.set_ylabel('loss', color='white')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor='#141C25', labelcolor='white', edgecolor='#35485E', loc='upper left')
fig.tight_layout()
fig.savefig('figures/gradnorm_first.png', dpi=130)
RESULTS['gradnorm_events'] = events
""")

md("""
## 5. My own MFU

MFU = (FLOPs the model needs per second) / (FLOPs the machine can do per second).

- **Model FLOPs per token** = 6N for the weights (forward 2N + backward 4N), plus the attention scores, 12·L·T·d.
  N counts the non-embedding weights plus the head (the embedding lookup costs no matmul).
- **Machine peak**: Apple does not publish an fp32 TFLOPS figure for the M5 GPU, so I **measure it**:
  the best throughput of large square matmuls (4096 × 4096) on the same device. That is the most this GPU can do
  through PyTorch, so MFU against it is the fair number.

Two model widths are timed, to see what is costing the distance to 40%.
""")

code("""
def sync():
    if DEVICE == 'mps': torch.mps.synchronize()
    elif DEVICE == 'cuda': torch.cuda.synchronize()

def matmul_peak(dtype, sizes=(1024, 2048, 3072, 4096), reps=20, trials=3):
    best = 0
    for n in sizes:
        a = torch.randn(n, n, device=DEVICE, dtype=dtype); b = torch.randn(n, n, device=DEVICE, dtype=dtype)
        for _ in range(3): a @ b
        for _ in range(trials):
            sync(); t = time.perf_counter()
            for _ in range(reps): a @ b
            sync(); best = max(best, 2 * n ** 3 * reps / (time.perf_counter() - t))
    return best

peak32 = matmul_peak(torch.float32); peak16 = matmul_peak(torch.bfloat16)
print(f'measured matmul peak: fp32 {peak32 / 1e12:.2f} TFLOP/s   bf16 {peak16 / 1e12:.2f} TFLOP/s')

def mfu_run(cfg, bs, T=256, steps=30, dtype=None):
    torch.manual_seed(0); m = TinyLM(V, **cfg).to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    n_body = sum(p.numel() for n_, p in m.named_parameters() if not n_.startswith(('emb', 'pos')))
    N = n_body + V * cfg['d']                                       # + head matmul (tied weights still cost FLOPs)
    flops_tok = 6 * N + 12 * cfg['n_layers'] * T * cfg['d']
    def step():
        xb = get_batch(train_stream, bs=bs, T=T)
        if dtype is None:
            loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
        else:
            with torch.autocast(device_type=DEVICE, dtype=dtype):
                loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, V).float(), xb[:, 1:].reshape(-1))
        o.zero_grad(); loss.backward(); o.step()
    for _ in range(5): step()
    sync(); t = time.perf_counter()
    for _ in range(steps): step()
    sync(); dt = time.perf_counter() - t
    tps = bs * T * steps / dt
    return dict(N=N, flops_per_token=flops_tok, tokens_per_s=tps, achieved_tflops=tps * flops_tok / 1e12)

rows = []
for label, cfg, bs, dtype, peak in [('d=192, 4 layers, fp32', CFG, 16, None, peak32),
                                    ('d=512, 6 layers, fp32', dict(d=512, n_layers=6, n_heads=8, ctx=256), 16, None, peak32),
                                    ('d=512, 6 layers, bf16 autocast', dict(d=512, n_layers=6, n_heads=8, ctx=256), 16, torch.bfloat16, peak16)]:
    r = mfu_run(cfg, bs, dtype=dtype); r['label'] = label; r['mfu'] = r['achieved_tflops'] * 1e12 / peak; rows.append(r)
    print(f"{label:32s} N={r['N'] / 1e6:5.2f}M  {r['tokens_per_s']:9,.0f} tok/s  {r['achieved_tflops']:.2f} TFLOP/s  MFU {r['mfu']:.1%}")
RESULTS['mfu'] = {'peak_fp32_tflops': peak32 / 1e12, 'peak_bf16_tflops': peak16 / 1e12, 'runs': rows}
""")

md("""
## 6. The number 0.1 in fp32, bf16 and fp8 E4M3

Worked by hand first (in the README), then checked against the bits PyTorch actually stores.
""")

code("""
def bits_of(x, dtype):
    t = torch.tensor([x], dtype=dtype)
    if dtype == torch.float32:
        u = struct.unpack('>I', struct.pack('>f', x))[0]; return format(u, '032b'), t.item()
    if dtype == torch.bfloat16:
        u = t.view(torch.int16).item() & 0xFFFF; return format(u, '016b'), t.float().item()
    u = t.view(torch.uint8).item(); return format(u, '08b'), t.float().item()

layout = {torch.float32: (8, 23, 127), torch.bfloat16: (8, 7, 127), torch.float8_e4m3fn: (4, 3, 7)}
fp_rows = []
for name, dt in [('fp32', torch.float32), ('bf16', torch.bfloat16), ('fp8 E4M3', torch.float8_e4m3fn)]:
    b, val = bits_of(0.1, dt); e_bits, m_bits, bias = layout[dt]
    s, e, mnt = b[0], b[1:1 + e_bits], b[1 + e_bits:]
    E = int(e, 2) - bias; frac = 1 + int(mnt, 2) / 2 ** m_bits
    fp_rows.append(dict(format=name, bits=b, sign=s, exponent=e, mantissa=mnt, value=val, error=val - 0.1))
    print(f'{name:9s} sign {s} | exponent {e} (= {int(e, 2)} - {bias} = {E}) | mantissa {mnt}')
    print(f'          value = {frac:.10f} x 2^{E} = {val:.12f}   error {val - 0.1:+.3e} ({abs(val - 0.1) / 0.1:.4%})')
RESULTS['point_one'] = fp_rows
json.dump(RESULTS, open('results.json', 'w'), indent=2, default=str)
print('saved results.json')
""")

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'session10_training_loop.ipynb')
print('wrote session10_training_loop.ipynb with', len(cells), 'cells')
