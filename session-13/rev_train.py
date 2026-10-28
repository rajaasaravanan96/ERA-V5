"""Session 13: a ~20M-parameter LM trained for 50M tokens, three ways.

  standard     ordinary pre-norm residual stack, activations stored for backward
  reversible   the same blocks in a reversible stack; activations rebuilt during backward
  rev_maxbatch reversible, at the largest batch that fits the same memory budget

Reversible variants (pilot):
  midpoint  p[l+1] = p[l-1] + 2h * f_l(p[l])            (Gal et al. 2025, the lesson's rule)
  revnet    two streams: x1 += f_l(x2); x2 += g_l(x1)    (additive coupling, RevNet / Reformer)

Usage:
  python rev_train.py prep                         tokenize WikiText-103 once (data/*.npy)
  python rev_train.py pilot                        short runs of each variant, checks reconstruction error
  python rev_train.py probe                        peak memory against batch size, finds the max batch
  python rev_train.py train <name> <variant> <bs>  one 50M-token run (name: standard | reversible | rev_maxbatch)
"""
import json, math, os, sys, threading, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
D, LAYERS, HEADS, CTX = 384, 9, 6, 256
TOKENS = 50_000_000
PEAK_LR, WARM = 1e-3, 200
H = 0.5                      # midpoint step size: 2h = 1 keeps the update the size of an ordinary residual
os.makedirs('logs', exist_ok=True); os.makedirs('data', exist_ok=True)


# --------------------------------------------------------------------------- data
def prep():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('../session-2/tokenizer.json')
    for split, files, limit in [('train', ['train-00000-of-00002', 'train-00001-of-00002'], 175_000_000),
                                ('val', ['validation-00000-of-00001'], None)]:
        text = ''.join(''.join(pq.read_table(f'data_raw/{f}.parquet').to_pandas()['text']) for f in files)
        if limit: text = text[:limit]
        chunks = [text[i:i + 1_000_000] for i in range(0, len(text), 1_000_000)]
        ids = np.concatenate([np.array(e.ids, dtype=np.uint16) for e in tok.encode_batch(chunks)])
        np.save(f'data/{split}.npy', ids)
        print(split, f'{len(text):,} chars -> {len(ids):,} tokens')


def load():
    return np.load('data/train.npy'), np.load('data/val.npy')


# --------------------------------------------------------------------------- blocks
class RMSNorm(nn.Module):
    def __init__(s, d):
        super().__init__(); s.w = nn.Parameter(torch.ones(d))

    def forward(s, x):
        return s.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.n = RMSNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)

    def forward(s, x):
        B, T, Dm = x.shape
        q, k, v = s.qkv(s.n(x)).view(B, T, 3, s.h, Dm // s.h).permute(2, 0, 3, 1, 4)
        return s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, Dm))


class MLP(nn.Module):
    def __init__(s, d):
        super().__init__(); s.n = RMSNorm(d); hid = int(8 * d / 3) // 64 * 64
        s.w1 = nn.Linear(d, hid, bias=False); s.w3 = nn.Linear(d, hid, bias=False); s.w2 = nn.Linear(hid, d, bias=False)

    def forward(s, x):
        y = s.n(x); return s.w2(F.silu(s.w1(y)) * s.w3(y))


class Block(nn.Module):
    """f(p) = attn(p) + mlp(p + attn(p)): the whole transformer block as one function of its input."""
    def __init__(s, d, h):
        super().__init__(); s.a = Attn(d, h); s.m = MLP(d)

    def forward(s, x):
        a = s.a(x); return a + s.m(x + a)


# --------------------------------------------------------------------------- reversible stacks
class MidpointStack(torch.autograd.Function):
    """p[l+1] = p[l-1] + 2h f_l(p[l]),  p[-1] = p[0] = x.  Stores only the last two states."""

    @staticmethod
    def forward(ctx, x, blocks, h, *params):
        with torch.no_grad():
            prev, cur = x, x
            for b in blocks:
                prev, cur = cur, prev + 2 * h * b(cur)
        ctx.blocks, ctx.h = blocks, h
        ctx.ac = torch.is_autocast_enabled(DEVICE)
        ctx.save_for_backward(prev.detach(), cur.detach())
        return cur

    @staticmethod
    def backward(ctx, g_out):
        blocks, h = ctx.blocks, ctx.h
        prev, cur = ctx.saved_tensors                    # p[L-1], p[L]
        g_cur, g_prev = g_out, torch.zeros_like(g_out)   # dL/dp[L], dL/dp[L-1]
        params = [p for b in blocks for p in b.parameters()]
        grads = {p: torch.zeros_like(p) for p in params}
        for b in reversed(blocks):
            # p[l+1] = cur, p[l] = prev; rebuild p[l-1] = p[l+1] - 2h f(p[l])
            with torch.enable_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=ctx.ac):
                pl = prev.detach().requires_grad_()
                fy = b(pl)
            with torch.no_grad():
                before = cur - 2 * h * fy.detach()
            bp = list(b.parameters())
            gs = torch.autograd.grad(fy, [pl] + bp, grad_outputs=2 * h * g_cur, allow_unused=True)
            for p, g in zip(bp, gs[1:]):
                if g is not None: grads[p] += g
            # dL/dp[l-1] gets g_cur (identity path); dL/dp[l] gets g_prev + vjp
            g_cur, g_prev = g_prev + gs[0], g_cur
            cur, prev = prev, before
        MidpointStack.recon = (cur - prev).abs().max().item()   # p[0] and p[-1] should both equal x
        g_x = g_cur + g_prev
        return (g_x, None, None) + tuple(grads[p] for p in params)


class RevNetStack(torch.autograd.Function):
    """two streams: y1 = x1 + F(x2); y2 = x2 + G(y1). Exactly invertible, stores only the output."""

    @staticmethod
    def forward(ctx, x1, x2, fs, gs, *params):
        with torch.no_grad():
            for f, g in zip(fs, gs):
                x1 = x1 + f(x2); x2 = x2 + g(x1)
        ctx.fs, ctx.gs = fs, gs
        ctx.ac = torch.is_autocast_enabled(DEVICE)
        ctx.save_for_backward(x1.detach(), x2.detach())
        return x1, x2

    @staticmethod
    def backward(ctx, g1, g2):
        y1, y2 = ctx.saved_tensors
        fs, gs = ctx.fs, ctx.gs
        params = [p for m in list(fs) + list(gs) for p in m.parameters()]
        grads = {p: torch.zeros_like(p) for p in params}
        for f, g in zip(reversed(list(fs)), reversed(list(gs))):
            with torch.enable_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=ctx.ac):
                y1r = y1.detach().requires_grad_(); gy = g(y1r)
            with torch.no_grad(): x2 = y2 - gy.detach()
            gp = list(g.parameters())
            out = torch.autograd.grad(gy, [y1r] + gp, grad_outputs=g2, allow_unused=True)
            for p, gg in zip(gp, out[1:]):
                if gg is not None: grads[p] += gg
            g1 = g1 + out[0]
            with torch.enable_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=ctx.ac):
                x2r = x2.detach().requires_grad_(); fx = f(x2r)
            with torch.no_grad(): x1 = y1 - fx.detach()
            fp = list(f.parameters())
            out = torch.autograd.grad(fx, [x2r] + fp, grad_outputs=g1, allow_unused=True)
            for p, gg in zip(fp, out[1:]):
                if gg is not None: grads[p] += gg
            g2 = g2 + out[0]
            y1, y2 = x1, x2
        return (g1, g2, None, None) + tuple(grads[p] for p in params)


class LM(nn.Module):
    def __init__(s, vocab, variant='standard'):
        super().__init__()
        s.variant = variant
        s.emb = nn.Embedding(vocab, D); s.pos = nn.Embedding(CTX, D)
        if variant == 'revnet':                         # half the layers each for F and G, same parameter count
            s.fs = nn.ModuleList(Attn(D, HEADS) for _ in range(LAYERS))
            s.gs = nn.ModuleList(MLP(D) for _ in range(LAYERS))
        else:
            s.blocks = nn.ModuleList(Block(D, HEADS) for _ in range(LAYERS))
        s.norm = RMSNorm(D); s.head = nn.Linear(D, vocab, bias=False); s.head.weight = s.emb.weight
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        for n_, p in s.named_parameters():            # scaled init of the residual outputs
            if n_.endswith(('o.weight', 'w2.weight')): nn.init.normal_(p, std=0.02 / math.sqrt(2 * LAYERS))

    def forward(s, t):
        x = s.emb(t) + s.pos(torch.arange(t.shape[1], device=t.device))
        if s.variant == 'standard':
            for b in s.blocks: x = x + b(x)
        elif s.variant == 'midpoint':
            x = MidpointStack.apply(x, s.blocks, H, *[p for b in s.blocks for p in b.parameters()])
        elif s.variant == 'revnet':
            x1, x2 = RevNetStack.apply(x, x, s.fs, s.gs, *[p for m in list(s.fs) + list(s.gs) for p in m.parameters()])
            x = (x1 + x2) / 2
        return s.head(s.norm(x))


# --------------------------------------------------------------------------- helpers
def sync():
    if DEVICE == 'mps': torch.mps.synchronize()
    elif DEVICE == 'cuda': torch.cuda.synchronize()


def allocated():
    if DEVICE == 'mps': return torch.mps.current_allocated_memory()
    if DEVICE == 'cuda': return torch.cuda.memory_allocated()
    return 0


class PeakWatch:
    """samples the allocator every 0.2 ms in a background thread"""
    def __enter__(s):
        sync(); s.base = allocated(); s.peak = s.base; s.stop = False
        s.th = threading.Thread(target=s._run, daemon=True); s.th.start(); return s

    def _run(s):
        while not s.stop:
            s.peak = max(s.peak, allocated()); time.sleep(0.0002)

    def __exit__(s, *a):
        sync(); s.peak = max(s.peak, allocated()); s.stop = True; s.th.join()


def batch(arr, bs, rng):
    ix = rng.integers(0, len(arr) - CTX - 1, bs)
    x = np.stack([arr[i:i + CTX + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(DEVICE)


def step_fn(model, opt, xb):
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        logits = model(xb[:, :-1])
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss, gn


@torch.no_grad()
def evaluate(model, val, n=20, bs=16):
    model.eval(); rng = np.random.default_rng(123); tot = 0.0
    for _ in range(n):
        xb = batch(val, bs, rng)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            lg = model(xb[:, :-1])
        tot += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), xb[:, 1:].reshape(-1)).item()
    model.train(); return tot / n


def n_params(m): return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- commands
def pilot():
    """same seed, same data, 400 steps of batch 32 for each variant; plus a gradient check against autograd."""
    train, val = load(); V = 10000; res = {}
    # 1. gradient equivalence: reversible backward vs autograd through the same forward rule
    torch.manual_seed(0)
    m = LM(V, 'midpoint').to(DEVICE).float()
    xb = batch(train, 4, np.random.default_rng(0))
    loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1)); loss.backward()
    g_rev = torch.cat([p.grad.flatten() for p in m.blocks.parameters()]); m.zero_grad()
    x = m.emb(xb[:, :-1]) + m.pos(torch.arange(CTX, device=DEVICE)); prev, cur = x, x
    for b in m.blocks: prev, cur = cur, prev + 2 * H * b(cur)
    loss2 = F.cross_entropy(m.head(m.norm(cur)).reshape(-1, V), xb[:, 1:].reshape(-1)); loss2.backward()
    g_ref = torch.cat([p.grad.flatten() for p in m.blocks.parameters()])
    res['grad_check_rel_err_fp32'] = ((g_rev - g_ref).norm() / g_ref.norm()).item()
    res['reconstruction_err_fp32'] = MidpointStack.recon
    print(f"gradient check (fp32): reversible vs autograd rel. error {res['grad_check_rel_err_fp32']:.2e}, "
          f"reconstruction max |p[0]-p[-1]| {MidpointStack.recon:.2e}")
    # 2. short training pilots
    for variant in ['standard', 'midpoint', 'revnet']:
        torch.manual_seed(0); rng = np.random.default_rng(1)
        m = LM(V, variant).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1)
        t0 = time.time(); losses = []; recon = []
        for s in range(400):
            for g in opt.param_groups: g['lr'] = PEAK_LR * min(1, (s + 1) / WARM)
            loss, gn = step_fn(m, opt, batch(train, 32, rng)); losses.append(loss.item())
            if variant == 'midpoint': recon.append(MidpointStack.recon)
        sync(); dt = time.time() - t0
        res[variant] = dict(params=n_params(m), final_train=sum(losses[-20:]) / 20, val=evaluate(m, val),
                            tok_s=400 * 32 * CTX / dt, recon_max=max(recon) if recon else None)
        print(f"{variant:9s} params {n_params(m):,}  train {res[variant]['final_train']:.3f}  val {res[variant]['val']:.3f}  "
              f"{res[variant]['tok_s']:,.0f} tok/s" + (f"  recon err (bf16) max {max(recon):.2e}" if recon else ''))
    json.dump(res, open('logs/pilot.json', 'w'), indent=1)


def peak_for(variant, bs):
    torch.manual_seed(0); V = 10000
    m = LM(V, variant).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    rng = np.random.default_rng(0); xb = batch(load()[0][:5_000_000], bs, rng)
    step_fn(m, opt, xb)                                   # creates optimizer state
    import gc; gc.collect()
    if DEVICE == 'mps': torch.mps.empty_cache()
    with PeakWatch() as pw:
        step_fn(m, opt, xb)
    state = pw.base
    del m, opt, xb; gc.collect()
    if DEVICE == 'mps': torch.mps.empty_cache()
    return pw.peak, state


BUDGET = 4 * 2 ** 30      # an emulated 4 GiB card: weights + grads + Adam + activations must fit


def probe():
    """The budget is what the standard model needs at the batch we can run (32). Then: how large a batch does the
    reversible model fit inside the same budget?"""
    rows = []
    budget, _ = peak_for('standard', 32)
    for variant in ['standard', 'midpoint']:
        best = None
        for bs in [8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512]:
            pk, st = peak_for(variant, bs)
            fits = pk <= budget * 1.02
            rows.append(dict(variant=variant, bs=bs, peak_GiB=pk / 2 ** 30, state_GiB=st / 2 ** 30, fits=fits))
            print(f'{variant:9s} batch {bs:4d}: peak {pk / 2 ** 30:6.2f} GiB (state {st / 2 ** 30:.2f})  {"fits" if fits else "over budget"}', flush=True)
            if fits: best = bs
            if pk > budget * 1.6: break
        rows.append(dict(variant=variant, max_batch_in_budget=best))
    json.dump(dict(budget_GiB=budget / 2 ** 30, rule='peak of the standard model at batch 32', rows=rows,
                   max_batch={v: r['max_batch_in_budget'] for r in rows for v in [r.get('variant')] if 'max_batch_in_budget' in r}),
              open('logs/memory_probe.json', 'w'), indent=1)


def train_run(name, variant, bs):
    train, val = load(); V = 10000
    torch.manual_seed(0); rng = np.random.default_rng(1)
    m = LM(V, variant).to(DEVICE)
    lr = PEAK_LR * math.sqrt(bs / 32)                     # sqrt rule from the batch-32 baseline
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps = TOKENS // (bs * CTX); warm = max(50, WARM * 32 // bs)
    log = open(f'logs/{name}.jsonl', 'w'); txt = open(f'logs/{name}.log', 'w')
    def say(s): print(s, flush=True); txt.write(s + '\n'); txt.flush()
    say(f'{name}: variant {variant}, batch {bs}, {steps} steps x {bs * CTX} tokens = {steps * bs * CTX:,} tokens, '
        f'peak lr {lr:.2e}, params {n_params(m):,}, device {DEVICE}')
    peak = 0; t0 = time.time(); seen = 0; t_last = t0; seen_last = 0
    for s in range(steps):
        cur = lr * min(1, (s + 1) / warm) if s < warm else lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
        for g in opt.param_groups: g['lr'] = cur
        xb = batch(train, bs, rng)
        if s in (5, 50, steps // 2):
            with PeakWatch() as pw: loss, gn = step_fn(m, opt, xb)
            peak = max(peak, pw.peak)
        else:
            loss, gn = step_fn(m, opt, xb)
        seen += bs * CTX
        rec = dict(step=s, tokens=seen, loss=loss.item(), grad_norm=gn.item(), lr=cur)
        if s % 200 == 0 or s == steps - 1:
            sync(); now = time.time()
            rec['val'] = evaluate(m, val); rec['tok_s'] = (seen - seen_last) / (now - t_last)
            t_last, seen_last = time.time(), seen
            if variant == 'midpoint': rec['recon_err'] = MidpointStack.recon
            say(f'[{name}] step {s:5d}/{steps}  tokens {seen / 1e6:6.1f}M  loss {loss.item():.4f}  val {rec["val"]:.4f}  '
                f'grad {gn.item():.3f}  {rec["tok_s"]:,.0f} tok/s  peak {peak / 2 ** 30:.2f} GiB  {now - t0:6.0f}s'
                + (f'  recon err {MidpointStack.recon:.1e}' if variant == 'midpoint' else ''))
        log.write(json.dumps(rec) + '\n')
    sync(); wall = time.time() - t0
    out = dict(name=name, variant=variant, batch=bs, steps=steps, tokens=seen, params=n_params(m),
               final_train_loss=sum(json.loads(l)['loss'] for l in open(f'logs/{name}.jsonl').readlines()[-50:]) / 50,
               final_val_loss=evaluate(m, val, n=50), tok_s=seen / wall, wall_s=wall, peak_GiB=peak / 2 ** 30)
    json.dump(out, open(f'logs/{name}_summary.json', 'w'), indent=1)
    say(json.dumps(out))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prep': prep()
    elif cmd == 'pilot': pilot()
    elif cmd == 'probe': probe()
    elif cmd == 'train': train_run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
