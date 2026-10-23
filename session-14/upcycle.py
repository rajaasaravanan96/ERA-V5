"""Session 14: train a dense model, convert its feed-forward layers into a mixture of experts, keep training.

Three runs continue from the same dense checkpoint, with the same data order and the same number of steps:
  dense      the dense model keeps training (the control)
  copy       sparse upcycling: every expert is a full copy of the dense FFN, only the router is new
  drop       drop-upcycling: every expert is a copy with half of its neurons re-drawn (r = 0.5)

Data: the Session 4 cleaned Opus 4.7 reasoning traces (session-4/data_clean/clean.jsonl.gz), my Session 2 tokenizer.
Run:  python upcycle.py      (about 30 minutes on an Apple M5)
Writes logs/*.jsonl, logs/train.log, results.json and figures/*.png
"""
import copy, json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
D, LAYERS, HEADS, FFN, CTX = 256, 4, 4, 768, 128
N_EXP, TOP_K = 8, 2
DENSE_STEPS, CONT_STEPS, BS = 1500, 1000, 32
PEAK_LR = 2e-3
os.makedirs('logs', exist_ok=True); os.makedirs('figures', exist_ok=True)
LOG = open('logs/train.log', 'w')


def log(msg):
    print(msg); LOG.write(msg + '\n'); LOG.flush()


# ----------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(s, d):
        super().__init__(); s.w = nn.Parameter(torch.ones(d))

    def forward(s, x):
        return s.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class SwiGLU(nn.Module):
    def __init__(s, d, hid):
        super().__init__()
        s.w1 = nn.Linear(d, hid, bias=False); s.w3 = nn.Linear(d, hid, bias=False); s.w2 = nn.Linear(hid, d, bias=False)

    def forward(s, x):
        return s.w2(F.silu(s.w1(x)) * s.w3(x))


class MoE(nn.Module):
    """Top-k routed experts. Gate weights are the softmax over the chosen k scores, so they sum to 1.
    Balancing is auxiliary-loss-free (DeepSeek-V3 style): a per-expert bias is added to the scores
    only for choosing experts, and nudged after every step towards an even load."""

    def __init__(s, d, hid, n_exp, k):
        super().__init__()
        s.k, s.n = k, n_exp
        s.experts = nn.ModuleList(SwiGLU(d, hid) for _ in range(n_exp))
        s.router = nn.Linear(d, n_exp, bias=False)
        s.register_buffer('bias', torch.zeros(n_exp))
        s.last_load = None

    def forward(s, x):
        B, T, Dm = x.shape
        flat = x.reshape(-1, Dm)
        scores = s.router(flat)                                     # (tokens, experts)
        _, idx = torch.topk(scores + s.bias, s.k, dim=-1)           # choose with the bias
        gates = torch.softmax(scores.gather(1, idx), dim=-1)        # weight without it
        out = torch.zeros_like(flat)
        load = torch.zeros(s.n, device=x.device)
        for e in range(s.n):
            hit = (idx == e)
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            load[e] = rows.numel()
            if rows.numel() == 0:
                continue
            g = (gates * hit).sum(-1)[rows, None]
            out.index_add_(0, rows, g * s.experts[e](flat[rows]))
        s.last_load = load
        return out.view(B, T, Dm)


class Block(nn.Module):
    def __init__(s, d, h, ffn):
        super().__init__()
        s.h = h; s.n1 = RMSNorm(d); s.n2 = RMSNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.ffn = ffn

    def forward(s, x):
        B, T, Dm = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, Dm // s.h).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, Dm))
        return x + s.ffn(s.n2(x))


class LM(nn.Module):
    def __init__(s, vocab):
        super().__init__()
        s.emb = nn.Embedding(vocab, D); s.pos = nn.Embedding(CTX, D)
        s.blocks = nn.ModuleList(Block(D, HEADS, SwiGLU(D, FFN)) for _ in range(LAYERS))
        s.norm = RMSNorm(D); s.head = nn.Linear(D, vocab, bias=False); s.head.weight = s.emb.weight
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)

    def forward(s, t):
        x = s.emb(t) + s.pos(torch.arange(t.shape[1], device=t.device))
        for b in s.blocks: x = b(x)
        return s.head(s.norm(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def active_params(m):
    tot = n_params(m)
    for b in m.blocks:
        if isinstance(b.ffn, MoE):
            per = n_params(b.ffn.experts[0]); tot -= per * (b.ffn.n - b.ffn.k)
    return tot


# ----------------------------------------------------------------- upcycling
def upcycle(dense, mode, seed=0):
    g = torch.Generator().manual_seed(seed)
    moe = copy.deepcopy(dense)
    for b in moe.blocks:
        src = b.ffn
        layer = MoE(D, FFN, N_EXP, TOP_K).to(DEVICE)
        for e, ex in enumerate(layer.experts):
            ex.load_state_dict(src.state_dict())
            if mode == 'drop':                                  # redraw half of the neurons of this expert
                with torch.no_grad():
                    pick = torch.randperm(FFN, generator=g)[: FFN // 2].to(DEVICE)
                    std1, std3, std2 = src.w1.weight.std(), src.w3.weight.std(), src.w2.weight.std()
                    ex.w1.weight[pick] = torch.randn(len(pick), D, generator=g).to(DEVICE) * std1
                    ex.w3.weight[pick] = torch.randn(len(pick), D, generator=g).to(DEVICE) * std3
                    ex.w2.weight[:, pick] = torch.randn(D, len(pick), generator=g).to(DEVICE) * std2
        nn.init.normal_(layer.router.weight, std=0.02 / math.sqrt(D))  # tiny, so routing starts near-uniform
        b.ffn = layer
    return moe


# ----------------------------------------------------------------- data
tok = Tokenizer.from_file('../session-2/tokenizer.json'); V = tok.get_vocab_size()
# data: the reasoning traces cleaned in Session 4 (prompt + reasoning + answer of every row)
import gzip
rows = [json.loads(l) for l in gzip.open('../session-4/data_clean/clean.jsonl.gz', 'rt')]
docs = ['\n'.join(m['content'] for m in r['messages']) + '\n' + r['reasoning'] for r in rows]
stream = torch.tensor([t for e in tok.encode_batch(docs) for t in e.ids])
cut = int(0.95 * len(stream)); train_ids, val = stream[:cut], stream[cut:]
gval = torch.Generator().manual_seed(1)
VAL = [torch.stack([val[i:i + CTX + 1] for i in torch.randint(0, len(val) - CTX - 1, (32,), generator=gval)]).to(DEVICE)
       for _ in range(10)]


def batch(gen):
    ix = torch.randint(0, len(train_ids) - CTX - 1, (BS,), generator=gen)
    return torch.stack([train_ids[i:i + CTX + 1] for i in ix]).to(DEVICE)


@torch.no_grad()
def val_loss(m):
    m.eval()
    l = sum(F.cross_entropy(m(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1)).item() for b in VAL) / len(VAL)
    m.train(); return l


def moe_stats(m):
    """per layer: MaxVio (busiest load / mean - 1), dead experts, mean pairwise cosine between expert w1 matrices."""
    out = []
    for b in m.blocks:
        if not isinstance(b.ffn, MoE): continue
        ld = b.ffn.last_load; mean = ld.mean()
        W = torch.stack([ex.w1.weight.flatten() for ex in b.ffn.experts]); W = F.normalize(W, dim=1)
        sim = (W @ W.T)[~torch.eye(N_EXP, dtype=torch.bool, device=W.device)].mean().item()
        out.append(dict(maxvio=(ld.max() / mean - 1).item(), dead=int((ld == 0).sum()), expert_cos=sim,
                        load=[int(x) for x in ld.tolist()]))
    return out


def train(m, steps, name, start_step=0, warm=0, total=None, gen_seed=0):
    total = total or steps
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(gen_seed)
    f = open(f'logs/{name}.jsonl', 'w'); t0 = time.time()
    for s in range(steps):
        step = start_step + s
        if warm and s < warm:
            lr = PEAK_LR * (s + 1) / warm
        else:
            lr = PEAK_LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / total)))
        for g in opt.param_groups: g['lr'] = lr
        xb = batch(gen)
        loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item(); opt.step()
        with torch.no_grad():                        # aux-loss-free balancing: push bias towards even load
            for b in m.blocks:
                if isinstance(b.ffn, MoE):
                    ld = b.ffn.last_load
                    b.ffn.bias += 1e-3 * torch.sign(ld.mean() - ld)
        rec = dict(step=step, loss=loss.item(), grad_norm=gn, lr=lr)
        if s % 50 == 0 or s == steps - 1:
            rec['val_loss'] = val_loss(m)
            if any(isinstance(b.ffn, MoE) for b in m.blocks): rec['moe'] = moe_stats(m)
            extra = ''
            if 'moe' in rec:
                extra = (f"  MaxVio {max(x['maxvio'] for x in rec['moe']):.2f}  dead {sum(x['dead'] for x in rec['moe'])}"
                         f"  expert cos {sum(x['expert_cos'] for x in rec['moe']) / len(rec['moe']):.3f}")
            log(f'[{name:5s}] step {step:5d}  loss {loss.item():.4f}  val {rec["val_loss"]:.4f}  '
                f'grad {gn:.3f}  lr {lr:.2e}  {time.time() - t0:5.0f}s{extra}')
        f.write(json.dumps(rec) + '\n')
    f.close()
    return m


if __name__ == '__main__':
    log(f'device {DEVICE} | vocab {V} | {len(stream):,} tokens | d {D} layers {LAYERS} ffn {FFN} | '
        f'{N_EXP} experts top-{TOP_K}')
    torch.manual_seed(0)
    dense = LM(V).to(DEVICE)
    log(f'dense model: {n_params(dense):,} parameters')
    train(dense, DENSE_STEPS, 'dense_pretrain', warm=100, total=DENSE_STEPS, gen_seed=0)
    torch.save(dense.state_dict(), 'logs/dense_checkpoint.pt')
    base_val = val_loss(dense)
    log(f'dense checkpoint after {DENSE_STEPS} steps: val loss {base_val:.4f}')

    results = dict(dense_steps=DENSE_STEPS, cont_steps=CONT_STEPS, dense_val_at_conversion=base_val,
                   dense_params=n_params(dense), runs={})
    for mode in ('dense', 'copy', 'drop'):
        if mode == 'dense':
            m = copy.deepcopy(dense)
        else:
            m = upcycle(dense, mode)
        v0 = val_loss(m)
        log(f'--- {mode}: total {n_params(m):,} params, active {active_params(m):,}; '
            f'val loss right after conversion {v0:.4f} (dense was {base_val:.4f})')
        train(m, CONT_STEPS, mode, start_step=DENSE_STEPS, warm=50, total=CONT_STEPS, gen_seed=1)
        v1 = val_loss(m)
        results['runs'][mode] = dict(total_params=n_params(m), active_params=active_params(m),
                                     val_at_conversion=v0, val_final=v1,
                                     moe_final=moe_stats(m) if mode != 'dense' else None)
        log(f'--- {mode}: final val loss {v1:.4f} (change since conversion {v1 - v0:+.4f})')
    json.dump(results, open('results.json', 'w'), indent=1)

    # ------------------------------------------------------------- figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rd = lambda n: [json.loads(l) for l in open(f'logs/{n}.jsonl')]
    def dark(ax, t, xl, yl):
        ax.set_facecolor('#0F151C'); ax.figure.set_facecolor('#0A0E13')
        for s_ in ax.spines.values(): s_.set_color('#35485E')
        ax.tick_params(colors='white'); ax.grid(color='#25323F')
        ax.set_title(t, color='white'); ax.set_xlabel(xl, color='white'); ax.set_ylabel(yl, color='white')
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    pre = [r for r in rd('dense_pretrain') if 'val_loss' in r]
    axes[0].plot([r['step'] for r in pre], [r['val_loss'] for r in pre], color='#4A6076', lw=2, label='dense pre-training')
    for n, c, ls, lab in [('dense', '#3FA7FF', '-', 'dense, continued'), ('copy', '#FF9F35', '--', 'MoE, copy-upcycled'),
                          ('drop', '#26D7C8', ':', 'MoE, drop-upcycled (r=0.5)')]:
        rr = [r for r in rd(n) if 'val_loss' in r]
        axes[0].plot([r['step'] for r in rr], [r['val_loss'] for r in rr], color=c, lw=2.2, ls=ls, label=lab)
    axes[0].axvline(DENSE_STEPS, color='#F5D33B', lw=1.5)
    dark(axes[0], 'Validation loss (gold line: dense model converted to MoE)', 'step', 'validation loss')
    axes[0].legend(facecolor='#141C25', labelcolor='white', edgecolor='#35485E')
    for n, c, ls in [('copy', '#FF9F35', '--'), ('drop', '#26D7C8', ':')]:
        rr = [r for r in rd(n) if 'moe' in r]
        axes[1].plot([r['step'] for r in rr], [sum(x['expert_cos'] for x in r['moe']) / LAYERS for r in rr],
                     color=c, lw=2.2, ls=ls, label=f'{n}: mean cosine between experts')
    dark(axes[1], 'How different the experts become (1.0 = identical copies)', 'step', 'cosine similarity')
    axes[1].legend(facecolor='#141C25', labelcolor='white', edgecolor='#35485E')
    fig.tight_layout(); fig.savefig('figures/upcycling.png', dpi=130)
    log('done')
