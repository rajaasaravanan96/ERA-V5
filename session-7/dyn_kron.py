"""Session 7, problem 3: a Kronecker embedding with no fixed 32-byte window.

Kronecker (the lesson):  code(w) = 1/sqrt(L) * sum_{p < 32} onehot(byte_p) (x) onehot(p)        256 x 32 = 8,192 dims
    - bytes after position 31 are cropped, so words that agree on their first 32 bytes collide forever
    - a 3-byte word still owns a 32-slot grid: most of it is empty, and the projection multiplies all of it

Dynamic Kronecker (mine): code(w) = 1/sqrt(L) * sum_{p < L} onehot(byte_p) (x) phi(p)              256 x P dims
    - phi(p) is a P-dimensional multi-frequency (sin, cos) position code, defined for every p: no crop, no length limit
    - the projection never needs the 8,192-dim vector: W is stored as a (256, P, d) tensor and
          embed(w) = sum_p  phi(p) @ W[byte_p]        cost = L * P * d   (grows with the word, not with the window)
    - same parameter count as Kronecker at P = 32; P is now a free knob for capacity, not a length limit

Run: python dyn_kron.py      writes results.json, figures/*.png, logs/*.jsonl
"""
import json, math, os, re, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
P = 32                         # position features (same parameter budget as the lesson's pos_dim = 32)
os.makedirs('figures', exist_ok=True); os.makedirs('logs', exist_ok=True)
R = {}


# --------------------------------------------------------------------------- the two codecs
def phi(p, P=P, base=64.0):
    """multi-frequency position code; geometric frequencies from 1 to 1/base radians per byte"""
    k = np.arange(P // 2)
    w = base ** (-k / (P // 2 - 1))
    return np.concatenate([np.sin(p * w), np.cos(p * w)])


PHI = np.stack([phi(p) for p in range(1024)])          # table for positions 0..1023 (any length works)


def kron_code(word):
    b = word.encode('utf-8'); L = len(b)
    g = np.zeros((256, 32))
    for p, x in enumerate(b[:32]): g[x, p] = 1.0       # crop at 32 bytes
    return g.ravel() / math.sqrt(max(1, min(L, 32)))


def dyn_code(word):
    b = word.encode('utf-8'); L = len(b)
    g = np.zeros((256, P))
    for p, x in enumerate(b): g[x] += PHI[p]            # every byte, any position
    return g.ravel() / math.sqrt(max(1, L))


# --------------------------------------------------------------------------- vocabulary from real text
def words_from(text):
    return re.findall(r'(?:[^\W_]|[\u0900-\u0DFF\u200c\u200d])+', text)


en = open('../session-2/corpus/en.faithful.txt', encoding='utf-8').read()
hi = open('../session-2/corpus/hi.faithful.txt', encoding='utf-8').read()
TEST_PAIR = ['अंतर्राष्ट्रीयकरण', 'अंतर्राष्ट्रीयता']        # the lesson's colliding pair
wc = Counter(words_from(en) + words_from(hi))
vocab = sorted(wc, key=lambda w: (-wc[w], w)) + [w for w in TEST_PAIR if w not in wc]
script = lambda w: 'Devanagari' if re.search(r'[ऀ-ॿ]', w) else ('Latin' if re.search(r'[A-Za-z]', w) else 'other')
blen = np.array([len(w.encode('utf-8')) for w in vocab])
print(f'vocabulary: {len(vocab):,} distinct words ({sum(script(w) == "Devanagari" for w in vocab):,} Devanagari)')


# --------------------------------------------------------------------------- 1. collisions
def collisions(codefn):
    seen, groups = {}, {}
    for w in vocab:
        key = np.round(codefn(w), 6).tobytes()
        groups.setdefault(key, []).append(w)
    bad = [g for g in groups.values() if len(g) > 1]
    per = Counter(script(w) for g in bad for w in g)
    return dict(words_in_collisions=sum(len(g) for g in bad), groups=len(bad), by_script=dict(per),
                examples=[g[:3] for g in sorted(bad, key=len, reverse=True)[:5]])


R['collisions'] = {'kronecker_32': collisions(kron_code), 'dynamic': collisions(dyn_code)}
for k, v in R['collisions'].items():
    v['probe_accuracy_ceiling'] = 1 - (v['words_in_collisions'] - v['groups']) / len(vocab)
a, b = [torch.tensor(f(w)) for w in TEST_PAIR for f in [kron_code]], [torch.tensor(dyn_code(w)) for w in TEST_PAIR]
R['lesson_pair'] = dict(words=TEST_PAIR, bytes=[len(w.encode()) for w in TEST_PAIR],
                        kronecker_distance=float((a[0] - a[1]).norm()), dynamic_distance=float((b[0] - b[1]).norm()))
over = blen > 32
R['length'] = dict(mean_bytes=float(blen.mean()), mean_bytes_latin=float(blen[[script(w) == 'Latin' for w in vocab]].mean()),
                   mean_bytes_devanagari=float(blen[[script(w) == 'Devanagari' for w in vocab]].mean()),
                   words_over_32_bytes=int(over.sum()), devanagari_over_32=int(sum(o and script(w) == 'Devanagari' for o, w in zip(over, vocab))),
                   longest=int(blen.max()), kron_grid_unused_pct=float(100 * (1 - np.minimum(blen, 32) / 32).mean()))
print(json.dumps({k: R[k] for k in ['collisions', 'lesson_pair', 'length']}, ensure_ascii=False, indent=1))

# separation: smallest distance between two different words' codes, on a sample
def min_dist(codefn, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vocab), size=min(n, len(vocab)), replace=False)
    X = torch.tensor(np.stack([codefn(vocab[i]) for i in idx]), dtype=torch.float32)
    d = torch.cdist(X, X); d.fill_diagonal_(float('inf'))
    return float(d.min())
R['min_pairwise_distance_sample3000'] = {'kronecker_32': min_dist(kron_code), 'dynamic': min_dist(dyn_code)}


# --------------------------------------------------------------------------- the input paths as modules
class DenseEmb(nn.Module):
    def __init__(s, V, d): super().__init__(); s.e = nn.Embedding(V, d)
    def forward(s, ids): return s.e(ids)


class KronEmb(nn.Module):
    """Kronecker input path without ever building the 8,192-dim code.
    W has shape (256, P, d): one P x d block per byte value (= Linear(256 * P, d), the same parameter count).
      kronecker: positions are one-hot, so byte b at position p (< 32) reads row W[b, p]; later bytes are cropped.
      dynamic:   byte b at position p reads phi(p) @ W[b]; every position exists, so nothing is cropped.
    Each step builds a (256, Lmax, d) table T[b, p] once and every word gathers its own L rows: cost grows with the
    word's length, not with a fixed 32-slot window."""
    def __init__(s, words, d, dynamic=False):
        super().__init__()
        s.dynamic = dynamic
        enc = [w.encode('utf-8') for w in words]
        L = np.array([len(e) for e in enc]); Lmax = int(L.max()) if dynamic else 32
        B = np.zeros((len(words), Lmax), dtype=np.int64); M = np.zeros((len(words), Lmax), dtype=np.float32)
        for i, e in enumerate(enc):
            e = e[:Lmax]; B[i, :len(e)] = list(e); M[i, :len(e)] = 1.0
        scale = 1.0 / np.sqrt(np.maximum(1, np.minimum(L, Lmax)))
        s.register_buffer('B', torch.tensor(B)); s.register_buffer('M', torch.tensor((M * scale[:, None]).astype(np.float32)))
        s.register_buffer('phi', torch.tensor(PHI[:Lmax], dtype=torch.float32))           # (Lmax, P)
        s.W = nn.Parameter(torch.randn(256, P, d) * (1.0 / math.sqrt(P)))
    def forward(s, ids):
        if s.dynamic:
            T = torch.einsum('lp,bpd->bld', s.phi, s.W)                                  # (256, Lmax, d)
        else:
            T = s.W                                                                       # (256, 32, d): one-hot positions
        b, m = s.B[ids], s.M[ids]                                                         # (..., Lmax)
        pos = torch.arange(b.shape[-1], device=b.device).expand_as(b)
        return (T[b, pos] * m[..., None]).sum(-2)


def n_params(m): return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- 2. probe: can the word be recovered from its embedding?
def probe(kind, d=128, steps=4000, V=None):
    words = vocab if V is None else vocab[:V]
    torch.manual_seed(0)
    emb = (DenseEmb(len(words), d) if kind == 'dense' else KronEmb(words, d, dynamic=(kind == 'dynamic'))).to(DEVICE)
    head = nn.Linear(d, len(words)).to(DEVICE)
    opt = torch.optim.AdamW(list(emb.parameters()) + list(head.parameters()), lr=3e-3)
    for s in range(steps):
        ids = torch.randint(0, len(words), (1024,), device=DEVICE)
        loss = F.cross_entropy(head(emb(ids)), ids); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = []
        for i in range(0, len(words), 4096):
            ids = torch.arange(i, min(i + 4096, len(words)), device=DEVICE)
            acc.append((head(emb(ids)).argmax(-1) == ids).float())
        acc = torch.cat(acc)
    long_mask = torch.tensor(blen[:len(words)] > 32, device=DEVICE)
    return dict(input_params=n_params(emb), accuracy=float(acc.mean()),
                accuracy_words_over_32_bytes=float(acc[long_mask].mean()) if long_mask.any() else None)


# --------------------------------------------------------------------------- 3. language model with each input path
def lm_run(kind, steps=3000, d=192, layers=4, ctx=64, bs=64):
    """Word-level LM on WikiText-103 (a 60M-character slice); the input path is the only thing that changes."""
    import pyarrow.parquet as pq
    text = ''.join(pq.read_table('../session-13/data_raw/train-00000-of-00002.parquet').to_pandas()['text'])[:60_000_000]
    toks = words_from(text)
    cnt = Counter(toks)
    lm_vocab = ['<unk>'] + sorted([w for w, c in cnt.items() if c >= 5], key=lambda w: (-cnt[w], w))
    widx = {w: i for i, w in enumerate(lm_vocab)}
    ids = torch.tensor([widx.get(w, 0) for w in toks])
    cut = int(0.95 * len(ids)); tr, va = ids[:cut], ids[cut:]
    torch.manual_seed(0)
    V = len(lm_vocab)
    print(f'LM data: {len(toks):,} words, vocabulary {V:,} (min count 5), longest word {max(len(w.encode()) for w in lm_vocab)} bytes, '
          f'{sum(len(w.encode()) > 32 for w in lm_vocab)} words over 32 bytes', flush=True)
    class LM(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = DenseEmb(V, d) if kind == 'dense' else KronEmb(lm_vocab, d, dynamic=(kind == 'dynamic'))
            s.pos = nn.Embedding(ctx, d)
            layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            s.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); s.norm = nn.LayerNorm(d)
            s.head = nn.Linear(d, V, bias=False)
        def forward(s, x):
            h = s.inp(x) + s.pos(torch.arange(x.shape[1], device=x.device))
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
            return s.head(s.norm(s.tf(h, mask=mask, is_causal=True)))
    m = LM().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
    g = torch.Generator().manual_seed(1)
    def get(src, n, gen):
        ix = torch.randint(0, len(src) - ctx - 1, (n,), generator=gen)
        return torch.stack([src[i:i + ctx + 1] for i in ix]).to(DEVICE)
    log = open(f'logs/lm_{kind}.jsonl', 'w'); t0 = time.time()
    for s in range(steps):
        lr = 1e-3 * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for gp in opt.param_groups: gp['lr'] = lr
        xb = get(tr, bs, g)
        loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        rec = dict(step=s, loss=loss.item())
        if s % 250 == 0 or s == steps - 1:
            m.eval()
            with torch.no_grad():
                gv = torch.Generator().manual_seed(9)
                vl = [F.cross_entropy(m((vb := get(va, bs, gv))[:, :-1]).reshape(-1, V), vb[:, 1:].reshape(-1)).item() for _ in range(20)]
            m.train(); rec['val'] = sum(vl) / len(vl)
            print(f'[{kind:8s}] step {s:5d} loss {loss.item():.3f} val {rec["val"]:.3f}  {time.time() - t0:.0f}s', flush=True)
        log.write(json.dumps(rec) + '\n')
    return dict(input_params=n_params(m.inp), total_params=n_params(m), final_val=rec['val'], final_train=loss.item(),
                vocab=V, seconds=time.time() - t0)


if __name__ == '__main__':
    R['probe'] = {}
    for kind in ['dense', 'kronecker_32', 'dynamic']:
        k = 'kron' if kind == 'kronecker_32' else kind
        R['probe'][kind] = probe(k)
        print('probe', kind, R['probe'][kind])
    R['lm'] = {}
    for kind in ['dense', 'kronecker_32', 'dynamic']:
        k = 'kron' if kind == 'kronecker_32' else kind
        R['lm'][kind] = lm_run(k)
        print('lm', kind, R['lm'][kind])
    json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    def dark(ax, t, xl, yl):
        ax.set_facecolor('#0F151C'); ax.figure.set_facecolor('#0A0E13')
        for s_ in ax.spines.values(): s_.set_color('#35485E')
        ax.tick_params(colors='white'); ax.grid(color='#25323F')
        ax.set_title(t, color='white'); ax.set_xlabel(xl, color='white'); ax.set_ylabel(yl, color='white')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].hist(blen, bins=range(1, int(blen.max()) + 2), color='#3FA7FF')
    axes[0].axvline(32.5, color='#F5D33B', lw=2); axes[0].text(34, axes[0].get_ylim()[1] * 0.8, 'cropped beyond 32 bytes', color='white')
    dark(axes[0], 'UTF-8 length of every word in the vocabulary', 'bytes', 'words')
    for kind, c, ls in [('dense', '#4A6076', '-'), ('kronecker_32', '#FF9F35', '--'), ('dynamic', '#26D7C8', '-')]:
        k = 'kron' if kind == 'kronecker_32' else kind
        rr = [json.loads(l) for l in open(f'logs/lm_{k}.jsonl') if 'val' in l]
        axes[1].plot([r['step'] for r in rr], [r['val'] for r in rr], color=c, ls=ls, lw=2.2, label=kind)
    dark(axes[1], 'Word-level LM, validation loss by input path', 'step', 'validation loss')
    axes[1].legend(facecolor='#141C25', labelcolor='white', edgecolor='#35485E')
    fig.tight_layout(); fig.savefig('figures/results.png', dpi=130)
    print('done')
