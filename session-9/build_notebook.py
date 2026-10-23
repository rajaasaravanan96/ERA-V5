"""Builds session9_loss_harness.ipynb from the cells below.

Run:  python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace session9_loss_harness.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Session 9 — a loss harness that tells the truth

Starting point from the lesson:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(logits[:, :-1].reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
```

This notebook makes those lines **correct and observable**, step by step.
Tokenizer: my own Session 2 BPE tokenizer (10,000 tokens, English + Hindi).
Corpus: the Session 2 English and Hindi Wikipedia text.
""")

code("""
import json, math, os, sys, time, threading
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

torch.manual_seed(0)
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('torch', torch.__version__, '| device', DEVICE)

TOK_PATH = '../session-2/tokenizer.json'
tok = Tokenizer.from_file(TOK_PATH)
V = tok.get_vocab_size()
PAD_ID = V          # padding gets its own id, one past the real vocabulary
V_MODEL = V + 1     # rows in the embedding / head (real tokens + PAD)
print('real vocabulary V =', V, '| model rows =', V_MODEL, '| PAD id =', PAD_ID)

def load(path):
    return open(path, encoding='utf-8').read()
en = load('../session-2/corpus/en.faithful.txt')
hi = load('../session-2/corpus/hi.faithful.txt')
# documents = paragraphs; keep the ones with some substance
docs = [d.strip() for d in (en + '\\n\\n' + hi).split('\\n\\n') if len(d.strip()) > 200]
ids_per_doc = [tok.encode(d).ids for d in docs]
n_tokens = sum(map(len, ids_per_doc))
print(f'{len(docs)} documents, {n_tokens:,} tokens')
""")

md("""
## The model

A small pre-norm transformer: RMSNorm, causal self-attention, SwiGLU feed-forward.
`model(tokens)` returns hidden states; the head is a separate `nn.Linear`, exactly as in the lesson.
""")

code("""
class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x):
        return s.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps)

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h; s.n1 = RMSNorm(d); s.n2 = RMSNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3)
        s.w1 = nn.Linear(d, hid, bias=False); s.w3 = nn.Linear(d, hid, bias=False); s.w2 = nn.Linear(hid, d, bias=False)
    def forward(s, x, attn_mask=None):
        B, T, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if attn_mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        y = s.n2(x)
        return x + s.w2(F.silu(s.w1(y)) * s.w3(y))

class TinyLM(nn.Module):
    \"\"\"Returns hidden states, shape (batch, time, d_model). The head lives outside.\"\"\"
    def __init__(s, vocab, d=192, n_layers=4, n_heads=6, ctx=256):
        super().__init__()
        s.emb = nn.Embedding(vocab, d); s.pos = nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        s.norm = RMSNorm(d)
        nn.init.normal_(s.emb.weight, std=0.02); nn.init.normal_(s.pos.weight, std=0.02)
    def forward(s, tokens, attn_mask=None, pos_ids=None):
        B, T = tokens.shape
        if pos_ids is None:
            pos_ids = torch.arange(T, device=tokens.device).expand(B, T)
        x = s.emb(tokens) + s.pos(pos_ids)
        for b in s.blocks:
            x = b(x, attn_mask)
        return s.norm(x)

CFG = dict(d=192, n_layers=4, n_heads=6, ctx=256)
model = TinyLM(V_MODEL, **CFG).to(DEVICE)
output_head = nn.Linear(CFG['d'], V_MODEL, bias=False).to(DEVICE)
nn.init.normal_(output_head.weight, std=0.02)
print(CFG)
""")

md("""
## 1. Every tensor shape, and what each dimension means
""")

code("""
B, T = 4, 16
tokens = torch.tensor([ids_per_doc[i][:T] for i in range(B)], device=DEVICE)
hidden = model(tokens)
logits = output_head(hidden)
flat_logits = logits[:, :-1].reshape(-1, V_MODEL)
flat_targets = tokens[:, 1:].reshape(-1)
loss = F.cross_entropy(flat_logits, flat_targets)

rows = [
    ('tokens',       tokens,       '(batch, time): token id at each position of each sequence'),
    ('hidden',       hidden,       '(batch, time, d_model): one 192-number vector per position'),
    ('logits',       logits,       '(batch, time, vocab): one score per vocabulary entry per position'),
    ('logits[:, :-1]', logits[:, :-1], '(batch, time-1, vocab): drop the last position, it has no next token'),
    ('tokens[:, 1:]',  tokens[:, 1:],  '(batch, time-1): the targets, each position\\'s NEXT token'),
    ('flat_logits',  flat_logits,  '(batch*(time-1), vocab): every prediction in one list'),
    ('flat_targets', flat_targets, '(batch*(time-1),): the matching correct ids'),
    ('loss',         loss,         '(): one scalar, the mean surprise over all predictions'),
]
for name, t, meaning in rows:
    print(f'{name:16s} {str(tuple(t.shape)):22s} {meaning}')
""")

md("""
## 2. Verify the shift with strings, not ids

Position *t* of the input must sit beside token *t+1* as its target.
If the shift went the wrong way, the target would equal the input (copying), and the loss would still fall beautifully.
""")

code("""
seq = tokens[0].tolist()
inp, tgt = seq[:-1], seq[1:]
print(f\"{'pos':>3}  {'input (what the model sees)':30s} -> target (what it must predict)\")
for t_, (a, b) in enumerate(zip(inp, tgt)):
    print(f'{t_:>3}  {tok.id_to_token(a)!r:30s} -> {tok.id_to_token(b)!r}')
assert all(seq[i + 1] == tgt[i] for i in range(len(tgt))), 'shift is wrong'
assert tgt != inp, 'targets equal inputs: the shift is missing'
print('\\n[PASS] every target is the token one step to the right of its input')
""")

md("""
## 3. Mask padding, and watch the count of contributing tokens change

Short documents are padded to the batch length with `PAD_ID`. Targets that are padding get `ignore_index=-100`,
so they add nothing to the sum and are not counted in the mean.
""")

code("""
T = 32
short_docs = [ids_per_doc[i][:n] for i, n in zip(range(4), [32, 20, 11, 27])]
batch = torch.full((4, T), PAD_ID, dtype=torch.long)
for i, d in enumerate(short_docs):
    batch[i, :len(d)] = torch.tensor(d)
batch = batch.to(DEVICE)

targets = batch[:, 1:].clone()
logits = output_head(model(batch))[:, :-1]

unmasked = F.cross_entropy(logits.reshape(-1, V_MODEL), targets.reshape(-1), reduction='none')
masked_t = targets.masked_fill(targets == PAD_ID, -100)
masked = F.cross_entropy(logits.reshape(-1, V_MODEL), masked_t.reshape(-1), ignore_index=-100, reduction='none')

n_all = targets.numel()
n_real = int((masked_t != -100).sum())
print('document lengths      :', [len(d) for d in short_docs])
print('targets without mask  :', n_all)
print('targets with pad mask :', n_real, f'(dropped {n_all - n_real} padding targets)')
print(f'mean loss, no mask    : {unmasked.mean().item():.4f}')
print(f'mean loss, pad masked : {masked.sum().item() / n_real:.4f}')
RESULTS = {'padding': {'targets_unmasked': n_all, 'targets_masked': n_real}}
""")

md("""
## 4. Pack two documents into one sequence and mask the boundary

Packing puts document A and document B back to back with no padding. Two leaks appear at the seam:

1. **The boundary target.** The last token of A is asked to predict the first token of B. Nothing in A can know how B starts,
   so this target is pure noise. We set it to `-100`.
2. **Cross-document attention.** Tokens of B can attend back into A. A block-diagonal causal mask stops that, and position ids restart at 0 for B.

We compare the loss with everything left in, against the loss with the boundary target masked and attention blocked.
To make the effect visible we first train the model briefly (an untrained model is equally lost everywhere).
""")

code("""
# --- a short training run so the model has learned something ---
CTX = 256
stream = torch.tensor([t for d in ids_per_doc for t in d])
split = int(0.95 * len(stream))
train_stream, val_stream = stream[:split], stream[split:]

def get_batch(src, bs=16, T=CTX):
    ix = torch.randint(0, len(src) - T - 1, (bs,))
    return torch.stack([src[i:i + T + 1] for i in ix]).to(DEVICE)

params = list(model.parameters()) + list(output_head.parameters())
opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.1)
for step in range(300):
    xb = get_batch(train_stream)
    lg = output_head(model(xb[:, :-1]))
    l = F.cross_entropy(lg.reshape(-1, V_MODEL), xb[:, 1:].reshape(-1))
    opt.zero_grad(); l.backward(); opt.step()
    if step % 100 == 0 or step == 299:
        print(f'warm-up step {step:3d}  loss {l.item():.3f}')
""")

code("""
docA, docB = ids_per_doc[10][:40], ids_per_doc[200][:40]
packed = torch.tensor(docA + docB, device=DEVICE)[None]       # (1, 80)
LA, LP = len(docA), len(docA) + len(docB)
inputs, targets = packed[:, :-1], packed[:, 1:].clone()
seam = LA - 1   # input position whose target is B's first token
print(f'seam: input {tok.id_to_token(docA[-1])!r} (end of A) -> target {tok.id_to_token(docB[0])!r} (start of B)')

model.eval()
with torch.no_grad():
    # (a) naive packing: causal mask across the whole row, boundary target kept
    per_tok_naive = F.cross_entropy(output_head(model(inputs))[0], targets[0], reduction='none')

    # (b) correct packing: block-diagonal causal attention, restarted positions, boundary target masked
    Tin = inputs.shape[1]
    doc_id = torch.tensor([0] * LA + [1] * (Tin - LA), device=DEVICE)
    causal = torch.tril(torch.ones(Tin, Tin, dtype=torch.bool, device=DEVICE))
    same_doc = doc_id[:, None] == doc_id[None, :]
    attn_mask = (causal & same_doc)[None, None]
    pos_ids = torch.cat([torch.arange(LA), torch.arange(Tin - LA)]).to(DEVICE)[None]
    masked_targets = targets.clone(); masked_targets[0, seam] = -100
    per_tok_doc = F.cross_entropy(output_head(model(inputs, attn_mask, pos_ids))[0], masked_targets[0],
                                  ignore_index=-100, reduction='none')
model.train()

naive = per_tok_naive.mean().item()
correct = per_tok_doc.sum().item() / (Tin - 1)
print(f'boundary target loss (naive)          : {per_tok_naive[seam].item():.3f}  <- noise the model can never learn')
print(f'mean loss, naive packing ({Tin} targets) : {naive:.4f}')
print(f'mean loss, boundary masked ({Tin-1} targets): {correct:.4f}')
print(f'mean loss of B only, naive vs isolated : {per_tok_naive[LA:].mean().item():.4f} vs {per_tok_doc[LA:].mean().item():.4f}')
RESULTS['packing'] = {'loss_naive': naive, 'loss_masked': correct, 'boundary_token_loss': per_tok_naive[seam].item()}
""")

md("""
**What the difference means.** The boundary target is usually one of the highest losses in the row: the model is being
graded on guessing the first word of an unrelated document. Masking it removes a target that can only teach noise.
Blocking attention changes B's losses too: with the naive mask, B's tokens read A's context as if it were related.
""")

md("""
## 5. Perplexity — an untrained model must sit near the vocabulary size

A model that knows nothing spreads probability evenly: p = 1/V for every token, so loss = ln V and perplexity = e^loss = V.
If a fresh model shows a perplexity far from V, the init or the loss is wrong.
""")

code("""
fresh = TinyLM(V_MODEL, **CFG).to(DEVICE)
fresh_head = nn.Linear(CFG['d'], V_MODEL, bias=False).to(DEVICE)
nn.init.normal_(fresh_head.weight, std=0.02)

@torch.no_grad()
def eval_loss(m, h, src, n=20):
    m.eval(); tot = 0.0
    for _ in range(n):
        xb = get_batch(src)
        tot += F.cross_entropy(h(m(xb[:, :-1])).reshape(-1, V_MODEL), xb[:, 1:].reshape(-1)).item()
    m.train(); return tot / n

l0 = eval_loss(fresh, fresh_head, val_stream)
l1 = eval_loss(model, output_head, val_stream)
print(f'ln(V) = ln({V_MODEL}) = {math.log(V_MODEL):.4f}  ->  ideal untrained perplexity = {V_MODEL}')
print(f'untrained model : loss {l0:.4f}  perplexity {math.exp(l0):,.0f}')
print(f'after 300 steps : loss {l1:.4f}  perplexity {math.exp(l1):,.1f}')
assert abs(math.exp(l0) - V_MODEL) / V_MODEL < 0.15, 'untrained perplexity is far from V: find the bug'
print('[PASS] untrained perplexity is within 15% of V')
RESULTS['perplexity'] = {'V': V_MODEL, 'untrained_ppl': math.exp(l0), 'trained_ppl': math.exp(l1)}
""")

md("""
## 6. Tied against untied head parameter counts

Tied: the head reuses the embedding matrix (`head.weight = emb.weight`). Untied: the head has its own V × d matrix.
""")

code("""
def count(m): return sum(p.numel() for p in m.parameters())
body = count(TinyLM(V_MODEL, **CFG))
head = V_MODEL * CFG['d']
untied, tied = body + head, body
print(f'body incl. embedding : {body:,}')
print(f'head (V x d)         : {head:,}  = {V_MODEL} x {CFG[\"d\"]}')
print(f'untied total         : {untied:,}')
print(f'tied total           : {tied:,}')
print(f'tying saves          : {head:,} parameters = {100 * head / untied:.1f}% of the untied model')
RESULTS['tying'] = {'untied': untied, 'tied': tied, 'saved': head}
""")

md("""
## 7. Peak memory: ordinary cross-entropy against a chunked version I wrote

Ordinary: build the full `(N, V)` logits tensor, then softmax, then the loss; autograd keeps it alive for backward,
and backward builds a second `(N, V)` gradient.

Chunked (`ChunkedLinearCE` below): loop over the N rows in chunks of 512. For each chunk compute the logits,
the loss and **the gradient right away**, then throw the chunk's logits away. Only the gradients for the hidden states
(`N × d`) and the head weight (`V × d`) are kept. The full `(N, V)` tensor never exists.

Memory is measured on the device's allocator with a background thread that samples every 0.1 ms
(`torch.mps.current_allocated_memory()` on Apple silicon, `torch.cuda.max_memory_allocated()` on CUDA).
""")

code("""
class ChunkedLinearCE(torch.autograd.Function):
    \"\"\"loss = mean CE(hidden @ W.T, targets), computed chunk by chunk with the grad done in forward.\"\"\"
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        N = hidden.shape[0]
        n_valid = (targets != -100).sum().clamp(min=1)
        g_hidden = torch.zeros_like(hidden)
        g_weight = torch.zeros_like(weight)
        total = hidden.new_zeros(())
        for s in range(0, N, chunk):
            h = hidden[s:s + chunk]; y = targets[s:s + chunk]
            lg = (h @ weight.T).float()                       # (chunk, V) only
            lse = torch.logsumexp(lg, -1)
            valid = y != -100
            yy = y.clamp(min=0)
            total = total + ((lse - lg.gather(1, yy[:, None]).squeeze(1)) * valid).sum()
            p = torch.softmax(lg, -1)                         # dL/dlogits = softmax - onehot
            p[torch.arange(len(y), device=y.device), yy] -= 1
            p = p * valid[:, None] / n_valid
            g_hidden[s:s + chunk] = (p.to(h.dtype) @ weight)
            g_weight += p.to(h.dtype).T @ h
            del lg, p
        ctx.save_for_backward(g_hidden, g_weight)
        return total / n_valid
    @staticmethod
    def backward(ctx, g):
        g_hidden, g_weight = ctx.saved_tensors
        return g_hidden * g, g_weight * g, None, None

def ordinary_ce(hidden, weight, targets):
    return F.cross_entropy(hidden @ weight.T, targets, ignore_index=-100)

# correctness first: same loss and same gradients
h = torch.randn(1024, CFG['d'], device=DEVICE, requires_grad=True)
W = (torch.randn(V_MODEL, CFG['d'], device=DEVICE) * 0.02).requires_grad_()
y = torch.randint(0, V, (1024,), device=DEVICE); y[::7] = -100
la = ordinary_ce(h, W, y); ga = torch.autograd.grad(la, (h, W))
lb = ChunkedLinearCE.apply(h, W, y, 256); gb = torch.autograd.grad(lb, (h, W))
print(f'loss ordinary {la.item():.6f}  chunked {lb.item():.6f}')
print('max |grad diff| hidden', (ga[0] - gb[0]).abs().max().item(), ' weight', (ga[1] - gb[1]).abs().max().item())
assert torch.allclose(la, lb, atol=1e-4) and torch.allclose(ga[0], gb[0], atol=1e-5) and torch.allclose(ga[1], gb[1], atol=1e-5)
print('[PASS] chunked loss and gradients match ordinary cross-entropy')
""")

code("""
def sync():
    if DEVICE == 'mps': torch.mps.synchronize()
    elif DEVICE == 'cuda': torch.cuda.synchronize()
def allocated():
    if DEVICE == 'mps': return torch.mps.current_allocated_memory()
    if DEVICE == 'cuda': return torch.cuda.memory_allocated()
    return 0

def peak_bytes(fn):
    \"\"\"Peak extra device memory while fn() runs (forward + backward).\"\"\"
    import gc; gc.collect()
    if DEVICE == 'mps': torch.mps.empty_cache()
    sync(); base = allocated(); peak = [base]; stop = [False]
    if DEVICE == 'cuda': torch.cuda.reset_peak_memory_stats()
    def watch():
        while not stop[0]:
            peak[0] = max(peak[0], allocated()); time.sleep(0.0001)
    th = threading.Thread(target=watch); th.start()
    fn(); sync(); stop[0] = True; th.join()
    if DEVICE == 'cuda': peak[0] = torch.cuda.max_memory_allocated()
    return peak[0] - base

N = 16 * 512          # batch 16 x sequence 512 positions
hN = torch.randn(N, CFG['d'], device=DEVICE, requires_grad=True)
WN = (torch.randn(V_MODEL, CFG['d'], device=DEVICE) * 0.02).requires_grad_()
yN = torch.randint(0, V, (N,), device=DEVICE)

def run_ordinary():
    l = ordinary_ce(hN, WN, yN); l.backward(); hN.grad = None; WN.grad = None
def run_chunked():
    l = ChunkedLinearCE.apply(hN, WN, yN, 512); l.backward(); hN.grad = None; WN.grad = None

run_ordinary(); run_chunked()   # warm up kernels
p_ord = max(peak_bytes(run_ordinary) for _ in range(3))
p_chk = max(peak_bytes(run_chunked) for _ in range(3))
MB = 2 ** 20
print(f'N = {N} positions, V = {V_MODEL}, d = {CFG[\"d\"]}')
print(f'one (N, V) fp32 tensor would be      {N * V_MODEL * 4 / MB:8.1f} MB')
print(f'peak memory, ordinary cross-entropy  {p_ord / MB:8.1f} MB')
print(f'peak memory, chunked (512 rows)      {p_chk / MB:8.1f} MB')
print(f'ratio ordinary / chunked             {p_ord / p_chk:8.1f} x')
RESULTS['memory'] = {'N': N, 'V': V_MODEL, 'ordinary_MB': p_ord / MB, 'chunked_MB': p_chk / MB, 'ratio': p_ord / p_chk}
""")

md("""
# Part 2 — a second head that predicts token t+2

Head 1 predicts `tokens[t+1]` from position t (the normal target). Head 2 predicts `tokens[t+2]`.
Both share the same trunk. Total loss = loss₁ + loss₂. Both losses are logged separately for every step.
""")

code("""
torch.manual_seed(1)
trunk = TinyLM(V_MODEL, **CFG).to(DEVICE)
head1 = nn.Linear(CFG['d'], V_MODEL, bias=False).to(DEVICE)
head2 = nn.Linear(CFG['d'], V_MODEL, bias=False).to(DEVICE)
for hh in (head1, head2): nn.init.normal_(hh.weight, std=0.02)
opt = torch.optim.AdamW(list(trunk.parameters()) + list(head1.parameters()) + list(head2.parameters()),
                        lr=3e-3, weight_decay=0.1)
STEPS = 800
log = []
for step in range(STEPS):
    lr = 3e-3 * min(1, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
    for g in opt.param_groups: g['lr'] = lr
    xb = get_batch(train_stream, T=CTX + 1)          # need two tokens of look-ahead
    hdn = trunk(xb[:, :-2])                           # positions 0 .. T-1
    l1 = F.cross_entropy(head1(hdn).reshape(-1, V_MODEL), xb[:, 1:-1].reshape(-1))   # t+1
    l2 = F.cross_entropy(head2(hdn).reshape(-1, V_MODEL), xb[:, 2:].reshape(-1))     # t+2
    loss = l1 + l2
    opt.zero_grad(); loss.backward(); opt.step()
    log.append({'step': step, 'loss_t1': l1.item(), 'loss_t2': l2.item(), 'sum': loss.item()})
    if step % 100 == 0 or step == STEPS - 1:
        print(f'step {step:4d}  loss t+1 {l1.item():.3f}  loss t+2 {l2.item():.3f}  sum {loss.item():.3f}')

# shape check for the t+2 target, with strings
xb = get_batch(val_stream, bs=1, T=CTX + 1)[0].tolist()
print('\\nposition -> t+1 target -> t+2 target')
for t_ in range(5):
    print(f'{tok.id_to_token(xb[t_])!r:14s} -> {tok.id_to_token(xb[t_+1])!r:14s} -> {tok.id_to_token(xb[t_+2])!r}')
""")

code("""
@torch.no_grad()
def eval_two(n=30):
    trunk.eval(); a = b = 0.0
    for _ in range(n):
        xb = get_batch(val_stream, T=CTX + 1); hdn = trunk(xb[:, :-2])
        a += F.cross_entropy(head1(hdn).reshape(-1, V_MODEL), xb[:, 1:-1].reshape(-1)).item()
        b += F.cross_entropy(head2(hdn).reshape(-1, V_MODEL), xb[:, 2:].reshape(-1)).item()
    trunk.train(); return a / n, b / n
v1, v2 = eval_two()

def avg(key, lo, hi): return sum(r[key] for r in log[lo:hi]) / (hi - lo)
first1, first2 = avg('loss_t1', 0, 20), avg('loss_t2', 0, 20)
last1, last2 = avg('loss_t1', STEPS - 50, STEPS), avg('loss_t2', STEPS - 50, STEPS)
print(f'                     t+1 head   t+2 head')
print(f'first 20 steps mean  {first1:8.3f}   {first2:8.3f}')
print(f'last 50 steps mean   {last1:8.3f}   {last2:8.3f}')
print(f'drop                 {first1 - last1:8.3f}   {first2 - last2:8.3f}')
print(f'validation           {v1:8.3f}   {v2:8.3f}   sum {v1 + v2:.3f}')
print(f'gap (t+2 minus t+1) at the end: {last2 - last1:.3f} nats')
RESULTS['two_heads'] = {'val_loss_t1': v1, 'val_loss_t2': v2, 'val_sum': v1 + v2,
                        'train_last50_t1': last1, 'train_last50_t2': last2}

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0A0E13'); ax.set_facecolor('#0F151C')
sm = lambda xs, k=20: [sum(xs[max(0, i - k):i + 1]) / len(xs[max(0, i - k):i + 1]) for i in range(len(xs))]
ax.plot(sm([r['loss_t1'] for r in log]), color='#3FA7FF', lw=2, label='head 1: predicts t+1')
ax.plot(sm([r['loss_t2'] for r in log]), color='#FF9F35', lw=2, ls='--', label='head 2: predicts t+2')
for s in ax.spines.values(): s.set_color('#35485E')
ax.tick_params(colors='white'); ax.set_xlabel('step', color='white'); ax.set_ylabel('loss (nats)', color='white')
ax.set_title('Two heads on one trunk', color='white'); ax.legend(facecolor='#141C25', labelcolor='white', edgecolor='#35485E')
ax.grid(color='#25323F'); fig.tight_layout(); fig.savefig('figures/two_heads.png', dpi=130); plt.show()

json.dump(log, open('two_heads_log.json', 'w'))
json.dump(RESULTS, open('results.json', 'w'), indent=2)
print(json.dumps(RESULTS, indent=2))
""")

md("""
**What the second head shows.** Both heads start at the same loss (≈ ln V, they know nothing). Both fall, but the t+2 head
falls more slowly and levels off higher, and the gap stays open. Predicting two steps ahead is a harder question:
the model has not seen token t+1, so it must average over every possible next token before guessing the one after.
Some of that uncertainty is irreducible, which is why the gap does not close with more training.
The t+2 head still learns a lot (its loss drops by several nats), so its gradient pushes the trunk to carry
information about further-ahead structure — the reason DeepSeek-V3-style multi-token prediction is used as an extra training signal.
""")

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'session9_loss_harness.ipynb')
print('wrote session9_loss_harness.ipynb with', len(cells), 'cells')
