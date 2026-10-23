"""Small pre-norm transformer and data helpers shared by the Session 10 notebook."""
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer


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
        s.w1 = nn.Linear(d, hid, bias=False); s.w3 = nn.Linear(d, hid, bias=False)
        s.w2 = nn.Linear(hid, d, bias=False)

    def forward(s, x, trace=None):
        B, T, D = x.shape
        qkv = s.qkv(s.n1(x))
        q, k, v = qkv.view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        y = s.n2(x)
        gate, up = s.w1(y), s.w3(y)
        out = x + s.w2(F.silu(gate) * up)
        if trace is not None:
            trace += [('block.qkv', qkv, '(batch, time, 3*d_model): query, key and value side by side'),
                      ('block.q', q, '(batch, heads, time, head_dim): queries split into heads'),
                      ('block.attn_out', a, '(batch, heads, time, head_dim): each head\'s weighted mix of values'),
                      ('block.ffn_gate', gate, '(batch, time, ffn_hidden): SwiGLU gate, 8/3 * d_model wide'),
                      ('block.out', out, '(batch, time, d_model): residual stream after the block')]
        return out


class TinyLM(nn.Module):
    def __init__(s, vocab, d=192, n_layers=4, n_heads=6, ctx=256):
        super().__init__()
        s.emb = nn.Embedding(vocab, d); s.pos = nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        s.norm = RMSNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
        s.head.weight = s.emb.weight          # tied
        nn.init.normal_(s.emb.weight, std=0.02); nn.init.normal_(s.pos.weight, std=0.02)

    def forward(s, tokens, trace=None):
        B, T = tokens.shape
        x = s.emb(tokens) + s.pos(torch.arange(T, device=tokens.device))
        if trace is not None:
            trace.append(('x = emb + pos', x, '(batch, time, d_model): token vector plus position vector'))
        for i, b in enumerate(s.blocks):
            b_trace = [] if (trace is not None and i == 0) else None
            x = b(x, b_trace)
            if b_trace: trace += b_trace
        x = s.norm(x)
        logits = s.head(x)
        if trace is not None:
            trace.append(('final norm', x, '(batch, time, d_model): normalised residual stream'))
            trace.append(('logits', logits, '(batch, time, vocab): a score for every token in the vocabulary'))
        return logits


def load_corpus(tok_path='../session-2/tokenizer.json', corpus_dir='../session-2/corpus'):
    tok = Tokenizer.from_file(tok_path)
    text = open(f'{corpus_dir}/en.faithful.txt', encoding='utf-8').read() + '\n\n' + \
        open(f'{corpus_dir}/hi.faithful.txt', encoding='utf-8').read()
    docs = [d.strip() for d in text.split('\n\n') if len(d.strip()) > 200]
    ids = [tok.encode(d).ids for d in docs]
    return tok, ids
