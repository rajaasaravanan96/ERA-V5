"""A 32-GPU data-parallel / ZeRO-1 / ZeRO-2 / ZeRO-3 simulator.

Every virtual GPU is a `Rank` with its own storage dict. Nothing is shared between ranks except through the
collectives below (all-reduce, reduce-scatter, all-gather), and every collective counts the bytes each rank sends
using the ring algorithm's cost. The training maths is real: the four strategies train the same model on the same
data, and the simulator checks that they end with bit-identical weights.

Precision follows the 16-bytes-per-weight recipe from the lesson:
    bf16 weights (2) + bf16 gradients (2) + fp32 master weights (4) + fp32 Adam m (4) + fp32 Adam v (4)
"""
import math, time
from dataclasses import dataclass, field
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call

N_GPU = 32


# ------------------------------------------------------------------ the demo model (byte-level LM)
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h; s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.up = nn.Linear(d, 4 * d, bias=False); s.down = nn.Linear(4 * d, d, bias=False)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + s.down(F.gelu(s.up(s.n2(x))))


class ByteLM(nn.Module):
    def __init__(s, d=128, n_layers=4, n_heads=4, ctx=64):
        super().__init__()
        s.emb = nn.Embedding(256, d); s.pos = nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        s.norm = nn.LayerNorm(d); s.head = nn.Linear(d, 256, bias=False)

    def forward(s, t):
        x = s.emb(t) + s.pos(torch.arange(t.shape[1]))
        for b in s.blocks: x = b(x)
        return s.head(s.norm(x))


# ------------------------------------------------------------------ flat parameter layout
class Layout:
    """All weights live in one flat vector (like FSDP's flat parameter), padded to a multiple of N_GPU.
    `units` are the groups ZeRO-3 gathers one at a time: embeddings, each block, the final norm + head."""

    def __init__(s, model, n_gpu=N_GPU):
        s.names, s.shapes, s.offsets = [], [], []
        off = 0
        for n, p in model.named_parameters():
            s.names.append(n); s.shapes.append(p.shape); s.offsets.append(off); off += p.numel()
        s.P = off
        s.Ppad = math.ceil(off / n_gpu) * n_gpu
        s.shard = s.Ppad // n_gpu
        s.units = {}
        for n, o, sh in zip(s.names, s.offsets, s.shapes):
            u = n.split('.')[0] if not n.startswith('blocks') else '.'.join(n.split('.')[:2])
            u = 'embeddings' if u in ('emb', 'pos') else ('head' if u in ('norm', 'head') else u)
            s.units[u] = s.units.get(u, 0) + math.prod(sh)

    def flatten(s, model):
        v = torch.zeros(s.Ppad)
        for n, p in model.named_parameters():
            i = s.names.index(n); v[s.offsets[i]:s.offsets[i] + p.numel()] = p.detach().flatten()
        return v

    def views(s, flat):
        return {n: flat[o:o + math.prod(sh)].view(sh) for n, o, sh in zip(s.names, s.offsets, s.shapes)}


# ------------------------------------------------------------------ ranks and collectives
@dataclass
class Rank:
    id: int
    store: dict = field(default_factory=dict)     # persistent tensors this GPU owns
    sent_bytes: int = 0
    flops: float = 0.0
    peak_transient: int = 0

    def persistent_bytes(s):
        return sum(t.numel() * t.element_size() for t in s.store.values())


class Cluster:
    def __init__(s, n=N_GPU):
        s.ranks = [Rank(i) for i in range(n)]; s.n = n

    def _charge(s, nbytes_full):
        # ring algorithm: every rank sends (n-1)/n of the buffer in a reduce-scatter or an all-gather
        for r in s.ranks: r.sent_bytes += nbytes_full * (s.n - 1) // s.n

    def reduce_scatter(s, full_per_rank, out_dtype):
        """input: one full-length tensor per rank; output: rank i gets the SUM of everyone's slice i."""
        n, L = s.n, full_per_rank[0].numel(); k = L // n
        s._charge(L * full_per_rank[0].element_size())
        acc = torch.zeros(L, dtype=torch.float32)
        for t in full_per_rank:                     # fixed rank order, so every strategy sums identically
            acc += t.float()
        return [acc[i * k:(i + 1) * k].to(out_dtype).clone() for i in range(n)]

    def all_gather(s, shard_per_rank):
        """input: one slice per rank; output: every rank gets the full concatenated tensor."""
        full = torch.cat(shard_per_rank)
        s._charge(full.numel() * full.element_size())
        return [full.clone() for _ in range(s.n)]

    def all_reduce(s, full_per_rank, out_dtype):
        """= reduce-scatter followed by all-gather (and costs exactly the two together)."""
        return s.all_gather(s.reduce_scatter(full_per_rank, out_dtype))


# ------------------------------------------------------------------ one training strategy
ADAM = dict(lr=3e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.0)


def adam_update(master, m, v, g, step):
    """In-place AdamW on fp32 tensors (works on a whole vector or on one shard)."""
    a = ADAM
    m.mul_(a['b1']).add_(g, alpha=1 - a['b1'])
    v.mul_(a['b2']).addcmul_(g, g, value=1 - a['b2'])
    mh = m / (1 - a['b1'] ** step); vh = v / (1 - a['b2'] ** step)
    master.add_(-a['lr'] * (mh / (vh.sqrt() + a['eps']) + a['wd'] * master))


class Strategy:
    """stage = 'single' (one GPU, the reference), 'dp', 'zero1', 'zero2' or 'zero3'."""

    def __init__(s, stage, model, layout, n=N_GPU):
        s.stage, s.model, s.L = stage, model, layout
        s.n = 1 if stage == 'single' else n
        s.cl = Cluster(s.n)
        flat = layout.flatten(model)
        for r in s.cl.ranks:
            sl = slice(r.id * layout.shard, (r.id + 1) * layout.shard) if s.n > 1 else slice(0, layout.Ppad)
            full_or_shard = (lambda t: t.clone()) if stage in ('single', 'dp') else (lambda t: t[sl].clone())
            # weights: replicated except in ZeRO-3
            r.store['weights_bf16'] = flat.to(torch.bfloat16).clone() if stage != 'zero3' else flat[sl].to(torch.bfloat16).clone()
            # gradients: replicated in DP and ZeRO-1, sharded in ZeRO-2/3
            r.store['grads_bf16'] = torch.zeros(layout.Ppad, dtype=torch.bfloat16) if stage in ('single', 'dp', 'zero1') \
                else torch.zeros(layout.shard, dtype=torch.bfloat16)
            # optimizer state: replicated in DP, sharded in every ZeRO stage
            r.store['master_fp32'] = full_or_shard(flat)
            r.store['adam_m_fp32'] = torch.zeros_like(r.store['master_fp32'])
            r.store['adam_v_fp32'] = torch.zeros_like(r.store['master_fp32'])
        s.step_no = 0
        biggest_unit = max(layout.units.values())
        # transient buffers the schedule needs on top of the persistent store (per rank, bytes)
        s.transient = {'single': 0, 'dp': 0, 'zero1': 0,
                       'zero2': 2 * biggest_unit,                    # one unit's full bf16 grads before its reduce-scatter
                       'zero3': 2 * biggest_unit * 2 + 2 * biggest_unit}  # gathered unit + prefetch of the next, + grad bucket
        for r in s.cl.ranks: r.peak_transient = s.transient[stage]

    def _local_grad(s, weights_bf16_full, batch):
        flat = weights_bf16_full.float().requires_grad_()
        logits = functional_call(s.model, s.L.views(flat), (batch[:, :-1],))
        loss = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        loss.backward()
        return flat.grad.detach(), loss.item()

    def step(s, micro_batches, flops_fwd_bwd):
        """micro_batches: list of N_GPU batches (the global batch); the 'single' reference runs all of them."""
        s.step_no += 1; L, cl = s.L, s.cl
        # ---- 1. full weights for the forward/backward on every rank
        if s.stage == 'zero3':
            # each unit is all-gathered before its forward and again before its backward: 2 gathers of the weights
            full_w = cl.all_gather([r.store['weights_bf16'] for r in cl.ranks])
            for r in cl.ranks: r.sent_bytes += L.Ppad * 2 * (cl.n - 1) // cl.n   # the backward re-gather
        else:
            full_w = [r.store['weights_bf16'] for r in cl.ranks]
        # ---- 2. forward + backward, each rank on its own micro-batch
        if s.stage == 'single':
            grads, losses = [], []
            for mb in micro_batches:
                g, l = s._local_grad(full_w[0], mb); grads.append(g); losses.append(l)
            g_sum = torch.zeros(L.Ppad)
            for g in grads: g_sum += g.to(torch.bfloat16).float()    # same rounding as the multi-GPU paths
            local = [g_sum]
            cl.ranks[0].flops += flops_fwd_bwd * len(micro_batches)
        else:
            local, losses = [], []
            for r, mb in zip(cl.ranks, micro_batches):
                g, l = s._local_grad(full_w[r.id], mb)
                local.append(g.to(torch.bfloat16)); losses.append(l); r.flops += flops_fwd_bwd
        # ---- 3. gradient communication + 4. optimizer step
        n_micro = len(micro_batches)
        if s.stage == 'single':
            r = cl.ranks[0]
            g = (local[0] / n_micro); r.store['grads_bf16'] = g.to(torch.bfloat16)
            adam_update(r.store['master_fp32'], r.store['adam_m_fp32'], r.store['adam_v_fp32'], r.store['grads_bf16'].float(), s.step_no)
            r.flops += 12 * L.Ppad
            r.store['weights_bf16'] = r.store['master_fp32'].to(torch.bfloat16)
        elif s.stage == 'dp':
            summed = cl.all_reduce(local, torch.bfloat16)              # every rank gets the full summed gradient
            for r in cl.ranks:
                r.store['grads_bf16'] = (summed[r.id] / n_micro).to(torch.bfloat16)
                adam_update(r.store['master_fp32'], r.store['adam_m_fp32'], r.store['adam_v_fp32'], r.store['grads_bf16'].float(), s.step_no)
                r.flops += 12 * L.Ppad                               # every rank updates all P weights
                r.store['weights_bf16'] = r.store['master_fp32'].to(torch.bfloat16)
        else:
            shards = cl.reduce_scatter(local, torch.bfloat16)          # rank i gets the summed slice i only
            for r in cl.ranks:
                g_shard = (shards[r.id] / n_micro).to(torch.bfloat16)
                if s.stage == 'zero1':                                # keeps a full grad buffer, fills its slice
                    sl = slice(r.id * L.shard, (r.id + 1) * L.shard); r.store['grads_bf16'][sl] = g_shard
                else:
                    r.store['grads_bf16'] = g_shard
                adam_update(r.store['master_fp32'], r.store['adam_m_fp32'], r.store['adam_v_fp32'], g_shard.float(), s.step_no)
                r.flops += 12 * L.shard                              # each rank updates only its slice
            new_shards = [r.store['master_fp32'].to(torch.bfloat16) for r in cl.ranks]
            if s.stage == 'zero3':
                for r in cl.ranks: r.store['weights_bf16'] = new_shards[r.id]   # stays sharded
            else:
                gathered = cl.all_gather(new_shards)                  # everyone needs the full new weights
                for r in cl.ranks: r.store['weights_bf16'] = gathered[r.id]
        return sum(losses) / len(losses)

    def full_weights(s):
        if s.stage == 'zero3':
            return torch.cat([r.store['weights_bf16'] for r in s.cl.ranks])
        return s.cl.ranks[0].store['weights_bf16']

    def report(s):
        r0 = s.cl.ranks[0]
        return dict(stage=s.stage, persistent_bytes=r0.persistent_bytes(), transient_bytes=r0.peak_transient,
                    peak_bytes=r0.persistent_bytes() + r0.peak_transient,
                    breakdown={k: v.numel() * v.element_size() for k, v in r0.store.items()},
                    sent_bytes_per_step=r0.sent_bytes / max(s.step_no, 1), flops_per_step=r0.flops / max(s.step_no, 1))


def memory_ladder(n_params, gpus, transient_bytes=0):
    """GiB per GPU for the lesson's 16-bytes-per-weight recipe (training state only)."""
    W, G, O = 2, 2, 12
    rows = {}
    for st in ('dp', 'zero1', 'zero2', 'zero3'):
        rows[st] = []
        for n in gpus:
            b = {'dp': W + G + O, 'zero1': W + G + O / n, 'zero2': W + (G + O) / n, 'zero3': (W + G + O) / n}[st]
            rows[st].append(n_params * b / 2 ** 30)
    return rows
