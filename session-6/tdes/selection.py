"""Mixture schedule compiler and OPUS-style selection with protected floors.

The compiler turns curriculum stages into exact per-step lane quotas (blended over a warmup band at every stage
boundary, protected floors enforced). OPUS scores every candidate sequence by how well its gradient on the output
head aligns with a proxy direction computed on a validation batch, then accepts, rejects or defers it. Protected
lanes are rescued up to their floor even when their score is low (a protected-floor override).
"""
import hashlib, json, math
import numpy as np
import torch, torch.nn.functional as F

LANES = ['web', 'indic', 'code', 'reasoning']

STAGES = [
    dict(name='foundation', start=0, end=40, weights=dict(web=0.45, indic=0.20, code=0.15, reasoning=0.20)),
    dict(name='code_reasoning', start=40, end=80, weights=dict(web=0.25, indic=0.20, code=0.30, reasoning=0.25)),
]
FLOORS = dict(indic=0.125, reasoning=0.125)      # at least 1 of every 8 sequences each, whatever OPUS thinks
WARMUP = 10                                       # steps over which a stage boundary is blended
BATCH = 8


def compile_schedule(stages=STAGES, floors=FLOORS, batch=BATCH, warmup=WARMUP):
    steps = []
    for st_i, st in enumerate(stages):
        for step in range(st['start'], st['end']):
            w = dict(st['weights'])
            if st_i > 0 and step < st['start'] + warmup:          # blend from the previous stage
                a = (step - st['start'] + 1) / (warmup + 1); prev = stages[st_i - 1]['weights']
                w = {l: (1 - a) * prev[l] + a * w[l] for l in LANES}
            raw = {l: w[l] * batch for l in LANES}
            q = {l: int(math.floor(raw[l])) for l in LANES}
            for l in sorted(LANES, key=lambda l: (raw[l] - q[l], l), reverse=True)[:batch - sum(q.values())]:
                q[l] += 1                                         # largest remainder
            for l, f in floors.items():                           # protected floors
                need = math.ceil(f * batch)
                while q[l] < need:
                    donor = max((x for x in LANES if x not in floors or q[x] > math.ceil(floors[x] * batch)), key=lambda x: q[x])
                    q[donor] -= 1; q[l] += 1
            steps.append(dict(step=step, stage=st['name'], weights={l: round(w[l], 4) for l in LANES}, quota=q))
    blob = json.dumps(dict(stages=stages, floors=floors, batch=batch, warmup=warmup, steps=steps), sort_keys=True)
    return steps, hashlib.sha256(blob.encode()).hexdigest()


class Opus:
    """Scores candidates against a proxy direction; logs every decision."""
    REJECT_BELOW = 0.0     # negative alignment with the proxy direction
    MAX_DEFERRALS = 2

    def __init__(self, model, proxy_batch, refresh_every=10):
        self.model, self.proxy_batch, self.refresh_every = model, proxy_batch, refresh_every
        self.version, self.direction, self.made_at = 0, None, -1

    def head_grad(self, arrays):
        """gradient of the mean loss w.r.t. the output head, computed analytically from final hidden states."""
        m = self.model
        with torch.no_grad():
            x = torch.tensor(np.stack([a['input_ids'] for a in arrays]))
            y = torch.tensor(np.stack([a['labels'] for a in arrays]))
            h = m.hidden(x, torch.tensor(np.stack([a['doc_index'] for a in arrays])),
                         torch.tensor(np.stack([a['position'] for a in arrays])))
            logits = m.head(h)
            keep = y != -100
            p = torch.softmax(logits[keep], -1)
            p[torch.arange(p.shape[0]), y[keep]] -= 1
            return (p.T @ h[keep]) / max(1, int(keep.sum()))

    def maybe_refresh(self, step, model_step):
        if self.direction is None or step - self.made_at >= self.refresh_every:
            self.direction = self.head_grad(self.proxy_batch).flatten()
            self.version += 1; self.made_at = step; self.direction_step = model_step

    def score(self, arrays):
        g = self.head_grad([arrays]).flatten()
        return float(F.cosine_similarity(g, self.direction, dim=0))
