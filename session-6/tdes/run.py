"""The training run: model, batch assembly through OPUS, ledgers, checkpoints, crash, resume, replay and fork."""
import copy, hashlib, json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from .packing import LaneStream, materialise, batch_hash, loss_mask_hash, POLICY
from .selection import LANES, FLOORS, Opus, compile_schedule, STAGES

DATALOADER_VERSION = 'tdes-loader-1.0'
ATTN_POLICY = 'block-diagonal causal (no attention across documents); positions reset per segment'
torch.use_deterministic_algorithms(True)


class SimulatedCrash(RuntimeError):
    pass


# --------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, mask):
        B, T, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, T, D))
        return x + s.ff(s.n2(x))


class TinyLM(nn.Module):
    def __init__(s, vocab, d=128, layers=2, heads=4, ctx=128):
        super().__init__()
        s.emb, s.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        s.norm, s.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    @staticmethod
    def mask(doc_index):
        T = doc_index.shape[1]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        same = (doc_index[:, :, None] == doc_index[:, None, :]) & (doc_index[:, :, None] >= 0)
        m = causal & same
        m = m | torch.eye(T, dtype=torch.bool)          # padding rows attend to themselves (no NaN); they carry no loss
        return m[:, None]

    def hidden(s, x, doc_index, position):
        h = s.emb(x) + s.pos(position)
        m = s.mask(doc_index)
        for b in s.blocks: h = b(h, m)
        return s.norm(h)

    def forward(s, x, doc_index, position): return s.head(s.hidden(x, doc_index, position))


# --------------------------------------------------------------------------- the run
class Run:
    def __init__(s, out, store, tok, firewall, proxy_arrays, run_id='v5-demo', branch='main', seed=0,
                 seq_len=128, steps=80, ckpt_every=10, stages=STAGES, log=print):
        s.out, s.store, s.tok, s.fw, s.log = out, store, tok, firewall, log
        s.run_id, s.branch, s.seq_len, s.total_steps, s.ckpt_every = run_id, branch, seq_len, steps, ckpt_every
        s.schedule, s.schedule_hash = compile_schedule(stages)
        torch.manual_seed(seed)
        s.model = TinyLM(tok.vocab_size, ctx=seq_len)
        s.opt = torch.optim.AdamW(s.model.parameters(), lr=3e-3, weight_decay=0.1)
        s.streams = {l: LaneStream(l, store, seq_len, firewall) for l in LANES}
        s.deferred = []                       # [{cand_id, lane, segs, deferrals}]
        s.step, s.last_ckpt, s.parent = 0, None, None
        s.opus = Opus(s.model, proxy_arrays)
        s.dirs = {k: os.path.join(out, k) for k in ('ledgers', 'checkpoints')}
        for d in s.dirs.values(): os.makedirs(d, exist_ok=True)
        s.paths = {k: os.path.join(s.dirs['ledgers'], f'{branch}.{k}.jsonl') for k in ('consumption', 'opus', 'learning', 'tokens')}
        s.perf = dict(loader_s=0.0, train_s=0.0, positions=0, nonpad=0, loss_tokens=0, by_lane={})

    # ----------------------------------------------------------------- ledgers
    def _append(self, kind, rec):
        with open(self.paths[kind], 'a') as f: f.write(json.dumps(rec) + '\n')

    def _count(self, kind):
        p = self.paths[kind]
        return sum(1 for _ in open(p)) if os.path.exists(p) else 0

    # ----------------------------------------------------------------- one step
    def _candidates(self, quota):
        cands = []
        for l in LANES:
            for d in [d for d in self.deferred if d['lane'] == l]:
                cands.append(d)
            fresh = quota[l] + max(1, quota[l] // 2)          # oversample 1.5x so OPUS has a choice
            for _ in range(fresh):
                segs = self.streams[l].next_sequence()
                cid = hashlib.sha256(json.dumps([self.branch, l, segs]).encode()).hexdigest()[:16]
                cands.append(dict(cand_id=cid, lane=l, segs=segs, deferrals=0))
        self.deferred = []
        return cands

    def build_batch(self):
        t0 = time.perf_counter()
        plan = self.schedule[self.step]; quota = plan['quota']
        self.opus.maybe_refresh(self.step, self.step)
        cands = self._candidates(quota)
        decisions, accepted = [], {l: [] for l in LANES}
        for c in cands:
            c['arrays'] = materialise(c['segs'], self.store, self.seq_len, self.tok.pad)
            hit = self.fw.overlap(c['arrays']['input_ids'][c['arrays']['input_ids'] != self.tok.pad])
            c['blocked'] = hit
            c['score'] = None if hit else self.opus.score(c['arrays'])
        for l in LANES:
            pool = sorted([c for c in cands if c['lane'] == l and not c['blocked']], key=lambda c: -c['score'])
            for rank, c in enumerate(pool):
                status, reason, override, q_override = None, None, False, False
                if rank < quota[l] and c['score'] >= Opus.REJECT_BELOW:
                    status, reason = 'accepted', 'top of lane within quota'
                elif rank < quota[l] and l in FLOORS:
                    status, reason, override = 'accepted', 'protected floor: accepted despite negative proxy alignment', True
                elif rank < quota[l]:
                    status, reason, q_override = 'accepted', 'mixture quota: best remaining candidate in lane, negatively aligned', True
                elif c['score'] < Opus.REJECT_BELOW:
                    status, reason = 'rejected', 'negative alignment with the proxy direction'
                elif c['deferrals'] >= Opus.MAX_DEFERRALS:
                    status, reason = 'rejected', f'deferred {Opus.MAX_DEFERRALS} times without winning a slot'
                else:
                    status, reason = 'deferred', 'quota pressure: lane quota already filled by higher scores'
                    self.deferred.append(dict(cand_id=c['cand_id'], lane=l, segs=c['segs'], deferrals=c['deferrals'] + 1))
                c.update(status=status, reason=reason, override=override, q_override=q_override)
                if status == 'accepted': accepted[l].append(c)
        for c in cands:
            if c['blocked']:
                c.update(status='blocked', reason=f'eval firewall: shares a 13-gram with never-train shard {c["blocked"][0]} '
                                                  f'(doc {c["blocked"][1]})', override=False, q_override=False)
        # slots left empty (a lane had too few acceptable candidates) are filled by the best deferred candidates
        chosen = [c for l in LANES for c in accepted[l]]
        spare = sorted([c for c in cands if c['status'] == 'deferred'], key=lambda c: -c['score'])
        while len(chosen) < sum(quota.values()) and spare:
            c = spare.pop(0); c.update(status='accepted', reason='filled an empty slot left by another lane', q_override=True)
            self.deferred = [d for d in self.deferred if d['cand_id'] != c['cand_id']]; chosen.append(c)
        for c in cands:
            n_loss = int((c['arrays']['labels'] != -100).sum())
            rec = dict(decision_id=f'{self.branch}:{self.step}:{c["cand_id"]}', candidate_id=c['cand_id'], run_id=self.run_id,
                       branch_id=self.branch, step=self.step, lane=c['lane'], stage=plan['stage'],
                       shard_ids=sorted({sg[0] for sg in c['segs']}), scoring_checkpoint_step=self.step,
                       proxy_version=self.opus.version, opus_score=c['score'], status=c['status'], reason=c['reason'],
                       protected_floor_override=c['override'], mixture_quota_override=c.get('q_override', False),
                       times_deferred=c['deferrals'],
                       effective_token_estimate=round(n_loss * max(0.0, c['score'] or 0.0), 1))
            self._append('opus', rec); c['decision_id'] = rec['decision_id']
        self.perf['loader_s'] += time.perf_counter() - t0
        return plan, chosen

    def train_step(self, plan, chosen, token_trace=False):
        t0 = time.perf_counter()
        arrays = [c['arrays'] for c in chosen]
        x = torch.tensor(np.stack([a['input_ids'] for a in arrays])); y = torch.tensor(np.stack([a['labels'] for a in arrays]))
        di = torch.tensor(np.stack([a['doc_index'] for a in arrays])); po = torch.tensor(np.stack([a['position'] for a in arrays]))
        self.opt.zero_grad()
        n_tok = int((y != -100).sum())
        per_tok = torch.zeros_like(y, dtype=torch.float32)
        for mb in range(2):                          # two micro-batches, loss normalised by TOKENS, not by micro-batch
            sl = slice(mb * len(arrays) // 2, (mb + 1) * len(arrays) // 2)
            logits = self.model(x[sl], di[sl], po[sl])
            l_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[sl].reshape(-1), ignore_index=-100, reduction='none')
            (l_tok.sum() / n_tok).backward()
            per_tok[sl] = l_tok.view(y[sl].shape).detach()
        gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        self.opt.step()
        loss = float(per_tok.sum() / n_tok)
        bh = batch_hash(arrays)
        rec = dict(run_id=self.run_id, branch_id=self.branch, parent=self.parent, global_step=self.step,
                   checkpoint_id=self.last_ckpt, rank=0, microbatches=2, stage=plan['stage'], quota=plan['quota'],
                   samples=[dict(sample_id=c['cand_id'], lane=c['lane'], policy=POLICY[c['lane']], segments=c['segs'],
                                 loss_mask_hash=loss_mask_hash(c['arrays']), opus_decision_id=c['decision_id'])
                            for c in chosen],
                   attention_position_policy=ATTN_POLICY, tokenizer_sha256=self.tok.sha256,
                   dataloader_version=DATALOADER_VERSION, schedule_sha256=self.schedule_hash,
                   batch_sha256=bh, loss=loss, grad_norm=gn, loss_tokens=n_tok)
        self._append('consumption', rec)
        for i, c in enumerate(chosen):
            m = y[i] != -100
            self._append('learning', dict(branch_id=self.branch, step=self.step, sample_id=c['cand_id'], lane=c['lane'],
                                          shard_ids=sorted({sg[0] for sg in c['segs']}), doc_ids=[sg[1] for sg in c['segs']],
                                          loss=float(per_tok[i][m].mean()), loss_tokens=int(m.sum()),
                                          opus_score=c['score'], grad_norm_of_step=gn))
            if token_trace:
                seg = c['arrays']['doc_index']
                for t in torch.nonzero(m).flatten().tolist():
                    sg = c['segs'][seg[t]]
                    self._append('tokens', dict(step=self.step, sample_id=c['cand_id'], lane=c['lane'], position=t,
                                                token_id=int(y[i, t]), shard_id=sg[0], doc_id=sg[1],
                                                loss=round(float(per_tok[i, t]), 4), ppl=round(math.exp(float(per_tok[i, t])), 2)))
        a_np = np.stack([a['input_ids'] for a in arrays])
        self.perf['positions'] += a_np.size; self.perf['nonpad'] += int((a_np != self.tok.pad).sum()); self.perf['loss_tokens'] += n_tok
        for c in chosen:
            d = self.perf['by_lane'].setdefault(c['lane'], dict(positions=0, nonpad=0, loss_tokens=0))
            d['positions'] += self.seq_len; d['nonpad'] += int((c['arrays']['input_ids'] != self.tok.pad).sum())
            d['loss_tokens'] += int((c['arrays']['labels'] != -100).sum())
        self.perf['train_s'] += time.perf_counter() - t0
        return rec

    # ----------------------------------------------------------------- checkpoints
    def save(self):
        cid = f'{self.branch}-step{self.step:04d}'
        state = dict(model=self.model.state_dict(), opt=self.opt.state_dict(), torch_rng=torch.get_rng_state(),
                     streams={l: s.state() for l, s in self.streams.items()}, deferred=self.deferred, step=self.step,
                     opus=dict(version=self.opus.version, made_at=self.opus.made_at, direction=self.opus.direction),
                     ledger_offsets={k: self._count(k) for k in self.paths}, branch=self.branch, parent=self.parent,
                     schedule_sha256=self.schedule_hash)
        path = os.path.join(self.dirs['checkpoints'], cid + '.pt')
        torch.save(state, path)
        meta = {k: v for k, v in state.items() if k in ('step', 'streams', 'ledger_offsets', 'branch', 'parent', 'schedule_sha256')}
        meta.update(checkpoint_id=cid, n_deferred=len(self.deferred),
                    model_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in self.model.state_dict().values())).hexdigest())
        json.dump(meta, open(os.path.join(self.dirs['checkpoints'], cid + '.json'), 'w'), indent=1)
        self.last_ckpt = cid
        return cid, meta

    def load(self, path, branch=None, truncate=True):
        st = torch.load(path, weights_only=False)
        self.model.load_state_dict(st['model']); self.opt.load_state_dict(st['opt']); torch.set_rng_state(st['torch_rng'])
        for l, s in self.streams.items(): s.load_state(st['streams'][l])
        self.deferred, self.step = st['deferred'], st['step']
        self.opus.version, self.opus.made_at, self.opus.direction = st['opus']['version'], st['opus']['made_at'], st['opus']['direction']
        self.last_ckpt = os.path.basename(path)[:-3]
        if branch and branch != st['branch']:                 # fork: a new branch with an explicit parent
            self.parent = dict(branch=st['branch'], step=st['step'], checkpoint=self.last_ckpt)
            self.branch = branch
            self.paths = {k: os.path.join(self.dirs['ledgers'], f'{branch}.{k}.jsonl') for k in self.paths}
            for p in self.paths.values():
                if os.path.exists(p): os.remove(p)
        elif truncate:                                         # resume: drop ledger events written after the checkpoint
            rolled = {}
            for k, off in st['ledger_offsets'].items():
                p = self.paths[k]
                if not os.path.exists(p): continue
                lines = open(p).readlines()
                rolled[k] = lines[off:]
                open(p, 'w').writelines(lines[:off])
                if lines[off:]:
                    open(p.replace('.jsonl', '.rolled_back.jsonl'), 'w').writelines(lines[off:])
            return rolled
        return {}

    # ----------------------------------------------------------------- the loop
    def train(self, until, crash_at=None, trace_steps=(), log_every=10):
        while self.step < until:
            plan, chosen = self.build_batch()
            rec = self.train_step(plan, chosen, token_trace=self.step in trace_steps)
            if self.step % log_every == 0:
                self.log(f'[{self.branch}] step {self.step:3d} stage {plan["stage"]:15s} loss {rec["loss"]:.4f} '
                         f'grad {rec["grad_norm"]:.3f} lanes {"/".join(str(sum(1 for s_ in rec["samples"] if s_["lane"] == l)) for l in LANES)}')
            self.step += 1
            if self.step % self.ckpt_every == 0:
                cid, _ = self.save()
                self.log(f'[{self.branch}] checkpoint {cid} saved at ledger offset {self._count("consumption")}')
            if crash_at is not None and self.step == crash_at:
                raise SimulatedCrash(f'simulated crash after step {self.step - 1}')
