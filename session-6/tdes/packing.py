"""Packing policies. A packed sequence is fully described by its segments, so any sequence can be rebuilt from
the shards alone (that is what replay does).

segment = (shard_id, doc_id, start, end, loss_from)   token range [start, end) of one document in one shard;
          loss_from = offset inside the document before which tokens carry no loss (the SFT prompt).

Every packed sequence of length T yields:
  input_ids  (T,)  tokens the model reads
  labels     (T,)  next token, or -100 where no loss is allowed (document boundary, padding, SFT prompt)
  doc_index  (T,)  which segment each position belongs to (-1 for padding) -> block-diagonal causal attention
  position   (T,)  position inside its own segment (restarts at 0 for every document or document piece)
"""
import hashlib
import numpy as np

POLICY = {'web': 'concat_chop', 'indic': 'concat_chop', 'code': 'greedy_files', 'reasoning': 'sft_structured'}


class LaneStream:
    """Deterministic cursor over one lane's train shards. Produces candidate sequences with the lane's policy."""

    def __init__(self, lane, store, seq_len, firewall):
        self.lane, self.store, self.T = lane, store, seq_len
        self.policy = POLICY[lane]
        self.docs = []
        for m in sorted(store.lane_shards(lane), key=lambda m: m['shard_id']):
            firewall.check_split(m)
            for d in m['docs']:
                self.docs.append((m['shard_id'], d['doc_id'], d['start'], d['end'], d['loss_from']))
        self.cursor = 0          # index of the next document
        self.offset = 0          # tokens of that document already used (concat_chop may split a document)
        self.epoch = 0

    def state(self): return dict(cursor=self.cursor, offset=self.offset, epoch=self.epoch)
    def load_state(self, s): self.cursor, self.offset, self.epoch = s['cursor'], s['offset'], s['epoch']

    def _next_doc(self):
        d = self.docs[self.cursor]
        self.cursor += 1
        if self.cursor == len(self.docs): self.cursor, self.epoch = 0, self.epoch + 1
        return d

    def next_sequence(self):
        T = self.T + 1                     # one extra token so inputs and labels can both be cut
        segs, used = [], 0
        if self.policy == 'concat_chop':   # documents back to back; a long document continues in the next sequence
            while used < T:
                sid, did, s, e, lf = self.docs[self.cursor]
                s2 = s + self.offset; take = min(e - s2, T - used)
                segs.append((sid, did, s2, s2 + take, max(0, lf - self.offset)))
                used += take
                if s2 + take == e: self._next_doc(); self.offset = 0
                else: self.offset += take
        elif self.policy == 'greedy_files':  # whole files only, in stream order; a file that does not fit starts the next sequence
            while True:
                sid, did, s, e, lf = self.docs[self.cursor]
                n = e - s
                if used and used + n > T: break
                take = min(n, T - used)                      # a single file longer than the window is chopped (flagged)
                segs.append((sid, did, s, s + take, lf)); used += take; self._next_doc()
                if used >= T: break
        elif self.policy == 'sft_structured':   # whole conversations only, never split; prompt tokens carry no loss
            while True:
                sid, did, s, e, lf = self.docs[self.cursor]
                n = e - s
                if used and used + n > T: break
                take = min(n, T - used)
                segs.append((sid, did, s, s + take, lf)); used += take; self._next_doc()
                if used >= T: break
        return segs


def materialise(segs, store, seq_len, pad_id):
    """Build the training arrays for one packed sequence from its segments (used by training AND by replay)."""
    T = seq_len + 1
    toks = np.full(T, pad_id, dtype=np.int64); seg_of = np.full(T, -1, dtype=np.int64)
    pos = np.zeros(T, dtype=np.int64); can_be_target = np.zeros(T, dtype=bool)
    i = 0
    for k, (sid, did, s, e, lf) in enumerate(segs):
        n = e - s
        toks[i:i + n] = store.tokens(sid, s, e); seg_of[i:i + n] = k; pos[i:i + n] = np.arange(n)
        can_be_target[i:i + n] = np.arange(n) >= max(lf, 1)          # first token of a document is never a target
        i += n
    input_ids, labels = toks[:-1].copy(), toks[1:].copy()
    tgt_ok = can_be_target[1:] & (seg_of[1:] == seg_of[:-1]) & (seg_of[1:] >= 0)
    labels[~tgt_ok] = -100
    return dict(input_ids=input_ids, labels=labels, doc_index=seg_of[:-1], position=np.minimum(pos[:-1], seq_len - 1))


def batch_hash(arrays_list):
    h = hashlib.sha256()
    for a in arrays_list:
        for k in ('input_ids', 'labels', 'doc_index', 'position'): h.update(a[k].astype(np.int64).tobytes())
    return h.hexdigest()


def loss_mask_hash(a): return hashlib.sha256((a['labels'] != -100).tobytes()).hexdigest()[:16]
