"""Audit: reconstruct and verify what the run consumed, from ledgers + manifests + shards only."""
import json
from collections import Counter, defaultdict
import numpy as np

from .packing import materialise, batch_hash
from .selection import LANES


def read(path): return [json.loads(l) for l in open(path)]


def replay(consumption, store, seq_len, pad_id, steps):
    """Rebuild the batches of the given steps from the ledger's segments and compare their hashes."""
    out = []
    for rec in consumption:
        if rec['global_step'] not in steps: continue
        arrays = [materialise([tuple(s) for s in smp['segments']], store, seq_len, pad_id) for smp in rec['samples']]
        out.append(dict(step=rec['global_step'], original=rec['batch_sha256'], replayed=batch_hash(arrays),
                        sample_ids=[s['sample_id'] for s in rec['samples']]))
    return out


def audit(consumption, opus, learning, store, firewall, schedule, pad_id, seq_len):
    res = {}
    # 1. every consumed span exists in a TRAIN shard and lies inside one document
    bad_spans, eval_spans, n_seg = [], [], 0
    for rec in consumption:
        for smp in rec['samples']:
            for sid, did, s, e, lf in smp['segments']:
                n_seg += 1
                m = store.manifests.get(sid)
                if m is None or not any(d['doc_id'] == did and d['start'] <= s and e <= d['end'] for d in m['docs']):
                    bad_spans.append((rec['global_step'], sid, did))
                elif m['split'] != 'train':
                    eval_spans.append((rec['global_step'], sid))
    res['spans'] = dict(segments_checked=n_seg, missing_or_out_of_doc=len(bad_spans), non_train_shards=len(eval_spans))
    # 2. no loss-bearing token sequence overlaps a never-train benchmark (re-checked from scratch)
    leaks = 0
    for rec in consumption:
        for smp in rec['samples']:
            a = materialise([tuple(s) for s in smp['segments']], store, seq_len, pad_id)
            if firewall.overlap(a['labels'][a['labels'] != -100]): leaks += 1
    res['eval_leaks_in_loss_bearing_tokens'] = leaks
    # 3. planned vs actual lane shares
    plan, act = Counter(), Counter()
    for rec in consumption:
        for l, q in rec['quota'].items(): plan[l] += q
        for smp in rec['samples']: act[smp['lane']] += 1
    tot_p, tot_a = sum(plan.values()), sum(act.values())
    res['mixture'] = {l: dict(planned=round(plan[l] / tot_p, 4), actual=round(act[l] / tot_a, 4),
                              abs_diff=round(abs(plan[l] / tot_p - act[l] / tot_a), 4)) for l in LANES}
    floor_viol = sum(1 for rec in consumption for l in ('indic', 'reasoning')
                     if sum(1 for s in rec['samples'] if s['lane'] == l) < 1)
    res['protected_floor_violations'] = floor_viol
    # 4. OPUS trail: every consumed sample has an 'accepted' decision; status counts
    dec = {d['decision_id']: d for d in opus}
    missing = sum(1 for rec in consumption for s in rec['samples']
                  if dec.get(s['opus_decision_id'], {}).get('status') != 'accepted')
    res['opus'] = dict(decisions=len(opus), by_status=dict(Counter(d['status'] for d in opus)),
                       overrides=sum(d['protected_floor_override'] for d in opus),
                       mixture_quota_overrides=sum(d.get('mixture_quota_override', False) for d in opus),
                       consumed_without_accepted_decision=missing,
                       by_lane_status={l: dict(Counter(d['status'] for d in opus if d['lane'] == l)) for l in LANES})
    # 5. learning trace: every consumed sample has a learning record; per-shard report card
    learned = {(r['step'], r['sample_id']) for r in learning}
    res['learning_records_missing'] = sum(1 for rec in consumption for s in rec['samples']
                                          if (rec['global_step'], s['sample_id']) not in learned)
    card = defaultdict(lambda: dict(early=[], late=[], opus=[]))
    half = max(r['step'] for r in learning) // 2
    for r in learning:
        for sid in r['shard_ids']:
            card[sid]['early' if r['step'] < half else 'late'].append(r['loss']); card[sid]['opus'].append(r['opus_score'] or 0)
    report = {}
    for sid, c in sorted(card.items()):
        e, l = (np.mean(c['early']) if c['early'] else None), (np.mean(c['late']) if c['late'] else None)
        delta = None if e is None or l is None else float(l - e)
        cls = 'unseen-late' if delta is None else ('useful' if delta < -0.3 else ('neutral' if delta < 0.05 else 'harmful/no-gain'))
        report[sid] = dict(early_loss=None if e is None else round(float(e), 3), late_loss=None if l is None else round(float(l), 3),
                           loss_delta=None if delta is None else round(delta, 3), mean_opus=round(float(np.mean(c['opus'])), 4),
                           exposures=len(c['early']) + len(c['late']), classification=cls)
    res['shard_report_card'] = report
    return res
