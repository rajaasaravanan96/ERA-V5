"""One command runs the whole Training Data Execution System demonstration and writes submission_artifacts/.

    python run_demo.py

documents -> tokenized shards -> manifests -> mixture schedule -> packing -> batches -> training
-> consumption ledger -> learning ledger -> checkpoint -> crash -> resume -> replay -> fork -> audit -> performance
"""
import glob, json, os, shutil, sys, time
import numpy as np
import torch

from tdes.data import FrozenTokenizer, build_shards, ShardStore, EvalFirewall, sha256_file
from tdes.packing import LaneStream, materialise
from tdes.selection import compile_schedule, LANES, STAGES
from tdes.run import Run, SimulatedCrash
from tdes.audit import read, replay, audit

OUT = 'submission_artifacts'
SEQ, STEPS, CRASH_AT = 128, 80, 45


def main():
    t_start = time.time()
    if os.path.exists(OUT): shutil.rmtree(OUT)
    for d in ('manifests', 'ledgers', 'checkpoints', 'shards'): os.makedirs(os.path.join(OUT, d))
    logf = open(os.path.join(OUT, 'run.log'), 'w')
    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    EV = {}
    def evidence(key, passed, where, **detail):
        EV[key] = dict(passed=bool(passed), evidence=where, **detail)
        log(f'[{"PASS" if passed else "FAIL"}] {key}')

    # ------------------------------------------------------------ 1. tokenizer, documents, shards, manifests
    pinned = json.load(open('tokenizer.lock.json'))['sha256']
    tok = FrozenTokenizer('sources/tokenizer.json', expected_sha256=pinned)
    evidence('tokenizer_hash_verified', tok.sha256 == pinned, 'tokenizer.lock.json; every manifest.tokenizer_sha256',
             sha256=tok.sha256, vocab_size=tok.vocab_size)
    docs = [json.loads(l) for l in open('sources/documents.jsonl', encoding='utf-8')]
    # a deliberately contaminated training document: ordinary text with a GSM8K TEST question pasted inside
    test_q = next(d['text'] for d in docs if d['split'] == 'test')
    docs.insert(0, dict(lane='web', split='train', source='INJECTED CONTAMINATION (firewall drill)',
                        text='Here is a practice problem that a forum copied from a benchmark. ' + test_q))
    manifests = build_shards(docs, tok, os.path.join(OUT, 'shards'))
    for m in manifests: shutil.copy(os.path.join(OUT, 'shards', m['shard_id'] + '.manifest.json'), os.path.join(OUT, 'manifests'))
    log(f'shards created: {len(manifests)} shards, {sum(m["n_tokens"] for m in manifests):,} tokens, '
        f'{sum(m["n_docs"] for m in manifests)} documents')
    try:                                                      # immutability: rewriting a shard with other content must fail
        bad = [dict(d) for d in docs if d['lane'] == 'code'][:3]; bad[0]['text'] += ' # edited'
        build_shards([dict(d) for d in docs if d['lane'] == 'code'][:0] + bad + [d for d in docs if d['lane'] == 'code'][3:],
                     tok, os.path.join(OUT, 'shards'))
        immutable = False
    except PermissionError as e:
        immutable = True; log(f'shard rewrite refused: {e}')
    store = ShardStore(os.path.join(OUT, 'shards'), tok)
    problems = store.validate()
    log('manifests validated' + ('' if not problems else f' with problems: {problems}'))
    evidence('manifests_validated', not problems and immutable, 'manifests/*.manifest.json',
             shards=len(manifests), content_hash_mismatches=len(problems), rewrite_refused=immutable)

    # ------------------------------------------------------------ 2. firewall
    fw = EvalFirewall(store)
    refused = []
    for m in store.manifests.values():
        if m['split'] != 'train':
            try: fw.check_split(m)
            except PermissionError: refused.append(m['shard_id'])
    log(f'evaluation data blocked: {len(refused)} val/test shards refused as gradient-bearing; '
        f'{len(fw.fingerprints):,} test 13-gram fingerprints registered')
    json.dump(dict(registry=fw.registry, fingerprints=len(fw.fingerprints)), open(os.path.join(OUT, 'manifests', 'eval_registry.json'), 'w'), indent=1)
    evidence('eval_shard_blocked', len(refused) == sum(1 for m in store.manifests.values() if m['split'] != 'train'),
             'manifests/eval_registry.json', refused=refused)

    # ------------------------------------------------------------ 3. mixture schedule
    sched, sched_hash = compile_schedule()
    sched2, sched_hash2 = compile_schedule()
    json.dump(dict(sha256=sched_hash, stages=STAGES, steps=sched), open(os.path.join(OUT, 'manifests', 'mixture_schedule.json'), 'w'), indent=1)
    log(f'mixture compiled: {len(sched)} steps, schedule sha256 {sched_hash[:16]}… (recompiled identically: {sched_hash == sched_hash2})')

    # proxy batch for OPUS: validation documents (direction only, never gradient-bearing)
    val_m = [m for m in store.manifests.values() if m['split'] == 'val'][0]
    proxy = [materialise([(val_m['shard_id'], d['doc_id'], d['start'], min(d['end'], d['start'] + SEQ + 1), 0)], store, SEQ, tok.pad)
             for d in val_m['docs'][:8]]

    # ------------------------------------------------------------ 4. reference run (never crashes) for comparison
    ref_dir = os.path.join(OUT, 'reference'); os.makedirs(ref_dir)
    ref = Run(ref_dir, store, tok, fw, proxy, branch='reference', seq_len=SEQ, steps=STEPS, log=lambda m: None)
    ref.train(STEPS)
    log(f'reference run (uninterrupted) finished: {STEPS} steps')

    # ------------------------------------------------------------ 5. main run with a crash
    run = Run(OUT, store, tok, fw, proxy, branch='main', seq_len=SEQ, steps=STEPS, log=log)
    log('batches packed: policies web/indic=concat_chop, code=greedy_files, reasoning=sft_structured; '
        'block-diagonal attention, positions reset per segment')
    try:
        run.train(STEPS, crash_at=CRASH_AT, trace_steps=(5, 40, 75))
    except SimulatedCrash as e:
        log(f'crash simulated: {e}')
    opus_recs = read(run.paths['opus'])
    log(f'OPUS decisions recorded: {len(opus_recs)} so far')
    last_ckpt = sorted(glob.glob(os.path.join(OUT, 'checkpoints', 'main-step*.pt')))[-1]
    evidence('checkpoint_saved', os.path.exists(last_ckpt), os.path.relpath(last_ckpt, OUT).replace('.pt', '.json'))

    # ------------------------------------------------------------ 6. resume
    run2 = Run(OUT, store, tok, fw, proxy, branch='main', seq_len=SEQ, steps=STEPS, log=log)
    rolled = run2.load(last_ckpt)
    lost = [json.loads(l) for l in rolled.get('consumption', [])]
    log(f'run resumed from {os.path.basename(last_ckpt)} at step {run2.step}; {len(lost)} consumption events after the '
        f'checkpoint moved to ledgers/main.consumption.rolled_back.jsonl')
    expected_next = lost[0] if lost else None
    ckpt_step = int(os.path.basename(last_ckpt).split('step')[1].split('.')[0])
    run2.train(STEPS, trace_steps=(75,))
    cons = read(run2.paths['consumption'])
    resumed_next = next(r for r in cons if r['global_step'] == ckpt_step)
    ref_cons = {r['global_step']: r for r in read(ref.paths['consumption'])}
    same_next = expected_next is not None and resumed_next['batch_sha256'] == expected_next['batch_sha256'] == ref_cons[resumed_next['global_step']]['batch_sha256']
    all_same = all(r['batch_sha256'] == ref_cons[r['global_step']]['batch_sha256'] for r in cons)
    steps_seen = [r['global_step'] for r in cons]
    no_skip_repeat = steps_seen == list(range(STEPS))
    w_main = torch.cat([p.flatten() for p in run2.model.parameters()]); w_ref = torch.cat([p.flatten() for p in ref.model.parameters()])
    evidence('resume_next_batch_matched', same_next and all_same and no_skip_repeat, 'ledgers/main.consumption.jsonl; ledgers/main.consumption.rolled_back.jsonl',
             resumed_from_step=ckpt_step, next_step=resumed_next['global_step'],
             expected_batch_sha256=expected_next['batch_sha256'] if expected_next else None,
             resumed_batch_sha256=resumed_next['batch_sha256'],
             expected_sample_ids=[s['sample_id'] for s in expected_next['samples']] if expected_next else None,
             resumed_sample_ids=[s['sample_id'] for s in resumed_next['samples']],
             all_80_batches_equal_uninterrupted_run=all_same, steps_contiguous_no_skip_no_repeat=no_skip_repeat,
             final_weights_equal_uninterrupted_run=bool(torch.equal(w_main, w_ref)))

    # ------------------------------------------------------------ 7. replay a historical interval
    rp = replay(cons, store, SEQ, tok.pad, set(range(20, 31)))
    ok = all(r['original'] == r['replayed'] for r in rp)
    json.dump(rp, open(os.path.join(OUT, 'ledgers', 'replay_steps_20_30.json'), 'w'), indent=1)
    log(f'historical stream replayed: steps 20-30, {sum(r["original"] == r["replayed"] for r in rp)}/{len(rp)} batch hashes match')
    evidence('replay_hash_matched', ok and len(rp) == 11, 'ledgers/replay_steps_20_30.json', steps=len(rp))

    # ------------------------------------------------------------ 8. fork from an earlier checkpoint
    fork_stages = [dict(s) for s in STAGES]
    fork_stages[1] = dict(fork_stages[1], weights=dict(web=0.15, indic=0.35, code=0.25, reasoning=0.25))
    fk = Run(OUT, store, tok, fw, proxy, branch='fork-indic-up', seq_len=SEQ, steps=60, stages=fork_stages, log=log)
    fk.load(os.path.join(OUT, 'checkpoints', 'main-step0040.pt'), branch='fork-indic-up')
    fk.train(60)
    fcons = read(fk.paths['consumption'])
    diverged = sum(r['batch_sha256'] != ref_cons[r['global_step']]['batch_sha256'] for r in fcons)
    log(f'branch forked: fork-indic-up from main@40 for 20 steps; {diverged}/20 batches differ from main; parent recorded = {fcons[0]["parent"]}')
    evidence('branch_forked', fcons[0]['parent']['step'] == 40 and len(fcons) == 20, 'ledgers/fork-indic-up.consumption.jsonl',
             parent=fcons[0]['parent'], batches_differing_from_main=diverged)

    # ------------------------------------------------------------ 9. audit
    opus_recs = read(run2.paths['opus']); learning = read(run2.paths['learning'])
    au = audit(cons, opus_recs, learning, store, fw, sched, tok.pad, SEQ)
    json.dump(au, open(os.path.join(OUT, 'ledgers', 'audit.json'), 'w'), indent=1)
    blocked = [d for d in opus_recs if d['status'] == 'blocked']
    log(f'audit completed: {au["spans"]["segments_checked"]} segments checked, eval leaks {au["eval_leaks_in_loss_bearing_tokens"]}, '
        f'OPUS {au["opus"]["by_status"]}, overrides {au["opus"]["overrides"]}, blocked by firewall {len(blocked)}')
    evidence('packing_correct', au['spans']['missing_or_out_of_doc'] == 0, 'ledgers/audit.json#spans; tests/test_invariants.py',
             **au['spans'])
    evidence('eval_firewall_held', au['eval_leaks_in_loss_bearing_tokens'] == 0 and au['spans']['non_train_shards'] == 0 and len(blocked) >= 1,
             'ledgers/main.opus.jsonl (status=blocked); ledgers/audit.json', blocked_candidates=len(blocked),
             blocked_example=blocked[0]['reason'] if blocked else None)
    mix_ok = max(v['abs_diff'] for v in au['mixture'].values()) <= 0.03 and au['protected_floor_violations'] == 0
    evidence('mixture_compliance', mix_ok, 'ledgers/audit.json#mixture', shares=au['mixture'],
             protected_floor_violations=au['protected_floor_violations'])
    evidence('opus_audit_trail', au['opus']['consumed_without_accepted_decision'] == 0 and
             all(k in au['opus']['by_status'] for k in ('accepted', 'rejected', 'deferred')) and au['opus']['overrides'] >= 0,
             'ledgers/main.opus.jsonl', **{k: au['opus'][k] for k in ('decisions', 'by_status', 'overrides', 'mixture_quota_overrides')})
    evidence('learning_trace', au['learning_records_missing'] == 0 and os.path.getsize(run2.paths['tokens']) > 0,
             'ledgers/main.learning.jsonl; ledgers/main.tokens.jsonl; ledgers/audit.json#shard_report_card',
             shards_classified=dict(__import__('collections').Counter(v['classification'] for v in au['shard_report_card'].values())))

    # ------------------------------------------------------------ 10. performance
    # loader-only throughput, measured separately: pack + materialise without the model
    st = {l: LaneStream(l, store, SEQ, fw) for l in LANES}
    t0 = time.perf_counter(); n = 0
    for _ in range(50):
        for l in LANES:
            a = materialise(st[l].next_sequence(), store, SEQ, tok.pad); n += int((a['labels'] != -100).sum())
    loader_tps = n / (time.perf_counter() - t0)
    p = run2.perf; pr = run.perf
    positions = p['positions'] + pr['positions']; loss_tok = p['loss_tokens'] + pr['loss_tokens']; nonpad = p['nonpad'] + pr['nonpad']
    train_s = p['train_s'] + pr['train_s']; loader_s = p['loader_s'] + pr['loader_s']
    by_lane = {}
    for l in LANES:
        a = pr['by_lane'].get(l, dict(positions=0, nonpad=0, loss_tokens=0)); b = p['by_lane'].get(l, dict(positions=0, nonpad=0, loss_tokens=0))
        tot = a['positions'] + b['positions']
        by_lane[l] = dict(utilization=round((a['nonpad'] + b['nonpad']) / max(1, tot), 4),
                          loss_bearing_fraction=round((a['loss_tokens'] + b['loss_tokens']) / max(1, tot), 4))
    perf = dict(packing_utilization=round(nonpad / positions, 4), loss_bearing_fraction=round(loss_tok / positions, 4),
                by_lane=by_lane, loader_only_loss_bearing_tokens_per_s=round(loader_tps),
                end_to_end_loss_bearing_tokens_per_s=round(loss_tok / (train_s + loader_s)),
                train_seconds=round(train_s, 2), selection_and_packing_seconds=round(loader_s, 2),
                note='selection time includes OPUS scoring of 1.5x oversampled candidates; measured on CPU')
    json.dump(perf, open(os.path.join(OUT, 'performance.json'), 'w'), indent=1)
    log(f'performance measured: utilization {perf["packing_utilization"]:.1%}, loss-bearing {perf["loss_bearing_fraction"]:.1%}, '
        f'{perf["end_to_end_loss_bearing_tokens_per_s"]:,} loss-bearing tok/s end to end, loader alone {perf["loader_only_loss_bearing_tokens_per_s"]:,}')
    evidence('throughput', perf['packing_utilization'] > 0.9 and perf['end_to_end_loss_bearing_tokens_per_s'] > 0, 'performance.json',
             **{k: perf[k] for k in ('packing_utilization', 'loss_bearing_fraction', 'end_to_end_loss_bearing_tokens_per_s')})

    # ------------------------------------------------------------ evidence bundle (generated, never hand-written)
    EV['_meta'] = dict(command='python run_demo.py', seconds=round(time.time() - t_start, 1), steps=STEPS, crash_after_step=CRASH_AT - 1,
                       torch=torch.__version__, tokenizer_sha256=tok.sha256, schedule_sha256=sched_hash)
    json.dump(EV, open(os.path.join(OUT, 'evidence.json'), 'w'), indent=1, default=str)
    rows = [('Tokenizer integrity', 'tokenizer_hash_verified', 'Manifest record'),
            ('Shards and manifests', 'manifests_validated', 'Content hashes + refused rewrite'),
            ('Evaluation firewall', 'eval_firewall_held', 'Blocked-candidate event + audit'),
            ('Eval shards not trainable', 'eval_shard_blocked', 'Registry refusal'),
            ('Packing correctness', 'packing_correct', 'Span audit + tests'),
            ('Mixture compliance', 'mixture_compliance', 'Planned versus actual shares'),
            ('OPUS audit trail', 'opus_audit_trail', 'Candidate decision records'),
            ('Checkpoint', 'checkpoint_saved', 'Checkpoint tied to ledger offset'),
            ('Crash recovery', 'resume_next_batch_matched', 'Expected and resumed batch ids'),
            ('Replay', 'replay_hash_matched', 'Original and replay hashes'),
            ('Fork', 'branch_forked', 'Branch ledger with parent'),
            ('Learning trace', 'learning_trace', 'Loss linked to source data'),
            ('Throughput', 'throughput', 'Performance report')]
    md = ['# Evidence', '', f'Generated by `python run_demo.py` in {EV["_meta"]["seconds"]} s.', '',
          '| Requirement | Result | Evidence |', '| --- | --- | --- |']
    for name, key, what in rows:
        md.append(f'| {name} | {"PASS" if EV[key]["passed"] else "FAIL"} | {what}: `{EV[key]["evidence"]}` |')
    r = EV['resume_next_batch_matched']
    md += ['', '## Crash recovery', '', f'Crashed after step {CRASH_AT - 1}; resumed from step {r["resumed_from_step"]}.',
           f'- expected next batch (step {r["next_step"]}): `{r["expected_batch_sha256"]}`',
           f'- resumed next batch: `{r["resumed_batch_sha256"]}`',
           f'- sample ids expected: {r["expected_sample_ids"]}', f'- sample ids resumed: {r["resumed_sample_ids"]}',
           f'- all {STEPS} batches equal to an uninterrupted run: {r["all_80_batches_equal_uninterrupted_run"]}; '
           f'final weights equal: {r["final_weights_equal_uninterrupted_run"]}',
           '', '## Mixture, planned against actual', '', '| lane | planned | actual |', '| --- | --- | --- |']
    md += [f'| {l} | {v["planned"]:.3f} | {v["actual"]:.3f} |' for l, v in au['mixture'].items()]
    md += ['', '## OPUS decisions', '', f'{au["opus"]["by_status"]}; protected-floor overrides: {au["opus"]["overrides"]}; '
           f'mixture-quota overrides: {au["opus"]["mixture_quota_overrides"]}',
           f'Firewall block: {EV["eval_firewall_held"]["blocked_example"]}',
           '', '## Performance', '', f'```json\n{json.dumps(perf, indent=1)}\n```']
    open(os.path.join(OUT, 'evidence.md'), 'w').write('\n'.join(md) + '\n')
    log(f'evidence written: {sum(v["passed"] for k, v in EV.items() if not k.startswith("_"))}/{len(rows)} requirements passed')
    shutil.rmtree(os.path.join(OUT, 'reference', 'checkpoints'), ignore_errors=True)
    return all(v['passed'] for k, v in EV.items() if not k.startswith('_'))


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
