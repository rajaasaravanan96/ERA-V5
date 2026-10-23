"""Invariant tests. Run: python -m pytest -q tests"""
import json, os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdes.data import FrozenTokenizer, build_shards, ShardStore, EvalFirewall
from tdes.packing import LaneStream, materialise, batch_hash
from tdes.selection import compile_schedule, LANES, FLOORS, BATCH
from tdes.run import Run, SimulatedCrash
from tdes.audit import read, replay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = 64


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    out = str(tmp_path_factory.mktemp('tdes'))
    tok = FrozenTokenizer(os.path.join(ROOT, 'sources/tokenizer.json'),
                          expected_sha256=json.load(open(os.path.join(ROOT, 'tokenizer.lock.json')))['sha256'])
    docs = [json.loads(l) for l in open(os.path.join(ROOT, 'sources/documents.jsonl'), encoding='utf-8')]
    q = next(d['text'] for d in docs if d['split'] == 'test')
    docs.insert(0, dict(lane='web', split='train', source='injected', text='copied: ' + q))
    build_shards(docs, tok, os.path.join(out, 'shards'))
    store = ShardStore(os.path.join(out, 'shards'), tok); assert store.validate() == []
    fw = EvalFirewall(store)
    val = [m for m in store.manifests.values() if m['split'] == 'val'][0]
    proxy = [materialise([(val['shard_id'], d['doc_id'], d['start'], min(d['end'], d['start'] + SEQ + 1), 0)], store, SEQ, tok.pad)
             for d in val['docs'][:4]]
    return dict(out=out, tok=tok, store=store, fw=fw, proxy=proxy, docs=docs)


def test_tokenizer_hash_is_enforced():
    with pytest.raises(ValueError):
        FrozenTokenizer(os.path.join(ROOT, 'sources/tokenizer.json'), expected_sha256='0' * 64)


def test_shards_are_immutable(env):
    docs = [dict(d) for d in env['docs'] if d['lane'] == 'code']
    docs[0]['text'] += ' changed'
    with pytest.raises(PermissionError):
        build_shards(docs, env['tok'], os.path.join(env['out'], 'shards'))


@pytest.mark.parametrize('lane', LANES)
def test_packing_masks(env, lane):
    st = LaneStream(lane, env['store'], SEQ, env['fw'])
    for _ in range(20):
        segs = st.next_sequence()
        a = materialise(segs, env['store'], SEQ, env['tok'].pad)
        di, lab, pos = a['doc_index'], a['labels'], a['position']
        # a target never crosses a document boundary and padding never carries loss
        for t in range(SEQ - 1):
            if lab[t] != -100:
                assert di[t] >= 0 and di[t] == (di[t + 1] if t + 1 < SEQ else di[t])
        assert np.all(lab[di < 0] == -100)
        # positions restart at 0 at the start of every segment
        starts = np.flatnonzero(np.r_[True, di[1:] != di[:-1]] & (di >= 0))
        assert np.all(pos[starts] == 0)
        # SFT: the prompt carries no loss (the first loss-bearing token of a conversation is after <assistant>)
        if lane == 'reasoning':
            toks = a['input_ids']
            for k in np.unique(di[di >= 0]):
                idx = np.flatnonzero(di == k)
                asst = idx[toks[idx] == env['tok'].asst]
                if len(asst):
                    assert np.all(lab[idx[idx < asst[0]]] == -100)


def test_attention_is_block_diagonal(env):
    from tdes.run import TinyLM
    di = torch.tensor([[0, 0, 0, 1, 1, -1]])
    m = TinyLM.mask(di)[0, 0]
    assert not m[3, 0] and not m[4, 2]          # document 1 cannot see document 0
    assert m[2, 0] and m[4, 3] and not m[0, 2]  # causal inside a document


def test_val_and_test_shards_cannot_train(env):
    for m in env['store'].manifests.values():
        if m['split'] != 'train':
            with pytest.raises(PermissionError):
                env['fw'].check_split(m)


def test_firewall_blocks_contamination(env):
    st = LaneStream('web', env['store'], SEQ, env['fw'])
    a = materialise(st.next_sequence(), env['store'], SEQ, env['tok'].pad)   # the injected document comes first
    assert env['fw'].overlap(a['input_ids']) is not None


def test_schedule_is_deterministic_and_respects_floors():
    s1, h1 = compile_schedule(); s2, h2 = compile_schedule()
    assert h1 == h2 and s1 == s2
    for st in s1:
        assert sum(st['quota'].values()) == BATCH
        for l, f in FLOORS.items():
            assert st['quota'][l] >= int(np.ceil(f * BATCH))


def test_resume_replay_and_fork(env, tmp_path):
    kw = dict(seq_len=SEQ, steps=12, ckpt_every=5, log=lambda m: None)
    ref = Run(str(tmp_path / 'ref'), env['store'], env['tok'], env['fw'], env['proxy'], branch='ref', **kw)
    ref.train(12)
    run = Run(str(tmp_path / 'r'), env['store'], env['tok'], env['fw'], env['proxy'], branch='main', **kw)
    with pytest.raises(SimulatedCrash):
        run.train(12, crash_at=8)
    run2 = Run(str(tmp_path / 'r'), env['store'], env['tok'], env['fw'], env['proxy'], branch='main', **kw)
    run2.load(str(tmp_path / 'r' / 'checkpoints' / 'main-step0005.pt'))
    run2.train(12)
    a = [r['batch_sha256'] for r in read(run2.paths['consumption'])]
    b = [r['batch_sha256'] for r in read(ref.paths['consumption'])]
    assert [r['global_step'] for r in read(run2.paths['consumption'])] == list(range(12))   # no skip, no repeat
    assert a == b                                                                              # same batches as an uninterrupted run
    assert all(torch.equal(p, q) for p, q in zip(run2.model.parameters(), ref.model.parameters()))
    rp = replay(read(run2.paths['consumption']), env['store'], SEQ, env['tok'].pad, set(range(12)))
    assert all(x['original'] == x['replayed'] for x in rp)
    fk = Run(str(tmp_path / 'r'), env['store'], env['tok'], env['fw'], env['proxy'], branch='fork', **kw)
    fk.load(str(tmp_path / 'r' / 'checkpoints' / 'main-step0005.pt'), branch='fork')
    assert fk.parent['step'] == 5 and fk.step == 5


def test_every_consumed_sample_has_accepted_opus_decision(env, tmp_path):
    run = Run(str(tmp_path / 'o'), env['store'], env['tok'], env['fw'], env['proxy'], branch='o', seq_len=SEQ, steps=6,
              ckpt_every=100, log=lambda m: None)
    run.train(6)
    dec = {d['decision_id']: d for d in read(run.paths['opus'])}
    for rec in read(run.paths['consumption']):
        for s in rec['samples']:
            assert dec[s['opus_decision_id']]['status'] == 'accepted'
    assert any(d['status'] == 'blocked' for d in dec.values())      # the injected contamination was stopped
