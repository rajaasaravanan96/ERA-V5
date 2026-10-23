"""Frozen tokenizer, immutable tokenized shards with manifests, and the eval/validation registry."""
import hashlib, json, os
import numpy as np
from tokenizers import Tokenizer

SPECIAL = {'<eos>': 0, '<pad>': 1, '<user>': 2, '<assistant>': 3}   # offsets added after the BPE vocabulary


def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_file(p): return sha256_bytes(open(p, 'rb').read())


class FrozenTokenizer:
    """The BPE tokenizer from Session 2, pinned by the hash of its file. Special ids sit after the BPE ids."""

    def __init__(self, path, expected_sha256=None):
        self.path, self.sha256 = path, sha256_file(path)
        if expected_sha256 and expected_sha256 != self.sha256:
            raise ValueError(f'tokenizer hash {self.sha256} != expected {expected_sha256}')
        self.tok = Tokenizer.from_file(path)
        self.base = self.tok.get_vocab_size()
        self.eos, self.pad, self.user, self.asst = (self.base + SPECIAL[k] for k in SPECIAL)
        self.vocab_size = self.base + len(SPECIAL)

    def encode(self, text): return self.tok.encode(text).ids


def doc_id(text): return sha256_bytes(text.encode('utf-8'))[:16]


def build_shards(docs, tok, out_dir, shard_docs=64):
    """Tokenize documents into immutable shards. A shard is written once; rewriting different content raises."""
    os.makedirs(out_dir, exist_ok=True)
    by_key = {}
    for d in docs: by_key.setdefault((d['lane'], d['split']), []).append(d)
    manifests = []
    for (lane, split), items in sorted(by_key.items()):
        for k in range(0, len(items), shard_docs):
            chunk = items[k:k + shard_docs]
            ids, meta = [], []
            for d in chunk:
                if lane == 'reasoning':                          # SFT: <user> prompt <assistant> answer <eos>
                    p, a = tok.encode(d['prompt']), tok.encode(d['text'])
                    seq = [tok.user] + p + [tok.asst] + a + [tok.eos]
                    loss_from = 1 + len(p) + 1                   # loss only on the assistant answer and its <eos>
                else:
                    seq = tok.encode(d['text']) + [tok.eos]; loss_from = 0
                meta.append(dict(doc_id=doc_id(d.get('prompt', '') + d['text']), start=len(ids), end=len(ids) + len(seq),
                                 loss_from=loss_from, source=d['source']))
                ids += seq
            arr = np.array(ids, dtype=np.uint16)
            shard_id = f'{lane}-{split}-{k // shard_docs:03d}'
            path = os.path.join(out_dir, shard_id + '.npy')
            content = arr.tobytes()
            if os.path.exists(path):
                if sha256_bytes(np.load(path).tobytes()) != sha256_bytes(content):
                    raise PermissionError(f'shard {shard_id} is immutable and already exists with different content')
            else:
                np.save(path, arr)
            m = dict(shard_id=shard_id, lane=lane, split=split, path=os.path.basename(path), n_docs=len(chunk),
                     n_tokens=len(ids), content_sha256=sha256_bytes(content), tokenizer_sha256=tok.sha256,
                     never_train=split == 'test', gradient_bearing=split == 'train', docs=meta)
            json.dump(m, open(os.path.join(out_dir, shard_id + '.manifest.json'), 'w'), indent=1)
            manifests.append(m)
    return manifests


class ShardStore:
    """Loads shards and verifies them against their manifests (content hash and tokenizer hash) before use."""

    def __init__(self, shard_dir, tok):
        self.dir, self.tok, self.manifests, self.arrays = shard_dir, tok, {}, {}
        for f in sorted(os.listdir(shard_dir)):
            if f.endswith('.manifest.json'):
                m = json.load(open(os.path.join(shard_dir, f)))
                self.manifests[m['shard_id']] = m

    def validate(self):
        problems = []
        for sid, m in self.manifests.items():
            arr = np.load(os.path.join(self.dir, m['path']))
            if sha256_bytes(arr.tobytes()) != m['content_sha256']: problems.append(f'{sid}: content hash mismatch')
            if m['tokenizer_sha256'] != self.tok.sha256: problems.append(f'{sid}: tokenizer hash mismatch')
            self.arrays[sid] = arr
        return problems

    def tokens(self, shard_id, start, end):
        return self.arrays[shard_id][start:end]

    def lane_shards(self, lane, split='train'):
        return [m for m in self.manifests.values() if m['lane'] == lane and m['split'] == split]


class EvalFirewall:
    """Registry of never-train shards. Any candidate that shares a 13-gram of token ids with them is blocked,
    and validation/test shards can never be opened as gradient-bearing streams."""
    N = 13

    def __init__(self, store):
        self.fingerprints, self.registry = {}, []
        for m in store.manifests.values():
            if m['split'] in ('test', 'val'):
                self.registry.append(dict(shard_id=m['shard_id'], split=m['split'], content_sha256=m['content_sha256'],
                                          never_train=True))
            if m['split'] == 'test':
                arr = store.arrays[m['shard_id']]
                for d in m['docs']:
                    t = arr[d['start']:d['end']].tolist()
                    for i in range(len(t) - self.N + 1):
                        self.fingerprints[hash(tuple(t[i:i + self.N]))] = (m['shard_id'], d['doc_id'])

    def check_split(self, manifest):
        if not manifest['gradient_bearing']:
            raise PermissionError(f"{manifest['shard_id']} is a {manifest['split']} shard and can never be gradient-bearing")

    def overlap(self, token_ids):
        t = list(token_ids)
        for i in range(len(t) - self.N + 1):
            hit = self.fingerprints.get(hash(tuple(t[i:i + self.N])))
            if hit: return hit
        return None
