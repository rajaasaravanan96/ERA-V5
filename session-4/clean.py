"""Session 4 cleaning pipeline for lordx64/reasoning-distill-opus-4-7-max-sft (7,823 rows, 29 MB, Apache-2.0).

Stages, in order (the lesson's pipeline, adapted to chat-format reasoning data):
  1. format    parse the literal Qwen chat markers into structured messages (the ghost-tag fix)
  2. normalize NFC, drop invisible noise, repair mojibake, unescape HTML in prose, tidy whitespace; keep ZWJ/ZWNJ
  3. quality   drop rows with no real task, unclosed/empty reasoning, truncated answers, degenerate repetition
  4. dedup     exact (content hash) then near-duplicate prompts (MinHash + LSH, implemented here)
  5. language  detect the script/language of each row instead of trusting the dataset card
  6. pii       mask emails, phone numbers, IPs and secrets; keep documentation-style placeholders
  7. decontam  13-gram overlap against GSM8K, HumanEval and MATH-500 test sets
  8. manifest  content-addressed ids, hashes, token counts, script hash, licence

Run:  python clean.py        (needs data_raw/, see README)
Writes data_clean/clean.jsonl.gz, data_clean/manifest.json, report.json
"""
import gzip, hashlib, html, json, os, random, re, sys, time, unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq

RAW = 'data_raw/opus47_train.parquet'
SRC_URL = 'https://huggingface.co/datasets/lordx64/reasoning-distill-opus-4-7-max-sft'
os.makedirs('data_clean', exist_ok=True)
REPORT = {'stages': [], 'examples': defaultdict(list)}


def example(stage, before, after, note, limit=4):
    if len(REPORT['examples'][stage]) < limit:
        REPORT['examples'][stage].append({'before': before[:400], 'after': after[:400], 'note': note})


def stage_log(name, n_in, n_out, chars_in, chars_out, detail):
    REPORT['stages'].append(dict(stage=name, docs_in=n_in, docs_out=n_out, chars_in=chars_in,
                                 chars_out=chars_out, detail=detail))
    print(f'{name:10s} {n_in:6d} -> {n_out:6d} docs   {chars_in:>11,} -> {chars_out:>11,} chars   {detail}')


def doc_chars(d):
    return sum(len(m['content']) for m in d['messages']) + len(d.get('reasoning', ''))


# ---------------------------------------------------------------------------- 1. format
TURN = re.compile(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>', re.S)
OTHER_GHOSTS = re.compile(r'(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|user\|>|<\|assistant\|>|<\|system\|>|'
                          r'\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|^### (Human|Assistant|Instruction|Response):|'
                          r'<start_of_turn>|<end_of_turn>)', re.M)


def parse(text):
    turns = TURN.findall(text)
    leftover = TURN.sub('', text).strip()
    if not turns or leftover:
        return None, 'unparsable markers'
    roles = [r for r, _ in turns]
    if roles not in (['system', 'user', 'assistant'], ['user', 'assistant']):
        return None, f'unexpected turn order {roles}'
    msgs = [{'role': r, 'content': c} for r, c in turns]
    a = msgs[-1]['content']
    if '<think>' not in a and '</think>' not in a:
        return {'messages': msgs, 'reasoning': '', 'has_reasoning': False}, None   # a plain answer: kept, labelled
    m = re.match(r'\s*<think>\n?(.*?)</think>\s*(.*)\Z', a, re.S)
    if not m:
        return None, 'broken <think> block (opened but not closed, or text before it)'
    msgs[-1]['content'] = m.group(2)
    return {'messages': msgs, 'reasoning': m.group(1), 'has_reasoning': True}, None


# ---------------------------------------------------------------------------- 2. normalize
NOISE = {'​': 'zero-width space', '﻿': 'byte-order mark', '⁠': 'word joiner',
         '­': 'soft hyphen', '‪': 'bidi embed', '‫': 'bidi embed', '‬': 'bidi pop',
         '‭': 'bidi override', '‮': 'bidi override', '⁦': 'bidi isolate', '⁧': 'bidi isolate',
         '⁨': 'bidi isolate', '⁩': 'bidi isolate'}
KEEP = {'‌': 'ZWNJ', '‍': 'ZWJ'}       # meaning in Indic scripts and emoji sequences: never removed
MOJIBAKE = re.compile(r'[Â-ô][\u0080-¿]{1,3}')
FENCE = re.compile(r'(```.*?```|`[^`\n]+`)', re.S)


def fix_mojibake(s, counts):
    def rep(m):
        try:
            fixed = m.group(0).encode('latin-1').decode('utf-8')
            counts['mojibake repaired'] += 1
            return fixed
        except UnicodeError:
            return m.group(0)
    return MOJIBAKE.sub(rep, s)


def normalize_prose(seg, counts):
    un = html.unescape(seg)
    if un != seg: counts['html entities unescaped (prose only)'] += len(re.findall(r'&[#\w]+;', seg))
    # Runs of spaces are NOT collapsed: the harm test showed that rule rewrote 9.1% of good documents
    # (Markdown tables, double spaces after a sentence). Only blank-line runs of 4+ are shortened.
    seg2 = re.sub(r'\n{4,}', '\n\n\n', un)
    if seg2 != un: counts['blank-line runs shortened'] += 1
    return seg2


def normalize(s, counts):
    s0 = s
    s = fix_mojibake(s, counts)
    s = unicodedata.normalize('NFC', s)                # NFC, not NFKC: NFKC rewrites ² ﬁ ＡＢ and breaks code/maths
    out = []
    for ch in s:
        if ch in KEEP:
            counts[f'{KEEP[ch]} kept'] += 1; out.append(ch)
        elif ch in NOISE:
            counts[f'{NOISE[ch]} removed'] += 1
        elif unicodedata.category(ch) == 'Cc' and ch not in '\n\t':
            counts['control character removed'] += 1
        elif unicodedata.category(ch) == 'Co':
            counts['private-use character removed'] += 1
        elif ch == '�':
            counts['replacement character removed'] += 1
        else:
            out.append(ch)
    s = ''.join(out)
    parts = FENCE.split(s)                             # code blocks and inline code are left byte-exact
    s = ''.join(p if i % 2 else normalize_prose(p, counts) for i, p in enumerate(parts))
    return s.strip('\n ') if s != s0 else s


# ---------------------------------------------------------------------------- 3. quality
NO_TASK = re.compile(r"(haven't|have not|didn't|did not) (included|include|provided|provide|shared|share|specified|"
                     r"given)[^.]{0,60}(problem|question|task|details|statement)|"
                     r"(could|can) you (please )?(share|provide|clarify|tell me)[^.]{0,40}(problem|question|task)", re.I)


def ngram_repeat_run(text, n=4):
    """longest run of a repeated n-gram (OLMo 2 removes docs with >= 32 repeated n-grams)."""
    w = text.split(); best = run = 0
    for i in range(n, len(w) - n + 1, 1):
        if w[i:i + n] == w[i - n:i]: run += 1; best = max(best, run)
        else: run = 0
    return best


def quality(d):
    user = d['messages'][-2]['content']; ans = d['messages'][-1]['content']; think = d['reasoning']
    if len(user.strip()) < 15: return 'user prompt shorter than 15 characters'
    if NO_TASK.search(ans[:600]) and len(ans) < 1500: return 'no real task: the answer only asks for the missing problem'
    if d['has_reasoning'] and len(think.strip()) == 0: return 'empty <think></think> block'
    if len(ans.strip()) == 0: return 'empty final answer'
    if ans.count('```') % 2: return 'truncated: unclosed code block in the answer'
    lines = [l for l in ans.split('\n') if l.strip()]
    if len(lines) >= 10 and 1 - len(set(lines)) / len(lines) > 0.3: return 'more than 30% duplicated lines'
    if ngram_repeat_run(think + ' ' + ans) >= 32: return 'degenerate repetition (32+ repeated 4-grams)'
    return None


# ---------------------------------------------------------------------------- 4. dedup (MinHash + LSH)
MERSENNE = (1 << 61) - 1
NUM_PERM, BANDS, ROWS = 128, 16, 8                    # threshold ~ (1/16)^(1/8) = 0.71 Jaccard
rnd = random.Random(1234)
PERMS = [(rnd.randrange(1, MERSENNE), rnd.randrange(0, MERSENNE)) for _ in range(NUM_PERM)]


def shingles(text, k=5):
    w = re.findall(r'\w+', text.lower())
    return {' '.join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def minhash(sh):
    hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little') for s in sh]
    return [min((a * h + b) % MERSENNE for h in hs) for a, b in PERMS]


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------------------- 5. language
SCRIPTS = [('Devanagari', 0x0900, 0x097F), ('Bengali', 0x0980, 0x09FF), ('Tamil', 0x0B80, 0x0BFF),
           ('Telugu', 0x0C00, 0x0C7F), ('Kannada', 0x0C80, 0x0CFF), ('Malayalam', 0x0D00, 0x0D7F),
           ('Gujarati', 0x0A80, 0x0AFF), ('Gurmukhi', 0x0A00, 0x0A7F), ('Cyrillic', 0x0400, 0x04FF),
           ('Arabic', 0x0600, 0x06FF), ('CJK', 0x4E00, 0x9FFF), ('Hiragana/Katakana', 0x3040, 0x30FF),
           ('Hangul', 0xAC00, 0xD7AF), ('Greek', 0x0370, 0x03FF)]
EN_STOP = set('the of and to in is that it for as with on be this are by an or from at not which you can we if'.split())


def detect_language(text):
    letters = [c for c in text if c.isalpha()]
    if not letters: return 'none'
    counts = Counter()
    for c in letters:
        o = ord(c)
        counts[next((n for n, lo, hi in SCRIPTS if lo <= o <= hi), 'Latin' if o < 0x0250 else 'other')] += 1
    script, n = counts.most_common(1)[0]
    if script != 'Latin':
        return script if n / len(letters) > 0.3 else 'mixed'
    words = re.findall(r'[a-z]+', text.lower())
    sw = sum(w in EN_STOP for w in words) / max(1, len(words))
    return 'English' if sw > 0.08 else 'Latin, not English (or code-heavy)'


# ---------------------------------------------------------------------------- 6. pii
SAFE_DOMAINS = re.compile(r'@(example\.(com|org|net)|test\.com|domain\.com|email\.com|company\.com|localhost|'
                          r'.*\.(test|example|invalid|local))$', re.I)
EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b')
PHONE = re.compile(r'(?<![\w.])(?:\+?\d{1,3}[ -])?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}(?![\w.])')
IPV4 = re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b')
SECRETS = [('OpenAI-style key', re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b')),
           ('AWS access key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
           ('GitHub token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{36,}\b')),
           ('Slack token', re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b')),
           ('private key block', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'))]
SAFE_IP = re.compile(r'^(127\.|0\.0\.0\.0|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|255\.|1\.1\.1\.1|8\.8\.)')


def scrub(s, counts):
    def em(m):
        if SAFE_DOMAINS.search(m.group(0)): counts['email kept (documentation domain)'] += 1; return m.group(0)
        counts['email masked'] += 1; return '<EMAIL>'
    s = EMAIL.sub(em, s)
    def ph(m):
        if re.search(r'555[ -]?01\d\d$', m.group(0)): counts['phone kept (fictional 555-01xx)'] += 1; return m.group(0)
        if re.match(r'(\+?1[ -])?\(?8(00|33|44|55|66|77|88)\)?[ -]', m.group(0)):
            counts['phone kept (public toll-free line)'] += 1; return m.group(0)
        counts['phone masked'] += 1; return '<PHONE>'
    s = PHONE.sub(ph, s)
    def ip(m):
        pre = s[max(0, m.start() - 1):m.start()]
        if pre in ('/', 'v', 'V') or m.group(0).endswith('.0.0'):
            counts['version number kept'] += 1; return m.group(0)
        if SAFE_IP.match(m.group(0)): counts['IP kept (private/reserved range)'] += 1; return m.group(0)
        if all(int(x) < 10 for x in m.group(0).split('.')): counts['version number kept'] += 1; return m.group(0)
        counts['public IP masked'] += 1; return '<IP>'
    s = IPV4.sub(ip, s)
    for name, rx in SECRETS:
        s, n = rx.subn(f'<SECRET:{name}>', s)
        if n: counts[f'secret masked: {name}'] += n
    return s


# ---------------------------------------------------------------------------- 7. decontamination
def ngrams(text, n=13):
    w = re.findall(r'\w+', text.lower())
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_evals():
    import json as _j
    ev = {}
    t = pq.read_table('data_raw/evals/gsm8k_test.parquet').to_pandas()
    ev['GSM8K test'] = list(t['question'])
    t = pq.read_table('data_raw/evals/humaneval_test.parquet').to_pandas()
    ev['HumanEval'] = list(t['prompt'])
    ev['MATH-500'] = [_j.loads(l)['problem'] for l in open('data_raw/evals/math500_test.jsonl')]
    return ev


# ---------------------------------------------------------------------------- run
def run():
    t0 = time.time()
    script_hash = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    raw_bytes = open(RAW, 'rb').read()
    texts = list(pq.read_table(RAW).to_pandas()['text'])
    raw_chars = sum(map(len, texts))
    REPORT['raw'] = dict(rows=len(texts), chars=raw_chars, parquet_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                         rows_with_system_prompt=sum('<|im_start|>system' in t for t in texts))
    print(f'raw: {len(texts)} rows, {raw_chars:,} chars')

    # 1. format
    docs, drops = [], Counter()
    for i, t in enumerate(texts):
        d, why = parse(t)
        if d is None:
            drops[why] += 1; continue
        d['raw_index'] = i
        stray = sum(len(OTHER_GHOSTS.findall(m['content'])) for m in d['messages']) + len(OTHER_GHOSTS.findall(d['reasoning']))
        if stray:
            drops['ghost markers left inside the content'] += 1; continue
        docs.append(d)
    ex = texts[0]
    example('format', ex[:300], json.dumps(parse(ex)[0]['messages'][:2], ensure_ascii=False)[:300],
            'literal <|im_start|> markers become structured roles; the model will only ever see real special tokens')
    stage_log('format', len(texts), len(docs), raw_chars, sum(map(doc_chars, docs)), dict(drops))

    # 2. normalize
    c_in = sum(map(doc_chars, docs)); counts = Counter(); changed = 0
    for d in docs:
        before = json.dumps(d, ensure_ascii=False)
        for m in d['messages']:
            old = m['content']; m['content'] = normalize(m['content'], counts)
            if old != m['content']:
                diff = next((k for k in range(min(len(old), len(m['content']))) if old[k] != m['content'][k]), 0)
                example('normalize', repr(old[max(0, diff - 60):diff + 80]), repr(m['content'][max(0, diff - 60):diff + 80]),
                        'first difference shown, repr() makes invisible characters visible')
        d['reasoning'] = normalize(d['reasoning'], counts)
        changed += json.dumps(d, ensure_ascii=False) != before
    stage_log('normalize', len(docs), len(docs), c_in, sum(map(doc_chars, docs)),
              dict(rows_changed=changed, **dict(counts.most_common())))

    # 3. quality
    c_in = sum(map(doc_chars, docs)); kept, drops = [], Counter()
    for d in docs:
        why = quality(d)
        if why:
            drops[why] += 1
            example('quality', d['messages'][-2]['content'][:250], d['messages'][-1]['content'][:250], why, limit=6)
        else:
            kept.append(d)
    stage_log('quality', len(docs), len(kept), c_in, sum(map(doc_chars, kept)), dict(drops))
    docs = kept

    # 4. dedup
    c_in = sum(map(doc_chars, docs)); seen = {}; kept = []; exact = 0
    for d in docs:
        h = hashlib.sha256(json.dumps(d['messages'], ensure_ascii=False).encode()).hexdigest()
        if h in seen: exact += 1; continue
        seen[h] = 1; kept.append(d)
    docs = kept
    sh = [shingles(d['messages'][-2]['content']) for d in docs]
    sigs = [minhash(s) for s in sh]
    buckets = defaultdict(list)
    for i, sg in enumerate(sigs):
        for b in range(BANDS):
            buckets[(b, tuple(sg[b * ROWS:(b + 1) * ROWS]))].append(i)
    parent = list(range(len(docs)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    cand = verified = 0; sims = []
    for members in buckets.values():
        for j in members[1:]:
            i = members[0]; cand += 1
            jac = jaccard(sh[i], sh[j])
            if jac >= 0.7:
                verified += 1; sims.append(jac)
                a, b = find(i), find(j)
                if a != b: parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for i in range(len(docs)): groups[find(i)].append(i)
    keep_idx = set()
    for root, mem in groups.items():
        best = max(mem, key=lambda k: len(docs[k]['messages'][-1]['content']))   # keep the fullest answer
        keep_idx.add(best)
        if len(mem) > 1:
            other = next(k for k in mem if k != best)
            example('dedup', docs[best]['messages'][-2]['content'][:300], docs[other]['messages'][-2]['content'][:300],
                    f'near-duplicate prompts, Jaccard {jaccard(sh[best], sh[other]):.2f}; kept the one with the fuller answer')
    REPORT['dedup_jaccard_hist'] = Counter(round(x, 1) for x in sims)
    low = sorted((jaccard(sh[i], sh[j]), i, j) for root, mem in groups.items() if len(mem) > 1
                 for i, j in [(mem[0], mem[1])])[:3]
    REPORT['dedup_lowest_pairs'] = [dict(jaccard=round(jv, 3), a=docs[i]['messages'][-2]['content'][:300],
                                         b=docs[j]['messages'][-2]['content'][:300]) for jv, i, j in low]
    near = len(docs) - len(keep_idx)
    docs = [d for i, d in enumerate(docs) if i in keep_idx]
    stage_log('dedup', len(docs) + near + exact, len(docs), c_in, sum(map(doc_chars, docs)),
              dict(exact_duplicates=exact, near_duplicate_prompts_removed=near, lsh_candidate_pairs=cand,
                   verified_pairs=verified, clusters=sum(1 for m in groups.values() if len(m) > 1),
                   minhash=f'{NUM_PERM} perms, {BANDS} bands x {ROWS} rows, 5-word shingles, verify Jaccard >= 0.70'))

    # 5. language
    langs = Counter()
    for d in docs:
        d['language'] = detect_language(d['messages'][-2]['content'] + ' ' + d['messages'][-1]['content'])
        langs[d['language']] += 1
    for d in docs:
        if d['language'] not in ('English',):
            example('language', d['messages'][-2]['content'][:250], d['language'], 'dataset card says "en"; detector disagrees', limit=5)
    stage_log('language', len(docs), len(docs), sum(map(doc_chars, docs)), sum(map(doc_chars, docs)), dict(langs.most_common()))

    # 6. pii
    c_in = sum(map(doc_chars, docs)); counts = Counter(); rows = 0
    for d in docs:
        before = json.dumps(d, ensure_ascii=False)
        for m in d['messages']:
            old = m['content']; m['content'] = scrub(m['content'], counts)
            if old != m['content']:
                k = next(i for i in range(len(old)) if i >= len(m['content']) or old[i] != m['content'][i])
                example('pii', old[max(0, k - 80):k + 60], m['content'][max(0, k - 80):k + 60], 'masked with a typed placeholder')
        d['reasoning'] = scrub(d['reasoning'], counts)
        rows += json.dumps(d, ensure_ascii=False) != before
    stage_log('pii', len(docs), len(docs), c_in, sum(map(doc_chars, docs)), dict(rows_changed=rows, **dict(counts.most_common())))

    # 7. decontamination
    c_in = sum(map(doc_chars, docs)); evals = load_evals()
    index = {}
    for name, items in evals.items():
        for k, q in enumerate(items):
            for g in ngrams(q): index.setdefault(g, (name, k))
    kept, hits = [], Counter()
    for d in docs:
        g = ngrams(d['messages'][-2]['content'] + ' ' + d['messages'][-1]['content'])
        found = [index[x] for x in g if x in index]
        if found:
            name, k = Counter(found).most_common(1)[0][0]; hits[name] += 1
            example('decontam', d['messages'][-2]['content'][:300], evals[name][k][:300],
                    f'shares {len(found)} 13-grams with {name} item {k}', limit=6)
        else:
            kept.append(d)
    stage_log('decontam', len(docs), len(kept), c_in, sum(map(doc_chars, kept)),
              dict(eval_items_indexed={k: len(v) for k, v in evals.items()}, rows_removed_by_benchmark=dict(hits)))
    docs = kept

    # 8. manifest + tokens
    from tokenizers import Tokenizer
    try:
        tk = Tokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); tk_name = 'Qwen/Qwen2.5-0.5B'
    except Exception:
        tk = Tokenizer.from_file('../session-2/tokenizer.json'); tk_name = 'session-2 BPE'
    def tok_count(texts_): return sum(len(e.ids) for e in tk.encode_batch(texts_))
    raw_tokens = tok_count(texts)
    out_rows = []
    for d in docs:
        rec = {'messages': d['messages'], 'reasoning': d['reasoning'], 'has_reasoning': d['has_reasoning'],
               'language': d['language'], 'raw_index': d['raw_index']}
        rec['id'] = hashlib.sha256(json.dumps(rec['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        out_rows.append(rec)
    out_rows.sort(key=lambda r: r['raw_index'])
    payload = '\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out_rows) + '\n'
    with gzip.GzipFile('data_clean/clean.jsonl.gz', 'wb', mtime=0) as f: f.write(payload.encode())
    clean_tokens = tok_count([' '.join(m['content'] for m in r['messages']) + ' ' + r['reasoning'] for r in out_rows])
    manifest = dict(
        name='era-v5-s4-opus47-reasoning-clean', source=SRC_URL,
        upstream='https://huggingface.co/datasets/lordx64/reasoning-distill-claude-opus-4-7-max',
        licence='apache-2.0 (as declared by the source; responses are model-generated, prompts reused from earlier corpora)',
        contributor='rajaasaravanan96', cleaning_script='session-4/clean.py', cleaning_script_sha256=script_hash,
        raw_parquet_sha256=REPORT['raw']['parquet_sha256'], content_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        rows=len(out_rows), chars=sum(doc_chars(r) for r in out_rows), tokens=clean_tokens, raw_tokens=raw_tokens,
        tokenizer=tk_name, languages=dict(Counter(r['language'] for r in out_rows)),
        rows_with_reasoning=sum(r['has_reasoning'] for r in out_rows),
        format='one JSON object per line: messages[{role,content}], reasoning, language, id (sha256 of messages)',
        status='ok')
    json.dump(manifest, open('data_clean/manifest.json', 'w'), indent=1)
    REPORT['manifest'] = manifest
    REPORT['seconds'] = round(time.time() - t0, 1)
    REPORT['examples'] = dict(REPORT['examples'])
    print(f"final: {len(out_rows)} rows, {manifest['chars']:,} chars, {clean_tokens:,} tokens ({tk_name}); "
          f"content sha256 {manifest['content_sha256'][:16]}...  ({REPORT['seconds']}s)")
    return manifest


def harm_test():
    """The medicine must not change more than 1% of good documents, and must never remove ZWJ/ZWNJ."""
    hi = open('../session-2/corpus/hi.faithful.txt', encoding='utf-8').read().split('\n\n')
    en = open('../session-2/corpus/en.faithful.txt', encoding='utf-8').read().split('\n\n')
    good = [d.strip() for d in hi + en if len(d.strip()) > 200]
    indic = ['\u0c38\u0c4d\u200c\u0c24\u0c4d\u0c30\u0c40',        # Telugu with ZWNJ
             '\u0d28\u0d4d\u200d \u0d15\u0d47\u0d30\u0d33\u0d02',  # Malayalam chillu with ZWJ
             '\u0915\u094d\u200d\u0937 \u0930\u094d\u200d\u092f', # Hindi half-forms with ZWJ
             '\u09b0\u200d\u09cd\u09af\u09be\u0995',               # Bengali ya-phala with ZWJ
             'family emoji \U0001F468\u200d\U0001F469\u200d\U0001F467']
    code = ['def f(x):\n    if x &lt; 2:\n        return  x\n', '```python\nprint("a  b")\nx = 1 &amp; 2\n```']
    docs = good + indic + code
    changed, zw_before, zw_after = 0, 0, 0
    noise_ops = ('byte-order mark removed', 'zero-width space removed', 'control character removed', 'word joiner removed',
                 'mojibake repaired', 'html entities unescaped (prose only)', 'replacement character removed',
                 'private-use character removed', 'soft hyphen removed')
    harmful, reasons = 0, Counter()
    for d in docs:
        c = Counter(); out = scrub(normalize(d, c), c)
        changed += out != d
        if out != d:
            ops = [k for k in c if 'kept' not in k]; reasons.update(ops)
            harmful += any(k not in noise_ops for k in ops)
        zw_before += d.count('\u200c') + d.count('\u200d'); zw_after += out.count('\u200c') + out.count('\u200d')
    code_ok = all(normalize(x, Counter()) == x for x in code[1:])
    res = dict(good_docs=len(docs), changed=changed, changed_pct=round(100 * changed / len(docs), 3),
               changed_by=dict(reasons), harmful_changes=harmful, harmful_pct=round(100 * harmful / len(docs), 3),
               zwj_zwnj_before=zw_before, zwj_zwnj_after=zw_after, code_block_untouched=code_ok,
               passed=harmful / len(docs) <= 0.01 and zw_before == zw_after and code_ok)
    print('harm test:', res)
    return res


if __name__ == '__main__':
    m1 = run()
    REPORT['harm_test'] = harm_test()
    json.dump(REPORT, open('report.json', 'w'), indent=1, ensure_ascii=False)
    if '--check-determinism' in sys.argv:
        first = m1['content_sha256']
        REPORT.clear(); REPORT.update({'stages': [], 'examples': defaultdict(list)})
        m2 = run()
        print('determinism:', 'PASS identical content hash' if m2['content_sha256'] == first else 'FAIL')
        rep = json.load(open('report.json')); rep['determinism'] = dict(run1=first, run2=m2['content_sha256'],
                                                                         identical=m2['content_sha256'] == first)
        json.dump(rep, open('report.json', 'w'), indent=1, ensure_ascii=False)
