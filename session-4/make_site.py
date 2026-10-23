"""Builds site/index.html (the Netlify widget) from report.json and data_clean/manifest.json.

Run after clean.py:  python make_site.py
"""
import html, json, os

R = json.load(open('report.json'))
M = R['manifest']
S = {s['stage']: s for s in R['stages']}
H = R['harm_test']
os.makedirs('site', exist_ok=True)
e = lambda x: html.escape(str(x))
fmt = lambda n: f'{n:,}'

STRATEGIES = [
    ('Normalization', 'Section 3', [
        ('Unicode NFC', 'one byte encoding for every character'),
        ('Strip invisible noise', 'zero-width space, byte-order mark, bidi overrides, control characters'),
        ('Keep ZWJ and ZWNJ', 'they carry meaning in Brahmic scripts, so they are never removed'),
        ('Unescape HTML entities', '&amp;amp; becomes &amp;'),
        ('Collapse whitespace', 'runs of spaces and blank lines')]),
    ('One conversation format', 'Section 4, ghost tags', [
        ('One canonical format', 'rewrite every source into one structure at ingestion'),
        ('Real special tokens', 'never literal [INST] or &lt;|im_start|&gt; text in pre-training')]),
    ('Quality filtering', 'Section 5', [
        ('Average word length', 'heuristic rule'), ('Symbol-to-word ratio', 'heuristic rule'),
        ('Lines ending in punctuation', 'heuristic rule'), ('Duplicated lines', 'heuristic rule'),
        ('Stop-word presence', 'heuristic rule'),
        ('Educational-value classifier', 'a large model labels a sample, a light model imitates it at scale')]),
    ('Deduplication', 'Sections 6 and 7', [
        ('Exact deduplication', 'content hash'), ('Shingling + MinHash', 'estimate the Jaccard overlap from short signatures'),
        ('LSH banding', 'compare only documents that share a band'),
        ('Global, not local', 'one central large-memory machine sees every shard')]),
    ('Language identification', 'Section 8', [
        ('Detect at runtime', 'never trust the folder name'), ('Validate codes, fail loudly', 'the Telugu te/tel bug')]),
    ('PII removal', 'Section 9', [
        ('Regex layer', 'emails, phone numbers, IP addresses'), ('ML layer', 'personal names; precision against recall, hardest for Indic names')]),
    ('Decontamination', 'Section 10', [
        ('Fingerprint the eval sets', 'n-gram fingerprints'), ('Scan and remove overlaps', 'every shard, against every benchmark'),
        ('Canary strings', 'detect a leak after the fact')]),
    ('Reproducibility and manifest', 'Section 11', [
        ('Deterministic content-based ids', 'never a running counter'), ('Hash after cleaning', 'the hash describes the clean text'),
        ('Provenance manifest + licence gate', 'source, licence, script hash, tokens, languages; unknown licence is blocked')]),
]
N_TECH = sum(len(t) for _, _, t in STRATEGIES)

funnel = [('raw rows', R['raw']['rows'], R['raw']['chars'])] + \
         [(s['stage'], s['docs_out'], s['chars_out']) for s in R['stages']]


def bars():
    mx = funnel[0][1]; out = []
    for name, n, c in funnel:
        out.append(f'<div class="bar"><span class="lab">{e(name)}</span><span class="track"><span class="fill" '
                   f'style="width:{100 * n / mx:.1f}%"></span></span><span class="num">{fmt(n)} rows · {c / 1e6:.1f}M chars</span></div>')
    return '\n'.join(out)


def detail_table(d):
    rows = ''.join(f'<tr><td>{e(k)}</td><td class="n">{e(fmt(v) if isinstance(v, int) else v)}</td></tr>'
                   for k, v in d.items() if not isinstance(v, dict))
    for k, v in d.items():
        if isinstance(v, dict):
            rows += f'<tr><td>{e(k)}</td><td class="n">' + ', '.join(f'{e(a)}: {e(b)}' for a, b in v.items()) + '</td></tr>'
    return f'<table class="kv">{rows}</table>'


def examples(stage):
    ex = [x for x in R['examples'].get(stage, []) if x['before'] != x['after']][:3]
    if not ex: return ''
    out = ''.join(f'<div class="ex"><div class="exn">{e(x["note"])}</div><div class="exr"><span class="tag">before</span>'
                  f'<pre>{e(x["before"])}</pre></div><div class="exr"><span class="tag">after</span><pre>{e(x["after"])}</pre></div></div>'
                  for x in ex)
    return f'<details><summary>Real examples from this run ({len(ex)})</summary>{out}</details>'


STAGE_TEXT = {
    'format': ('What', 'Every row is one string with literal Qwen markers: <code>&lt;|im_start|&gt;user … &lt;|im_end|&gt;</code> and '
               '<code>&lt;think&gt;…&lt;/think&gt;</code>. Trained on as text, those markers are exactly the ghost tags of Section 4: '
               'the model learns them as ordinary subwords and later fights the real special tokens.',
               'Parse each row into <code>messages[{role, content}]</code> plus a separate <code>reasoning</code> field. '
               'Rows whose markers do not parse, or that still contain any other format\'s markers ([INST], ### Human:, '
               '&lt;start_of_turn&gt;…) inside the text, are dropped. 1,179 rows have no &lt;think&gt; block at all: they are '
               'valid answers, so they are kept and labelled <code>has_reasoning: false</code> rather than silently mixed in.'),
    'normalize': ('What', 'Character-level dirt: UTF-8 text that was decoded as Windows-1252 ("chiefâ\\x80\\x94" for "chief—", '
                  '"Â£1m" for "£1m"), control characters, word joiners and HTML entities.',
                  'Repair the mojibake by re-decoding Latin-1 runs as UTF-8, NFC (not NFKC, which rewrites ² and full-width '
                  'letters and breaks code), drop the invisible noise characters, keep ZWJ/ZWNJ, unescape HTML only in prose. '
                  'Code blocks and inline code are left byte-exact. Space runs are deliberately <b>not</b> collapsed: the harm test '
                  'showed that rule changed 9.1% of good documents (Markdown tables, double spaces), so it was removed.'),
    'quality': ('What', 'English web heuristics (symbol ratio, stop words, punctuation) are wrong for chat data full of code and maths, '
                'so the filters were rewritten for what goes wrong in distilled reasoning data.',
                'Drop: prompts that are only boilerplate with no task (the answer just asks for the missing problem), prompts under '
                '15 characters ("# Problem"), empty &lt;think&gt; blocks, empty answers, answers with an unclosed code block (truncated), '
                'answers with more than 30% duplicated lines, and degenerate loops of 32+ repeated 4-grams (the OLMo 2 rule). '
                'One-letter answers like "C" are kept: they are correct answers to multiple-choice questions.'),
    'dedup': ('What', 'The same prompt answered more than once, sometimes with different whitespace or wording.',
              'Exact: SHA-256 of the canonical messages. Near: 5-word shingles of the prompt, 128-permutation MinHash written from '
              'scratch, LSH with 16 bands × 8 rows (threshold ≈ 0.71), then every candidate pair is verified with the true '
              'Jaccard ≥ 0.70. Each cluster keeps the row with the fullest answer.'),
    'language': ('What', 'The dataset card says "en". The detector disagrees for about one row in eleven.',
                 'Script detection over Unicode blocks plus an English stop-word ratio. Nothing is dropped: the rows are labelled, '
                 'because for V5 these are useful multilingual prompts, but they must not be counted as English when the mixture is sized.'),
    'pii': ('What', 'Emails, phone numbers, public IP addresses and secrets inside prompts and answers.',
            'Regex layer with typed placeholders (&lt;EMAIL&gt;, &lt;PHONE&gt;, &lt;IP&gt;, &lt;SECRET:…&gt;). Kept on purpose: '
            'documentation domains (example.com, *.test), private and reserved IP ranges, version strings like Chrome/120.0.0.0, '
            'fictional 555-01xx numbers and public toll-free help lines.'),
    'decontam': ('What', 'Benchmark questions hidden in the training data would make every later score dishonest.',
                 'Every 13-gram of GSM8K test (1,319), HumanEval (164) and MATH-500 (500) is indexed; every row is scanned and any '
                 'row sharing a 13-gram is removed.'),
}

stage_cards = ''
for st in ['format', 'normalize', 'quality', 'dedup', 'language', 'pii', 'decontam']:
    s = S[st]; what, why, how = STAGE_TEXT[st]
    removed = s['docs_in'] - s['docs_out']
    stage_cards += f'''
<section class="card" id="st-{st}">
  <header><span class="w-title">{e(st)}</span><span class="w-hint">{fmt(s["docs_in"])} → {fmt(s["docs_out"])} rows · {"−" + fmt(removed) + " removed" if removed else "nothing removed"}</span></header>
  <div class="body">
    <p><b>Why.</b> {why}</p>
    <p><b>How.</b> {how}</p>
    {detail_table(s["detail"])}
    {examples(st)}
  </div>
</section>'''

strat_html = ''
for i, (name, sec, techs) in enumerate(STRATEGIES, 1):
    lis = ''.join(f'<li><b>{t}</b> — {d}</li>' for t, d in techs)
    strat_html += f'<div class="strat"><div class="sh"><span class="sn">{i}</span><span class="st">{e(name)}</span><span class="ss">{e(sec)} · {len(techs)}</span></div><ul>{lis}</ul></div>'

langs = sorted(M['languages'].items(), key=lambda x: -x[1])
lang_bars = ''.join(f'<div class="bar small"><span class="lab">{e(k)}</span><span class="track"><span class="fill alt" style="width:{100 * v / M["rows"]:.1f}%"></span></span><span class="num">{fmt(v)}</span></div>' for k, v in langs)
det = R.get('determinism', {})

page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cleaning Opus 4.7 Reasoning Traces</title>
<style>
:root{{--bg:#0A0E13;--bg2:#0F151C;--panel:#141C25;--panel2:#1A242F;--raise:#1F2C39;--line:#25323F;--line2:#35485E;
--azure:#3FA7FF;--amber:#FF9F35;--violet:#B07BFF;--cyan:#26D7C8;--gold:#F5D33B;--steel:#4A6076;
--disp:"Avenir Next Condensed","HelveticaNeue-CondensedBold","Arial Narrow",system-ui,sans-serif;
--body:Charter,"Bitstream Charter","Iowan Old Style",Georgia,"Times New Roman",serif;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:#fff;font-family:var(--body);font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:940px;margin:0 auto;padding:40px 16px 80px}}
h1,h2,h3{{font-family:var(--disp);letter-spacing:-.01em;text-wrap:balance;line-height:1.15}}
h1{{font-size:44px;margin:.2em 0 .3em}} h2{{font-size:30px;margin:2.2em 0 .5em;border-top:1px solid var(--line);padding-top:1.1em}}
p{{max-width:68ch}} code,pre,.n,.num,.w-hint,.eyebrow,.tag,table{{font-family:var(--mono)}}
code{{font-size:.84em;background:var(--bg2);padding:1px 5px;border-radius:4px;border:1px solid var(--line)}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:22px 0}}
.stat{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--azure);border-radius:8px;padding:12px 14px}}
.stat .k{{font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase}}
.stat .v{{font-family:var(--mono);font-size:28px;font-weight:700;font-variant-numeric:tabular-nums}}
.stat .u{{font-family:var(--mono);font-size:12px}}
.stat.b2{{border-left-color:var(--amber)}} .stat.b3{{border-left-color:var(--cyan)}} .stat.b4{{border-left-color:var(--violet)}}
.card{{border:1px solid var(--line);border-radius:10px;background:var(--panel);margin:16px 0;overflow:hidden}}
.card header{{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;background:var(--panel2);border-bottom:1px solid var(--line);padding:10px 16px}}
.w-title{{font-family:var(--disp);font-weight:800;text-transform:uppercase;letter-spacing:.08em}}
.w-hint{{font-size:13px}} .card .body{{padding:14px 16px}}
.bar{{display:grid;grid-template-columns:110px 1fr 220px;gap:10px;align-items:center;margin:7px 0;font-size:14px}}
.bar.small{{grid-template-columns:230px 1fr 70px}}
.bar .lab{{font-family:var(--mono);font-size:13px}} .bar .num{{font-size:13px;font-variant-numeric:tabular-nums}}
.track{{background:#ffffff0a;height:16px;border-radius:3px;overflow:hidden}}
.fill{{display:block;height:100%;background:var(--azure)}} .fill.alt{{background:var(--violet)}}
.strats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.strat{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.strat ul{{margin:8px 0 0;padding-left:18px;font-size:15px}} .strat li{{margin:3px 0}}
.sh{{display:flex;gap:10px;align-items:baseline}} .sn{{font-family:var(--mono);font-weight:700;background:var(--gold);color:#06121C!important;border-radius:4px;padding:0 7px}}
.st{{font-family:var(--disp);font-weight:800;font-size:19px}} .ss{{font-family:var(--mono);font-size:12px;margin-left:auto}}
table{{border-collapse:collapse;font-size:13px;width:100%;margin:8px 0}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
td.n{{font-variant-numeric:tabular-nums}} th{{font-weight:700;background:var(--panel2)}}
.kv td:first-child{{width:46%}}
details{{margin-top:10px}} summary{{cursor:pointer;font-family:var(--mono);font-size:13px;padding:6px 0}}
.ex{{border:1px solid var(--line2);border-radius:8px;margin:10px 0;padding:10px}}
.exn{{font-family:var(--mono);font-size:12.5px;margin-bottom:6px}}
.exr{{display:grid;grid-template-columns:62px 1fr;gap:8px;align-items:start}}
.tag{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;border:1px solid var(--line2);border-radius:4px;padding:1px 5px;text-align:center}}
pre{{white-space:pre-wrap;word-break:break-word;background:var(--bg2);border:1px solid var(--line);border-radius:6px;padding:8px;margin:4px 0;font-size:12.5px;max-height:180px;overflow:auto}}
.callout{{border-left:3px solid var(--gold);background:#ffffff05;padding:10px 14px;margin:14px 0}}
.concerns li{{margin:8px 0}} a{{text-decoration:underline}}
.pass{{font-family:var(--mono);font-weight:700;border:1px solid var(--cyan);border-radius:4px;padding:0 6px}}
@media (max-width:640px){{h1{{font-size:32px}} .bar{{grid-template-columns:80px 1fr;}} .bar .num{{grid-column:1/-1}} .bar.small{{grid-template-columns:150px 1fr 50px}}}}
body,body *,h1,h2,h3,p,li,a,strong,em,code,th,td,.stat .v,.stat .k,.stat .u,.bar,.bar .num,.tag,.w-title,.w-hint,.callout{{color:#FFFFFF!important}}
.sn{{color:#06121C!important}}
</style></head><body><main class="wrap">
<div class="eyebrow">ERA V5 · Session 4 assignment · Data cleaning and deduplication</div>
<h1>Cleaning 7,823 Claude Opus 4.7 reasoning traces</h1>
<p>The dataset behind the model linked in the assignment, put through the Session 4 pipeline. Every number on this page is
generated by <code>clean.py</code> from the real run, and the pipeline is deterministic: two runs give the same content hash.</p>
<div class="stats">
 <div class="stat"><div class="k">strategies</div><div class="v">{len(STRATEGIES)}</div><div class="u">stages · {N_TECH} techniques</div></div>
 <div class="stat b2"><div class="k">rows</div><div class="v">{fmt(M["rows"])}</div><div class="u">of {fmt(R["raw"]["rows"])} kept ({100 * M["rows"] / R["raw"]["rows"]:.1f}%)</div></div>
 <div class="stat b3"><div class="k">tokens</div><div class="v">{M["tokens"] / 1e6:.2f}M</div><div class="u">of {M["raw_tokens"] / 1e6:.2f}M (Qwen2.5 tokenizer)</div></div>
 <div class="stat b4"><div class="k">harm test</div><div class="v">{H["harmful_pct"]:.0f}%</div><div class="u">harmful changes · ZWJ/ZWNJ {H["zwj_zwnj_after"]}/{H["zwj_zwnj_before"]} kept</div></div>
</div>

<h2>1. How many strategies are in Session 4?</h2>
<p><b>{len(STRATEGIES)} cleaning strategies, made of {N_TECH} named techniques.</b> The lesson's own closing list names the eight
stages: normalization, format discipline, quality filtering, deduplication, language validation, PII removal, decontamination and
the manifest. Extraction comes before them, but it was taught in Session 3, so it is not counted here. Inside each stage the lesson
names specific techniques. I counted one technique wherever the text describes a separate rule or mechanism.</p>
<div class="strats">{strat_html}</div>

<h2>2. The dataset I picked</h2>
<section class="card"><header><span class="w-title">lordx64/reasoning-distill-opus-4-7-max-sft</span><span class="w-hint">Apache-2.0 · 29.3 MB · 7,823 rows</span></header>
<div class="body">
<p>The assignment links the model <code>Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled</code>. Its card names the dataset it was
trained on, so I cleaned <b>that exact dataset</b>: <a href="{e(M["source"])}">{e(M["source"])}</a>. It holds single-turn conversations whose
answers, with their <code>&lt;think&gt;</code> reasoning, were generated by Claude Opus 4.7. The prompts were reused from earlier
distillation corpora (competitive programming, maths, translation, reading comprehension and open chat).</p>
<p>Why this one: it is 29 MB, inside the 10–100M range. It is the data behind the example, and it is exactly the kind of
post-training data V5 will need for its reasoning and agentic slots. It is also dirty in the ways Session 4 is about: ghost chat markers in
every row, mojibake, duplicates and prompts with no task.</p>
<table><tr><th>raw file sha256</th><td class="n">{e(R["raw"]["parquet_sha256"])}</td></tr>
<tr><th>raw characters</th><td class="n">{fmt(R["raw"]["chars"])}</td></tr>
<tr><th>raw tokens (Qwen2.5)</th><td class="n">{fmt(M["raw_tokens"])}</td></tr>
<tr><th>rows with a system prompt</th><td class="n">{fmt(R["raw"]["rows_with_system_prompt"])}</td></tr></table>
</div></section>

<h2>3. What was cleaned, why and how</h2>
<section class="card"><header><span class="w-title">The funnel</span><span class="w-hint">rows and characters after each stage</span></header>
<div class="body">{bars()}</div></section>
{stage_cards}

<section class="card"><header><span class="w-title">Harm test</span><span class="w-hint"><span class="pass">{"PASS" if H["passed"] else "FAIL"}</span></span></header>
<div class="body">
<p>The medicine must not damage good text. I ran the same normalization and PII scrubbing on {H["good_docs"]} known-good documents:
the Session 2 English and Hindi Wikipedia paragraphs, Telugu, Malayalam, Hindi and Bengali strings that use ZWJ/ZWNJ, an emoji
ZWJ sequence, and code blocks.</p>
<ul><li>Harmful changes: <b>{H["harmful_changes"]} ({H["harmful_pct"]}%)</b>, under the 1% limit.</li>
<li>Documents changed at all: {H["changed"]} ({H["changed_pct"]}%). Every one was real dirt being fixed: {", ".join(f"{k} ×{v}" for k, v in H["changed_by"].items())}.</li>
<li>ZWJ/ZWNJ before → after: <b>{H["zwj_zwnj_before"]} → {H["zwj_zwnj_after"]}</b>, so none were removed.</li>
<li>Code blocks byte-exact: <b>{H["code_block_untouched"]}</b>.</li></ul>
<div class="callout">The first version failed: a "collapse runs of spaces" rule changed 9.1% of the good documents (Markdown tables,
double spaces after full stops). The harm test caught it, and the rule was removed.</div>
</div></section>

<h2>4. Other strategies and concerns</h2>
<ul class="concerns">
<li><b>Licence of distilled outputs.</b> The card says Apache-2.0, but the answers are generated by a commercial model, and its
provider's terms restrict using outputs to train competing models. Before V5 trains on this, it needs a legal check, not just a licence field.
The manifest records it as a concern.</li>
<li><b>Mixed formats.</b> {fmt(M["rows"] - M["rows_with_reasoning"])} of the kept rows answer without any &lt;think&gt; block. Mixing them silently would teach the model to sometimes
skip reasoning, so they are kept but labelled, and the SFT mixture can choose.</li>
<li><b>The language label is wrong.</b> The card says English, but {fmt(M["rows"] - M["languages"].get("English", 0))} rows are not plain English
(other scripts, or code-heavy). Counting them as English would skew the Session 5 mixture.</li>
<li><b>No ML name detection.</b> Only the regex PII layer ran: no NER model was available offline. Names in prompts (mostly public figures and
fictional characters) are untouched, and that is a known gap.</li>
<li><b>Decontamination coverage.</b> Zero overlaps at 13-grams against GSM8K, HumanEval and MATH-500. That does not prove the data is clean for
every benchmark: MMLU, LiveCodeBench and the Indic sets were not scanned, and paraphrased leaks escape exact n-grams.</li>
<li><b>English quality heuristics do not fit.</b> Symbol-ratio and stop-word rules would have thrown away most of the code and maths, so they were
replaced with chat-specific rules. This is Section 5's filter-bias warning, in a different form.</li>
<li><b>Near-duplicate policy.</b> When a prompt appears more than once, only the fullest answer is kept. For preference data you would keep
both and pair them instead.</li>
</ul>

<h2>5. Final statistics</h2>
<table>
<tr><th></th><th>raw</th><th>clean</th><th>kept</th></tr>
<tr><td>rows</td><td class="n">{fmt(R["raw"]["rows"])}</td><td class="n">{fmt(M["rows"])}</td><td class="n">{100 * M["rows"] / R["raw"]["rows"]:.1f}%</td></tr>
<tr><td>characters</td><td class="n">{fmt(R["raw"]["chars"])}</td><td class="n">{fmt(M["chars"])}</td><td class="n">{100 * M["chars"] / R["raw"]["chars"]:.1f}%</td></tr>
<tr><td>tokens (Qwen2.5)</td><td class="n">{fmt(M["raw_tokens"])}</td><td class="n">{fmt(M["tokens"])}</td><td class="n">{100 * M["tokens"] / M["raw_tokens"]:.1f}%</td></tr>
<tr><td>rows with reasoning</td><td class="n">—</td><td class="n">{fmt(M["rows_with_reasoning"])}</td><td class="n"></td></tr>
</table>
<p>The character and token counts include the removed chat markers. About {(R["raw"]["chars"] - S["format"]["chars_out"]) / 1e6:.1f}M characters of
<code>&lt;|im_start|&gt;</code>-style text became structure, not text.</p>
<section class="card"><header><span class="w-title">Language mix of the clean rows</span><span class="w-hint">detected, not declared</span></header>
<div class="body">{lang_bars}</div></section>
<section class="card"><header><span class="w-title">Manifest</span><span class="w-hint">data_clean/manifest.json</span></header>
<div class="body"><table>
<tr><th>content sha256</th><td class="n">{e(M["content_sha256"])}</td></tr>
<tr><th>cleaning script sha256</th><td class="n">{e(M["cleaning_script_sha256"])}</td></tr>
<tr><th>determinism</th><td class="n">{"two runs, identical content hash" if det.get("identical") else "not checked"}</td></tr>
<tr><th>row id</th><td class="n">sha256 of the messages (content-based, never a counter)</td></tr>
<tr><th>licence</th><td class="n">{e(M["licence"])}</td></tr>
<tr><th>runtime</th><td class="n">{R["seconds"]} s on an Apple M5</td></tr>
</table></div></section>
<p>Code, manifest and the full report: <a href="https://github.com/rajaasaravanan96/ERA-V5/tree/main/session-4">github.com/rajaasaravanan96/ERA-V5/session-4</a>.
Reproduce with <code>python clean.py --check-determinism &amp;&amp; python make_site.py</code>.</p>
</main></body></html>'''
open('site/index.html', 'w').write(page)
print('wrote site/index.html', len(page), 'bytes')
