"""Checks every arXiv date in mechanisms.js against the arXiv API (the v1 'published' date).

Run: python verify_dates.py
"""
import re, time, urllib.request

src = open('mechanisms.js', encoding='utf-8').read()
entries = re.findall(r"name:'([^']+)', date:'([\d-]+)'(?:, arxiv:'([\d.]+)')?", src)
released = set(re.findall(r"name:'([^']+)'[^\n]*arxiv:'[\d.]+', src:", src))   # dated by an earlier official release
ids = [a for _, _, a in entries if a]
xml = urllib.request.urlopen('https://export.arxiv.org/api/query?max_results=100&id_list=' + ','.join(ids), timeout=60).read().decode()
published = {}
for e in xml.split('<entry>')[1:]:
    i = re.search(r'<id>https?://arxiv.org/abs/([\d.]+)v\d+</id>', e).group(1)
    published[i] = re.search(r'<published>([\d-]+)T', e).group(1)
bad = 0
for name, date, a in entries:
    if not a:
        print(f'  (no arXiv id)  {date:10s}  {name}  -> see the date note on its card'); continue
    ok = published.get(a) == date
    if not ok and name in released and date < published.get(a, ''):
        print(f'OK   {a:11s} {date}  released before the arXiv report ({published.get(a)})  {name}'); continue
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'}  {a:11s} {date}  arXiv says {published.get(a)}  {name}")
print(f'{len(entries)} mechanisms, {len(ids)} arXiv dates checked, {bad} mismatches')
