"""Clean throughput and memory benchmark on an otherwise idle machine (the training runs shared the GPU with
other jobs and the laptop slept during one of them, so their wall-clock numbers are not comparable)."""
import json, sys, time
import numpy as np, torch
sys.argv = ['x']
import rev_train as r

train, _ = r.load(); out = []
for name, variant, bs in [('standard', 'standard', 32), ('reversible', 'midpoint', 32), ('rev_maxbatch', 'midpoint', 96)]:
    torch.manual_seed(0); m = r.LM(10000, variant).to(r.DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4); rng = np.random.default_rng(0)
    for _ in range(15): r.step_fn(m, opt, r.batch(train, bs, rng))
    r.sync(); t0 = time.time(); n = 40
    with r.PeakWatch() as pw:
        for _ in range(n): r.step_fn(m, opt, r.batch(train, bs, rng))
    dt = time.time() - t0
    out.append(dict(name=name, variant=variant, batch=bs, tok_s=round(n * bs * r.CTX / dt), step_s=round(dt / n, 3),
                    peak_GiB=round(pw.peak / 2 ** 30, 2)))
    print(out[-1], flush=True)
    del m, opt; torch.mps.empty_cache()
json.dump(out, open('logs/bench_idle.json', 'w'), indent=1)
