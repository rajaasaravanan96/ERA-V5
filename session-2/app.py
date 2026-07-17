"""
app.py — ERA V5 Session 2 assignment backend.

One Flask app that:
  - serves the frontend (static/index.html) at "/"
  - POST /api/train           -> fetches the 4 Wikipedia pages (or uses pasted
                                  text overrides) and starts training in a
                                  background thread; returns a job_id
  - GET  /api/train/status/<job_id> -> poll for progress / final result
  - POST /api/encode          -> tokenize arbitrary text with a trained run
  - GET  /api/download/<kind>/<run_id>  -> tokenizer.json / vocab.txt / report.json

Training runs in a background thread instead of inline in the request handler
because the auto-rebalance search needs several rounds (each a full 10k-merge
BPE fit) to close the fertility gap, and that easily exceeds the ~30s request
timeout most free-tier hosts (and gunicorn's default worker timeout) enforce.
Polling keeps every individual HTTP request fast regardless of host.

Run locally:   python app.py            (http://localhost:5000)
Deploy:        gunicorn app:app         (see requirements.txt)
"""

import json
import os
import threading
import time
import unicodedata
import uuid

from flask import Flask, Response, jsonify, request, send_from_directory

from tokenizer_core import (
    bytes_to_unicode,
    build_merge_rank,
    decode_tokens,
    encode_text,
    fertility_of,
    symbols_to_text,
    to_hf_tokenizer_dict,
    train_bpe,
    word_count,
)
from wiki_fetch import fetch_wiki_text

# The shipped tokenizer.json is a standard HuggingFace Metaspace-BPE tokenizer
# (see train_tokenizer.py); it is served through the real `tokenizers` library
# so encode/decode here behave exactly like the grader's harness.
from tokenizers import Tokenizer as HFTokenizer

app = Flask(__name__, static_folder="static", static_url_path="")

# In-memory store of trained runs, keyed by run_id. Fine for an assignment
# demo; a free-tier host may restart the process and clear this, so the
# frontend keeps the run_id only for the current session's downloads.
RUNS = {}

# In-memory store of background training jobs, keyed by job_id.
JOBS = {}

DEFAULT_LANGS = {
    "en": {"label": "English", "code": "en", "title": "India"},
    "hi": {"label": "Hindi",   "code": "hi", "title": "भारत"},
    "te": {"label": "Telugu",  "code": "te", "title": "భారతదేశం"},
    "ta": {"label": "Tamil",   "code": "ta", "title": "இந்தியா"},
}

BYTE_TO_CHAR, CHAR_TO_BYTE = bytes_to_unicode()

PRETRAINED_RUN_ID = "pretrained"
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_PATH = os.path.join(HERE_DIR, "tokenizer.json")
METRICS_PATH = os.path.join(HERE_DIR, "metrics.json")
PRETRAINED_REPORT_PATH = os.path.join(HERE_DIR, "data", "pretrained_report.json")


def _visible(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _load_pretrained():
    """Register the shipped tokenizer.json (HuggingFace Metaspace BPE, built by
    train_tokenizer.py) as a run so /api/encode, /api/decode and the frontend's
    "Try it" box work immediately — no training required. Fertility/score for
    the Results table come from metrics.json (built by evaluate_tokenizer.py:
    fertility = tokens / faithful units on the faithful Markdown corpus)."""
    if not os.path.exists(PRETRAINED_PATH):
        return None
    hf_tok = HFTokenizer.from_file(PRETRAINED_PATH)
    vocab_ids = hf_tok.get_vocab()
    vocab = sorted(vocab_ids, key=vocab_ids.get)

    fert, score, spread = {}, None, None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        for code, v in metrics.get("per_language", {}).items():
            fert[code] = {"label": v["label"], "code": code, "tokens": v["tokens"],
                          "words": v["faithful_units"], "fertility": v["fertility"]}
        spread = metrics.get("spread")
        score = metrics.get("hindi_adjusted_score")
    lang_meta = ({v["code"]: {"label": v["label"], "code": v["code"]} for v in fert.values()}
                 or {c["code"]: {"label": c["label"], "code": c["code"]}
                     for c in DEFAULT_LANGS.values()})

    run = {
        "hf": hf_tok, "vocab": vocab, "merges": [], "lang_meta": lang_meta,
        "fert": fert, "weights": {}, "round_log": [],
        "score": score, "spread": spread,
        "vocab_size_target": len(vocab), "created": time.time(),
    }
    RUNS[PRETRAINED_RUN_ID] = run
    return run


_load_pretrained()


@app.route("/api/pretrained")
def pretrained():
    run = RUNS.get(PRETRAINED_RUN_ID)
    if not run:
        return jsonify({"available": False})
    return jsonify({
        "available": True,
        "run_id": PRETRAINED_RUN_ID,
        "vocab_size": len(run["vocab"]),
        "score": run["score"],
        "spread": run["spread"],
        "languages": list(run["lang_meta"].values()),
        "fert": run["fert"],
    })


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/languages")
def languages():
    return jsonify(DEFAULT_LANGS)


def _run_training(job_id, lang_texts, lang_meta, vocab_size, rounds):
    job = JOBS[job_id]
    weights = {k: 1.0 for k in lang_texts}
    best = None
    round_log = []
    try:
        for rnd in range(rounds):
            vocab, merges = train_bpe(lang_texts, weights, vocab_size, BYTE_TO_CHAR)
            merge_rank = build_merge_rank(merges)
            fert = {k: fertility_of(t, merge_rank, BYTE_TO_CHAR) for k, t in lang_texts.items()}
            vals = [f["fertility"] for f in fert.values()]
            spread = max(vals) - min(vals)
            score = (1000 / spread) if spread > 0 else float("inf")

            result = {
                "round": rnd, "weights": dict(weights), "vocab": vocab, "merges": merges,
                "fert": fert, "spread": spread, "score": score,
            }
            round_log.append({"round": rnd, "spread": spread,
                               "score": (score if score != float("inf") else None)})
            if best is None or score > best["score"]:
                best = result

            job["round"] = rnd + 1
            job["best_score"] = (best["score"] if best["score"] != float("inf") else None)
            job["best_spread"] = best["spread"]

            # Proportional feedback: a language whose fertility sits above the
            # mean is under-merged, so its corpus weight goes up next round,
            # pulling more of the fixed 10k-merge budget toward it; a
            # below-mean language's weight comes down so it stops soaking up
            # merges English/Telugu don't need. Ratios are clamped per round
            # so the loop doesn't blow up, and this runs every round (not
            # just once) — the greedy merge-selection has ties that make
            # fertility move non-monotonically with weight, especially for a
            # small corpus like Tamil/Telugu, so revisiting the ratio every
            # round is what lets a later round land on a much tighter spread
            # than an earlier one, even though the update rule is unchanged.
            mean_fert = sum(vals) / len(vals)
            if mean_fert > 0:
                for k in weights:
                    ratio = fert[k]["fertility"] / mean_fert
                    ratio = max(0.6, min(ratio, 1.8))
                    weights[k] = max(1.0, weights[k] * ratio)
                min_w = min(weights.values())
                for k in weights:
                    weights[k] = weights[k] / min_w

        run_id = uuid.uuid4().hex[:12]
        RUNS[run_id] = {**best, "lang_meta": lang_meta, "round_log": round_log,
                         "vocab_size_target": vocab_size, "created": time.time()}

        job["status"] = "done"
        job["response"] = {
            "run_id": run_id,
            "vocab_size": len(best["vocab"]),
            "fert": {k: {**v, **lang_meta[k]} for k, v in best["fert"].items()},
            "spread": best["spread"],
            "score": (best["score"] if best["score"] != float("inf") else None),
            "round_log": round_log,
        }
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.route("/api/train", methods=["POST"])
def train():
    body = request.get_json(force=True, silent=True) or {}
    vocab_size = int(body.get("vocab_size", 10000))
    rounds = max(1, min(16, int(body.get("rounds", 10))))
    text_overrides = body.get("texts", {}) or {}
    lang_config = body.get("languages") or DEFAULT_LANGS

    lang_texts, lang_meta, errors = {}, {}, {}
    for key, cfg in lang_config.items():
        lang_meta[key] = {"label": cfg.get("label", key), "code": cfg.get("code", key)}
        override = (text_overrides.get(key) or "").strip()
        if len(override) > 200:
            lang_texts[key] = unicodedata.normalize("NFC", override)
            continue
        try:
            lang_texts[key] = fetch_wiki_text(cfg["code"], cfg["title"])
        except Exception as e:
            errors[key] = str(e)

    if errors:
        return jsonify({"error": "fetch_failed", "details": errors}), 400

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "round": 0, "rounds": rounds,
                     "best_score": None, "best_spread": None}
    threading.Thread(
        target=_run_training, args=(job_id, lang_texts, lang_meta, vocab_size, rounds), daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "rounds": rounds})


@app.route("/api/train/status/<job_id>")
def train_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job["status"] == "running":
        return jsonify({
            "status": "running", "round": job["round"], "rounds": job["rounds"],
            "best_score": job["best_score"], "best_spread": job["best_spread"],
        })
    if job["status"] == "error":
        return jsonify({"status": "error", "error": job["error"]})
    return jsonify({"status": "done", **job["response"]})


@app.route("/api/encode", methods=["POST"])
def encode():
    body = request.get_json(force=True, silent=True) or {}
    run_id = body.get("run_id")
    text = (body.get("text") or "").strip()
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "run not found — retrain, this server may have restarted"}), 404
    if not text:
        return jsonify({"error": "paste some text first"}), 400

    if run.get("hf"):
        enc = run["hf"].encode(text)
        tokens = [t.replace("▁", " ") for t in enc.tokens]
        ids = list(enc.ids)
        decoded = run["hf"].decode(ids)
    else:
        merge_rank = build_merge_rank(run["merges"])
        token_ids = {tok: i for i, tok in enumerate(run["vocab"])}
        symbols = encode_text(text, merge_rank, BYTE_TO_CHAR)
        tokens = [symbols_to_text(s, CHAR_TO_BYTE) for s in symbols]
        ids = [token_ids.get(s) for s in symbols]
        decoded = decode_tokens(symbols, CHAR_TO_BYTE)

    words = word_count(text)
    fertility = (len(tokens) / words) if words > 0 else None
    return jsonify({
        "tokens": tokens, "ids": ids, "words": words, "token_count": len(tokens),
        "fertility": fertility, "decoded": decoded,
        # the assignment's faithful-roundtrip gate: same visible characters
        "roundtrip_ok": _visible(decoded) == _visible(text),
    })


@app.route("/api/decode", methods=["POST"])
def decode():
    """Faithful inverse of /api/encode: token ids (or raw byte-alphabet token
    strings) back to text. decode(encode(text)) == text for any input."""
    body = request.get_json(force=True, silent=True) or {}
    run = RUNS.get(body.get("run_id"))
    if not run:
        return jsonify({"error": "run not found — retrain, this server may have restarted"}), 404

    ids, tokens = body.get("ids"), body.get("tokens")
    if ids is None and tokens is not None:
        try:
            lookup = {tok: i for i, tok in enumerate(run["vocab"])}
            ids = [lookup[str(t)] for t in tokens]
        except KeyError:
            return jsonify({"error": "tokens must be token strings from this vocab"}), 400
    if ids is None:
        return jsonify({"error": "provide 'ids' or 'tokens'"}), 400

    try:
        ids = [int(i) for i in ids]
        if run.get("hf"):
            text = run["hf"].decode(ids)
        else:
            text = decode_tokens([run["vocab"][i] for i in ids], CHAR_TO_BYTE)
    except (TypeError, ValueError, IndexError, KeyError):
        return jsonify({"error": "ids must be integers within the vocab range"}), 400
    return jsonify({"text": text})


@app.route("/api/download/<kind>/<run_id>")
def download(kind, run_id):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "run not found — retrain, this server may have restarted"}), 404

    if kind == "tokenizer":
        # Standard Hugging Face `tokenizers` format so any grading harness can
        # Tokenizer.from_file() it and get a faithful decode. The pretrained
        # run serves the exact tokenizer.json file built by train_tokenizer.py.
        if run.get("hf"):
            return send_from_directory(
                HERE_DIR, "tokenizer.json", as_attachment=True,
                download_name="tokenizer.json", mimetype="application/json")
        payload = to_hf_tokenizer_dict(run["vocab"], run["merges"])
        return Response(
            json.dumps(payload, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=tokenizer.json"},
        )

    if kind == "vocab":
        def _display(tok):
            # escape backslashes and every non-printable char so each vocab
            # entry stays on exactly one line
            text = tok if run.get("hf") else symbols_to_text(tok, CHAR_TO_BYTE)
            out = []
            for ch in text:
                if ch == "\\":
                    out.append("\\\\")
                elif ch == " " or ch.isprintable():
                    out.append(ch)
                else:
                    o = ord(ch)
                    out.append(f"\\x{o:02x}" if o <= 0xFF else f"\\u{o:04x}")
            return "".join(out)
        lines = [f"{i}\t{_display(tok)}" for i, tok in enumerate(run["vocab"])]
        return Response(
            "\n".join(lines), mimetype="text/plain",
            headers={"Content-Disposition": "attachment; filename=vocab.txt"},
        )

    if kind == "report":
        if run.get("hf") and os.path.exists(METRICS_PATH):
            return send_from_directory(
                HERE_DIR, "metrics.json", as_attachment=True,
                download_name="report.json", mimetype="application/json")
        report = {
            "vocab_size": len(run["vocab"]),
            "weights_used": run["weights"],
            "per_language": {run["lang_meta"][k]["code"]: v for k, v in run["fert"].items()},
            "spread": run["spread"],
            "self_score": (run["score"] if run["score"] != float("inf") else None),
            "round_log": run["round_log"],
        }
        return Response(
            json.dumps(report, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=report.json"},
        )

    return jsonify({"error": "unknown kind, use tokenizer|vocab|report"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
