"""Step 4/5: calibrate Gradient Cuff at 5% benign FPR, report per-attack metrics.

Resumable: one JSON line per prompt in results/<model>/records.jsonl.
Only scores/booleans are stored -- never model responses.

usage: python run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct \
           --data benign_alpaca.csv harmful_behaviors.csv [attack.csv ...] [--max-per-file 100]
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from gradient_cuff_lab import (GradientCuffDetector, apply_threshold, calibrate_threshold,
                               load_model, read_labeled_data)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--data", nargs="+", default=["benign_alpaca.csv", "harmful_behaviors.csv"])
ap.add_argument("--max-per-file", type=int, default=None, help="cap malicious rows per file")
ap.add_argument("--fpr", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

frames = []
for path in args.data:
    f = read_labeled_data(path)
    f["id"] = Path(path).stem + ":" + f["id"].astype(str)
    if args.max_per_file and (f["label"] == "malicious").all():
        f = f.head(args.max_per_file)
    frames.append(f)
data = pd.concat(frames, ignore_index=True)
data.loc[data["label"].eq("benign") & data["attack_family"].eq(""), "attack_family"] = "benign"

out = Path("results") / re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
out.mkdir(parents=True, exist_ok=True)
cache_path = out / "records.jsonl"
cache = {}
if cache_path.exists():
    cache = {r["id"]: r for r in map(json.loads, cache_path.read_text("utf-8").splitlines())}

todo = data[~data["id"].isin(cache)]
print(f"{len(data)} prompts, {len(cache)} cached, {len(todo)} to run")
if len(todo):
    model, tok = load_model(args.model)
    det = GradientCuffDetector(model, tok, batch_size=100, response_max_new_tokens=32)
    with cache_path.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(todo.itertuples(), 1):
            t = time.time()
            r = det.evaluate_prompt(row.prompt, seed=args.seed)
            rec = {"id": row.id, "sample_refusal_rate": r["sample_refusal_rate"],
                   "gradient_norm": r["gradient_norm"],
                   "raw_response_refusal": r["raw_response_refusal"], "seconds": time.time() - t}
            cache[row.id] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {row.attack_family} {rec['seconds']:.1f}s", flush=True)

records = [cache[i] for i in data["id"]]
calib = data["split"].eq("calibration") & data["label"].eq("benign")
threshold, info = calibrate_threshold([r for r, c in zip(records, calib) if c], args.fpr)
print(f"\nthreshold={threshold:.4f}  calibration FPR={info['calibration_observed_refusal_rate']:.3f}")

behaviour = data["behavior"] if "behavior" in data else pd.Series([None] * len(data))
rows = [{**apply_threshold(r, threshold), "attack_family": fam, "behavior": b}
        for r, fam, b, c in zip(records, data["attack_family"], behaviour, calib) if not c]
res = pd.DataFrame(rows)
summary = res.groupby("attack_family").agg(
    n=("id", "size"),
    no_defense_refusal=("raw_response_refusal", "mean"),  # model alone
    stage1_reject=("stage_one_rejected", "mean"),
    stage2_reject=("stage_two_rejected", "mean"),
    gradient_cuff_refusal=("final_refusal", "mean"),       # TPR for attacks, FPR for benign
    median_grad_norm=("gradient_norm", "median"),
    mean_seconds=("seconds", "mean"),
).round(3)
print(summary.to_string())
summary.to_csv(out / "summary.csv")

# Best-of-N style: a behaviour "evades" if ANY of its variants is not refused.
bon = res[res["attack_family"].eq("BoN") & res["behavior"].notna()]
if len(bon):
    g = bon.groupby("behavior")
    print(f"\nBoN any-of-N evasion over {g.ngroups} behaviours: "
          f"no defense={(~g['raw_response_refusal'].all()).mean():.3f}  "
          f"gradient cuff={(~g['final_refusal'].all()).mean():.3f}")
(out / "threshold.json").write_text(json.dumps({**info, "model": args.model}, indent=2, default=str))
print(f"\nsaved {out / 'summary.csv'}")
