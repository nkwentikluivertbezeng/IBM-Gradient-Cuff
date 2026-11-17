"""Use Gradient Cuff as a live guard: type prompts, get accept/reject plus the model's reply.

Needs a calibrated threshold, i.e. run_eval.py must have been run for the same model first
(it writes results/<model>/threshold.json).

usage: python live_demo.py --model Qwen/Qwen2.5-1.5B-Instruct
       python live_demo.py --model meta-llama/Llama-3.2-3B-Instruct --threshold 30
Type a prompt and press Enter. An empty line or Ctrl+C quits.
"""
import argparse
import json
import re
import time
from pathlib import Path

from gradient_cuff_lab import GradientCuffDetector, apply_threshold, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--threshold", type=float, default=None,
                help="override the calibrated threshold from results/<model>/threshold.json")
ap.add_argument("--max-new-tokens", type=int, default=256, help="length of the reply shown to the user")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

threshold = args.threshold
if threshold is None:
    path = Path("results") / re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-") / "threshold.json"
    if not path.exists():
        raise SystemExit(f"No calibrated threshold at {path}. Run run_eval.py for this model "
                         "first, or pass --threshold.")
    threshold = json.loads(path.read_text("utf-8"))["threshold"]

model, tok = load_model(args.model)
det = GradientCuffDetector(model, tok, batch_size=100, response_max_new_tokens=args.max_new_tokens)
print(f"\nGradient Cuff live guard | {args.model} | threshold {threshold:.2f}")
print("Type a prompt (empty line to quit).\n")

REFUSAL = "I'm sorry, but I can't help with that request."
while True:
    try:
        prompt = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt:
        break
    t = time.time()
    r = apply_threshold(det.evaluate_prompt(prompt, seed=args.seed), threshold)
    if r["stage_one_rejected"]:
        verdict = f"REJECTED at stage 1 (refusal rate {r['sample_refusal_rate']:.1f} > 0.5)"
    elif r["stage_two_rejected"]:
        verdict = f"REJECTED at stage 2 (gradient norm {r['gradient_norm']:.1f} > {threshold:.1f})"
    else:
        verdict = (f"ACCEPTED (refusal rate {r['sample_refusal_rate']:.1f}, "
                   f"gradient norm {r['gradient_norm']:.1f})")
    print(f"[guard] {verdict}  [{time.time() - t:.1f}s]")
    print("model>", REFUSAL if r["detector_rejected"] else r["raw_response"], "\n")
