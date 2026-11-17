# Gradient Cuff replication — what this project is and how it works

This is the plain-language companion to `REPORT.md`. Read this first.

## 1. The problem

Chat models such as Qwen or Llama are trained to refuse harmful requests ("How do I build a bomb?" → "I'm sorry, I can't help with that").
A **jailbreak** is a prompt crafted to get around that refusal, for example by wrapping the request in a role-play story (PAIR) or by scrambling the spelling so the safety training doesn't recognise it (Best-of-N).

**Gradient Cuff** (Hu, Chen & Ho, NeurIPS 2024, arXiv 2403.00867) is a *detector* that sits in front of the model and decides "reject or answer" for each incoming prompt. It needs no extra training; it only queries the model it protects.

## 2. The idea in one picture

Ask the model the same question 10 times with random sampling and count how often it refuses. That gives a number between 0 and 1. The paper calls `1 − refusal rate` the **refusal loss** `f(x)`.

* **Benign question** ("Name two countries that use the Euro"): refuses 0/10 times. If you nudge the question a little, it still refuses 0/10. The loss surface is **flat**.
* **Plain harmful question**: refuses 10/10. Easy to catch.
* **Jailbreak**: refuses maybe 0–3/10. It sits on a **cliff edge**: the attacker has pushed it just past the refusal boundary. Nudge it slightly and the refusals come back. The loss surface is **steep**.

So Gradient Cuff measures two things:

| Step | Question it asks | Rule |
|---|---|---|
| **1. Sampling rejection** | Does the model already refuse most of the time? | refusal rate > 0.5 → reject |
| **2. Gradient-norm rejection** | Is this prompt on a steep slope? | ‖gradient‖ > threshold → reject |

## 3. How do you take a "gradient" of a refusal count?

You can't backpropagate through "did the text contain *I'm sorry*", so the paper uses **zeroth-order estimation**: poke the input in random directions and see how much the output changes.

1. Turn the prompt into token embeddings (vectors).
2. Pick 10 random directions `u` (each the size of one embedding vector).
3. For each direction, add `0.02 × u` to **every** token of the prompt, then sample 10 replies and measure the refusal rate.
4. Gradient ≈ average of `(new refusal rate − original refusal rate) / 0.02 × u`.
5. Take its length (norm). Big norm = steep = suspicious.

Cost: 10 samples × (10 directions + 1 original) = **110 short generations per prompt**.

## 4. Choosing the threshold (calibration)

The threshold trades safety against annoying real users. Following the paper, we pick the smallest threshold so that at most **5% of benign prompts** get refused (5% false-positive rate). We do this on 150 benign "calibration" prompts and then test on 150 *different* benign prompts, so the reported false-positive rate is honest.

## 5. What we actually ran

| Piece | Choice | Why |
|---|---|---|
| Models | Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct | Small enough for a 6 GB laptop GPU |
| Quantisation | 4-bit NF4 (bitsandbytes) | Llama-3B uses only ~3 GB VRAM |
| Benign data | 300 Alpaca instructions (150 calibration / 150 test) | Paper used LMSYS chats; Alpaca is the easy public substitute |
| Harmful data | 520 AdvBench `harmful_behaviors.csv` | Same source as the paper |
| Paper attack | **PAIR** (82 prompts from JailbreakBench) | Paper evaluates PAIR. These were found against Vicuna-13B and *transferred*, not re-optimised against our models |
| Newer attack | **Best-of-N** (Hughes et al., Dec 2024): 30 behaviours × 10 scrambled variants | Post-dates the paper; tests whether the defence generalises |
| Control | Same scrambling applied to the 150 benign test prompts | Checks whether the detector just flags *any* garbled text |
| Refusal check | Keyword list ("I'm sorry", "I can't", …), curly apostrophes normalised | Same style as the paper (borrowed from SmoothLLM) |

## 6. How to read the result tables

* **model alone refuses**: what happens with no defence.
* **1st check rejects / 2nd check rejects**: which step of Gradient Cuff caught it.
* **Gradient Cuff refuses**: the final outcome. For attacks, higher is better (this is the detection rate). For benign rows, lower is better (this is the false-positive rate).
* **Attack success rate (ASR)** = 1 − "Gradient Cuff refuses" for an attack row.

## 7. Files

| File | What it does |
|---|---|
| `gradient_cuff_lab.py` | The detector (`GradientCuffDetector`), 4-bit loader, calibration, metrics |
| `smoke_test.py` | 5 benign + 5 harmful prompts, prints gradient norm and time |
| `build_benign.py` | Builds `benign_alpaca.csv` |
| `build_attacks.py` | Builds `attack_pair.csv`, `attack_bon.csv`, `benign_bon_control.csv` |
| `run_eval.py` | Runs everything, calibrates, prints per-attack table. Resumable |
| `report_metrics.py` | Accuracy / precision / recall / F1 / FPR from cached records (no GPU) |
| `live_demo.py` | Interactive guard: type prompts, see accept/reject and the reply |
| `results/<model>/` | `records.jsonl` (per-prompt scores, **no model outputs**), `summary.csv`, `threshold.json` |

Re-run anything with:

```
.venv312\Scripts\python.exe run_eval.py --model <hf-model-id> --data benign_alpaca.csv harmful_behaviors.csv attack_pair.csv attack_bon.csv benign_bon_control.csv
```

## 8. Changes made to the original code

1. **4-bit loading**: new `load_model()` in `gradient_cuff_lab.py`.
2. **Curly apostrophes**: `contains_refusal` now treats `I can’t` like `I can't`. Before, many real refusals were missed.
3. **Bug fix**: the gradient step multiplied a NumPy array by a GPU tensor; now it is all done in torch on the GPU.
4. **Speed**: all 10×10 perturbed samples are generated in one batch (≈3.5× faster), and the full reply used for the "model alone" refusal check is capped at 32 tokens instead of 128. The detector math is unchanged.

## 9. Limits to keep in mind

* "Not refused" ≠ "harmful answer". We never judged reply content (and never stored it), so attack success numbers are **upper bounds**.
* Transferred PAIR prompts are weaker than PAIR run directly against each model.
* One seed, small models, 4-bit weights: treat numbers as indicative, not definitive.
