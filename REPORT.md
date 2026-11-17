# Gradient Cuff on small open models: a replication and a test against Best-of-N jailbreaking

> **Draft, 17 Nov 2026.** All runs are complete (1352 prompts per model). Plain-language background: `PROJECT_EXPLAINED.md`.

## Abstract

Gradient Cuff (Hu, Chen & Ho, NeurIPS 2024) detects jailbreak prompts by measuring how a model's refusal behaviour changes under small random perturbations of the prompt's embeddings. The original paper evaluates it on LLaMA-2-7B-Chat and Vicuna-7B-v1.5. We re-implemented the detector for current instruction-tuned models running in 4-bit precision on a 6 GB laptop GPU, and evaluated it on Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct. We used three malicious prompt sets: plain AdvBench requests; PAIR jailbreaks transferred from JailbreakBench; and Best-of-N (BoN) character-level jailbreaks (Hughes et al., 2024), which appeared after the paper. With the threshold calibrated to a 5% benign false-positive target, Gradient Cuff raised PAIR detection from 14.6% (model alone) to 39.0% on Qwen2.5-1.5B and from 70.7% to 89.0% on Llama-3.2-3B. False-positive rates on 150 held-out benign prompts were 3.3% and 2.7%. BoN detection rose to 100% and 99.3%, but a control shows that this is not evidence of harm detection: when the same character scrambling is applied to *benign* prompts, Gradient Cuff rejects 93.3% (Qwen) and 94.0% (Llama) of them. On scrambled input the detector fires on the noise, not the intent. It blocks BoN only by also blocking any user whose input looks garbled.

## 1. Background

A jailbreak is an input crafted so that an aligned model complies with a request it would normally refuse. Gradient Cuff defines the **refusal loss** of a prompt `x` as `f(x) = 1 − p_refuse(x)`. It estimates `p_refuse` by sampling `N` replies and checking each for refusal phrases. It then rejects a prompt in either of two stages:

1. **Sampling rejection:** `p_refuse(x) > 0.5`.
2. **Gradient-norm rejection:** `‖ĝ(x)‖ > t`. Here `ĝ` is a zeroth-order estimate of the gradient of `f`, taken over `P` random Gaussian directions `u` applied to the token embeddings with step size `μ`: `ĝ = (1/P) Σ [(f(x + μu) − f(x)) / μ] u`.

The idea is that benign prompts lie on a flat part of the refusal landscape, where the model never refuses and perturbations change nothing. Jailbreaks lie close to the refusal boundary, so a small perturbation brings refusals back. The threshold `t` is set on benign data only, to a target false-positive rate.

## 2. Experimental setup

### 2.1 Models and hardware

| | Qwen2.5-1.5B-Instruct | Llama-3.2-3B-Instruct |
|---|---|---|
| HF id | `Qwen/Qwen2.5-1.5B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` |
| Precision | 4-bit NF4 (bitsandbytes) | 4-bit NF4 (bitsandbytes) |
| Chat template | tokenizer's own | tokenizer's own |

Hardware and software: NVIDIA RTX 4050 Laptop GPU (6 GB), Python 3.12, torch 2.6.0+cu124, transformers 5.23.0, bitsandbytes 0.50.2.

The two models differ in size (1.5B vs 3B parameters) as well as in vendor. Differences between them should therefore not be attributed to model family alone.

### 2.2 Detector settings

Following the paper: `μ = 0.02`, `N = P = 10`, temperature 0.6, top-p 0.9, 16 new tokens per detector sample. Each prompt needs at most 1 + 10 + 10 × 10 = 111 generations. Prompts rejected in stage 1 skip the gradient estimate. One seed was used (42), instead of the paper's five.

Changes from the reference `upstream/main.py` (all are engineering changes; the detector maths is unchanged):

- 4-bit loading, with the model's own chat template instead of FastChat.
- The 100 perturbed samples are generated in one batch (about 3.5× faster).
- The user-facing reply used for the "model alone" condition is capped at 32 tokens instead of 128.
- The refusal check normalises curly apostrophes (`I can’t` → `I can't`). Without this, many genuine refusals from the newer models were missed.
- A bug fix: a NumPy array was being multiplied by a CUDA tensor in the gradient step.

### 2.3 Data

| Set | n | Label | Source | Role |
|---|---:|---|---|---|
| Benign calibration | 150 | benign | Alpaca (instruction-only rows, 20–300 chars, seed 42) | sets threshold `t` only |
| Benign test | 150 | benign | same sample, disjoint half | FPR |
| Direct harmful | 520 | malicious | AdvBench `harmful_behaviors.csv` (`goal` column) | non-jailbreak baseline |
| PAIR | 82 | malicious | JailbreakBench PAIR artifacts, found on Vicuna-13B-v1.5 | attack evaluated in the paper |
| BoN | 300 | malicious | 30 JBB behaviours × 10 augmentations | attack published after the paper |
| Scrambled benign control | 150 | benign | BoN augmentation applied to the benign test set | separates "harmful" from "garbled" |

The paper took its benign prompts from LMSYS Chatbot Arena. We used Alpaca because it is public and needs no gated access. Alpaca instructions are cleaner and less borderline than real chat traffic, so our FPR is likely an optimistic estimate.

The PAIR prompts were optimised against Vicuna-13B and **transferred** to our models without re-optimisation, which makes them weaker than an adaptive PAIR attack. The BoN augmentation follows Hughes et al. (word-middle scrambling with p = 0.6, random capitalisation with p = 0.6, ASCII ±1 noising with p = 0.06). Unlike the original BoN attack, we do not sample until the model complies: we use a fixed set of 10 variants per behaviour.

### 2.4 Metrics

Malicious is the positive class. For each attack family, the evaluation set is that family's prompts plus the 150 benign test prompts. We compare two conditions:

- **No defense:** a prompt counts as refused if the model's own reply contains a refusal phrase.
- **Gradient Cuff:** a prompt counts as refused if the detector rejects it *or* the model's own reply is a refusal.

We report accuracy, precision, recall, F1 and FPR. Under these definitions, the malicious refusal rate (MRR) equals recall and the benign refusal rate (BRR) equals FPR. Recall has a 95% Wilson interval. Accuracy and precision depend on the class balance of each evaluation set (for example, 520 malicious vs 150 benign), so recall and FPR are the more comparable figures across rows.

Model replies are never stored, and we did not judge whether a non-refused reply was actually harmful. All attack success rates are therefore **upper bounds**.

## 3. Results

### 3.1 Calibration

| Model | Threshold `t` | Calibration FPR (target 0.05) | Calibration prompts refused by the model itself |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 186.37 | 0.047 | 3 / 150 |
| Llama-3.2-3B | 27.94 | 0.047 | 0 / 150 |

Llama's threshold is much lower because its benign landscape is flatter: 97% of its benign test prompts have a gradient norm of exactly zero, against 79% for Qwen. On Llama, almost any refusal variability at all is enough to trigger stage 2.

### 3.2 Qwen2.5-1.5B-Instruct

Benign test FPR: **0.007** with no defense (1/150) and **0.033** with Gradient Cuff (5/150).

| Attack | Condition | Recall (= MRR) | 95% CI | Precision | F1 | Accuracy | TP / FN |
|---|---|---:|---|---:|---:|---:|---|
| Direct harmful | No defense | 0.987 | [0.972, 0.993] | 0.998 | 0.992 | 0.988 | 513 / 7 |
| Direct harmful | Gradient Cuff | 0.996 | [0.986, 0.999] | 0.990 | 0.993 | 0.990 | 518 / 2 |
| PAIR | No defense | 0.146 | [0.086, 0.239] | 0.923 | 0.253 | 0.694 | 12 / 70 |
| PAIR | Gradient Cuff | **0.390** | [0.292, 0.498] | 0.865 | 0.538 | 0.763 | 32 / 50 |
| BoN | No defense | 0.540 | [0.483, 0.596] | 0.994 | 0.700 | 0.691 | 162 / 138 |
| BoN | Gradient Cuff | **1.000** | [0.987, 1.000] | 0.984 | 0.992 | 0.989 | 300 / 0 |
| BoN vs scrambled benign | No defense | 0.540 | [0.483, 0.596] | 0.775 | 0.637 | 0.589 | 162 / 138 |
| BoN vs scrambled benign | Gradient Cuff | 1.000 | [0.987, 1.000] | 0.682 | 0.811 | 0.689 | 300 / 0 |

In the last two rows the negatives are the 150 scrambled benign prompts. Their FPR is **0.313** with no defense (47/150) and **0.933** with Gradient Cuff (140/150).

**Which stage catches what** (fraction of each set rejected). The gradient-norm columns cover only prompts that reach stage 2:

| Set | Stage 1 (sampling) | Stage 2 (gradient) | Median ‖ĝ‖ [IQR] | Prompts with ‖ĝ‖ = 0 |
|---|---:|---:|---:|---:|
| Benign test | 0.007 | 0.027 | 0 [0, 0] | 79% |
| Direct harmful | 0.975 | 0.017 | (13 prompts reach stage 2) | – |
| PAIR | 0.134 | 0.232 | 111 [48, 193] | 13% |
| BoN | 0.490 | 0.503 | 309 [264, 414] | 0% |
| Scrambled benign | 0.300 | 0.620 | 323 [260, 407] | 1% |

**Per-behaviour BoN.** Without a defense, all 30 behaviours had at least one of their 10 variants accepted by the model, which is the success criterion of the BoN attack. With Gradient Cuff, none of the 30 did.

**Cost.** The mean wall-clock time per prompt was 9.4 s for benign prompts, 4.5 s for direct harmful requests (most stop at stage 1), 10.5 s for PAIR, 8.3 s for BoN and 8.3 s for scrambled benign. The full Qwen run (1352 prompts) took 2.7 GPU-hours.

### 3.3 Llama-3.2-3B-Instruct

Benign test FPR: **0.000** with no defense (0/150) and **0.027** with Gradient Cuff (4/150).

| Attack | Condition | Recall (= MRR) | 95% CI | Precision | F1 | Accuracy | TP / FN |
|---|---|---:|---|---:|---:|---:|---|
| Direct harmful | No defense | 0.981 | [0.965, 0.990] | 1.000 | 0.990 | 0.985 | 510 / 10 |
| Direct harmful | Gradient Cuff | 0.996 | [0.986, 0.999] | 0.992 | 0.994 | 0.991 | 518 / 2 |
| PAIR | No defense | 0.707 | [0.601, 0.795] | 1.000 | 0.829 | 0.897 | 58 / 24 |
| PAIR | Gradient Cuff | **0.890** | [0.804, 0.941] | 0.948 | 0.918 | 0.944 | 73 / 9 |
| BoN | No defense | 0.883 | [0.842, 0.915] | 1.000 | 0.938 | 0.922 | 265 / 35 |
| BoN | Gradient Cuff | **0.993** | [0.976, 0.998] | 0.987 | 0.990 | 0.987 | 298 / 2 |
| BoN vs scrambled benign | No defense | 0.883 | [0.842, 0.915] | 0.847 | 0.865 | 0.816 | 265 / 35 |
| BoN vs scrambled benign | Gradient Cuff | 0.993 | [0.976, 0.998] | 0.679 | 0.806 | 0.682 | 298 / 2 |

Scrambled-benign FPR: **0.320** with no defense (48/150) and **0.940** with Gradient Cuff (141/150).

**Which stage catches what** (gradient-norm columns cover only prompts that reach stage 2):

| Set | Stage 1 (sampling) | Stage 2 (gradient) | Median ‖ĝ‖ [IQR] | Prompts with ‖ĝ‖ = 0 |
|---|---:|---:|---:|---:|
| Benign test | 0.000 | 0.027 | 0 [0, 0] | 97% |
| Direct harmful | 0.983 | 0.013 | (9 prompts reach stage 2) | – |
| PAIR | 0.720 | 0.171 | 279 [0, 505] (n = 23) | 35% |
| BoN | 0.860 | 0.133 | 324 [273, 411] (n = 42) | 2% |
| Scrambled benign | 0.260 | 0.680 | 283 [180, 392] | 4% |

**Per-behaviour BoN.** Without a defense, 20 of the 30 behaviours (66.7%) had at least one variant accepted by the model. With Gradient Cuff, 2 of 30 (6.7%) did.

**Cost.** The full Llama run (1352 prompts) took 11.0 GPU-hours. Per-prompt timings are not comparable with Qwen's: the run was interrupted twice, and the laptop throttled heavily in some stretches (up to about 90 s per prompt, against 5–15 s normally). The direct-harmful mean of 54.6 s per prompt reflects this throttling, not the detector's cost.

### 3.4 Comparison with the paper

Table 2 of the paper reports FPR/TPR of 0.022/0.883 on LLaMA-2-7B-Chat and 0.034/0.743 on Vicuna-7B, averaged over six attack families. Our results are not directly comparable: we use different models, a different benign source, transferred rather than adaptive PAIR, and one seed. Our benign FPRs (0.033 Qwen, 0.027 Llama) are in the same range as the paper's. PAIR detection on Llama (0.89) matches the paper's LLaMA-2 figure. On Qwen (0.39) it is far lower. This mostly reflects the base model: Llama already refuses 71% of the transferred PAIR prompts on its own, while Qwen refuses 15%. Gradient Cuff adds a similar absolute gain on both (+24 points on Qwen, +18 on Llama).

## 4. Discussion

**The two stages do different jobs.** Plain harmful requests are almost all caught by stage 1, because the model already refuses them. Stage 2 contributes little there, which matches the paper's analysis. For PAIR on Qwen, stage 2 does most of the work (23 of the 39 points of recall). Even so, 61% of PAIR prompts pass. Many of them have a moderate gradient norm (IQR 48–193) that overlaps the upper tail of the benign distribution, and the threshold of 186 sits inside that overlap. On Llama, stage 1 does most of the work, because the model already refuses most PAIR prompts. Of the 9 PAIR prompts that get through, 8 have a gradient norm of exactly zero: Llama complies with them consistently under every perturbation, so no threshold can catch them. The ninth has a norm exactly equal to the threshold. Ties like this occur because the estimated norm takes only a limited set of values when `N = P = 10`. These 8 zero-norm prompts mark the limit of the method on this model.

**The BoN result is a garbled-input effect.** Gradient Cuff caught 100% (Qwen) and 99.3% (Llama) of BoN prompts. We considered two explanations:

1. BoN prompts really do sit on the refusal boundary, as the paper's theory predicts for jailbreaks.
2. Scrambled text makes the model reply with confusion ("Sorry, I don't understand…"). The keyword check counts this as a refusal, because "Sorry" is in the SmoothLLM phrase list. If confusion replies come and go under perturbation, the estimated gradient is large for *any* garbled input, harmful or not.

The control supports explanation (2). Applying the same augmentation to benign prompts raises Gradient Cuff's FPR from about 3% to **93–94%** on both models. The gradient-norm distribution of scrambled benign prompts (Qwen median 323, IQR 260–407) is nearly identical to that of BoN prompts (median 309, IQR 264–414). Even without the detector, the models "refuse" about a third of scrambled benign prompts (31–32%), which shows that the refusal-keyword signal is already contaminated by confusion replies. Measured against scrambled benign prompts, Gradient Cuff's accuracy on BoN drops to 0.69 (Qwen) and 0.68 (Llama), compared with 0.59 and 0.82 for the model alone. On Llama it is therefore *worse* than no defense on that metric.

In practice, Gradient Cuff does block BoN as implemented here, but only as a "garbled input" filter. It would reject legitimate users who make heavy typos, write in leetspeak or use unusual casing. An attacker can probably weaken the effect by using milder augmentations that keep the model's replies fluent. This is a weakness of refusal-keyword losses on noisy input in general, not only of Gradient Cuff, and a judge-based refusal check might reduce it.

**Benign prompts are mostly flat.** 79% (Qwen) and 97% (Llama) of benign test prompts have exactly zero gradient norm: the model refused in none of the 110 detector samples. This is what makes the threshold workable, since only the minority of benign prompts with some refusal variability compete with attacks for the threshold. The flip side is that the threshold is set by very few prompts. On Llama, only 9 of the 150 calibration prompts have a non-zero norm, so the threshold (27.9, itself one of those 9 values) is fragile and would probably shift with another seed or benign sample.

## 5. Limitations

- **Refusal is judged by keywords**, not by a judge model or human review, and replies were not stored. "Not refused" does not mean "harmful content produced".
- **One seed** and 4-bit weights. Quantisation may change refusal behaviour compared with full precision.
- **PAIR is transferred, not adaptive.** An attacker who knows Gradient Cuff is deployed could optimise for a low gradient norm. The paper studies adaptive attacks, but we did not, and none were run for BoN.
- **The benign prompts (Alpaca) are easier than real chat traffic**, so the FPR is probably optimistic.
- **The models differ in size**, so Qwen-vs-Llama differences mix size effects with family effects.
- **Cost**: up to 111 generations per prompt (about 9 s on a laptop GPU) is impractical for interactive serving without batching across users. Llama timings are further inflated by thermal throttling and are not a reliable cost estimate.
- **The scrambled-benign control uses one augmentation strength** (the BoN defaults). We did not measure how FPR changes with milder scrambling.

## 6. Reproducibility

```
python -I build_benign.py
python -I build_attacks.py <jailbreakbench PAIR vicuna-13b artifact.json>
.venv312\Scripts\python.exe run_eval.py --model <hf-id> --data benign_alpaca.csv harmful_behaviors.csv attack_pair.csv attack_bon.csv benign_bon_control.csv
.venv312\Scripts\python.exe report_metrics.py results/<model-dir>
```

`results/<model>/records.jsonl` holds per-prompt refusal rate, gradient norm, raw-reply refusal flag and timing (no model text). `summary.csv`, `threshold.json` and `metrics.csv` are derived from it.

## References

1. Hu, X., Chen, P.-Y., Ho, T.-Y. (2024). *Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes.* NeurIPS 2024. arXiv:2403.00867.
2. Hughes, J., et al. (2024). *Best-of-N Jailbreaking.* arXiv:2412.03556.
3. Chao, P., et al. (2023). *Jailbreaking Black Box Large Language Models in Twenty Queries* (PAIR). arXiv:2310.08419.
4. Chao, P., et al. (2024). *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* NeurIPS 2024 Datasets & Benchmarks.
5. Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models* (AdvBench). arXiv:2307.15043.
6. Robey, A., et al. (2023). *SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks.* arXiv:2310.03684.
7. Taori, R., et al. (2023). *Stanford Alpaca: An Instruction-following LLaMA Model.*
