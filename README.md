# Gradient Cuff on small open models

A replication of **Gradient Cuff** (Hu, Chen & Ho, NeurIPS 2024, [arXiv:2403.00867](https://arxiv.org/abs/2403.00867)) on Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct in 4-bit on a 6 GB laptop GPU, plus a test against the newer **Best-of-N (BoN)** jailbreak.

- Full write-up: [`REPORT.md`](REPORT.md) (LaTeX version: `report.tex` / `report.pdf`)

## Key results (5% benign false-positive target)

| | Qwen2.5-1.5B | Llama-3.2-3B |
|---|---|---|
| Benign FPR, Gradient Cuff | 0.033 | 0.027 |
| PAIR detection: model alone → Gradient Cuff | 0.146 → 0.390 | 0.707 → 0.890 |
| BoN detection: model alone → Gradient Cuff | 0.540 → 1.000 | 0.883 → 0.993 |
| **Scrambled *benign* FPR, Gradient Cuff** | **0.933** | **0.940** |

Gradient Cuff blocks BoN, but it also rejects about 94% of harmless prompts given the same character scrambling. On noisy input it behaves as a "garbled text" filter rather than a harm detector. See REPORT.md §4.

## Repository layout

| Path | Purpose |
|---|---|
| `gradient_cuff_lab.py` | Detector (`GradientCuffDetector`), 4-bit loader, threshold calibration, metrics |
| `build_benign.py` | Builds `benign_alpaca.csv` (150 calibration + 150 test Alpaca prompts) |
| `build_attacks.py` | Builds `attack_pair.csv`, `attack_bon.csv`, `benign_bon_control.csv` |
| `harmful_behaviors.csv` | AdvBench direct harmful requests (520) |
| `smoke_test.py` | Quick check: 5 benign + 5 harmful prompts, prints gradient norm and timing |
| `run_eval.py` | Main experiment: scores every prompt, calibrates the threshold, prints a summary. Resumable |
| `report_metrics.py` | Accuracy / precision / recall / F1 / FPR from the cached records (no GPU) |
| `live_demo.py` | Interactive guard: type prompts, see accept/reject and the model's reply |
| `results/<model>/` | `records.jsonl` (per-prompt scores, no model text), `threshold.json`, `summary.csv`, `metrics.csv` |
| `upstream/` | The original authors' reference implementation (`main.py`), kept for comparison |

## Setup

Python 3.12, an NVIDIA GPU with at least 6 GB of memory, and a Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct` (Qwen is ungated).

```bash
python -m venv .venv312
.venv312\Scripts\activate          # Linux/macOS: source .venv312/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
huggingface-cli login              # only needed for Llama
```

## Reproduce

The prompt CSVs are already included. To rebuild them:

```bash
python build_benign.py
python build_attacks.py <JailbreakBench PAIR vicuna-13b artifact.json>
```

Run the evaluation. This takes about 3 GPU-hours for Qwen on a laptop RTX 4050. If it stops, rerunning the same command resumes from `records.jsonl`:

```bash
python run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --data benign_alpaca.csv harmful_behaviors.csv attack_pair.csv attack_bon.csv benign_bon_control.csv
python report_metrics.py results/qwen-qwen2-5-1-5b-instruct
```

`report_metrics.py` works on the committed `results/` straight away, so you can check the reported numbers without a GPU.

## Live input

After `run_eval.py` has calibrated a threshold for a model:

```bash
python live_demo.py --model Qwen/Qwen2.5-1.5B-Instruct
```

Type a prompt and press Enter. For each prompt the script runs the two Gradient Cuff checks, prints whether the prompt was accepted (with its refusal rate and gradient norm), and then shows the model's reply or a refusal. Each prompt takes about 5–15 s on a 6 GB laptop GPU, because the check needs up to 111 generations. Use `--threshold X` to try a different threshold.

## Credits and licence

The detector follows the method and reference code of Hu, Chen & Ho ([IBM/Gradient-Cuff](https://github.com/IBM/Gradient-Cuff)); the original licence is kept in `LICENSE`. Datasets: AdvBench (Zou et al., 2023), JailbreakBench PAIR artifacts (Chao et al., 2024), Stanford Alpaca (Taori et al., 2023; CC BY-NC 4.0). BoN augmentation follows Hughes et al. (2024).
