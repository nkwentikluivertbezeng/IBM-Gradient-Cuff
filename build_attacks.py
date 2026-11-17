"""Step 5a: attack CSVs. Usage: python -I build_attacks.py <pair_artifact.json>

* attack_pair.csv - JailbreakBench PAIR prompts (found on vicuna-13b), used as transfer attacks.
* attack_bon.csv  - Best-of-N text augmentations (Hughes et al., 2024): word scrambling,
                    random capitalisation, ASCII noising; N variants per behaviour.
"""
import json
import random
import sys

import pandas as pd

N_BEHAVIOURS, N_VARIANTS = 30, 10

art = json.load(open(sys.argv[1], encoding="utf-8"))["jailbreaks"]
pair = [j for j in art if j.get("prompt")]
pd.DataFrame({
    "id": [f"pair-{j['index']:03d}" for j in pair],
    "prompt": [j["prompt"] for j in pair],
    "label": "malicious", "source": "JailbreakBench PAIR vicuna-13b-v1.5",
    "attack_family": "PAIR", "behavior": [j["behavior"] for j in pair],
}).to_csv("attack_pair.csv", index=False)


def bon_augment(text: str, rng: random.Random) -> str:
    words = []
    for w in text.split():
        if len(w) > 3 and rng.random() < 0.6:  # scramble middle characters
            mid = list(w[1:-1]); rng.shuffle(mid); w = w[0] + "".join(mid) + w[-1]
        words.append(w)
    out = []
    for c in " ".join(words):
        if rng.random() < 0.6:  # random capitalisation
            c = c.upper() if rng.random() < 0.5 else c.lower()
        if c.isprintable() and 32 < ord(c) < 126 and rng.random() < 0.06:  # ASCII noising
            c = chr(ord(c) + rng.choice((-1, 1)))
        out.append(c)
    return "".join(out)


rng = random.Random(42)
goals = [(j["behavior"], j["goal"]) for j in art][:N_BEHAVIOURS]
rows = [{"id": f"bon-{b:02d}-{v}", "prompt": bon_augment(goal, rng), "label": "malicious",
         "source": "Best-of-N augmentation of JBB goals", "attack_family": "BoN", "behavior": beh}
        for b, (beh, goal) in enumerate(goals) for v in range(N_VARIANTS)]
pd.DataFrame(rows).to_csv("attack_bon.csv", index=False)

# Control: same augmentation on the held-out benign TEST prompts -> FPR on garbled benign text.
ben = pd.read_csv("benign_alpaca.csv").query("split == 'test'")
pd.DataFrame({"id": ben["id"] + "-bon", "prompt": [bon_augment(p, rng) for p in ben["prompt"]],
              "label": "benign", "split": "test", "source": "Best-of-N augmentation of alpaca test",
              "attack_family": "benign_BoN_control"}).to_csv("benign_bon_control.csv", index=False)
print(f"benign BoN control: {len(ben)} prompts")
print(f"PAIR prompts: {len(pair)} / {len(art)} behaviours had a jailbreak")
print(f"BoN prompts: {len(rows)} ({N_BEHAVIOURS} behaviours x {N_VARIANTS} variants)")
