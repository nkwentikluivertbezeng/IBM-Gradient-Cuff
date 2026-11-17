"""Step 4a: sample 300 benign prompts from tatsu-lab/alpaca (instruction-only rows)."""
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train").to_pandas()
ds = ds[(ds["input"].str.strip() == "") & ds["instruction"].str.len().between(20, 300)]
df = ds.sample(300, random_state=42).reset_index(drop=True)
df = df.assign(
    id=[f"alpaca-{i:03d}" for i in range(len(df))],
    prompt=df["instruction"],
    label="benign",
    split=["calibration" if i < 150 else "test" for i in range(len(df))],
    source="tatsu-lab/alpaca",
    attack_family="benign",
)[["id", "prompt", "label", "split", "source", "attack_family"]]
df.to_csv("benign_alpaca.csv", index=False)
print(df["split"].value_counts().to_string())
print(df.head(5)[["id", "prompt"]].to_string(index=False))
