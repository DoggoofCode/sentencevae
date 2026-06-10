from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True,
)

target_bytes = 1 * 1024**3  # 1 GB
written = 0

with open("fineweb_1gb.jsonl", "w", encoding="utf-8") as f:
    for sample in dataset:
        line = sample["text"] + "\n"
        encoded = line.encode("utf-8")

        if written + len(encoded) > target_bytes:
            break

        f.write(line)
        written += len(encoded)

print(f"Wrote {written / 1024**3:.2f} GB")
