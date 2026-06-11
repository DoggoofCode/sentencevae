from datasets import load_dataset
import re
import argparse
import os

def main() ->None:
    parser = argparse.ArgumentParser(
        prog="Data downloader"
    )
    parser.add_argument("--length", type=int)
    parser.add_argument("--max_seq_len", type=int, default=64)
    args = parser.parse_args()
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train",
        streaming=True,
        cache_dir=f"{os.getcwd()}/dts_cache"
    )

    target_sentences = args.length if args.length else 100_000
    written = 0

    with open("fineweb.jsonl", "w", encoding="utf-8") as f:
        for sample in dataset:
            line_str: str = sample["text"] + "\n"
            line: list = re.split(r"[\.]+", line_str)
            line = [s.strip().replace("\n", "") for s in line if s.strip()]
            line = [s[:args.max_seq_len] for s in line if len(s)>20]

            f.write("\n".join(line))
            written += len(line)

            if written > target_sentences:
                break
            print(f"Completed {written/target_sentences*100:.2f}%", end="\r")


    del dataset
    print(f"\nWrote {written} lines")

if __name__ == "__main__":
    main()
