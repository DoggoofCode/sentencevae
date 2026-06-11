from datasets import load_dataset
import re
import argparse

def main() ->None:
    parser = argparse.ArgumentParser(
        prog="Data downloader"
    )
    parser.add_argument("--length", type=int)
    parser.add_argument("--max_seq_len", type=int, default=128)
    args = parser.parse_args()
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train",
        streaming=True,
        cache_dir="/home/vedj/code/sentencevae/dts_cache"
    )

    target_sentences = args.length if args.length else 100_000
    written = 0

    with open("fineweb.jsonl", "w", encoding="utf-8") as f:
        for sample in dataset:
            line = sample["text"] + "\n"
            line: list = re.split(r"[\.]+", line)
            line = [s.strip() for s in line if s.strip()]
            lines_for_remove = []
            for idx, ln in enumerate(line):
                line[idx] = ln[:args.max_seq_len]
                if len(ln) < 20:
                    lines_for_remove.append(idx)
            for idx, line_num in enumerate(lines_for_remove):
                line.pop(line_num - idx)
            f.write("\n".join(line))
            written += len(line)

            if written > target_sentences:
                break
            print(f"Completed {written/target_sentences*100:.2f}%", end="\r")


    print(f"\nWrote {written} lines")

if __name__ == "__main__":
    main()
