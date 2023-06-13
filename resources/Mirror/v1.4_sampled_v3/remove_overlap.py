from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines
from rex.utils.progress_bar import pbar

# outdir = Path("resources/Mirror/v1.4_sampled_v3/merged/all_woZeroShotNER")
outdir = Path("resources/Mirror/uie/merged")

train_filepath = outdir / "train.jsonl"
test_filepath = outdir / "test.jsonl"

train_data = load_jsonlines(train_filepath)
test_data = load_jsonlines(test_filepath)

test_texts = set()
new_train_data = []
for d in test_data:
    test_texts.add(d["text"])
for d in pbar(train_data):
    if d["text"] not in test_texts:
        new_train_data.append(d)
dump_jsonlines(new_train_data, outdir / "train_wo_overlap.jsonl")
print(f"#original: {len(train_data)}, #new: {len(new_train_data)}")
