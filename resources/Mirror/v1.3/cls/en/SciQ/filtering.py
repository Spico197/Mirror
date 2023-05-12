from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines

input_dir = "cls/en/SciQ/new"
out_dir = Path("cls/en/SciQ/instructed")
if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)

for dname in ["train", "dev", "test"]:
    data = load_jsonlines(f"{input_dir}/{dname}.jsonl")
    for ins in data:
        ins["schema"]["cls"] = list(filter(lambda x: len(x) > 0, ins["schema"]["cls"]))
        ins["ans"]["cls"] = list(filter(lambda x: len(x) > 0, ins["ans"]["cls"]))
    dump_jsonlines(data, f"{out_dir}/{dname}.jsonl")
