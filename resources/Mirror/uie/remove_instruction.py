from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines

data_dir = Path("resources/Mirror/uie/rel/scierc")

for fname in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
    p = data_dir / fname
    data = load_jsonlines(p)
    new_data = []
    for d in data:
        del d["instruction"]
        new_data.append(d)
    dump_dir = data_dir / "remove_instruction"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonlines(new_data, dump_dir / fname)
