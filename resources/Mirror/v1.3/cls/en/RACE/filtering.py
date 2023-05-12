from rex.utils.io import dump_jsonlines, load_jsonlines

input_dir = "cls/en/RACE/new"
out_dir = "cls/en/RACE/instructed"

for dname in ["train", "dev", "test"]:
    data = load_jsonlines(f"{input_dir}/{dname}.jsonl")
    for ins in data:
        ins["schema"]["cls"] = list(filter(lambda x: len(x) > 0, ins["schema"]["cls"]))
        ins["ans"]["cls"] = list(filter(lambda x: len(x) > 0, ins["ans"]["cls"]))
    dump_jsonlines(data, f"{out_dir}/{dname}.jsonl")
