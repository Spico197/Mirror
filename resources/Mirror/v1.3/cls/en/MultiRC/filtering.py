from rex.utils.io import dump_jsonlines, load_jsonlines

for dname in ["train", "dev"]:
    data = load_jsonlines(f"cls/en/MultiRC/new/{dname}.jsonl")
    for ins in data:
        ins["schema"]["cls"] = list(filter(lambda x: len(x) > 0, ins["schema"]["cls"]))
        ins["ans"]["cls"] = list(filter(lambda x: len(x) > 0, ins["ans"]["cls"]))
    dump_jsonlines(data, f"cls/en/MultiRC/instructed/{dname}.jsonl")
