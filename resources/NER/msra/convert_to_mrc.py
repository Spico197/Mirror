from collections import defaultdict
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_json, load_jsonlines

dest_path = Path("resources/NER/msra/mrc")
dest_path.mkdir(parents=True, exist_ok=True)
type2query = load_json("resources/query.json")


def convert_ins(ins):
    type2index_list = defaultdict(list)
    for ent in ins["ents"]:
        type2index_list[ent["type"]].append(ent["index"])

    data = []
    for etype, index_list in type2index_list.items():
        data.append(
            {
                "id": ins["id"],
                "query_tokens": list(type2query[etype]),
                "context_tokens": ins["tokens"],
                "answer_index": index_list,
            }
        )

    return data


for dname in ["train", "test"]:
    data = []
    raw = load_jsonlines(f"resources/NER/msra/formatted/{dname}.jsonl")
    for ins in raw:
        ret = convert_ins(ins)
        data.extend(ret)
    dump_jsonlines(data, dest_path / f"{dname}.jsonl")
