import re
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_json, load_jsonlines


def convert_ins(ins) -> list:
    results = []
    bg, qas, _ = ins
    background_tokens = list("".join(bg))
    for qa in qas:
        query_tokens = list(qa["question"])
        context_string = ",".join(qa["choice"])
        context_tokens = list(context_string)
        answer_start_index = context_string.find(qa["answer"])
        answer_endp1_index = answer_start_index + len(qa["answer"])
        answer_index = list(range(answer_start_index, answer_endp1_index))
        new_ins = dict(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            background_tokens=background_tokens,
            answer_index=[answer_index],
        )
        results.append(new_ins)
    return results


if __name__ == "__main__":
    data_dir = "resources/MRC/C3/raw"
    dump_dir = "resources/MRC/C3/formatted"
    data_dir = Path(data_dir)
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    for dname in ["m-train", "m-dev", "d-train", "d-dev"]:
        p = data_dir / f"{dname}.json"
        data = load_json(p)
        new_data = []
        for ins in data:
            new_ins_list = convert_ins(ins)
            new_data.extend(new_ins_list)
        dump_jsonlines(new_data, dump_dir / f"{dname}.jsonl")
