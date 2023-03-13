import re
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_json, load_jsonlines


def convert_ins(ins, ans_map) -> list:
    results = []
    candidates = ins["candidates"]
    context_string = ",".join(candidates)
    context_tokens = list(context_string)
    for sent in ins["content"]:
        placeholder = re.search(r"#idiom\d+#", sent)
        if placeholder:
            ph = placeholder.group(0)
            ans = candidates[ans_map[ph]]
            query_tokens = list(f"{ph}处应该填什么成语")
            answer_start_index = context_string.find(ans)
            answer_endp1_index = answer_start_index + len(ans)
            answer_index = list(range(answer_start_index, answer_endp1_index))
            new_ins = dict(
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                answer_index=[answer_index],
            )
            results.append(new_ins)
    return results


if __name__ == "__main__":
    data_dir = "resources/MRC/CHID/raw"
    dump_dir = "resources/MRC/CHID/formatted"
    data_dir = Path(data_dir)
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    for dname in ["train", "dev"]:
        p = data_dir / f"{dname}.json"
        data = load_jsonlines(p)
        ans_map = load_json(data_dir / f"{dname}_answer.json")
        new_data = []
        for ins in data:
            new_ins_list = convert_ins(ins, ans_map)
            new_data.extend(new_ins_list)
        dump_jsonlines(new_data, dump_dir / f"{dname}.jsonl")
