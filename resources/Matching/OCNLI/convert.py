import random
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines

LABEL_MAP = {
    "entailment": "蕴含",
    "contradiction": "矛盾",
    "neutral": "中立",
    "-": "不确定",
}
LABELS = list(LABEL_MAP.values())


def convert_ins(ins):
    background_tokens = list(f"前提：{ins['sentence1']}；假设：{ins['sentence2']}")
    query_tokens = list("判断前提句和假设句之间的蕴含关系")
    label = LABEL_MAP[ins["label"]]
    context_string = ",".join(LABELS)
    context_tokens = list(context_string)
    answer_start_index = context_string.find(label)
    answer_index = list(range(answer_start_index, answer_start_index + len(label)))
    answer_index_list = [answer_index]
    return dict(
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        background_tokens=background_tokens,
        answer_index=answer_index_list,
    )


if __name__ == "__main__":
    data_dir = "resources/Matching/OCNLI/raw"
    dump_dir = "resources/Matching/OCNLI/formatted"
    data_dir = Path(data_dir)
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    for dname in ["train.50k", "dev"]:
        p = data_dir / f"{dname}.json"
        data = load_jsonlines(p)
        new_data = []
        for ins in data:
            if ins["label"] == "-":
                continue
            new_ins = convert_ins(ins)
            new_data.append(new_ins)
        dump_jsonlines(new_data, dump_dir / f"{dname}.jsonl")
