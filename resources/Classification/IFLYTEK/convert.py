import random
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines

LABELS = [
    label["label_des"]
    for label in load_jsonlines("resources/Classification/IFLYTEK/raw/labels.json")
]


def convert_ins(ins):
    background_tokens = list(ins["sentence"])
    query_tokens = list("下面这段文本属于什么类别")
    label = ins["label_des"]
    labels = random.choices(list(set(LABELS) - {label}), k=9)
    labels.append(label)
    random.shuffle(labels)
    context_string = ",".join(labels)
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
    data_dir = "resources/Classification/IFLYTEK/raw"
    dump_dir = "resources/Classification/IFLYTEK/formatted"
    data_dir = Path(data_dir)
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    for dname in ["train", "dev"]:
        p = data_dir / f"{dname}.json"
        data = load_jsonlines(p)
        new_data = []
        for ins in data:
            new_ins = convert_ins(ins)
            new_data.append(new_ins)
        dump_jsonlines(new_data, dump_dir / f"{dname}.jsonl")
