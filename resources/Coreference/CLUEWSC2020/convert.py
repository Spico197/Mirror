from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines


def convert_ins(ins):
    assert ins["label"] == "true"
    context_tokens = list(ins["text"])
    query_tokens = list(f"这里的{ins['target']['span2_text']}指的是谁")
    span1_start_index = ins["target"]["span1_index"]
    span1_endp1_index = span1_start_index + len(ins["target"]["span1_text"])
    assert (
        ins["text"][span1_start_index:span1_endp1_index] == ins["target"]["span1_text"]
    )
    answer_index = list(range(span1_start_index, span1_endp1_index))

    answer_index_list = [answer_index]
    return dict(
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        answer_index=answer_index_list,
    )


if __name__ == "__main__":
    data_dir = "resources/Coreference/CLUEWSC2020/raw"
    dump_dir = "resources/Coreference/CLUEWSC2020/formatted"
    data_dir = Path(data_dir)
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    for dname in ["train", "dev"]:
        p = data_dir / f"{dname}.json"
        data = load_jsonlines(p)
        new_data = []
        for ins in data:
            if ins["label"] == "true":
                new_ins = convert_ins(ins)
                new_data.append(new_ins)
        dump_jsonlines(new_data, dump_dir / f"{dname}.jsonl")
