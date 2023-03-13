from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines


def convert_ins(ins):
    query_tokens = list("两句话是否表达同一个意思")
    context_tokens = list("是,否,不确定")
    background_tokens = list(f"句子1：{ins['sentence1']}；句子2：{ins['sentence2']}")
    if ins["label"] == "0":
        answer_index = [2]
    elif ins["label"] == "1":
        answer_index = [0]
    else:
        answer_index = [4, 5, 6]
    answer_index_list = [answer_index]
    return dict(
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        background_tokens=background_tokens,
        answer_index=answer_index_list,
    )


if __name__ == "__main__":
    data_dir = "resources/Matching/AFQMC/raw"
    dump_dir = "resources/Matching/AFQMC/formatted"
    data_dir = Path(data_dir)
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    for dname in ["train", "dev", "test"]:
        p = data_dir / f"{dname}.json"
        data = load_jsonlines(p)
        new_data = []
        for ins in data:
            new_ins = convert_ins(ins)
            new_data.append(new_ins)
        dump_jsonlines(new_data, dump_dir / f"{dname}.jsonl")
