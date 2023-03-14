from pathlib import Path

from rex.utils.io import dump_jsonlines, load_json


def convert_ins(ins) -> list:
    results = []
    context_tokens = []

    offset = 0
    context_string = ""
    if ins.get("title"):
        title = ins["title"].strip()
        offset = 1 + len(title)
        context_string += f"{title} "
    context_string += ins["context"].strip()
    context_tokens = list(context_string)

    for qa in ins["qas"]:
        query_tokens = list(qa["question"].strip())
        answer_index_list = []
        if "answers" in qa:
            for ans in qa["answers"]:
                if ans["id"] != "1":
                    continue
                if (
                    ans["answer_start"] == -1
                    and len(ans["text"]) > 0
                    and ans["text"].strip() in context_string
                ):
                    ans["answer_start"] = (
                        ins["context"].strip().find(ans["text"].strip())
                    )
                if ans["answer_start"] > -1:
                    ans_index = [
                        i + offset
                        for i in range(
                            ans["answer_start"], ans["answer_start"] + len(ans["text"])
                        )
                    ]
                    answer_index_list.append(ans_index)

            new_ins = dict(
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                answer_index=answer_index_list,
            )
            results.append(new_ins)
    return results


def convert_normal(data_dir, dump_dir):
    data_dir_path = Path(data_dir)
    dump_dir_path = Path(dump_dir)
    if not dump_dir_path.exists():
        dump_dir_path.mkdir(parents=True)

    for dname in ["train", "dev", "test"]:
        new_data = []
        data = load_json(data_dir_path / f"{dname}.json")
        for paragraph in data["data"]:
            for ins in paragraph["paragraphs"]:
                converted_instances = convert_ins(ins)
                new_data.extend(converted_instances)
        dump_jsonlines(new_data, dump_dir_path / f"{dname}.jsonl")


if __name__ == "__main__":
    data_dir = "resources/MRC/DRCD2018/raw"
    dump_dir = "resources/MRC/DRCD2018/formatted"
    convert_normal(data_dir, dump_dir)
