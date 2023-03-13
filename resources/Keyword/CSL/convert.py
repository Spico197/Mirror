import re
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_csv


def convert_ins(ins):
    _, abstract, keyword_string = ins
    abstract = re.sub(r"\s+", " ", abstract).strip()
    keyword_string = re.sub(r"\s+", " ", keyword_string).strip()
    keywords = keyword_string.split("_")
    query_tokens = list("关键词")
    context_tokens = list(abstract)
    answer_index_list = []
    for keyword in keywords:
        if keyword in abstract:
            k_start = abstract.find(keyword)
            k_end = k_start + len(keyword)
            answer_index_list.append(list(range(k_start, k_end)))
    return dict(
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        answer_index=answer_index_list,
    )


if __name__ == "__main__":
    data_dir = "resources/Keyword/CSL/raw"
    dump_dir = "resources/Keyword/CSL/formatted"
    data_dir = Path(data_dir)
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    # for dname in ["train", "dev", "test"]:
    #     p = data_dir / f"{dname}.tsv"
    #     data = load_csv(p, False, None, sep="\t")
    #     new_data = []
    #     for ins in data:
    #         new_ins = convert_ins(ins)
    #         new_data.append(new_ins)
    #     dump_jsonlines(new_data, dump_dir / f"{dname}.jsonl")

    taboo_texts = set()
    dev = load_csv("resources/Keyword/CSL/raw/dev.tsv", False)
    test = load_csv("resources/Keyword/CSL/raw/test.tsv", False)
    data = dev + test
    for d in data:
        taboo_texts.add(d[1].strip())

    all40k = load_csv("resources/Keyword/CSL/raw/csl_40k.tsv", True, sep="\t")
    filtered_all40k = []
    for ins in all40k:
        if ins["abstract"].strip() in taboo_texts:
            continue
        new_ins = convert_ins([None, ins["abstract"], ins["keywords"]])
        filtered_all40k.append(new_ins)
    dump_jsonlines(filtered_all40k, "resources/Keyword/CSL/formatted/csl_40k.jsonl")
