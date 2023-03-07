import os

from datasets import load_dataset
from rex.utils.io import dump_jsonlines

cache_dir = "resources/MRC/cmrc2018/cache"
dump_dir = "resources/MRC/cmrc2018/formatted"

datasets = load_dataset("cmrc2018", cache_dir=cache_dir)

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)


def convert_to_span_ner(ins):
    question_tokens = list(ins["question"])
    context_tokens = list(ins["context"])
    answer_index = [
        list(range(s, s + len(t)))
        for t, s in zip(ins["answers"]["text"], ins["answers"]["answer_start"])
    ]
    ins.update(
        {
            "query_tokens": question_tokens,
            "context_tokens": context_tokens,
            "answer_index": answer_index,
        }
    )
    return ins


for key in datasets:
    dataset = datasets[key]
    final_data = []
    for ins in dataset:
        final_data.append(convert_to_span_ner(ins))
    dump_jsonlines(final_data, os.path.join(dump_dir, f"{key}.jsonl"))
