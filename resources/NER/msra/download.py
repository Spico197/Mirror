import os

# pip install datasets
from datasets import load_dataset
from rex.utils.io import dump_jsonlines
from rex.utils.tagging import get_entities_from_tag_seq

cache_dir = "./resources/msra_ner/cache"
dump_dir = "./resources/msra_ner/formatted"

datasets = load_dataset("msra_ner", cache_dir=cache_dir)

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)


def extract(dataset):
    to_tag = dataset.info.features["ner_tags"].feature.int2str
    data = []
    for d in dataset:
        tokens = d["tokens"]
        if len(tokens) < 1:
            continue
        text = "".join(tokens)
        ner_tags = d["ner_tags"]
        ner_tags = list(map(to_tag, ner_tags))
        ents = get_entities_from_tag_seq(tokens, ner_tags)
        retained_ents = []
        for ent in ents:
            if text[ent[2][0] : ent[2][1]] == ent[0]:
                retained_ents.append((ent[0], ent[1], *ent[2]))
        data.append(
            {
                "text": text,
                "ents": retained_ents,
            }
        )
    return data


for key in datasets:
    data = extract(datasets[key])
    dump_jsonlines(data, os.path.join(dump_dir, f"{key}.jsonl"))
