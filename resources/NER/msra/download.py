import os

# pip install datasets
from datasets import load_dataset
from rex.utils.io import dump_jsonlines
from rex.utils.tagging import get_entities_from_tag_seq

cache_dir = "resources/NER/msra/cache"
dump_dir = "resources/NER/msra/formatted"

datasets = load_dataset("msra_ner", cache_dir=cache_dir)

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)


def extract(dataset, id_prefix):
    to_tag = dataset.info.features["ner_tags"].feature.int2str
    data = []
    for i, d in enumerate(dataset):
        tokens = d["tokens"]
        ner_tags = d["ner_tags"]
        ner_tags = list(map(to_tag, ner_tags))
        ents = get_entities_from_tag_seq(tokens, ner_tags)
        retained_ents = []
        for ent in ents:
            retained_ents.append(
                {
                    "type": ent[1],
                    "index": list(range(*ent[2])),
                }
            )
        data.append(
            {
                "id": f"{id_prefix}{i}",
                "tokens": tokens,
                "ents": retained_ents,
            }
        )
    return data


for key in datasets:
    data = extract(datasets[key], f"ner.msra.{key}.")
    dump_jsonlines(data, os.path.join(dump_dir, f"{key}.jsonl"))
