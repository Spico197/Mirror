import os

# pip install datasets
from datasets import load_dataset
from rex.data.label_encoder import LabelEncoder
from rex.utils.io import dump_iterable, dump_json, dump_jsonlines

cache_dir = "./data/cache"
dump_dir = "./data/formatted"

datasets = load_dataset("peoples_daily_ner", cache_dir=cache_dir)
lbe = LabelEncoder()

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)


def extract(dataset):
    to_tag = dataset.info.features["ner_tags"].feature.int2str
    data = []
    tokens = set()
    for d in dataset:
        if len(d["tokens"]) < 1:
            continue
        tokens.update(d["tokens"])
        d["ner_tags"] = list(map(to_tag, d["ner_tags"]))
        lbe.update(d["ner_tags"])
        data.append(d)
    return data, tokens


vocab_files = []
for key in datasets:
    data, tokens = extract(datasets[key])
    dump_jsonlines(data, os.path.join(dump_dir, f"{key}.jsonl"))
    vocab_filepath = os.path.join(cache_dir, f"{key}.vocab")
    vocab_files.append(vocab_filepath)
    dump_iterable(tokens, vocab_filepath)

lbe.save_pretrained(os.path.join(dump_dir, "label2id.json"))

dump_json(
    {
        "LOC": "地名地理位置",
        "PER": "人名公民居民老百姓名人明星",
        "ORG": "公司法院企业集团学校医院单位",
    },
    os.path.join(dump_dir, "role2query.json"),
)
