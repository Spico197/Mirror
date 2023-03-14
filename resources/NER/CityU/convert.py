from collections import defaultdict
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_csv
from rex.utils.tagging import get_entities_from_tag_seq

if __name__ == "__main__":
    type2query = {
        "LOC": "地名地理位置",
        "PER": "人名公民居民老百姓名人明星",
        "ORG": "公司法院企业集团学校医院单位",
    }

    data_dir = Path("resources/NER/CityU/raw/CityU/trans")
    out_dir = Path("resources/NER/CityU/formatted")
    if not out_dir.exists():
        out_dir.mkdir()

    for dname in ["cityu_train_bio", "cityu_test_bio"]:
        data = load_csv(data_dir / dname, False)
        new_data = []
        tokens = []
        ner_tags = []
        for line in data:
            if len(line) == 2:
                tokens.append(line[0])
                ner_tags.append(line[1])
            else:
                ents = get_entities_from_tag_seq(tokens, ner_tags)
                type2index_list = defaultdict(list)
                for _, ent_type, (ent_sid, ent_eid) in ents:
                    ent_index = list(range(ent_sid, ent_eid))
                    type2index_list[ent_type].append(ent_index)

                for etype, index_list in type2index_list.items():
                    new_data.append(
                        {
                            "query_tokens": list(type2query[etype]),
                            "context_tokens": tokens,
                            "answer_index": index_list,
                        }
                    )

                tokens = []
                ner_tags = []
        dump_jsonlines(new_data, out_dir / f"{dname}.jsonl")
