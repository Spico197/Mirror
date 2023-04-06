from collections import defaultdict
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_csv
from rex.utils.tagging import get_entities_from_tag_seq

if __name__ == "__main__":
    type2query = {
        "NR": "人名和虚构的人物形象",
        "NS": "按照地理位置划分的国家,城市,乡镇,大洲",
        "NT": "组织包括公司,政府党派,学校,政府,新闻机构",
    }

    data_dir = Path("resources/NER/MSRA_v2/raw")
    out_dir = Path("resources/NER/MSRA_v2/formatted")
    if not out_dir.exists():
        out_dir.mkdir()

    for dname in ["train.char.bmes", "dev.char.bmes", "test.char.bmes"]:
        data = load_csv(data_dir / dname, False, sep=" ")
        new_data = []
        tokens = []
        ner_tags = []
        idx = 0
        for line in data:
            if len(line) == 2:
                tokens.append(line[0])
                ner_tags.append(line[1])
            else:
                if len(tokens) < 1:
                    continue
                ents = get_entities_from_tag_seq(tokens, ner_tags)
                retained_ents = []
                for ent in ents:
                    retained_ents.append(
                        {
                            "type": ent[1],
                            "index": list(range(*ent[2])),
                        }
                    )
                new_data.append(
                    {
                        "id": f"{dname}.{idx}",
                        "tokens": tokens,
                        "ents": retained_ents,
                    }
                )
                idx += 1

                tokens = []
                ner_tags = []
        dump_jsonlines(new_data, out_dir / f"{dname}.jsonl")
