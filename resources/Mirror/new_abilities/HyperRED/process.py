from rex.utils.io import dump_line_json, load_line_json


def convert(raw_data: list, data_type: str):
    new_data = []
    for idx, data in enumerate(raw_data):
        text = " ".join(data["tokens"])
        hyper_rel = []
        for relation in data["relations"]:
            head_start_id = relation["head"][0]
            head_end_id = relation["head"][1]
            if head_start_id == 0:
                head_start = 0
            else:
                head_start = len(" ".join(data["tokens"][0:head_start_id])) + 1
            head_end = len(" ".join(data["tokens"][0:head_end_id]))
            head_span = [head_start, head_end]

            tail_start_id = relation["tail"][0]
            tail_end_id = relation["tail"][1]
            if tail_start_id == 0:
                tail_start = 0
            else:
                tail_start = len(" ".join(data["tokens"][0:tail_start_id])) + 1
            tail_end = len(" ".join(data["tokens"][0:tail_end_id]))
            tail_span = [tail_start, tail_end]

            qualifiers = []
            for qualifier in relation["qualifiers"]:
                start_id = qualifier["span"][0]
                end_id = qualifier["span"][1]
                if start_id == 0:
                    start = 0
                else:
                    start = len(" ".join(data["tokens"][0:start_id])) + 1
                end = len(" ".join(data["tokens"][0:end_id]))
                span = [start, end]
                qualifiers.append(
                    {
                        "text": " ".join(
                            data["tokens"][qualifier["span"][0] : qualifier["span"][1]]
                        ),
                        "span": span,
                        "label": qualifier["label"],
                    }
                )
            one_rel = {
                "relation": relation["label"],
                "head": {
                    "text": " ".join(
                        data["tokens"][relation["head"][0] : relation["head"][1]]
                    ),
                    "span": head_span,
                },
                "tail": {
                    "text": " ".join(
                        data["tokens"][relation["tail"][0] : relation["tail"][1]]
                    ),
                    "span": tail_span,
                },
                "qualifiers": qualifiers,
            }
            hyper_rel.append(one_rel)
        one_data = {
            "id": f"HyperRED.{data_type}.{idx}",
            "instruction": "Please extract hyper relations from the text with the given schema.",
            "schema": {
                "hyper_rel": [
                    "statement is subject of",
                    "place of birth",
                    "present in work",
                    "number of matches played/races/starts",
                    "academic degree",
                    "parent organization",
                    "occupation",
                    "military branch",
                    "object has role",
                    "country",
                    "start time",
                    "ticker symbol",
                    "publication date",
                    "series ordinal",
                    "followed by",
                    "director / manager",
                    "nominee",
                    "head of state",
                    "participating team",
                    "capital of",
                    "part of",
                    "statement disputed by",
                    "subject has role",
                    "location",
                    "member of sports team",
                    "winner",
                    "number of points/goals/set scored",
                    "follows",
                    "used by",
                    "subclass of",
                    "applies to part",
                    "sports league level",
                    "notable work",
                    "national team appearances",
                    "sport",
                    "instance of",
                    "located on street",
                    "child",
                    "member of political party",
                    "employer",
                    "located in the administrative territorial entity",
                    "performer",
                    "affiliation",
                    "operator",
                    "home venue",
                    "residence",
                    "sports season of league or competition",
                    "award received",
                    "quantity",
                    "country of citizenship",
                    "incarnation of",
                    "ranking",
                    "street number",
                    "for work",
                    "electoral district",
                    "owned by",
                    "league",
                    "adjacent station",
                    "diocese",
                    "coach of sports team",
                    "participant",
                    "replaces",
                    "head of government",
                    "original broadcaster",
                    "noble title",
                    "spouse",
                    "headquarters location",
                    "together with",
                    "candidacy in election",
                    "connecting line",
                    "significant event",
                    "shares border with",
                    "chairperson",
                    "end time",
                    "point in time",
                    "nominated for",
                    "member of",
                    "character role",
                    "voice actor",
                    "manufacturer",
                    "partner in business or sport",
                    "educated at",
                    "part of the series",
                    "of",
                    "stock exchange",
                    "narrative role",
                    "occupant",
                    "position held",
                    "academic major",
                    "position played on team / speciality",
                    "mother",
                    "cast member",
                    "towards",
                    "legislative body",
                    "has part",
                ]
            },
            "ans": {"hyper_rel": hyper_rel},
            "text": text,
            "bg": "",
        }
        new_data.append(one_data)
    return new_data


def main():
    raw_train = load_line_json("newdata/HyperRED/raw/train.json")
    raw_dev = load_line_json("newdata/HyperRED/raw/dev.json")
    raw_test = load_line_json("newdata/HyperRED/raw/test.json")
    new_train = convert(raw_train, "train")
    new_dev = convert(raw_dev, "dev")
    new_test = convert(raw_test, "test")
    dump_line_json(new_train, "newdata/HyperRED/new/train.jsonl")
    dump_line_json(new_dev, "newdata/HyperRED/new/dev.jsonl")
    dump_line_json(new_test, "newdata/HyperRED/new/test.jsonl")


if __name__ == "__main__":
    main()
