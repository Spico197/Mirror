import random

from rex.utils.io import dump_line_json, load_line_json


def convert(raw_data: list, data_type: str):
    new_data = []
    instructions = [
        "Please extract hyper relations from the text with the given schema.",
        "Extract hyper relations from the text using the provided schema.",
        "Identify and extract hyper relations from the text based on the given schema.",
        "Extract hyper relations from the text according to the provided schema.",
        "Use the given schema to extract hyper relations from the text.",
        "Extract hyper relations from the text by following the schema provided.",
    ]
    for idx, data in enumerate(raw_data):
        text = " ".join(data["tokens"])
        hyper_rel = []
        instruction = random.choice(instructions)
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
            "instruction": instruction,
            "schema": {
                "hyper_rel": {
                    "notable work": [
                        "publication date",
                        "point in time",
                        "together with",
                        "subject has role",
                        "end time",
                        "start time",
                        "applies to part",
                        "character role",
                        "object has role",
                        "follows",
                    ],
                    "child": ["mother", "series ordinal", "start time", "of"],
                    "shares border with": [
                        "start time",
                        "end time",
                        "statement disputed by",
                        "applies to part",
                        "object has role",
                        "statement is subject of",
                        "replaces",
                        "located in the administrative territorial entity",
                        "country",
                        "of",
                        "location",
                    ],
                    "winner": [
                        "point in time",
                        "for work",
                        "series ordinal",
                        "country",
                        "together with",
                        "statement is subject of",
                        "number of points/goals/set scored",
                        "ranking",
                    ],
                    "award received": [
                        "point in time",
                        "together with",
                        "winner",
                        "for work",
                        "statement is subject of",
                        "series ordinal",
                        "start time",
                        "subject has role",
                        "follows",
                        "end time",
                        "quantity",
                        "of",
                    ],
                    "part of": [
                        "follows",
                        "series ordinal",
                        "start time",
                        "end time",
                        "subject has role",
                        "quantity",
                        "sports league level",
                        "of",
                        "applies to part",
                        "instance of",
                        "replaces",
                        "country",
                        "statement is subject of",
                        "point in time",
                        "publication date",
                        "object has role",
                        "ranking",
                        "location",
                        "has part",
                        "located in the administrative territorial entity",
                    ],
                    "subclass of": [
                        "follows",
                        "series ordinal",
                        "of",
                        "start time",
                        "location",
                        "applies to part",
                        "sports league level",
                        "country",
                        "end time",
                    ],
                    "nominated for": [
                        "point in time",
                        "nominee",
                        "for work",
                        "statement is subject of",
                        "winner",
                        "of",
                        "together with",
                    ],
                    "significant event": [
                        "point in time",
                        "end time",
                        "start time",
                        "quantity",
                        "location",
                        "located in the administrative territorial entity",
                        "of",
                        "statement is subject of",
                        "country",
                    ],
                    "instance of": [
                        "of",
                        "replaces",
                        "start time",
                        "follows",
                        "end time",
                        "series ordinal",
                        "statement is subject of",
                        "quantity",
                        "point in time",
                        "has part",
                        "country",
                        "applies to part",
                        "statement disputed by",
                    ],
                    "parent organization": [
                        "start time",
                        "end time",
                        "series ordinal",
                        "quantity",
                        "location",
                        "follows",
                        "country",
                        "object has role",
                    ],
                    "head of government": [
                        "end time",
                        "start time",
                        "member of political party",
                        "series ordinal",
                        "position held",
                        "replaces",
                        "statement is subject of",
                        "point in time",
                        "subject has role",
                        "of",
                    ],
                    "position held": [
                        "series ordinal",
                        "end time",
                        "start time",
                        "country",
                        "of",
                        "replaces",
                        "statement is subject of",
                        "instance of",
                        "together with",
                        "location",
                        "diocese",
                        "follows",
                        "position held",
                        "electoral district",
                        "located in the administrative territorial entity",
                        "subject has role",
                        "statement disputed by",
                        "point in time",
                        "member of political party",
                    ],
                    "director / manager": [
                        "start time",
                        "end time",
                        "object has role",
                        "position held",
                        "replaces",
                        "subject has role",
                    ],
                    "spouse": [
                        "series ordinal",
                        "end time",
                        "start time",
                        "point in time",
                    ],
                    "capital of": [
                        "end time",
                        "start time",
                        "replaces",
                        "statement is subject of",
                        "together with",
                        "statement disputed by",
                        "follows",
                        "series ordinal",
                        "located in the administrative territorial entity",
                        "subject has role",
                        "point in time",
                        "instance of",
                    ],
                    "country": [
                        "start time",
                        "end time",
                        "statement disputed by",
                        "applies to part",
                        "located in the administrative territorial entity",
                        "together with",
                        "object has role",
                        "has part",
                        "series ordinal",
                        "point in time",
                    ],
                    "place of birth": [
                        "country",
                        "located in the administrative territorial entity",
                        "location",
                    ],
                    "located in the administrative territorial entity": [
                        "start time",
                        "instance of",
                        "subject has role",
                        "end time",
                        "statement disputed by",
                        "object has role",
                        "series ordinal",
                        "statement is subject of",
                        "applies to part",
                        "country",
                        "located in the administrative territorial entity",
                        "replaces",
                        "location",
                        "point in time",
                        "of",
                    ],
                    "occupant": ["start time", "end time", "follows", "point in time"],
                    "operator": [
                        "start time",
                        "end time",
                        "statement is subject of",
                        "point in time",
                        "of",
                        "country",
                        "subject has role",
                        "applies to part",
                        "quantity",
                    ],
                    "owned by": [
                        "start time",
                        "end time",
                        "country",
                        "together with",
                        "point in time",
                    ],
                    "home venue": [
                        "start time",
                        "end time",
                        "located in the administrative territorial entity",
                    ],
                    "headquarters location": [
                        "country",
                        "location",
                        "start time",
                        "end time",
                        "located in the administrative territorial entity",
                        "street number",
                        "point in time",
                    ],
                    "residence": ["start time", "end time", "country", "position held"],
                    "part of the series": [
                        "follows",
                        "series ordinal",
                        "subject has role",
                    ],
                    "replaces": [
                        "point in time",
                        "start time",
                        "of",
                        "together with",
                        "applies to part",
                        "series ordinal",
                        "statement is subject of",
                        "end time",
                        "object has role",
                    ],
                    "original broadcaster": [
                        "start time",
                        "end time",
                        "country",
                        "point in time",
                    ],
                    "country of citizenship": [
                        "start time",
                        "end time",
                        "series ordinal",
                        "statement disputed by",
                    ],
                    "educated at": [
                        "academic major",
                        "end time",
                        "start time",
                        "academic degree",
                        "point in time",
                        "location",
                    ],
                    "member of": [
                        "start time",
                        "end time",
                        "follows",
                        "point in time",
                        "series ordinal",
                        "subject has role",
                        "replaces",
                        "together with",
                        "electoral district",
                        "location",
                        "affiliation",
                    ],
                    "member of sports team": [
                        "number of points/goals/set scored",
                        "start time",
                        "end time",
                        "number of matches played/races/starts",
                        "position played on team / speciality",
                        "national team appearances",
                        "point in time",
                        "series ordinal",
                    ],
                    "coach of sports team": [
                        "start time",
                        "end time",
                        "follows",
                        "replaces",
                        "point in time",
                    ],
                    "performer": [
                        "of",
                        "publication date",
                        "start time",
                        "together with",
                        "point in time",
                        "end time",
                        "object has role",
                        "applies to part",
                        "follows",
                        "character role",
                        "subject has role",
                        "country",
                    ],
                    "cast member": [
                        "character role",
                        "end time",
                        "start time",
                        "series ordinal",
                    ],
                    "head of state": [
                        "end time",
                        "start time",
                        "position held",
                        "member of political party",
                        "replaces",
                        "series ordinal",
                    ],
                    "employer": [
                        "start time",
                        "end time",
                        "position held",
                        "of",
                        "subject has role",
                        "point in time",
                        "statement is subject of",
                        "location",
                        "replaces",
                    ],
                    "member of political party": [
                        "end time",
                        "start time",
                        "point in time",
                    ],
                    "present in work": [
                        "performer",
                        "of",
                        "subject has role",
                        "point in time",
                        "character role",
                        "series ordinal",
                    ],
                    "followed by": [
                        "of",
                        "point in time",
                        "country",
                        "start time",
                        "applies to part",
                        "end time",
                        "location",
                        "located in the administrative territorial entity",
                        "together with",
                    ],
                    "adjacent station": ["towards", "connecting line", "end time"],
                    "participating team": [
                        "ranking",
                        "number of points/goals/set scored",
                        "number of matches played/races/starts",
                        "point in time",
                        "country",
                    ],
                    "chairperson": [
                        "end time",
                        "start time",
                        "series ordinal",
                        "replaces",
                        "position held",
                        "follows",
                        "object has role",
                        "point in time",
                        "located in the administrative territorial entity",
                    ],
                    "noble title": [
                        "start time",
                        "end time",
                        "of",
                        "series ordinal",
                        "follows",
                        "point in time",
                        "replaces",
                    ],
                    "candidacy in election": [
                        "member of political party",
                        "electoral district",
                        "ranking",
                        "start time",
                    ],
                    "located on street": ["street number"],
                    "league": ["start time", "end time", "point in time"],
                    "location": [
                        "start time",
                        "located in the administrative territorial entity",
                        "country",
                        "end time",
                        "point in time",
                        "applies to part",
                    ],
                    "partner in business or sport": [
                        "affiliation",
                        "start time",
                        "end time",
                    ],
                    "occupation": [
                        "start time",
                        "point in time",
                        "end time",
                        "instance of",
                        "location",
                        "of",
                        "together with",
                    ],
                    "used by": ["of", "start time", "subject has role", "end time"],
                    "participant": [
                        "point in time",
                        "country",
                        "ranking",
                        "series ordinal",
                        "performer",
                        "together with",
                        "character role",
                        "number of points/goals/set scored",
                        "electoral district",
                        "end time",
                        "start time",
                        "object has role",
                        "position held",
                        "number of matches played/races/starts",
                        "location",
                        "of",
                        "quantity",
                    ],
                    "voice actor": [
                        "character role",
                        "series ordinal",
                        "end time",
                        "start time",
                        "of",
                    ],
                    "connecting line": [
                        "adjacent station",
                        "together with",
                        "start time",
                        "end time",
                    ],
                    "military branch": ["end time", "start time"],
                    "legislative body": [
                        "has part",
                        "object has role",
                        "start time",
                        "end time",
                    ],
                    "manufacturer": [
                        "start time",
                        "point in time",
                        "location",
                        "end time",
                        "follows",
                        "applies to part",
                        "quantity",
                    ],
                    "stock exchange": ["ticker symbol", "start time"],
                    "sport": [
                        "statement is subject of",
                        "start time",
                        "end time",
                        "point in time",
                    ],
                    "incarnation of": ["subject has role"],
                    "sports season of league or competition": [
                        "follows",
                        "series ordinal",
                    ],
                    "narrative role": ["of"],
                }
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
