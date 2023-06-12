from rex.utils.io import dump_line_json, load_json


def get_span(index: list, sentence: list):
    span = []
    split_index = []
    one_index = [index[0]]
    for i in range(len(index) - 1):
        if index[i + 1] - index[i] != 1:
            split_index.append(one_index)
            one_index = [index[i + 1]]
        else:
            one_index.append(index[i + 1])
    if one_index != []:
        split_index.append(one_index)
    for one_index in split_index:
        start_id = one_index[0]
        end_id = one_index[-1]
        if start_id == 0:
            start = 0
        else:
            start = len(" ".join(sentence[0:start_id])) + 1
        end = len(" ".join(sentence[0 : end_id + 1]))
        span.append([start, end])
    return span


def convert(raw_data: list, data_type: str):
    type_map = {"ADR": "adverse drug reaction"}
    new_data = []
    for idx, data in enumerate(raw_data):
        text = " ".join(data["sentence"])
        discontinuous_ent = []
        for entity in data["ner"]:
            index = entity["index"]
            new_sentence = []
            for id in index:
                new_sentence.append(data["sentence"][id])
            new_text = " ".join(new_sentence)
            span = get_span(entity["index"], data["sentence"])
            ent_type = type_map[entity["type"]]
            one_ent = {"type": ent_type, "text": new_text, "span": span}
            discontinuous_ent.append(one_ent)
        one_data = {
            "id": f"cadec.{data_type}.{idx}",
            "instruction": "Please extract discontinuous entities from the given types.",
            "schema": {"discontinuous_ent": ["adverse drug reaction"]},
            "ans": {"discontinuous_ent": discontinuous_ent},
            "text": text,
            "bg": "",
        }
        new_data.append(one_data)
    return new_data


def main():
    raw_train = load_json("newdata/cadec/raw/train.json")
    raw_dev = load_json("newdata/cadec/raw/dev.json")
    raw_test = load_json("newdata/cadec/raw/test.json")
    new_train = convert(raw_train, "train")
    new_dev = convert(raw_dev, "dev")
    new_test = convert(raw_test, "test")
    dump_line_json(new_train, "newdata/cadec/new/train.jsonl")
    dump_line_json(new_dev, "newdata/cadec/new/dev.jsonl")
    dump_line_json(new_test, "newdata/cadec/new/test.jsonl")


if __name__ == "__main__":
    main()
