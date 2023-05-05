from rex.utils.io import dump_jsonlines, load_jsonlines

if __name__ == "__main__":
    data = load_jsonlines("resources/RE/T-REx/raw_spo/data.jsonl")
    output_filepath = "resources/RE/T-REx/raw_spo/t-rex.udi.jsonl"

    new_data = []
    for ins in data:
        triples = []
        for triple in ins["triples"]:
            triples.append(
                {
                    "relation": triple["predicate"]["surfaceform"],
                    "head": {
                        "text": triple["subject"]["surfaceform"],
                        "span": triple["subject"]["boundaries"],
                    },
                    "tail": {
                        "text": triple["object"]["surfaceform"],
                        "span": triple["object"]["boundaries"],
                    },
                }
            )
        relations = list({triple["relation"] for triple in triples})
        new_ins = {
            "id": f"trex.{ins['id']}",
            "instruction": "",
            "schema": {
                "rel": relations,
            },
            "ans": {"rel": triples},
            "text": ins["sentence"],
            "bg": "",
        }
        new_data.append(new_ins)
    dump_jsonlines(new_data, output_filepath)
