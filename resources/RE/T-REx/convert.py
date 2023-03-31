import glob
from collections import defaultdict

from rex.utils.io import dump_jsonlines, load_json
from rex.utils.progress_bar import pbar

if __name__ == "__main__":
    filepaths = glob.glob("resources/RE/T-REx/raw/re-nlg_*.json")
    new_data = []
    progress = pbar(filepaths)
    for path in progress:
        data = load_json(path)
        for ins in data:
            text = ins["text"]
            triples = []
            sent_id_to_triples = defaultdict(list)
            for triple in ins["triples"]:
                if all(
                    triple[x]["boundaries"] is not None
                    for x in ["predicate", "subject", "object"]
                ):
                    sent_id = triple["sentence_id"]
                    sent_id_to_triples[sent_id].append(triple)
            for sent_id, triples in sent_id_to_triples.items():
                sent_boundary = ins["sentences_boundaries"][sent_id]
                ins_id = f"{ins['docid']}-{sent_id}"
                sent = text[sent_boundary[0] : sent_boundary[1]]
                offset = sent_boundary[0]
                for triple in triples:
                    for t in ["predicate", "subject", "object"]:
                        triple[t]["boundaries"] = list(
                            map(lambda x: x - offset, triple[t]["boundaries"])
                        )
                new_data.append(
                    {
                        "id": ins_id,
                        "sentence": sent,
                        "triples": triples,
                    }
                )

    dump_jsonlines(new_data, "resources/RE/T-REx/raw_spo/data.jsonl")
