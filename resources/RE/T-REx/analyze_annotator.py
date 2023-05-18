import glob

from rex.utils.io import dump_json, load_json
from rex.utils.progress_bar import pbar

if __name__ == "__main__":
    filepaths = glob.glob("resources/RE/T-REx/raw/re-nlg_*.json")
    annotator_to_example = {}
    progress = pbar(filepaths)
    for path in progress:
        data = load_json(path)
        for ins in data:
            for ent in ins["entities"]:
                annotator_to_example[ent["annotator"]] = ent

    dump_json(annotator_to_example, "resources/RE/T-REx/annotator_to_example.json")
