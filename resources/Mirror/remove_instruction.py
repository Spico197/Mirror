from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines

# data_dir = Path("resources/Mirror/uie/rel/scierc")
# data_dir = Path("resources/Mirror/v1.5/merged/t-rex-200k-woInstruction")


def remove_instruction(data_dir):
    for fname in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        p = data_dir / fname
        data = load_jsonlines(p)
        new_data = []
        for d in data:
            del d["instruction"]
            new_data.append(d)
        dump_dir = data_dir / "remove_instruction"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonlines(new_data, dump_dir / fname)


if __name__ == "__main__":
    data_dirs = [
        "resources/Mirror/v1.4/ent/en/CrossNER_AI/instructed",
        "resources/Mirror/v1.4/ent/en/CrossNER_literature/instructed",
        "resources/Mirror/v1.4/ent/en/CrossNER_music/instructed",
        "resources/Mirror/v1.4/ent/en/CrossNER_politics/instructed",
        "resources/Mirror/v1.4/ent/en/CrossNER_science/instructed",
        "resources/Mirror/v1.4/ent/en/MIT_MOVIE_Review/instructed",
        "resources/Mirror/v1.4/ent/en/MIT_Restaurant_Review/instructed",
    ]
    for data_dir in data_dirs:
        remove_instruction(Path(data_dir))
