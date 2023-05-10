from rex.utils.io import load_jsonlines


def main():
    middle_filepath = "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05EN_labelmap_Rel_updateTag_bs32/middle/test.final.jsonl"
    data = load_jsonlines(middle_filepath)
    for ins in data:
        gold = ins["gold"]
        pred = ins["pred"]
        if gold["spans"] != pred["spans"]:
            breakpoint()


if __name__ == "__main__":
    main()
