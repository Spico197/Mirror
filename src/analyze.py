from collections import defaultdict

from rex.metrics.tagging import tagging_prf1
from rex.utils.io import load_jsonlines
from rex.utils.position import find_all_positions


def main():
    middle_filepath = "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05EN_labelmap_Rel_updateTag_bs32/middle/test.final.jsonl"
    data = load_jsonlines(middle_filepath)
    for ins in data:
        gold = ins["gold"]
        pred = ins["pred"]
        if gold["spans"] != pred["spans"]:
            breakpoint()


def check_ent_string_matching_upper_bound(filepath: str, strategy: str = "first"):
    def _check_overlap(x, y):
        if x[0] > y[1] or y[0] > x[1]:
            return False
        else:
            return True

    data = load_jsonlines(filepath)
    golds = []
    preds = []
    for ins in data:
        text = ins["text"]
        gold_ents = ins["ans"]["ent"]
        gold_ents = list(
            set([(ent["text"], ent["type"], tuple(ent["span"])) for ent in gold_ents])
        )
        gold_ents.sort(key=lambda x: len(x[0]), reverse=True)
        pred_ents = []
        matched = set()
        for gold_ent in gold_ents:
            ent_string = gold_ent[0]
            ent_type = gold_ent[1]
            positions = find_all_positions(text, ent_string)
            if strategy == "first":
                for position in positions:
                    if (ent_type, position) not in matched:
                        matched.add((ent_type, position))
                        pred_ents.append((ent_string, ent_type, tuple(position)))
            else:
                flag = False
                for position in positions:
                    for _, g in matched:
                        if _check_overlap(g, position):
                            flag = True
                    if flag:
                        continue

                    if (ent_type, position) not in matched:
                        matched.add((ent_type, position))
                        pred_ents.append((ent_string, ent_type, tuple(position)))
                        break

        golds.append(gold_ents)
        preds.append(pred_ents)

    results = tagging_prf1(golds, preds)

    print(f"filepath: {filepath}, Strategy: {strategy}")
    print(f"Results: {results['micro']}")


def check_rel_tanl_upper_bound(filepath):
    data = load_jsonlines(filepath)
    golds = []
    preds = []
    for ins in data:
        text = ins["text"]
        gold_rels = ins["ans"]["rel"]
        ent_text_to_spans = defaultdict(set)
        for ent in ins["ans"]["ent"]:
            ent_text_to_spans[ent["text"]].add(tuple(ent["span"]))
        gold_rels = list(
            set(
                [
                    (
                        tuple(rel["head"]["span"]),
                        rel["relation"],
                        tuple(rel["tail"]["span"]),
                    )
                    for rel in gold_rels
                ]
            )
        )
        pred_rels = []
        for pred_rel in ins["ans"]["rel"]:
            # pred_triple = ()
            tail_text = pred_rel["tail"]["text"]
            if (
                tail_text in ent_text_to_spans
                and len(ent_text_to_spans[tail_text]) == 1
            ):
                tail_span = list(ent_text_to_spans[tail_text])[0]
                pred_rels.append(
                    (tuple(pred_rel["head"]["span"]), pred_rel["relation"], tail_span)
                )
            # if tail_text in ent_text_to_spans:
            #     tail_span = list(ent_text_to_spans[tail_text])[0]
            # else:
            #     tail_span = find_all_positions(text, tail_text)[0]
            # pred_rels.append((tuple(pred_rel["head"]["span"]), pred_rel["relation"], tail_span))

        golds.append(gold_rels)
        preds.append(pred_rels)

    results = tagging_prf1(golds, preds)

    print(f"filepath: {filepath}")
    print(f"Results: {results['micro']}")


if __name__ == "__main__":
    # main()

    # for filepath in [
    #     "/data/tzhu/Mirror/resources/Mirror/uie/ent/ace04/test.jsonl",
    #     "/data/tzhu/Mirror/resources/Mirror/uie/ent/ace05/test.jsonl",
    #     "/data/tzhu/Mirror/resources/Mirror/uie/ent/conll03/test.jsonl",
    # ]:
    #     for strategy in ["first", "longer_first"]:
    #         check_ent_string_matching_upper_bound(filepath, strategy)

    for filepath in [
        "/data/tzhu/Mirror/resources/Mirror/uie/rel/ace05-rel/test.jsonl",
        "/data/tzhu/Mirror/resources/Mirror/uie/rel/conll04/test.jsonl",
        "/data/tzhu/Mirror/resources/Mirror/uie/rel/nyt/test.jsonl",
        "/data/tzhu/Mirror/resources/Mirror/uie/rel/scierc/test.jsonl",
    ]:
        check_rel_tanl_upper_bound(filepath)
