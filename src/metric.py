from collections import defaultdict
from typing import Tuple

from rex.metrics.base import MetricBase
from rex.metrics.tagging import tagging_prf1
from rex.utils.batch import decompose_batch_into_instances
from rex.utils.random import generate_random_string_with_datetime


class MrcNERMetric(MetricBase):
    def get_instances_from_batch(self, raw_batch: dict, out_batch: dict) -> Tuple:
        gold_instances = []
        pred_instances = []

        batch_gold = decompose_batch_into_instances(raw_batch)
        assert len(batch_gold) == len(out_batch["pred"])

        for i, gold in enumerate(batch_gold):
            gold_instances.append(
                {
                    "id": gold["id"],
                    "ents": {(gold["ent_type"], gent) for gent in gold["gold_ents"]},
                }
            )
            pred_instances.append(
                {
                    "id": gold["id"],
                    "ents": {(gold["ent_type"], pent) for pent in out_batch["pred"][i]},
                }
            )

        return gold_instances, pred_instances

    def calculate_scores(self, golds: list, preds: list) -> dict:
        id2gold = defaultdict(set)
        id2pred = defaultdict(set)
        # aggregate all ents with diff queries before evaluating
        for gold in golds:
            id2gold[gold["id"]].update(gold["ents"])
        for pred in preds:
            id2pred[pred["id"]].update(pred["ents"])
        assert len(id2gold) == len(id2pred)

        gold_ents = []
        pred_ents = []
        for _id in id2gold:
            gold_ents.append(id2gold[_id])
            pred_ents.append(id2pred[_id])

        return tagging_prf1(gold_ents, pred_ents, type_idx=0)


class MrcSpanMetric(MetricBase):
    def get_instances_from_batch(self, raw_batch: dict, out_batch: dict) -> Tuple:
        gold_instances = []
        pred_instances = []

        batch_gold = decompose_batch_into_instances(raw_batch)
        assert len(batch_gold) == len(out_batch["pred"])

        for i, gold in enumerate(batch_gold):
            gold_instances.append(
                {
                    "id": gold["id"],
                    "spans": set(tuple(span) for span in gold["gold_spans"]),
                }
            )
            pred_instances.append(
                {
                    "id": gold["id"],
                    "spans": set(out_batch["pred"][i]),
                }
            )

        return gold_instances, pred_instances

    def calculate_scores(self, golds: list, preds: list) -> dict:
        id2gold = defaultdict(set)
        id2pred = defaultdict(set)
        # aggregate all ents with diff queries before evaluating
        for gold in golds:
            id2gold[gold["id"]].update(gold["spans"])
        for pred in preds:
            id2pred[pred["id"]].update(pred["spans"])
        assert len(id2gold) == len(id2pred)

        gold_spans = []
        pred_spans = []
        for _id in id2gold:
            gold_spans.append(id2gold[_id])
            pred_spans.append(id2pred[_id])

        return tagging_prf1(gold_spans, pred_spans, type_idx=None)


class MultiPartSpanMetric(MetricBase):
    def get_instances_from_batch(self, raw_batch: dict, out_batch: dict) -> Tuple:
        gold_instances = []
        pred_instances = []

        batch_gold = decompose_batch_into_instances(raw_batch)
        assert len(batch_gold) == len(out_batch["pred"])

        for i, gold in enumerate(batch_gold):
            ins_id = gold.get("id", generate_random_string_with_datetime())
            gold_instances.append(
                {
                    "id": ins_id,
                    "spans": set(
                        tuple(multi_part_span) for multi_part_span in gold["spans"]
                    ),
                }
            )
            pred_instances.append(
                {
                    "id": ins_id,
                    "spans": set(
                        tuple(multi_part_span)
                        for multi_part_span in out_batch["pred"][i]
                    ),
                }
            )

        return gold_instances, pred_instances

    def calculate_scores(self, golds: list, preds: list) -> dict:
        id2gold = defaultdict(set)
        id2pred = defaultdict(set)
        # aggregate all ents with diff queries before evaluating
        for gold in golds:
            id2gold[gold["id"]].update(gold["spans"])
        for pred in preds:
            id2pred[pred["id"]].update(pred["spans"])
        assert len(id2gold) == len(id2pred)

        gold_spans = []
        pred_spans = []
        for _id in id2gold:
            gold_spans.append(id2gold[_id])
            pred_spans.append(id2pred[_id])

        return tagging_prf1(gold_spans, pred_spans, type_idx=None)
