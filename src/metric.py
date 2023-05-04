from collections import defaultdict
from typing import Tuple

from rex.metrics import calc_p_r_f1_from_tp_fp_fn, safe_division
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


def calc_char_event(golds, preds):
    """
    Calculate char-level event argument scores

    References:
        - https://aistudio.baidu.com/aistudio/competition/detail/46/0/submit-result

    Args:
        golds: a list of gold answers (a list of `event_list`), len=#data,
            format is a list of `event_list`
        preds: a list of pred answers, len=#data
    """

    def _match_arg_char_f1(gold_arg, pred_args):
        gtype, grole, gstring = gold_arg
        gchars = set(gstring)
        garg_len = len(gchars)
        cands = []
        for parg in pred_args:
            if parg[0] == gtype and parg[1] == grole:
                pchars = set(str(parg[-1]))
                parg_len = len(pchars)
                pmatch = len(pchars & gchars)
                p = safe_division(pmatch, parg_len)
                r = safe_division(pmatch, garg_len)
                f1 = safe_division(2 * p * r, p + r)
                cands.append(f1)
        if len(cands) > 0:
            f1 = sorted(cands)[-1]
            return f1
        else:
            return 0.0

    pscore = num_gargs = num_pargs = 0
    for _golds, _preds in zip(golds, preds):
        # _golds and _preds pair in one data instance
        gold_args = []
        pred_args = []
        for gold in _golds:
            for arg in gold.get("arguments", []):
                gold_args.append(
                    (gold.get("event_type"), arg.get("role"), arg.get("argument"))
                )
        for pred in _preds:
            for arg in pred.get("arguments", []):
                pred_args.append(
                    (pred.get("event_type"), arg.get("role"), arg.get("argument"))
                )

        num_gargs += len(gold_args)
        num_pargs += len(pred_args)
        for gold_arg in gold_args:
            pscore += _match_arg_char_f1(gold_arg, pred_args)

    p = safe_division(pscore, num_pargs)
    r = safe_division(pscore, num_gargs)
    f1 = safe_division(2 * p * r, p + r)
    return {
        "p": p,
        "r": r,
        "f1": f1,
        "pscore": pscore,
        "num_pargs": num_pargs,
        "num_gargs": num_gargs,
    }


def calc_trigger_identification_metrics(golds, preds):
    tp = fp = fn = 0
    for _golds, _preds in zip(golds, preds):
        gold_triggers = {gold["trigger"] for gold in _golds}
        pred_triggers = {pred["trigger"] for pred in _preds}
        tp += len(gold_triggers & pred_triggers)
        fp += len(pred_triggers - gold_triggers)
        fn += len(gold_triggers - pred_triggers)
    metrics = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def calc_trigger_classification_metrics(golds, preds):
    tp = fp = fn = 0
    for _golds, _preds in zip(golds, preds):
        gold_tgg_cls = {(gold["trigger"], gold["event_type"]) for gold in _golds}
        pred_tgg_cls = {(pred["trigger"], pred["event_type"]) for pred in _preds}
        tp += len(gold_tgg_cls & pred_tgg_cls)
        fp += len(pred_tgg_cls - gold_tgg_cls)
        fn += len(gold_tgg_cls - pred_tgg_cls)
    metrics = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def calc_arg_identification_metrics(golds, preds):
    """Calculate argument identification metrics

    Notice:
        An entity could take different roles in an event,
            so the base number must be calculated by
            (arg, event type, pos, role)
    """
    tp = fp = fn = 0
    for _golds, _preds in zip(golds, preds):
        gold_args = set()
        pred_args = set()
        for gold in _golds:
            _args = {
                (arg["role"], arg["argument"], gold["event_type"])
                for arg in gold["arguments"]
            }
            gold_args.update(_args)
        for pred in _preds:
            _args = {
                (arg["role"], arg["argument"], pred["event_type"])
                for arg in pred["arguments"]
            }
            pred_args.update(_args)
        # logic derived from OneIE
        _tp = 0
        _tp_fp = len(pred_args)
        _tp_fn = len(gold_args)
        _gold_args_wo_role = {_ga[1:] for _ga in gold_args}
        for pred_arg in pred_args:
            if pred_arg[1:] in _gold_args_wo_role:
                _tp += 1
        tp += _tp
        fp += _tp_fp - _tp
        fn += _tp_fn - _tp
    metrics = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def calc_arg_classification_metrics(golds, preds):
    tp = fp = fn = 0
    for _golds, _preds in zip(golds, preds):
        gold_arg_cls = set()
        pred_arg_cls = set()
        for gold in _golds:
            _args = {
                (arg["argument"], arg["role"], gold["event_type"])
                for arg in gold["arguments"]
            }
            gold_arg_cls.update(_args)
        for pred in _preds:
            _args = {
                (arg["argument"], arg["role"], pred["event_type"])
                for arg in pred["arguments"]
            }
            pred_arg_cls.update(_args)
        tp += len(gold_arg_cls & pred_arg_cls)
        fp += len(pred_arg_cls - gold_arg_cls)
        fn += len(gold_arg_cls - pred_arg_cls)
    metrics = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def calc_ent(golds, preds):
    """
    Args:
        golds, preds: [(type, index list), ...]
    """
    res = tagging_prf1(golds, preds, type_idx=0)
    return res


def calc_rel(golds, preds):
    gold_ents = []
    pred_ents = []
    for gold, pred in zip(golds, preds):
        gold_ins_ents = []
        for t in gold:
            gold_ins_ents.extend(t[1:])
        gold_ents.append(gold_ins_ents)
        pred_ins_ents = []
        for t in pred:
            pred_ins_ents.extend(t[1:])
        pred_ents.append(pred_ins_ents)

    metrics = {
        "ent": tagging_prf1(gold_ents, pred_ents, type_idx=None),
        "rel": tagging_prf1(golds, preds, type_idx=None),
    }
    return metrics


class MultiPartSpanMetric(MetricBase):
    def _encode_span_to_label_dict(self, span_to_label: dict) -> list:
        span_to_label_list = []
        for key, val in span_to_label.items():
            span_to_label_list.append({"key": key, "val": val})
        return span_to_label_list

    def _decode_span_to_label(self, span_to_label_list: list) -> dict:
        span_to_label = {}
        for content in span_to_label_list:
            span_to_label[tuple(content["key"])] = content["val"]
        return span_to_label

    def get_instances_from_batch(self, raw_batch: dict, out_batch: dict) -> Tuple:
        gold_instances = []
        pred_instances = []

        batch_gold = decompose_batch_into_instances(raw_batch)
        assert len(batch_gold) == len(out_batch["pred"])

        for i, gold in enumerate(batch_gold):
            ins_id = gold["raw"].get("id", generate_random_string_with_datetime())
            # encode to list to make the span_to_label dict json-serializable
            # where the original dict key is a tuple
            span_to_label_list = self._encode_span_to_label_dict(gold["span_to_label"])
            gold["span_to_label"] = span_to_label_list
            gold_instances.append(
                {
                    "id": ins_id,
                    "span_to_label_list": span_to_label_list,
                    "raw_gold_content": gold,
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
        # for general purpose evaluation
        general_gold_spans, general_pred_spans = [], []
        # cls task
        gold_cls_list, pred_cls_list = [], []
        # ent task
        gold_ent_list, pred_ent_list = [], []
        # rel task
        gold_rel_list, pred_rel_list = [], []
        # event task
        gold_event_list, pred_event_list = [], []
        # span task
        gold_span_list, pred_span_list = [], []

        for gold, pred in zip(golds, preds):
            general_gold_spans.append(gold["spans"])
            general_pred_spans.append(pred["spans"])
            span_to_label = self._decode_span_to_label(gold["span_to_label_list"])
            gold_clses, pred_clses = [], []
            gold_ents, pred_ents = [], []
            gold_rels, pred_rels = [], []
            gold_trigger_to_event = defaultdict(
                lambda: {"event_type": "", "arguments": []}
            )
            pred_trigger_to_event = defaultdict(
                lambda: {"event_type": "", "arguments": []}
            )
            gold_events, pred_events = [], []
            gold_spans, pred_spans = [], []

            for span in gold["spans"]:
                if span[0] in span_to_label:
                    label = span_to_label[span[0]]
                    if label["task"] == "cls":
                        gold_clses.append(label["string"])
                    elif label["task"] == "ent":
                        gold_ents.append((label["string"], *span[1:]))
                    elif label["task"] == "rel":
                        gold_rels.append((label["string"], *span[1:]))
                    elif label["task"] == "event":
                        if label["type"] == "lm" and len(span) == 2:
                            gold_trigger_to_event[span[1]]["event_type"] = label[
                                "string"
                            ]
                        elif label["type"] == "lr" and len(span) == 3:
                            gold_trigger_to_event[span[1]]["arguments"].append(
                                {"argument": span[2], "role": label["string"]}
                            )
                else:
                    # span task has no labels
                    gold_spans.append(tuple(span))
            for trigger, item in gold_trigger_to_event.items():
                gold_events.append(
                    {
                        "trigger": trigger,
                        "event_type": item["event_type"],
                        "arguments": item["arguments"],
                    }
                )

            for span in pred["spans"]:
                if span[0] in span_to_label:
                    label = span_to_label[span[0]]
                    if label["task"] == "cls":
                        pred_clses.append(label["string"])
                    elif label["task"] == "ent":
                        pred_ents.append((label["string"], *span[1:]))
                    elif label["task"] == "rel":
                        pred_rels.append((label["string"], *span[1:]))
                    elif label["task"] == "event":
                        if label["type"] == "lm" and len(span) == 2:
                            pred_trigger_to_event[span[1]]["event_type"] = label[
                                "string"
                            ]
                        elif label["type"] == "lr" and len(span) == 3:
                            pred_trigger_to_event[span[1]]["arguments"].append(
                                {"argument": span[2], "role": label["string"]}
                            )
                else:
                    # span task has no labels
                    pred_spans.append(tuple(span))
            for trigger, item in pred_trigger_to_event.items():
                pred_events.append(
                    {
                        "trigger": trigger,
                        "event_type": item["event_type"],
                        "arguments": item["arguments"],
                    }
                )

            gold_cls_list.append(gold_clses)
            pred_cls_list.append(pred_clses)
            gold_ent_list.append(gold_ents)
            pred_ent_list.append(pred_ents)
            gold_rel_list.append(gold_rels)
            pred_rel_list.append(pred_rels)
            gold_event_list.append(gold_events)
            pred_event_list.append(pred_events)
            gold_span_list.append(gold_spans)
            pred_span_list.append(pred_spans)

        metrics = {
            "general_spans": tagging_prf1(
                general_gold_spans, general_pred_spans, type_idx=None
            ),
            "cls": tagging_prf1(gold_cls_list, pred_cls_list, type_idx=None),
            "ent": calc_ent(gold_ent_list, pred_ent_list),
            "rel": calc_rel(gold_rel_list, pred_rel_list),
            "event": {
                "trigger_id": calc_trigger_identification_metrics(
                    gold_event_list, pred_event_list
                ),
                "trigger_cls": calc_trigger_classification_metrics(
                    gold_event_list, pred_event_list
                ),
                "arg_id": calc_arg_identification_metrics(
                    gold_event_list, pred_event_list
                ),
                "arg_cls": calc_arg_classification_metrics(
                    gold_event_list, pred_event_list
                ),
                "char_event": calc_char_event(gold_event_list, pred_event_list),
            },
            "span": tagging_prf1(gold_span_list, pred_span_list, type_idx=None),
        }

        return metrics
