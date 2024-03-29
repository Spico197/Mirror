import random
import re
from collections import defaultdict
from typing import Iterable, Iterator, List, MutableSet, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from rex.data.collate_fn import GeneralCollateFn
from rex.data.transforms.base import CachedTransformBase, CachedTransformOneBase
from rex.metrics import calc_p_r_f1_from_tp_fp_fn
from rex.utils.io import load_json
from rex.utils.iteration import windowed_queue_iter
from rex.utils.logging import logger
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import (
    DebertaV2TokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding

from src.utils import (
    decode_nnw_nsw_thw_mat,
    decode_nnw_thw_mat,
    encode_nnw_nsw_thw_mat,
    encode_nnw_thw_mat,
)

Filled = TypeVar("Filled")


class PaddingMixin:
    max_seq_len: int

    def pad_seq(self, batch_seqs: Iterable[Filled], fill: Filled) -> Iterable[Filled]:
        max_len = max(len(seq) for seq in batch_seqs)
        assert max_len <= self.max_seq_len
        for i in range(len(batch_seqs)):
            batch_seqs[i] = batch_seqs[i] + [fill] * (max_len - len(batch_seqs[i]))
        return batch_seqs

    def pad_mat(
        self, mats: List[torch.Tensor], fill: Union[int, float]
    ) -> List[torch.Tensor]:
        max_len = max(mat.shape[0] for mat in mats)
        assert max_len <= self.max_seq_len
        for i in range(len(mats)):
            num_add = max_len - mats[i].shape[0]
            mats[i] = F.pad(
                mats[i], (0, 0, 0, num_add, 0, num_add), mode="constant", value=fill
            )
        return mats


class PointerTransformMixin:
    tokenizer: BertTokenizerFast
    max_seq_len: int
    space_token: str = "[unused1]"

    def build_ins(
        self,
        query_tokens: list[str],
        context_tokens: list[str],
        answer_indexes: list[list[int]],
        add_context_tokens: list[str] = None,
    ) -> Tuple:
        # -2: cls and sep
        reserved_seq_len = self.max_seq_len - 3 - len(query_tokens)
        # reserve at least 20 tokens
        if reserved_seq_len < 20:
            raise ValueError(
                f"Query {query_tokens} too long: {len(query_tokens)} "
                f"while max seq len is {self.max_seq_len}"
            )

        input_tokens = [self.tokenizer.cls_token]
        input_tokens += query_tokens
        input_tokens += [self.tokenizer.sep_token]
        offset = len(input_tokens)
        input_tokens += context_tokens[:reserved_seq_len]
        available_token_range = range(
            offset, offset + len(context_tokens[:reserved_seq_len])
        )
        input_tokens += [self.tokenizer.sep_token]

        add_context_len = 0
        max_add_context_len = self.max_seq_len - len(input_tokens) - 1
        add_context_flag = False
        if add_context_tokens and len(add_context_tokens) > 0:
            add_context_flag = True
            add_context_len = len(add_context_tokens[:max_add_context_len])
            input_tokens += add_context_tokens[:max_add_context_len]
            input_tokens += [self.tokenizer.sep_token]
        new_tokens = []
        for t in input_tokens:
            if len(t.strip()) > 0:
                new_tokens.append(t)
            else:
                new_tokens.append(self.space_token)
        input_tokens = new_tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        mask = [1]
        mask += [2] * len(query_tokens)
        mask += [3]
        mask += [4] * len(context_tokens[:reserved_seq_len])
        mask += [5]
        if add_context_flag:
            mask += [6] * add_context_len
            mask += [7]
        assert len(mask) == len(input_ids) <= self.max_seq_len

        available_spans = [tuple(i + offset for i in index) for index in answer_indexes]
        available_spans = list(
            filter(
                lambda index: all(i in available_token_range for i in index),
                available_spans,
            )
        )

        token_len = len(input_ids)
        pad_len = self.max_seq_len - token_len
        input_tokens += pad_len * [self.tokenizer.pad_token]
        input_ids += pad_len * [self.tokenizer.pad_token_id]
        mask += pad_len * [0]

        return input_tokens, input_ids, mask, offset, available_spans

    def update_labels(self, data: dict) -> dict:
        bs = len(data["input_ids"])
        seq_len = self.max_seq_len
        labels = torch.zeros((bs, 2, seq_len, seq_len))
        for i, batch_spans in enumerate(data["available_spans"]):
            # offset = data["offset"][i]
            # pad_len = data["mask"].count(0)
            # token_len = seq_len - pad_len
            for span in batch_spans:
                if len(span) == 1:
                    labels[i, :, span[0], span[0]] = 1
                else:
                    for s, e in windowed_queue_iter(span, 2, 1, drop_last=True):
                        labels[i, 0, s, e] = 1
                    labels[i, 1, span[-1], span[0]] = 1
            # labels[i, :, 0:offset, :] = -100
            # labels[i, :, :, 0:offset] = -100
            # labels[i, :, :, token_len:] = -100
            # labels[i, :, token_len:, :] = -100
        data["labels"] = labels
        return data

    def update_consecutive_span_labels(self, data: dict) -> dict:
        bs = len(data["input_ids"])
        seq_len = self.max_seq_len
        labels = torch.zeros((bs, 1, seq_len, seq_len))
        for i, batch_spans in enumerate(data["available_spans"]):
            for span in batch_spans:
                assert span == tuple(sorted(set(span)))
                if len(span) == 1:
                    labels[i, 0, span[0], span[0]] = 1
                else:
                    labels[i, 0, span[0], span[-1]] = 1
        data["labels"] = labels
        return data


class CachedPointerTaggingTransform(CachedTransformBase, PointerTransformMixin):
    def __init__(
        self,
        max_seq_len: int,
        plm_dir: str,
        ent_type2query_filepath: str,
        mode: str = "w2",
        negative_sample_prob: float = 1.0,
    ) -> None:
        super().__init__()

        self.max_seq_len: int = max_seq_len
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(plm_dir)
        self.ent_type2query: dict = load_json(ent_type2query_filepath)
        self.negative_sample_prob = negative_sample_prob

        self.collate_fn: GeneralCollateFn = GeneralCollateFn(
            {
                "input_ids": torch.long,
                "mask": torch.long,
                "labels": torch.long,
            },
            guessing=False,
            missing_key_as_null=True,
        )
        if mode == "w2":
            self.collate_fn.update_before_tensorify = self.update_labels
        elif mode == "cons":
            self.collate_fn.update_before_tensorify = (
                self.update_consecutive_span_labels
            )
        else:
            raise ValueError(f"Mode: {mode} not recognizable")

    def transform(
        self,
        transform_loader: Iterator,
        dataset_name: str = None,
        **kwargs,
    ) -> Iterable:
        final_data = []
        # tp = fp = fn = 0
        for data in transform_loader:
            ent_type2ents = defaultdict(set)
            for ent in data["ents"]:
                ent_type2ents[ent["type"]].add(tuple(ent["index"]))
            for ent_type in self.ent_type2query:
                gold_ents = ent_type2ents[ent_type]
                if (
                    len(gold_ents) < 1
                    and dataset_name == "train"
                    and random.random() > self.negative_sample_prob
                ):
                    # skip negative samples
                    continue
                # res = self.build_ins(ent_type, data["tokens"], gold_ents)
                query = self.ent_type2query[ent_type]
                query_tokens = self.tokenizer.tokenize(query)
                try:
                    res = self.build_ins(query_tokens, data["tokens"], gold_ents)
                except (ValueError, AssertionError):
                    continue
                input_tokens, input_ids, mask, offset, available_spans = res
                ins = {
                    "id": data.get("id", str(len(final_data))),
                    "ent_type": ent_type,
                    "gold_ents": gold_ents,
                    "raw_tokens": data["tokens"],
                    "input_tokens": input_tokens,
                    "input_ids": input_ids,
                    "mask": mask,
                    "offset": offset,
                    "available_spans": available_spans,
                    # labels are dynamically padded in collate fn
                    "labels": None,
                    # "labels": labels.tolist(),
                }
                final_data.append(ins)

        #         # upper bound analysis
        #         pred_spans = set(decode_nnw_thw_mat(labels.unsqueeze(0))[0])
        #         g_ents = set(available_spans)
        #         tp += len(g_ents & pred_spans)
        #         fp += len(pred_spans - g_ents)
        #         fn += len(g_ents - pred_spans)

        # # upper bound results
        # measures = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
        # logger.info(f"Upper Bound: {measures}")

        return final_data

    def predict_transform(self, texts: List[str]):
        dataset = []
        for text_id, text in enumerate(texts):
            data_id = f"Prediction#{text_id}"
            tokens = self.tokenizer.tokenize(text)
            dataset.append(
                {
                    "id": data_id,
                    "tokens": tokens,
                    "ents": [],
                }
            )
        final_data = self(dataset, disable_pbar=True)
        return final_data


class CachedPointerMRCTransform(CachedTransformBase, PointerTransformMixin):
    def __init__(
        self,
        max_seq_len: int,
        plm_dir: str,
        mode: str = "w2",
    ) -> None:
        super().__init__()

        self.max_seq_len: int = max_seq_len
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(plm_dir)

        self.collate_fn: GeneralCollateFn = GeneralCollateFn(
            {
                "input_ids": torch.long,
                "mask": torch.long,
                "labels": torch.long,
            },
            guessing=False,
            missing_key_as_null=True,
        )

        if mode == "w2":
            self.collate_fn.update_before_tensorify = self.update_labels
        elif mode == "cons":
            self.collate_fn.update_before_tensorify = (
                self.update_consecutive_span_labels
            )
        else:
            raise ValueError(f"Mode: {mode} not recognizable")

    def transform(
        self,
        transform_loader: Iterator,
        dataset_name: str = None,
        **kwargs,
    ) -> Iterable:
        final_data = []
        for data in transform_loader:
            try:
                res = self.build_ins(
                    data["query_tokens"],
                    data["context_tokens"],
                    data["answer_index"],
                    data.get("background_tokens"),
                )
            except (ValueError, AssertionError):
                continue
            input_tokens, input_ids, mask, offset, available_spans = res
            ins = {
                "id": data.get("id", str(len(final_data))),
                "gold_spans": sorted(set(tuple(x) for x in data["answer_index"])),
                "raw_tokens": data["context_tokens"],
                "input_tokens": input_tokens,
                "input_ids": input_ids,
                "mask": mask,
                "offset": offset,
                "available_spans": available_spans,
                "labels": None,
            }
            final_data.append(ins)

        return final_data

    def predict_transform(self, data: list[dict]):
        """
        Args:
            data: a list of dict with query, context, and background strings
        """
        dataset = []
        for idx, ins in enumerate(data):
            idx = f"Prediction#{idx}"
            dataset.append(
                {
                    "id": idx,
                    "query_tokens": list(ins["query"]),
                    "context_tokens": list(ins["context"]),
                    "background_tokens": list(ins.get("background")),
                    "answer_index": [],
                }
            )
        final_data = self(dataset, disable_pbar=True, num_samples=0)
        return final_data


class CachedLabelPointerTransform(CachedTransformOneBase):
    """Transform for label-token linking for skip consecutive spans"""

    def __init__(
        self,
        max_seq_len: int,
        plm_dir: str,
        mode: str = "w2",
        label_span: str = "tag",
        include_instructions: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.max_seq_len: int = max_seq_len
        self.mode = mode
        self.label_span = label_span
        self.include_instructions = include_instructions

        self.tokenizer: DebertaV2TokenizerFast = DebertaV2TokenizerFast.from_pretrained(
            plm_dir
        )
        self.lc_token = "[LC]"
        self.lm_token = "[LM]"
        self.lr_token = "[LR]"
        self.i_token = "[I]"
        self.tl_token = "[TL]"
        self.tp_token = "[TP]"
        self.b_token = "[B]"
        num_added = self.tokenizer.add_tokens(
            [
                self.lc_token,
                self.lm_token,
                self.lr_token,
                self.i_token,
                self.tl_token,
                self.tp_token,
                self.b_token,
            ]
        )
        assert num_added == 7

        self.collate_fn: GeneralCollateFn = GeneralCollateFn(
            {
                "input_ids": torch.long,
                "mask": torch.long,
                "labels": torch.long,
                "spans": None,
            },
            guessing=False,
            missing_key_as_null=True,
            # only for pre-training
            discard_missing=False,
        )

        self.collate_fn.update_before_tensorify = self.skip_consecutive_span_labels

    def transform(self, instance: dict, **kwargs):
        # input
        tokens = [self.tokenizer.cls_token]
        mask = [1]
        label_map = {"lc": {}, "lm": {}, "lr": {}}
        # (2, 3): {"type": "lc", "task": "cls/ent/rel/event/hyper_rel/discontinuous_ent", "string": ""}
        span_to_label = {}

        def _update_seq(
            label: str,
            label_type: str,
            task: str = "",
            label_mask: int = 4,
            content_mask: int = 5,
        ):
            if label not in label_map[label_type]:
                label_token_map = {
                    "lc": self.lc_token,
                    "lm": self.lm_token,
                    "lr": self.lr_token,
                }
                label_tag_start_idx = len(tokens)
                tokens.append(label_token_map[label_type])
                mask.append(label_mask)
                label_tag_end_idx = len(tokens) - 1  # exact end position
                label_tokens = self.tokenizer(label, add_special_tokens=False).tokens()
                label_content_start_idx = len(tokens)
                tokens.extend(label_tokens)
                mask.extend([content_mask] * len(label_tokens))
                label_content_end_idx = len(tokens) - 1  # exact end position

                if self.label_span == "tag":
                    start_idx = label_tag_start_idx
                    end_idx = label_tag_end_idx
                elif self.label_span == "content":
                    start_idx = label_content_start_idx
                    end_idx = label_content_end_idx
                else:
                    raise ValueError(f"label_span={self.label_span} is not supported")

                if end_idx == start_idx:
                    label_map[label_type][label] = (start_idx,)
                else:
                    label_map[label_type][label] = (start_idx, end_idx)
                span_to_label[label_map[label_type][label]] = {
                    "type": label_type,
                    "task": task,
                    "string": label,
                }
            return label_map[label_type][label]

        if self.include_instructions:
            instruction = instance.get("instruction")
            if not instruction:
                logger.warning(
                    "include_instructions=True, while the instruction is empty!"
                )
        else:
            instruction = ""
        if instruction:
            tokens.append(self.i_token)
            mask.append(2)
            instruction_tokens = self.tokenizer(
                instruction, add_special_tokens=False
            ).tokens()
            tokens.extend(instruction_tokens)
            mask.extend([3] * len(instruction_tokens))
        types = instance["schema"].get("cls")
        if types:
            for t in types:
                _update_seq(t, "lc", task="cls")
        mention_types = instance["schema"].get("ent")
        if mention_types:
            for mt in mention_types:
                _update_seq(mt, "lm", task="ent")
        discon_ent_types = instance["schema"].get("discontinuous_ent")
        if discon_ent_types:
            for mt in discon_ent_types:
                _update_seq(mt, "lm", task="discontinuous_ent")
        rel_types = instance["schema"].get("rel")
        if rel_types:
            for rt in rel_types:
                _update_seq(rt, "lr", task="rel")
        hyper_rel_schema = instance["schema"].get("hyper_rel")
        if hyper_rel_schema:
            for rel, qualifiers in hyper_rel_schema.items():
                _update_seq(rel, "lr", task="hyper_rel")
                for qualifier in qualifiers:
                    _update_seq(qualifier, "lr", task="hyper_rel")
        event_schema = instance["schema"].get("event")
        if event_schema:
            for event_type, roles in event_schema.items():
                _update_seq(event_type, "lm", task="event")
                for role in roles:
                    _update_seq(role, "lr", task="event")

        text = instance.get("text")
        if text:
            text_tokenized = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            if any(val for val in label_map.values()):
                text_label_token = self.tl_token
            else:
                text_label_token = self.tp_token
            tokens.append(text_label_token)
            mask.append(6)
            remain_token_len = self.max_seq_len - 1 - len(tokens)
            if remain_token_len < 5 and kwargs.get("dataset_name", "train") == "train":
                return None
            text_off = len(tokens)
            text_tokens = text_tokenized.tokens()[:remain_token_len]
            tokens.extend(text_tokens)
            mask.extend([7] * len(text_tokens))
        else:
            text_tokenized = None

        bg = instance.get("bg")
        if bg:
            bg_tokenized = self.tokenizer(
                bg, return_offsets_mapping=True, add_special_tokens=False
            )
            tokens.append(self.b_token)
            mask.append(8)
            remain_token_len = self.max_seq_len - 1 - len(tokens)
            if remain_token_len < 5 and kwargs.get("dataset_name", "train") == "train":
                return None
            bg_tokens = bg_tokenized.tokens()[:remain_token_len]
            tokens.extend(bg_tokens)
            mask.extend([9] * len(bg_tokens))
        else:
            bg_tokenized = None

        tokens.append(self.tokenizer.sep_token)
        mask.append(10)

        # labels
        # spans: [[(ent_type start, ent_type end + 1), (ent s, ent e + 1)]]
        spans = []  # one span may have many parts
        if "cls" in instance["ans"]:
            for t in instance["ans"]["cls"]:
                part = label_map["lc"][t]
                spans.append([part])
        if "ent" in instance["ans"]:
            for ent in instance["ans"]["ent"]:
                label_part = label_map["lm"][ent["type"]]
                position_seq = self.char_to_token_span(
                    ent["span"], text_tokenized, text_off
                )
                spans.append([label_part, position_seq])
        if "discontinuous_ent" in instance["ans"]:
            for ent in instance["ans"]["discontinuous_ent"]:
                label_part = label_map["lm"][ent["type"]]
                ent_span = [label_part]
                for part in ent["span"]:
                    position_seq = self.char_to_token_span(
                        part, text_tokenized, text_off
                    )
                    ent_span.append(position_seq)
                spans.append(ent_span)
        if "rel" in instance["ans"]:
            for rel in instance["ans"]["rel"]:
                label_part = label_map["lr"][rel["relation"]]
                head_position_seq = self.char_to_token_span(
                    rel["head"]["span"], text_tokenized, text_off
                )
                tail_position_seq = self.char_to_token_span(
                    rel["tail"]["span"], text_tokenized, text_off
                )
                spans.append([label_part, head_position_seq, tail_position_seq])
        if "hyper_rel" in instance["ans"]:
            for rel in instance["ans"]["hyper_rel"]:
                label_part = label_map["lr"][rel["relation"]]
                head_position_seq = self.char_to_token_span(
                    rel["head"]["span"], text_tokenized, text_off
                )
                tail_position_seq = self.char_to_token_span(
                    rel["tail"]["span"], text_tokenized, text_off
                )
                # rel_span = [label_part, head_position_seq, tail_position_seq]
                for q in rel["qualifiers"]:
                    q_label_part = label_map["lr"][q["label"]]
                    q_position_seq = self.char_to_token_span(
                        q["span"], text_tokenized, text_off
                    )
                    spans.append(
                        [
                            label_part,
                            head_position_seq,
                            tail_position_seq,
                            q_label_part,
                            q_position_seq,
                        ]
                    )
        if "event" in instance["ans"]:
            for event in instance["ans"]["event"]:
                event_type_label_part = label_map["lm"][event["event_type"]]
                trigger_position_seq = self.char_to_token_span(
                    event["trigger"]["span"], text_tokenized, text_off
                )
                trigger_part = [event_type_label_part, trigger_position_seq]
                spans.append(trigger_part)
                for arg in event["args"]:
                    role_label_part = label_map["lr"][arg["role"]]
                    arg_position_seq = self.char_to_token_span(
                        arg["span"], text_tokenized, text_off
                    )
                    arg_part = [role_label_part, trigger_position_seq, arg_position_seq]
                    spans.append(arg_part)
        if "span" in instance["ans"]:
            # Extractive-QA or Extractive-MRC tasks
            for span in instance["ans"]["span"]:
                span_position_seq = self.char_to_token_span(
                    span["span"], text_tokenized, text_off
                )
                spans.append([span_position_seq])

        if self.mode == "w2":
            new_spans = []
            for parts in spans:
                new_parts = []
                for part in parts:
                    new_parts.append(tuple(range(part[0], part[-1] + 1)))
                new_spans.append(new_parts)
            spans = new_spans
        elif self.mode == "span":
            spans = spans
        else:
            raise ValueError(f"mode={self.mode} is not supported")

        ins = {
            "raw": instance,
            "tokens": tokens,
            "input_ids": self.tokenizer.convert_tokens_to_ids(tokens),
            "mask": mask,
            "spans": spans,
            "label_map": label_map,
            "span_to_label": span_to_label,
            "labels": None,  # labels are calculated dynamically in collate_fn
        }
        return ins

    def char_to_token_span(
        self, span: list[int], tokenized: BatchEncoding, offset: int = 0
    ) -> list[int]:
        token_s = tokenized.char_to_token(span[0])
        token_e = tokenized.char_to_token(span[1] - 1)
        if token_e == token_s:
            position_seq = (offset + token_s,)
        else:
            position_seq = (offset + token_s, offset + token_e)
        return position_seq

    def skip_consecutive_span_labels(self, data: dict) -> dict:
        bs = len(data["input_ids"])
        max_seq_len = max(len(input_ids) for input_ids in data["input_ids"])
        batch_seq_len = min(self.max_seq_len, max_seq_len)
        for i in range(bs):
            data["input_ids"][i] = data["input_ids"][i][:batch_seq_len]
            data["mask"][i] = data["mask"][i][:batch_seq_len]
            assert len(data["input_ids"][i]) == len(data["mask"][i])
            pad_len = batch_seq_len - len(data["mask"][i])
            data["input_ids"][i] = (
                data["input_ids"][i] + [self.tokenizer.pad_token_id] * pad_len
            )
            data["mask"][i] = data["mask"][i] + [0] * pad_len
            data["labels"][i] = encode_nnw_nsw_thw_mat(data["spans"][i], batch_seq_len)

            # # for debugging only
            # pred_spans = decode_nnw_nsw_thw_mat(data["labels"][i].unsqueeze(0))[0]
            # sorted_gold = sorted(set(tuple(x) for x in data["spans"][i]))
            # sorted_pred = sorted(set(tuple(x) for x in pred_spans))
            # if sorted_gold != sorted_pred:
            #     breakpoint()

        # # for pre-training only
        # del data["spans"]

        return data
