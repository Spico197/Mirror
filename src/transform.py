import random
from collections import defaultdict
from typing import Iterable, Iterator, List, MutableSet, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from rex.data.collate_fn import GeneralCollateFn
from rex.data.transforms.base import CachedTransformBase
from rex.metrics import calc_p_r_f1_from_tp_fp_fn
from rex.utils.io import load_json
from rex.utils.logging import logger
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from src.utils import decode_nnw_thw_mat, encode_nnw_thw_matrix

Filled = TypeVar("Filled")


class CachedPointerTaggingTransform(CachedTransformBase):
    def __init__(
        self,
        max_seq_len: int,
        plm_dir: str,
        ent_type2query_filepath: str,
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
                "mask": torch.int8,
                "labels": torch.int8,
            },
            guessing=False,
            missing_key_as_null=True,
        )
        self.collate_fn.update_before_tensorify = self.dynamic_padding

    def transform(
        self,
        transform_loader: Iterator,
        inference_mode: Optional[bool] = False,
        **kwargs,
    ) -> Iterable:
        final_data = []
        tp = fp = fn = 0
        for data in transform_loader:
            ent_type2ents = defaultdict(set)
            for ent in data["ents"]:
                ent_type2ents[ent["type"]].add(tuple(ent["index"]))
            for ent_type in self.ent_type2query:
                _ents = ent_type2ents[ent_type]
                if (
                    len(_ents) < 1
                    and not inference_mode
                    and random.random() > self.negative_sample_prob
                ):
                    # skip negative samples
                    continue
                res = self.build_ins(ent_type, data["tokens"], _ents)
                input_tokens, input_ids, mask, labels, offset, available_ents = res
                ins = {
                    "id": data["id"],
                    "ent_type": ent_type,
                    "gold_ents": _ents,
                    "raw_tokens": data["tokens"],
                    "input_tokens": input_tokens,
                    "input_ids": input_ids,
                    "mask": mask,
                    "offset": offset,
                    "available_ents": available_ents,
                    # labels are dynamically padded in collate fn
                    "labels": labels,
                }
                final_data.append(ins)

                # upper bound analysis
                pred_spans = set(decode_nnw_thw_mat(labels.unsqueeze(0))[0])
                g_ents = set(available_ents)
                tp += len(g_ents & pred_spans)
                fp += len(pred_spans - g_ents)
                fn += len(g_ents - pred_spans)

        # upper bound results
        measures = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
        logger.info(f"Upper Bound: {measures}")

        return final_data

    def build_ins(
        self, ent_type: str, tokens: List[str], ents: MutableSet[Tuple[int]]
    ) -> Tuple:
        query = self.ent_type2query[ent_type]
        query_tokens = self.tokenizer.tokenize(query)

        # -2: cls and sep
        reserved_seq_len = self.max_seq_len - 3 - len(query_tokens)
        # reserve at least 20 tokens
        if reserved_seq_len < 20:
            raise ValueError(
                f"Query {query} too long: {len(query_tokens)} "
                f"while max seq len is {self.max_seq_len}"
            )

        input_tokens = [self.tokenizer.cls_token]
        input_tokens += query_tokens
        input_tokens += [self.tokenizer.sep_token]
        offset = len(input_tokens)
        input_tokens += tokens[:reserved_seq_len]
        available_token_range = range(offset, offset + len(tokens[:reserved_seq_len]))
        input_tokens += [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        mask = [1]
        mask += [2] * len(query_tokens)
        mask += [3]
        mask += [4] * len(tokens[:reserved_seq_len])
        mask += [5]
        assert len(mask) == len(input_ids) <= self.max_seq_len

        available_ents = [tuple(i + offset for i in index) for index in ents]
        available_ents = list(
            filter(
                lambda index: all(i in available_token_range for i in index),
                available_ents,
            )
        )
        labels = encode_nnw_thw_matrix(available_ents, len(input_ids))

        return input_tokens, input_ids, mask, labels, offset, available_ents

    def dynamic_padding(self, data: dict) -> dict:
        data["input_ids"] = self.pad_seq(data["input_ids"], self.tokenizer.pad_token_id)
        data["mask"] = self.pad_seq(data["mask"], 0)
        data["labels"] = self.pad_mat(data["labels"], -100)
        data["labels"] = [mat.tolist() for mat in data["labels"]]
        return data

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
        final_data = self(dataset, disable_pbar=True, inference_mode=True)
        return final_data