from collections import defaultdict
from typing import Any, Iterable, List, MutableSet, Optional, Tuple, TypeVar

import torch
from rex.data.collate_fn import GeneralCollateFn
from rex.data.transforms.base import TransformBase
from rex.utils.io import load_json
from rex.utils.logging import logger
from rex.utils.progress_bar import pbar
from rex.utils.tagging import get_entities_from_tag_seq
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

Filled = TypeVar("Filled")


class CachedPointerTaggingTransform(TransformBase):
    def __init__(
        self, max_seq_len: int, plm_dir: str, ent_type2query_filepath: str
    ) -> None:
        super().__init__()

        self.max_seq_len: int = max_seq_len
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(plm_dir)
        self.ent_type2query: dict = load_json(ent_type2query_filepath)

        self.collate_fn: GeneralCollateFn = GeneralCollateFn(
            {
                "input_ids": torch.long,
                "mask": torch.long,
                "start_labels": torch.long,
                "end_labels": torch.long,
            },
            guessing=False,
            missing_key_as_null=True,
        )
        self.collate_fn.update_before_tensorify = self.dynamic_padding

    def transform(
        self,
        dataset: List[Any],
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        disable_pbar: Optional[bool] = False,
        **kwargs,
    ) -> Iterable:
        final_data = []
        if debug:
            dataset = dataset[:50]
        transform_loader = pbar(dataset, desc=desc, disable=disable_pbar)

        num_tot_ins = 0
        for data in transform_loader:
            ents = get_entities_from_tag_seq(data["tokens"], data["ner_tags"])
            ent_type2ents = defaultdict(set)
            for ent in ents:
                ent_type2ents[ent[1]].add(ent)
            for ent_type in self.ent_type2query:
                _ents = ent_type2ents[ent_type]
                res = self.build_ins(ent_type, data["tokens"], _ents)
                cat_tokens, input_ids, mask, start_labels, end_labels, ent_offset = res
                ins = {
                    "id": data["id"],
                    "ent_type": ent_type,
                    "ent_offset": ent_offset,
                    "gold_ents": _ents,
                    "raw_tokens": data["tokens"],
                    "raw_ner_tags": data["ner_tags"],
                    "tokens": cat_tokens,
                    "input_ids": input_ids,
                    "mask": mask,
                    "start_labels": start_labels,
                    "end_labels": end_labels,
                }
                num_tot_ins += 1
                final_data.append(ins)

        logger.debug(f"Sample: {final_data[:3]}")
        logger.info(f"#ins: {len(final_data)}")
        return final_data

    def build_ins(
        self, ent_type: str, tokens: List[str], ents: MutableSet[Tuple]
    ) -> Tuple:
        query = self.ent_type2query[ent_type]
        query_tokens = self.tokenizer.tokenize(query)

        # -2: cls and sep
        reserved_seq_len = self.max_seq_len - 2 - len(query_tokens)
        # reserve at least 20 tokens
        assert reserved_seq_len >= 20

        cat_tokens = [self.tokenizer.cls_token]
        cat_tokens += query_tokens
        cat_tokens += tokens[:reserved_seq_len]
        cat_tokens += [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(cat_tokens)
        mask = [1]
        mask += [2] * len(query_tokens)
        mask += [3] * len(tokens[:reserved_seq_len])
        mask += [4]
        assert len(mask) == len(input_ids) <= self.max_seq_len

        # construct labels, +1 means cls
        start_labels = [0] * len(input_ids)
        end_labels = [0] * len(input_ids)
        ent_offset = 1 + len(query_tokens)
        for ent in ents:
            assert ent[1] == ent_type
            start_pos, end_pos_plus_one = ent[2]
            _sp = ent_offset + start_pos
            _ep_p1 = ent_offset + end_pos_plus_one
            if 0 <= _sp < _ep_p1 <= len(input_ids):
                # available ent
                start_labels[_sp] = 1
                # -1 to remove `end + 1` offset
                end_labels[_ep_p1 - 1] = 1

        start_labels[0] = start_labels[-1] = -100
        end_labels[0] = end_labels[-1] = -100
        start_labels[1 : 1 + len(query_tokens)] = [-100] * len(  # noqa: E203
            query_tokens
        )
        end_labels[1 : 1 + len(query_tokens)] = [-100] * len(query_tokens)  # noqa: E203

        return cat_tokens, input_ids, mask, start_labels, end_labels, ent_offset

    def dynamic_padding(self, data: dict) -> dict:
        data["input_ids"] = self.padding(data["input_ids"], self.tokenizer.pad_token_id)
        data["mask"] = self.padding(data["mask"], 0)
        data["start_labels"] = self.padding(data["start_labels"], -100)
        data["end_labels"] = self.padding(data["end_labels"], -100)
        return data

    def padding(self, batch_seqs: Iterable[Filled], fill: Filled) -> Iterable[Filled]:
        max_len = max(len(seq) for seq in batch_seqs)
        assert max_len <= self.max_seq_len
        for i in range(len(batch_seqs)):
            batch_seqs[i] = batch_seqs[i] + [fill] * (max_len - len(batch_seqs[i]))
        return batch_seqs

    def predict_transform(self, texts: List[str]):
        dataset = []
        for text_id, text in enumerate(texts):
            data_id = f"Prediction#{text_id}"
            tokens = self.tokenizer.tokenize(text)
            ner_tags = ["O"] * len(tokens)
            dataset.append(
                {
                    "id": data_id,
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
            )
        final_data = self.transform(dataset, disable_pbar=True)
        return final_data
