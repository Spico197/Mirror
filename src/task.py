import math
import re
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import List

import torch
import torch.optim as optim
from rex.data.data_manager import DataManager
from rex.data.dataset import CachedDataset, StreamReadDataset
from rex.tasks.simple_metric_task import SimpleMetricTask
from rex.utils.batch import decompose_batch_into_instances
from rex.utils.dict import flatten_dict
from rex.utils.io import load_jsonlines
from rex.utils.registry import register
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_cosine_schedule_with_warmup

from .metric import MrcNERMetric, MrcSpanMetric, MultiPartSpanMetric
from .model import (
    MrcGlobalPointerModel,
    SchemaGuidedInstructBertModel,
)
from .transform import (
    CachedLabelPointerTransform,
    CachedPointerMRCTransform,
    CachedPointerTaggingTransform,
)


@register("task")
class MrcTaggingTask(SimpleMetricTask):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

    def after_initialization(self):
        now_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.tb_logger: SummaryWriter = SummaryWriter(
            log_dir=self.task_path / "tb_summary" / now_string,
            comment=self.config.comment,
        )

    def after_whole_train(self):
        self.tb_logger.close()

    def get_grad_norm(self):
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         grads = param.grad.detach().data
        #         grad_norm = (grads.norm(p=2) / grads.numel()).item()
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def log_loss(
        self, idx: int, loss_item: float, step_or_epoch: str, dataset_name: str
    ):
        self.tb_logger.add_scalar(
            f"loss/{dataset_name}/{step_or_epoch}", loss_item, idx
        )
        # self.tb_logger.add_scalars(
        #     "lr",
        #     {
        #         str(i): self.optimizer.param_groups[i]["lr"]
        #         for i in range(len(self.optimizer.param_groups))
        #     },
        #     idx,
        # )
        self.tb_logger.add_scalar("lr", self.optimizer.param_groups[0]["lr"], idx)
        self.tb_logger.add_scalar("grad_norm_total", self.get_grad_norm(), idx)

    def log_metrics(
        self, idx: int, metrics: dict, step_or_epoch: str, dataset_name: str
    ):
        metrics = flatten_dict(metrics)
        self.tb_logger.add_scalars(f"{dataset_name}/{step_or_epoch}", metrics, idx)

    def init_transform(self):
        return CachedPointerTaggingTransform(
            self.config.max_seq_len,
            self.config.plm_dir,
            self.config.ent_type2query_filepath,
            mode=self.config.mode,
            negative_sample_prob=self.config.negative_sample_prob,
        )

    def init_data_manager(self):
        return DataManager(
            self.accelerator,
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            CachedDataset,
            self.transform,
            load_jsonlines,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.transform.collate_fn,
            use_stream_transform=False,
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.dump_cache_dir,
            regenerate_cache=self.config.regenerate_cache,
        )

    def init_model(self):
        # m = MrcPointerMatrixModel(
        m = MrcGlobalPointerModel(
            self.config.plm_dir,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
            mode=self.config.mode,
        )
        return m

    def init_metric(self):
        return MrcNERMetric()

    def init_optimizer(self):
        no_decay = r"(embedding|LayerNorm|\.bias$)"
        plm_lr = r"^plm\."
        non_trainable = r"^plm\.(emb|encoder\.layer\.[0-3])"

        param_groups = []
        for name, param in self.model.named_parameters():
            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            if re.search(non_trainable, name):
                param.requires_grad = False
            if not re.search(plm_lr, name):
                lr = self.config.other_learning_rate
            if re.search(no_decay, name):
                weight_decay = 0.0
            param_groups.append(
                {"params": param, "lr": lr, "weight_decay": weight_decay}
            )
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    def init_lr_scheduler(self):
        num_training_steps = int(
            len(self.data_manager.train_loader)
            * self.config.num_epochs
            * self.accelerator.num_processes
        )
        num_warmup_steps = math.floor(
            num_training_steps * self.config.warmup_proportion
        )
        # return get_linear_schedule_with_warmup(
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def predict_api(self, texts: List[str], **kwargs):
        raw_dataset = self.transform.predict_transform(texts)
        text_ids = sorted(list({ins["id"] for ins in raw_dataset}))
        loader = self.data_manager.prepare_loader(raw_dataset)
        # to prepare input device
        loader = self.accelerator.prepare_data_loader(loader)
        id2ents = defaultdict(set)
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            for _id, _pred in zip(batch["id"], batch_out["pred"]):
                id2ents[_id].update(_pred)
        results = [id2ents[_id] for _id in text_ids]

        return results


@register("task")
class MrcQaTask(MrcTaggingTask):
    def init_transform(self):
        return CachedPointerMRCTransform(
            self.config.max_seq_len,
            self.config.plm_dir,
            mode=self.config.mode,
        )

    def init_model(self):
        # m = MrcPointerMatrixModel(
        m = MrcGlobalPointerModel(
            self.config.plm_dir,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
            mode=self.config.mode,
        )
        return m

    def init_metric(self):
        return MrcSpanMetric()

    def predict_api(self, data: list[dict], **kwargs):
        """
        Args:
            data: a list of dict with query, context, and background strings
        """
        raw_dataset = self.transform.predict_transform(data)
        loader = self.data_manager.prepare_loader(raw_dataset)
        results = []
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            batch["pred"] = batch_out["pred"]
            instances = decompose_batch_into_instances(batch)
            for ins in instances:
                preds = ins["pred"]
                ins_results = []
                for index_list in preds:
                    ins_result = []
                    for i in index_list:
                        ins_result.append(ins["raw_tokens"][i])
                    ins_results.append(("".join(ins_result), tuple(index_list)))
                results.append(ins_results)

        return results


class StreamReadDatasetWithLen(StreamReadDataset):
    def __len__(self):
        return 631346


def load_jsonlines_with_num(filepath, num_lines: int = None, **kwargs):
    data = load_jsonlines(filepath, **kwargs)
    if num_lines is not None and isinstance(num_lines, int):
        return data[:num_lines]
    else:
        return data


@register("task")
class SchemaGuidedInstructBertTask(MrcTaggingTask):
    # def __init__(self, config, **kwargs) -> None:
    #     super().__init__(config, **kwargs)

    #     from watchmen import ClientMode, WatchClient

    #     client = WatchClient(
    #         id=config.task_name,
    #         gpus=[4],
    #         req_gpu_num=1,
    #         mode=ClientMode.SCHEDULE,
    #         server_host="127.0.0.1",
    #         server_port=62333,
    #     )
    #     client.wait()

    # def init_lr_scheduler(self):
    #     num_training_steps = int(
    #         631346 / self.config.train_batch_size
    #         * self.config.num_epochs
    #         * accelerator.num_processes
    #     )
    #     num_warmup_steps = math.floor(
    #         num_training_steps * self.config.warmup_proportion
    #     )
    #     # return get_linear_schedule_with_warmup(
    #     return get_cosine_schedule_with_warmup(
    #         self.optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,
    #     )

    def init_transform(self):
        self.transform: CachedLabelPointerTransform
        return CachedLabelPointerTransform(
            self.config.max_seq_len,
            self.config.plm_dir,
            mode=self.config.mode,
            label_span=self.config.label_span,
            include_instructions=self.config.get("include_instructions", True),
        )

    def init_data_manager(self):
        if self.config.get("stream_mode", False):
            DatasetClass = StreamReadDatasetWithLen
            transform = self.transform.transform
        else:
            DatasetClass = CachedDataset
            transform = self.transform
        return DataManager(
            self.accelerator,
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            DatasetClass,
            transform,
            partial(
                load_jsonlines_with_num,
                num_lines=self.config.get("train_data_num", None),
            ),
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.transform.collate_fn,
            use_stream_transform=self.config.get("stream_mode", False),
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.dump_cache_dir,
            regenerate_cache=self.config.regenerate_cache,
        )

    def init_model(self):
        self.model = SchemaGuidedInstructBertModel(
            self.config.plm_dir,
            vocab_size=len(self.transform.tokenizer),
            use_rope=self.config.use_rope,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
        )

        if self.config.get("base_model_path"):
            self.load(
                self.config.base_model_path,
                load_config=False,
                load_model=True,
                load_optimizer=False,
                load_history=False,
            )
        return self.model

    def init_optimizer(self):
        no_decay = r"(embedding|LayerNorm|\.bias$)"
        plm_lr = r"^plm\."
        # non_trainable = r"^plm\.(emb|encoder\.layer\.[0-3])"
        non_trainable = "no_non_trainable"

        param_groups = []
        for name, param in self.model.named_parameters():
            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            if re.search(non_trainable, name):
                param.requires_grad = False
            if not re.search(plm_lr, name):
                lr = self.config.other_learning_rate
            if re.search(no_decay, name):
                weight_decay = 0.0
            param_groups.append(
                {"params": param, "lr": lr, "weight_decay": weight_decay}
            )
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    def init_metric(self):
        return MultiPartSpanMetric()

    def _convert_span_to_string(self, span, token_ids, tokenizer):
        string = ""
        if len(span) == 0 or len(span) > 2:
            pass
        elif len(span) == 1:
            string = tokenizer.decode(token_ids[span[0]])
        elif len(span) == 2:
            string = tokenizer.decode(token_ids[span[0] : span[1] + 1])
        return (string, self.reset_position(token_ids, span))

    def reset_position(self, token_ids: list[int], span: list[int]) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            input_ids = token_ids.cpu().tolist()
        if len(span) < 1:
            return span

        tp_token_id, tl_token_id = self.transform.tokenizer.convert_tokens_to_ids(
            [self.transform.tp_token, self.transform.tl_token]
        )
        offset = 0
        if tp_token_id in input_ids:
            offset = input_ids.index(tp_token_id) + 1
        elif tl_token_id in input_ids:
            offset = input_ids.index(tl_token_id) + 1
        return [i - offset for i in span]

    def predict_api(self, data: list[dict], **kwargs):
        """
        Args:
            data: a list of dict in UDI:
                {
                    "id": str,
                    "instruction": str,
                    "schema": {
                        "ent": list,
                        "rel": list,
                        "event": dict,
                        "cls": list,
                        "discontinuous_ent": list,
                        "hyper_rel": dict
                    },
                    "text": str,
                    "bg": str,
                    "ans": {},  # empty dict
                }
        """
        raw_dataset = [self.transform.transform(d) for d in data]
        loader = self.data_manager.prepare_loader(raw_dataset)
        results = []
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            batch["pred"] = batch_out["pred"]
            instances = decompose_batch_into_instances(batch)
            for ins in instances:
                pred_clses = []
                pred_ents = []
                pred_rels = []
                pred_trigger_to_event = defaultdict(
                    lambda: {"event_type": "", "arguments": []}
                )
                pred_events = []
                pred_spans = []
                pred_discon_ents = []
                pred_hyper_rels = []
                raw_schema = ins["raw"]["schema"]
                for multi_part_span in ins["pred"]:
                    span = tuple(multi_part_span)
                    span_to_label = ins["span_to_label"]
                    if span[0] in span_to_label:
                        label = span_to_label[span[0]]
                        if label["task"] == "cls" and len(span) == 1:
                            pred_clses.append(label["string"])
                        elif label["task"] == "ent" and len(span) == 2:
                            string = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_ents.append((label["string"], string))
                        elif label["task"] == "rel" and len(span) == 3:
                            head = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            tail = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_rels.append((label["string"], head, tail))
                        elif label["task"] == "event":
                            if label["type"] == "lm" and len(span) == 2:
                                pred_trigger_to_event[span[1]]["event_type"] = label["string"]  # fmt: skip
                            elif label["type"] == "lr" and len(span) == 3:
                                arg = self._convert_span_to_string(
                                    span[2], ins["input_ids"], self.transform.tokenizer
                                )
                                pred_trigger_to_event[span[1]]["arguments"].append(
                                    {"argument": arg, "role": label["string"]}
                                )
                        elif label["task"] == "discontinuous_ent" and len(span) > 1:
                            parts = [
                                self._convert_span_to_string(
                                    part, ins["input_ids"], self.transform.tokenizer
                                )
                                for part in span[1:]
                            ]
                            string = " ".join([part[0] for part in parts])
                            position = []
                            for part in parts:
                                position.append(part[1])
                            pred_discon_ents.append(
                                (label["string"], string, self.reset_position(position))
                            )
                        elif label["task"] == "hyper_rel" and len(span) == 5 and span[3] in span_to_label:  # fmt: skip
                            q_label = span_to_label[span[3]]
                            span_1 = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            span_2 = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            span_4 = self._convert_span_to_string(
                                span[4], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_hyper_rels.append((label["string"], span_1, span_2, q_label["string"], span_4))  # fmt: skip
                    else:
                        # span task has no labels
                        pred_token_ids = []
                        for part in span:
                            _pred_token_ids = [ins["input_ids"][i] for i in part]
                            pred_token_ids.extend(_pred_token_ids)
                        span_string = self.transform.tokenizer.decode(pred_token_ids)
                        pred_spans.append(
                            (
                                span_string,
                                tuple(
                                    [
                                        tuple(
                                            self.reset_position(
                                                ins["input_ids"].cpu().tolist(), part
                                            )
                                        )
                                        for part in span
                                    ]
                                ),
                            )
                        )
                for trigger, item in pred_trigger_to_event.items():
                    trigger = self._convert_span_to_string(
                        trigger, ins["input_ids"], self.transform.tokenizer
                    )
                    if item["event_type"] not in raw_schema["event"]:
                        continue
                    legal_roles = raw_schema["event"][item["event_type"]]
                    pred_events.append(
                        {
                            "trigger": trigger,
                            "event_type": item["event_type"],
                            "arguments": [
                                arg
                                for arg in filter(
                                    lambda arg: arg["role"] in legal_roles,
                                    item["arguments"],
                                )
                            ],
                        }
                    )
                results.append(
                    {
                        "id": ins["raw"]["id"],
                        "results": {
                            "cls": pred_clses,
                            "ent": pred_ents,
                            "rel": pred_rels,
                            "event": pred_events,
                            "span": pred_spans,
                            "discon_ent": pred_discon_ents,
                            "hyper_rel": pred_hyper_rels,
                        },
                    }
                )

        return results


if __name__ == "__main__":
    pass
    # further_finetune()

    # from rex.utils.config import ConfigParser

    # config = ConfigParser.parse_cmd(cmd_args=["-dc", "conf/ner.yaml"])
    # config = ConfigParser.parse_cmd(cmd_args=["-dc", "conf/mirror-ace05en.yaml"])

    # task = MrcTaggingTask(
    #     config,
    #     initialize=True,
    #     makedirs=True,
    #     dump_configfile=True,
    # )
    # task = SchemaGuidedInstructBertTask.from_taskdir(
    #     "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05EN_Rel",
    #     initialize=True,
    #     load_config=True,
    #     dump_configfile=False,
    # )
    # task = SchemaGuidedInstructBertTask(
    #     config,
    #     initialize=True,
    #     makedirs=True,
    #     dump_configfile=False,
    # )
    # task.load(
    #     "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05EN_NerRelEvent/ckpt/SchemaGuidedInstructBertModel.epoch.0.pth",
    #     load_config=False,
    # )
    # task.eval("test", verbose=True, dump=True, dump_middle=True, postfix="re_eval")
    # task.load(
    #     # "outputs/Mirror_RobertaBaseWwm_Cons_MsraMrc/ckpt/MrcGlobalPointerModel.best.pth",
    #     # "outputs/Mirror_RobertaBaseWwm_W2_MsraMrc_HyperParamExp1/ckpt/MrcGlobalPointerModel.best.pth",
    #     config.base_model_path,
    #     load_config=False,
    #     load_model=True,
    #     load_optimizer=False,
    #     load_history=False,
    # )
    # task.train()
    # task = MrcTaggingTask.from_taskdir(
    #     "outputs/Mirror_W2_MSRAv2_NER",
    #     initialize=True,
    #     dump_configfile=False,
    #     load_config=True,
    # )
    # for name, _ in task.model.named_parameters():
    #     print(name)
    # task.eval("test", verbose=True, dump=True, dump_middle=True, postfix="re_eval.0.1")

    # task = MrcQaTask(
    #     config,
    #     initialize=True,
    #     makedirs=True,
    #     dump_configfile=True,
    # )
    # task.train()
    # task.eval("dev", verbose=True, dump=True, dump_middle=True, postfix="re_eval")
