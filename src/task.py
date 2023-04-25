import math
import re
from collections import defaultdict
from datetime import datetime
from typing import List

import torch.optim as optim
from rex import accelerator
from rex.data.data_manager import DataManager
from rex.data.dataset import CachedDataset
from rex.tasks.simple_metric_task import SimpleMetricTask
from rex.utils.batch import decompose_batch_into_instances
from rex.utils.dict import flatten_dict
from rex.utils.io import load_jsonlines
from rex.utils.registry import register
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from .metric import MrcNERMetric, MrcSpanMetric, MultiPartSpanMetric
from .model import (
    MrcGlobalPointerModel,
    MrcPointerMatrixModel,
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
        num_training_steps = (
            len(self.data_manager.train_loader) * self.config.num_epochs
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
        loader = accelerator.prepare_data_loader(loader)
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
        # to prepare input device
        loader = accelerator.prepare_data_loader(loader)
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


@register("task")
class SchemaGuidedInstructBertTask(MrcTaggingTask):
    def init_transform(self):
        return CachedLabelPointerTransform(
            self.config.max_seq_len,
            self.config.plm_dir,
            mode=self.config.mode,
        )

    def init_model(self):
        m = SchemaGuidedInstructBertModel(
            self.config.plm_dir,
            use_rope=self.config.use_rope,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
        )
        return m

    def init_metric(self):
        return MultiPartSpanMetric()


if __name__ == "__main__":
    from rex.utils.config import ConfigParser

    config = ConfigParser.parse_cmd(cmd_args=["-dc", "conf/ner.yaml"])

    task = MrcTaggingTask(
        config,
        initialize=True,
        makedirs=True,
        dump_configfile=True,
    )
    # task.load(
    #     # "outputs/Mirror_RobertaBaseWwm_Cons_MsraMrc/ckpt/MrcGlobalPointerModel.best.pth",
    #     # "outputs/Mirror_RobertaBaseWwm_W2_MsraMrc_HyperParamExp1/ckpt/MrcGlobalPointerModel.best.pth",
    #     config.base_model_path,
    #     load_config=False,
    #     load_model=True,
    #     load_optimizer=False,
    #     load_history=False,
    # )
    task.train()
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
