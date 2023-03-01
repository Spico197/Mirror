import math
from collections import defaultdict
from typing import List

import torch.optim as optim
from rex import accelerator
from rex.data.data_manager import DataManager
from rex.data.dataset import CachedDataset
from rex.tasks.simple_metric_task import SimpleMetricTask
from rex.utils.io import load_jsonlines
from rex.utils.registry import register
from transformers.optimization import get_linear_schedule_with_warmup

from .metric import MrcNERMetric
from .model import PlmMRCModel
from .transform import CachedPointerTaggingTransform


@register("task")
class MrcTaggingTask(SimpleMetricTask):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

    def init_transform(self):
        return CachedPointerTaggingTransform(
            self.config.max_seq_len,
            self.config.plm_dir,
            self.config.ent_type2query_filepath,
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
        m = PlmMRCModel(
            self.config.plm_dir,
            self.config.dropout,
        )
        return m

    def init_metric(self):
        return MrcNERMetric()

    def init_optimizer(self):
        rest_params = [
            x[1]
            for x in filter(
                lambda name_params: "plm" not in name_params[0],
                self.model.named_parameters(),
            )
        ]
        optimizer_grouped_parameters = [
            {"params": self.model.plm.parameters()},
            {"params": rest_params, "lr": self.config.other_learning_rate},
        ]
        return optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def init_lr_scheduler(self):
        num_training_steps = (
            len(self.data_manager.train_loader) * self.config.num_epochs
        )
        num_warmup_steps = math.floor(
            num_training_steps * self.config.warmup_proportion
        )
        return get_linear_schedule_with_warmup(
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
            batch_out = self.model(**batch, decode=True)
            for _id, _pred in zip(batch["id"], batch_out["pred"]):
                id2ents[_id].update(_pred)
        results = [id2ents[_id] for _id in text_ids]

        return results
