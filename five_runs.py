import numpy as np
from omegaconf import DictConfig
from rex.utils.config import ConfigParser
from rex.utils.dict import get_dict_content
from rex.utils.logging import logger

from src.task import MrcTaggingTask

if __name__ == "__main__":
    losses = []
    metric_results = []

    for random_seed in [17, 127, 1227, 12227, 122227]:
        config: DictConfig = ConfigParser.parse_cmd(cmd_args=["-dc", "custom.yaml"])
        config.final_eval_on_test = False
        config.random_seed = random_seed
        config.task_name += f"_{random_seed}"

        task = MrcTaggingTask(
            config,
            initialize=True,
            makedirs=True,
            dump_configfile=True,
        )
        task.train()

        logger.info("Loading best ckpt")
        task.load_best_ckpt()
        test_loss, test_measures = task.eval(
            "test", verbose=True, dump=True, postfix="final"
        )
        losses.append(test_loss)

        metric_result = get_dict_content(test_measures, config.best_metric_field)
        metric_results.append(metric_result)

    logger.info(f"losses: {losses}")
    logger.info(f"losses - mean: {np.mean(losses)}, std: {np.std(losses)}")
    logger.info(f"metric_results: {metric_results}")
    logger.info(
        f"metric_results - mean: {np.mean(metric_results)}, std: {np.std(metric_results)}"
    )
