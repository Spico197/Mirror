import os

from rex.utils.logging import logger

from src.task import MrcTaggingTask

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    task = MrcTaggingTask.from_taskdir(
        "outputs/bert_mrc_ner",
        load_best_model=True,
        update_config={
            "skip_train": True,
            "debug_mode": False,
        },
    )

    cases = ["李大嘴在同福有限责任公司当厨子。", "佟湘玉是同福有限责任公司的老板娘。"]
    logger.info(f"Cases: {cases}")

    ents = task.predict(cases)
    logger.info(f"Results: {ents}")
