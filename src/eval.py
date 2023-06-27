from pathlib import Path

import pandas as pd
from rex.utils.initialization import set_seed_and_log_path
from rex.utils.io import load_json
from rich.console import Console
from rich.table import Table

from src.task import SchemaGuidedInstructBertTask

set_seed_and_log_path(log_path="eval.log")


if __name__ == "__main__":
    task_dir = "mirror_outputs/Mirror_Pretrain_AllExcluded_2"
    task: SchemaGuidedInstructBertTask = SchemaGuidedInstructBertTask.from_taskdir(
        task_dir,
        load_best_model=True,
        initialize=False,
        dump_configfile=False,
        update_config={
            "regenerate_cache": True,
            "eval_on_data": ["dev"],
            "select_best_on_data": "dev",
            "select_best_by_key": "metric",
            "best_metric_field": "general_spans.micro.f1",
            "eval_batch_size": 32,
        },
    )
    table = Table(title=task_dir)

    data_pairs = [
        # fmt: off

        # UIE eval data
        ["ent_ace04_test", "resources/Mirror/uie/ent/ace04/test.jsonl"],
        ["ent_ace05_test", "resources/Mirror/uie/ent/ace05/test.jsonl"],
        ["ent_conll03_test", "resources/Mirror/uie/ent/conll03/test.jsonl"],
        ["rel_ace05_test", "resources/Mirror/uie/rel/ace05-rel/test.jsonl"],
        ["rel_conll04_test", "resources/Mirror/uie/rel/conll04/test.jsonl"],
        ["rel_nyt_test", "resources/Mirror/uie/rel/nyt/test.jsonl"],
        ["rel_scierc_test", "resources/Mirror/uie/rel/scierc/test.jsonl"],
        ["event_ace05_test", "resources/Mirror/uie/event/ace05-evt/test.jsonl"],
        ["event_casie_test", "resources/Mirror/uie/event/casie/test.jsonl"],
        ["absa_14res_test", "resources/Mirror/uie/absa/14res/test.jsonl"],
        ["absa_14lap_test", "resources/Mirror/uie/absa/14lap/test.jsonl"],
        ["absa_15res_test", "resources/Mirror/uie/absa/15res/test.jsonl"],
        ["absa_16res_test", "resources/Mirror/uie/absa/16res/test.jsonl"],

        # discontinuous NER
        ["discontinuous_ent", "resources/Mirror/new_abilities_v2/cadec/new/test.jsonl"],

        # hyper-RE
        ["hyper_rel", "resources/Mirror/new_abilities_v2/HyperRED/new/test.jsonl"],

        # zero-shot NER
        ["ent_movie", "resources/Mirror/v1.3/ent/en/MIT_MOVIE_Review/instructed/test.jsonl"],
        ["ent_restaurant", "resources/Mirror/v1.3/ent/en/MIT_Restaurant_Review/instructed/test.jsonl"],
        ["ent_ai", "resources/Mirror/v1.3/ent/en/CrossNER_AI/instructed/test.jsonl"],
        ["ent_literature", "resources/Mirror/v1.3/ent/en/CrossNER_literature/instructed/test.jsonl"],
        ["ent_music", "resources/Mirror/v1.3/ent/en/CrossNER_music/instructed/test.jsonl"],
        ["ent_politics", "resources/Mirror/v1.3/ent/en/CrossNER_politics/instructed/test.jsonl"],
        ["ent_science", "resources/Mirror/v1.3/ent/en/CrossNER_science/instructed/test.jsonl"],
        # mrc
        ["span_squad2", "resources/Mirror/v1.3/span/en/squad_v2/dev.jsonl"],
        # glue
        ["cls_glue_cola", "resources/Mirror/v1.3/cls/en/CoLA/formated/dev.jsonl"],
        ["cls_glue_qqp", "resources/Mirror/v1.3/cls/en/QQP/new/dev.jsonl"],
        ["cls_glue_mnli", "resources/Mirror/v1.3/cls/en/MNLI/formated/MNLI_dev.jsonl"],
        ["cls_glue_sst2", "resources/Mirror/v1.3/cls/en/SST-2/instructed/SST-2_dev.jsonl"],
        ["cls_glue_qnli", "resources/Mirror/v1.3/cls/en/QNLI/processed/QNLI_dev.jsonl"],
        ["cls_glue_rte", "resources/Mirror/v1.3/cls/en/RTE/formated/RTE_dev.jsonl"],
        ["cls_glue_mrpc", "resources/Mirror/v1.3/cls/en/MRPC/formated/dev.jsonl"],
        # fmt: on
    ]

    eval_res = {"task": [], "dataset": [], "metric_val": []}
    table.add_column("Task", justify="left", style="cyan")
    table.add_column("Dataset", justify="left", style="magenta")
    table.add_column("Metric (%)", justify="right", style="green")
    for dname, fpath in data_pairs:
        dname = dname.lower()
        task.data_manager.update_datapath(dname, fpath)
        _, res = task.eval(dname, verbose=True, dump=True, dump_middle=True)
        # res = load_json(Path(task_dir) / "measures" / f"{dname}.json")["metrics"]
        if dname.startswith("ent_"):
            eval_res["task"].append("ent")
            eval_res["dataset"].append(dname)
            eval_res["metric_val"].append(res["ent"]["micro"]["f1"])
        elif dname.startswith("rel_"):
            eval_res["task"].append("rel")
            eval_res["dataset"].append(dname)
            eval_res["metric_val"].append(res["rel"]["rel"]["micro"]["f1"])
        elif dname.startswith("event_"):
            eval_res["task"].append("event")
            eval_res["dataset"].append(dname + "_tgg")
            eval_res["metric_val"].append(res["event"]["trigger_cls"]["f1"])
            eval_res["task"].append("event")
            eval_res["dataset"].append(dname + "_arg")
            eval_res["metric_val"].append(res["event"]["arg_cls"]["f1"])
        elif dname.startswith("absa_"):
            eval_res["task"].append("absa")
            eval_res["dataset"].append(dname)
            eval_res["metric_val"].append(res["rel"]["rel"]["micro"]["f1"])
        elif dname.startswith("cls_"):
            eval_res["task"].append("cls")
            eval_res["dataset"].append(dname)
            if "_glue_" in dname:
                if "_cola" in dname:
                    eval_res["metric_val"].append(res["cls"]["mcc"])
                else:
                    eval_res["metric_val"].append(res["cls"]["acc"])
            else:
                eval_res["metric_val"].append(res["cls"]["mf1"]["micro"]["f1"])
        elif dname.startswith("span"):
            eval_res["task"].append("span_em")
            eval_res["dataset"].append(dname)
            eval_res["metric_val"].append(res["span"]["em"])
            eval_res["task"].append("span_f1")
            eval_res["dataset"].append(dname)
            eval_res["metric_val"].append(res["span"]["f1"]["f1"])
        elif dname.startswith("discontinuous_ent"):
            eval_res["task"].append("discontinuous_ent")
            eval_res["dataset"].append(dname)
            eval_res["metric_val"].append(res["discontinuous_ent"]["micro"]["f1"])
        elif dname.startswith("hyper_rel"):
            eval_res["task"].append("hyper_rel")
            eval_res["dataset"].append(dname)
            eval_res["metric_val"].append(res["hyper_rel"]["micro"]["f1"])
        else:
            raise ValueError

    for i in range(len(eval_res["task"])):
        table.add_row(
            eval_res["task"][i],
            eval_res["dataset"][i],
            f"{100*eval_res['metric_val'][i]:.3f}",
        )

    console = Console()
    console.print(table)

    df = pd.DataFrame(eval_res)
    df.to_excel(task.measures_path.joinpath("data_eval_res.xlsx"))
