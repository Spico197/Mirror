import pandas as pd
from rex.utils.initialization import set_seed_and_log_path
from rich.console import Console
from rich.table import Table

from src.task import SchemaGuidedInstructBertTask

set_seed_and_log_path(log_path="eval.log")


# task_dir = "mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData"
# task_dir = "mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2"
# task_dir = "mirror_outputs/InstructBert_NewMergedUIEData"
# task_dir = "mirror_outputs/InstructBert_Large_NewMergedUIEData"
# task_dir = "mirror_outputs/InstructBert_Large_NewMergedUIEData_bs10"
# task_dir = "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05ENPlus"
task_dir = "mirror_outputs/MirrorLarge_SamplingPretrain"
# task_dir = "mirror_outputs/MirrorLarge_SamplingPretrain_woZeroShotNER"
# task_dir = "mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap"
# task_dir = "mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap"
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
    },
)
table = Table(title=task_dir, width=len(task_dir))

data_pairs = [
    # fmt: off

    # # UIE eval data
    # ["ent_ace04_test", "resources/Mirror/uie/ent/ace04/test.jsonl"],
    # ["ent_ace05_test", "resources/Mirror/uie/ent/ace05/test.jsonl"],
    # ["ent_conll03_test", "resources/Mirror/uie/ent/conll03/test.jsonl"],

    # ["rel_ace05_test", "resources/Mirror/uie/rel/ace05-rel/test.jsonl"],
    # ["rel_conll04_test", "resources/Mirror/uie/rel/conll04/test.jsonl"],
    # ["rel_nyt_test", "resources/Mirror/uie/rel/nyt/test.jsonl"],
    # ["rel_scierc_test", "resources/Mirror/uie/rel/scierc/test.jsonl"],

    # ["event_ace05_test", "resources/Mirror/uie/event/ace05-evt/test.jsonl"],
    # ["event_casie_test", "resources/Mirror/uie/event/casie/test.jsonl"],

    # ["absa_14res_test", "resources/Mirror/uie/absa/14res/test.jsonl"],
    # ["absa_14lap_test", "resources/Mirror/uie/absa/14lap/test.jsonl"],
    # ["absa_15res_test", "resources/Mirror/uie/absa/15res/test.jsonl"],
    # ["absa_16res_test", "resources/Mirror/uie/absa/16res/test.jsonl"],

    # # zero-shot NER
    # ["ent_movie", "resources/Mirror/v1.4/ent/en/MIT_MOVIE_Review/instructed/test.jsonl"],
    # ["ent_restaurant", "resources/Mirror/v1.4/ent/en/MIT_Restaurant_Review/instructed/test.jsonl"],
    # ["ent_ai", "resources/Mirror/v1.4/ent/en/CrossNER_AI/instructed/test.jsonl"],
    # ["ent_literature", "resources/Mirror/v1.4/ent/en/CrossNER_literature/instructed/test.jsonl"],
    # ["ent_music", "resources/Mirror/v1.4/ent/en/CrossNER_music/instructed/test.jsonl"],
    # ["ent_politics", "resources/Mirror/v1.4/ent/en/CrossNER_politics/instructed/test.jsonl"],
    # ["ent_science", "resources/Mirror/v1.4/ent/en/CrossNER_science/instructed/test.jsonl"],

    # discontinuous NER
    ["discontinuous_ent", "resources/Mirror/new_abilities_v2/cadec/new/test.jsonl"],

    # hyper-RE
    ["hyper_rel", "resources/Mirror/new_abilities_v2/HyperRED/new/test.jsonl"],
    # glue
    # ["cls_glue_cola", "resources/Mirror/v1.4/cls/en/CoLA/formated/test.jsonl"],
    # ["cls_glue_qqp", "resources/Mirror/v1.4/cls/en/QQP/new/dev.jsonl"],
    # ["cls_glue_mnli", "resources/Mirror/v1.4/cls/en/MNLI/formated/MNLI_dev.jsonl"],
    # ["cls_glue_sst2", "resources/Mirror/v1.4/cls/en/SST-2/instructed/SST-2_dev.jsonl"],
    # ["cls_glue_qnli", "resources/Mirror/v1.4/cls/en/QNLI/processed/QNLI_dev.jsonl"],
    # ["cls_glue_rte", "resources/Mirror/v1.4/cls/en/RTE/formated/RTE_dev.jsonl"],
    # ["cls_glue_mrpc", "resources/Mirror/v1.4/cls/en/MRPC/formated/dev.jsonl"],
    # mrc
    # ["span_squad2", "resources/Mirror/v1.4/span/en/squad_v2/dev.jsonl"],
    # fmt: on
]

eval_res = {"task": [], "dataset": [], "metric_val": []}
table.add_column("Task", justify="left", style="cyan")
table.add_column("Dataset", justify="left", style="magenta")
table.add_column("Metric (%)", justify="right", style="green")
for dname, fpath in data_pairs:
    task.data_manager.update_datapath(dname, fpath)
    _, res = task.eval(dname, verbose=True, dump=True, dump_middle=True)
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

df = pd.DataFrame(eval_res)
df.to_excel(task.measures_path.joinpath("data_eval_res.xlsx"))

console = Console()
console.print(table)


"""
fixed upper bound

mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.934 │
│ ent       │ ent_ace05_test                 │         100.000 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │          96.444 │
│ rel       │ rel_conll04_test               │          96.009 │
│ rel       │ rel_nyt_test                   │          78.145 │
│ rel       │ rel_scierc_test                │          81.288 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │          92.987 │
│ event     │ event_casie_test_arg           │          93.376 │
│ absa      │ absa_14res_test                │          98.991 │
│ absa      │ absa_14lap_test                │          99.815 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.611 │
└───────────┴────────────────────────────────┴─────────────────┘

eval UIEData2
mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          84.912 │
│ ent       │ ent_ace05_test                 │          90.300 │
│ ent       │ ent_conll03_test               │          92.335 │
│ rel       │ rel_ace05_test                 │          59.879 │
│ rel       │ rel_conll04_test               │          45.272 │
│ rel       │ rel_nyt_test                   │          72.610 │
│ rel       │ rel_scierc_test                │          19.890 │
│ event     │ event_ace05_test_tgg           │          71.752 │
│ event     │ event_ace05_test_arg           │          51.140 │
│ event     │ event_casie_test_tgg           │          63.915 │
│ event     │ event_casie_test_arg           │          32.243 │
│ absa      │ absa_14res_test                │          75.456 │
│ absa      │ absa_14lap_test                │          64.251 │
│ absa      │ absa_15res_test                │          93.525 │
│ absa      │ absa_16res_test                │          76.505 │
└───────────┴────────────────────────────────┴─────────────────┘

eval UIEData2 upperbound fixed-v1 with constraint
mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.934 │
│ ent       │ ent_ace05_test                 │         100.000 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │          98.401 │
│ rel       │ rel_conll04_test               │          99.176 │
│ rel       │ rel_nyt_test                   │          78.573 │
│ rel       │ rel_scierc_test                │          89.655 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │          92.987 │
│ event     │ event_casie_test_arg           │          93.376 │
│ absa      │ absa_14res_test                │          99.091 │
│ absa      │ absa_14lap_test                │          99.815 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.708 │
└───────────┴────────────────────────────────┴─────────────────┘

eval UIEData2 fixed-v2 with constraint
mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          84.912 │
│ ent       │ ent_ace05_test                 │          90.300 │
│ ent       │ ent_conll03_test               │          92.335 │
│ rel       │ rel_ace05_test                 │          60.365 │
│ rel       │ rel_conll04_test               │          46.064 │
│ rel       │ rel_nyt_test                   │          73.048 │
│ rel       │ rel_scierc_test                │          20.084 │
│ event     │ event_ace05_test_tgg           │          71.752 │
│ event     │ event_ace05_test_arg           │          51.140 │
│ event     │ event_casie_test_tgg           │          63.915 │
│ event     │ event_casie_test_arg           │          32.243 │
│ absa      │ absa_14res_test                │          75.456 │
│ absa      │ absa_14lap_test                │          64.251 │
│ absa      │ absa_15res_test                │          93.525 │
│ absa      │ absa_16res_test                │          76.505 │
└───────────┴────────────────────────────────┴─────────────────┘

pso upper bound, find all nnw paths
mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.934 │
│ ent       │ ent_ace05_test                 │         100.000 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │          98.463 │
│ rel       │ rel_conll04_test               │          99.176 │
│ rel       │ rel_nyt_test                   │          98.392 │
│ rel       │ rel_scierc_test                │          92.593 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │          92.987 │
│ event     │ event_casie_test_arg           │          93.376 │
│ absa      │ absa_14res_test                │          98.947 │
│ absa      │ absa_14lap_test                │          99.815 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.709 │
└───────────┴────────────────────────────────┴─────────────────┘

MergedUIEDataMultitaskSFT
pso find all nnw paths
output label type to span len constraint
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.934 │
│ ent       │ ent_ace05_test                 │         100.000 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │         100.000 │
│ rel       │ rel_conll04_test               │          99.881 │
│ rel       │ rel_nyt_test                   │          99.362 │
│ rel       │ rel_scierc_test                │          97.113 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │          92.987 │
│ event     │ event_casie_test_arg           │          93.376 │
│ absa      │ absa_14res_test                │          99.496 │
│ absa      │ absa_14lap_test                │          99.908 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.903 │
└───────────┴────────────────────────────────┴─────────────────┘

pso upper bound
new merged uie data
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.951 │
│ ent       │ ent_ace05_test                 │          99.852 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │          99.957 │
│ rel       │ rel_conll04_test               │          99.643 │
│ rel       │ rel_nyt_test                   │          99.380 │
│ rel       │ rel_scierc_test                │          97.113 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │         100.000 │
│ event     │ event_casie_test_arg           │          99.991 │
│ absa      │ absa_14res_test                │          99.496 │
│ absa      │ absa_14lap_test                │          99.908 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.903 │
└───────────┴────────────────────────────────┴─────────────────┘

merged uie data v2 eval on new data
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          83.055 │
│ ent       │ ent_ace05_test                 │          89.497 │
│ ent       │ ent_conll03_test               │          86.333 │
│ rel       │ rel_ace05_test                 │          67.989 │
│ rel       │ rel_conll04_test               │           0.000 │
│ rel       │ rel_nyt_test                   │          91.656 │
│ rel       │ rel_scierc_test                │           7.509 │
│ event     │ event_ace05_test_tgg           │          71.170 │
│ event     │ event_ace05_test_arg           │          49.408 │
│ event     │ event_casie_test_tgg           │          30.459 │
│ event     │ event_casie_test_arg           │           7.966 │
│ absa      │ absa_14res_test                │          73.684 │
│ absa      │ absa_14lap_test                │          62.737 │
│ absa      │ absa_15res_test                │          90.928 │
│ absa      │ absa_16res_test                │          74.853 │
└───────────┴────────────────────────────────┴─────────────────┘

InstructBert_NewMergedUIEData middle
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task  ┃ Dataset             ┃ Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent   │ ent_ace04_test      │     85.686 │
│ ent   │ ent_ace05_test      │     90.354 │
│ ent   │ ent_conll03_test    │     92.456 │
│ rel   │ rel_ace05_test      │     61.343 │
│ rel   │ rel_conll04_test    │     67.797 │
│ rel   │ rel_nyt_test        │     92.122 │
│ rel   │ rel_scierc_test     │     21.911 │
│ event │ event_ace05_test_t… │     67.178 │
│ event │ event_ace05_test_a… │     43.394 │
│ event │ event_casie_test_t… │     59.827 │
│ event │ event_casie_test_a… │     37.390 │
│ absa  │ absa_14res_test     │     74.384 │
│ absa  │ absa_14lap_test     │     65.564 │
│ absa  │ absa_15res_test     │     85.775 │
│ absa  │ absa_16res_test     │     74.533 │
└───────┴─────────────────────┴────────────┘

mirror_outputs/InstructBert_NewMergedUIEData
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task  ┃ Dataset             ┃ Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent   │ ent_ace04_test      │     86.929 │
│ ent   │ ent_ace05_test      │     92.290 │
│ ent   │ ent_conll03_test    │     92.747 │
│ rel   │ rel_ace05_test      │     67.777 │
│ rel   │ rel_conll04_test    │     71.159 │
│ rel   │ rel_nyt_test        │     93.226 │
│ rel   │ rel_scierc_test     │     34.031 │
│ event │ event_ace05_test_t… │     72.372 │
│ event │ event_ace05_test_a… │     52.946 │
│ event │ event_casie_test_t… │     69.821 │
│ event │ event_casie_test_a… │     56.977 │
│ absa  │ absa_14res_test     │     75.732 │
│ absa  │ absa_14lap_test     │     66.401 │
│ absa  │ absa_15res_test     │     92.798 │
│ absa  │ absa_16res_test     │     74.138 │
└───────┴─────────────────────┴────────────┘

large model on new data
mirror_outputs/InstructBert_Large_NewMergedUIEData
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Task   ┃ Dataset                 ┃  Metric (%) ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ ent    │ ent_ace04_test          │      88.531 │
│ ent    │ ent_ace05_test          │      93.515 │
│ ent    │ ent_conll03_test        │      93.094 │
│ rel    │ rel_ace05_test          │      72.015 │
│ rel    │ rel_conll04_test        │      75.933 │
│ rel    │ rel_nyt_test            │      93.995 │
│ rel    │ rel_scierc_test         │      42.069 │
│ event  │ event_ace05_test_tgg    │      73.177 │
│ event  │ event_ace05_test_arg    │      57.833 │
│ event  │ event_casie_test_tgg    │      71.659 │
│ event  │ event_casie_test_arg    │      59.336 │
│ absa   │ absa_14res_test         │      76.899 │
│ absa   │ absa_14lap_test         │      63.448 │
│ absa   │ absa_15res_test         │      95.436 │
│ absa   │ absa_16res_test         │      75.624 │
└────────┴─────────────────────────┴─────────────┘

mirror_outputs/InstructBert_Large_NewMergedUIEData_bs10
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Task    ┃ Dataset                    ┃   Metric (%) ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ ent     │ ent_ace04_test             │       87.759 │
│ ent     │ ent_ace05_test             │       93.673 │
│ ent     │ ent_conll03_test           │       92.449 │
│ rel     │ rel_ace05_test             │       72.188 │
│ rel     │ rel_conll04_test           │       77.255 │
│ rel     │ rel_nyt_test               │       93.764 │
│ rel     │ rel_scierc_test            │       42.358 │
│ event   │ event_ace05_test_tgg       │       72.256 │
│ event   │ event_ace05_test_arg       │       58.561 │
│ event   │ event_casie_test_tgg       │       71.800 │
│ event   │ event_casie_test_arg       │       59.477 │
│ absa    │ absa_14res_test            │       77.663 │
│ absa    │ absa_14lap_test            │       66.142 │
│ absa    │ absa_15res_test            │       93.769 │
│ absa    │ absa_16res_test            │       74.835 │
└─────────┴────────────────────────────┴──────────────┘

pretrain direct infer
mirror_outputs/MirrorLarge_SamplingPretrain
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task  ┃ Dataset            ┃ Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent   │ ent_ace04_test     │     82.892 │
│ ent   │ ent_ace05_test     │     85.304 │
│ ent   │ ent_conll03_test   │     91.905 │
│ rel   │ rel_ace05_test     │      9.233 │
│ rel   │ rel_conll04_test   │     52.615 │
│ rel   │ rel_nyt_test       │     88.260 │
│ rel   │ rel_scierc_test    │      1.621 │
│ event │ event_ace05_test_… │     54.799 │
│ event │ event_ace05_test_… │     14.267 │
│ event │ event_casie_test_… │     19.541 │
│ event │ event_casie_test_… │      0.701 │
│ absa  │ absa_14res_test    │     59.305 │
│ absa  │ absa_14lap_test    │     57.208 │
│ absa  │ absa_15res_test    │     62.546 │
│ absa  │ absa_16res_test    │     67.333 │
└───────┴────────────────────┴────────────┘

pretrain direct infer
mirror_outputs/MirrorLarge_SamplingPretrain
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Task  ┃ Dataset           ┃  Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ ent   │ ent_movie         │      85.367 │
│ ent   │ ent_restaurant    │      81.790 │
│ ent   │ ent_ai            │      61.803 │
│ ent   │ ent_literature    │      66.210 │
│ ent   │ ent_music         │      75.516 │
│ ent   │ ent_politics      │      75.653 │
│ ent   │ ent_science       │      69.719 │
└───────┴───────────────────┴─────────────┘

pretrain w/o zero-shot NER, direct infer
mirror_outputs/MirrorLarge_SamplingPretrain_woZeroShotNER
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Task     ┃ Dataset                 ┃       Metric (%) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ ent      │ ent_movie               │           39.229 │
│ ent      │ ent_restaurant          │           22.413 │
│ ent      │ ent_ai                  │           51.155 │
│ ent      │ ent_literature          │           51.484 │
│ ent      │ ent_music               │           62.215 │
│ ent      │ ent_politics            │           62.087 │
│ ent      │ ent_science             │           52.632 │
└──────────┴─────────────────────────┴──────────────────┘

pretrain direct infer, upper-bound
mirror_outputs/MirrorLarge_SamplingPretrain
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task     ┃ Dataset         ┃ Metric (%) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cls      │ cls_glue_cola   │    100.000 │
│ cls      │ cls_glue_qqp    │    100.000 │
│ cls      │ cls_glue_mnli   │    100.000 │
│ cls      │ cls_glue_sst2   │    100.000 │
│ cls      │ cls_glue_qnli   │    100.000 │
│ cls      │ cls_glue_rte    │    100.000 │
│ cls      │ cls_glue_mrpc   │    100.000 │
│ span_em  │ span_squad2     │     95.614 │
│ span_f1  │ span_squad2     │     99.907 │
└──────────┴─────────────────┴────────────┘

pretrain direct infer on glue and mrc
mirror_outputs/MirrorLarge_SamplingPretrain
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task     ┃ Dataset         ┃ Metric (%) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cls      │ cls_glue_cola   │      0.000 │
│ cls      │ cls_glue_qqp    │     62.895 │
│ cls      │ cls_glue_mnli   │      0.000 │
│ cls      │ cls_glue_sst2   │     22.222 │
│ cls      │ cls_glue_qnli   │     43.053 │
│ cls      │ cls_glue_rte    │      0.000 │
│ cls      │ cls_glue_mrpc   │     68.382 │
│ span_em  │ span_squad2     │     38.664 │
│ span_f1  │ span_squad2     │     55.380 │
└──────────┴─────────────────┴────────────┘

mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Task    ┃ Dataset                  ┃   Metric (%) ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ ent     │ ent_ace04_test           │       84.426 │
│ ent     │ ent_ace05_test           │       86.357 │
│ ent     │ ent_conll03_test         │       92.716 │
│ rel     │ rel_ace05_test           │       22.812 │
│ rel     │ rel_conll04_test         │       53.652 │
│ rel     │ rel_nyt_test             │       89.094 │
│ rel     │ rel_scierc_test          │        7.832 │
│ event   │ event_ace05_test_tgg     │       63.624 │
│ event   │ event_ace05_test_arg     │       25.000 │
│ event   │ event_casie_test_tgg     │       50.017 │
│ event   │ event_casie_test_arg     │       17.642 │
│ absa    │ absa_14res_test          │       66.818 │
│ absa    │ absa_14lap_test          │       62.260 │
│ absa    │ absa_15res_test          │       62.896 │
│ absa    │ absa_16res_test          │       69.530 │
│ ent     │ ent_movie                │       85.942 │
│ ent     │ ent_restaurant           │       83.304 │
│ ent     │ ent_ai                   │       65.724 │
│ ent     │ ent_literature           │       67.932 │
│ ent     │ ent_music                │       78.245 │
│ ent     │ ent_politics             │       75.921 │
│ ent     │ ent_science              │       70.959 │
└─────────┴──────────────────────────┴──────────────┘

mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                          ┃       Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                   │           84.325 │
│ ent       │ ent_ace05_test                   │           86.262 │
│ ent       │ ent_conll03_test                 │           69.106 │
│ rel       │ rel_ace05_test                   │           20.818 │
│ rel       │ rel_conll04_test                 │           16.601 │
│ rel       │ rel_nyt_test                     │           88.332 │
│ rel       │ rel_scierc_test                  │            3.910 │
│ event     │ event_ace05_test_tgg             │            0.000 │
│ event     │ event_ace05_test_arg             │            0.000 │
│ event     │ event_casie_test_tgg             │           39.003 │
│ event     │ event_casie_test_arg             │            9.116 │
│ absa      │ absa_14res_test                  │           63.170 │
│ absa      │ absa_14lap_test                  │           60.268 │
│ absa      │ absa_15res_test                  │           60.633 │
│ absa      │ absa_16res_test                  │           68.119 │
│ ent       │ ent_movie                        │           40.964 │
│ ent       │ ent_restaurant                   │           20.022 │
│ ent       │ ent_ai                           │           51.130 │
│ ent       │ ent_literature                   │           44.803 │
│ ent       │ ent_music                        │           60.626 │
│ ent       │ ent_politics                     │           61.190 │
│ ent       │ ent_science                      │           53.649 │
└───────────┴──────────────────────────────────┴──────────────────┘
"""
