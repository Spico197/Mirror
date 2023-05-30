import pandas as pd
from rex.utils.initialization import set_seed_and_log_path
from rich.console import Console
from rich.table import Table

from src.task import SchemaGuidedInstructBertTask

set_seed_and_log_path(log_path="eval.log")


# task_dir = "mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData"
task_dir = "mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2"
# task_dir = "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05ENPlus"
task: SchemaGuidedInstructBertTask = SchemaGuidedInstructBertTask.from_taskdir(
    task_dir,
    load_best_model=True,
    initialize=False,
    dump_configfile=False,
    update_config={"regenerate_cache": True},
)
table = Table(title=task_dir, width=len(task_dir))

data_pairs = [
    # fmt: off

    # [
    #     "rel_conll2004",
    #     "resources/Mirror/Tasks/RE/CoNLL2004/formatted/CoNLL2004_RE_test.jsonl",
    # ],
    # ["rel_gids", "resources/Mirror/Tasks/RE/GIDS/formatted/GIDS_test.jsonl"],
    # [
    #     "rel_nyt11hrl",
    #     "resources/Mirror/Tasks/RE/NYT11HRL/formatted/NYT11HRL_test.jsonl",
    # ],
    # ["rel_webnlg", "resources/Mirror/Tasks/RE/WebNLG/formatted/WebNLG_test.jsonl"],
    # ["rel_ace05", "resources/Mirror/Tasks/EE/ACE05-EN/ACE2005_oneie_RE_test.jsonl"],

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
    else:
        raise ValueError

for i in range(len(eval_res["task"])):
    table.add_row(
        eval_res["task"][i],
        eval_res["dataset"][i],
        f"{100*eval_res['metric_val'][i]:.3f}",
    )

df = pd.DataFrame(eval_res)
df.to_excel(task.measures_path.joinpath("uie_data_eval_res.xlsx"))

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
"""
