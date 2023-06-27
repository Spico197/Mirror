import os
import re
import statistics as sts
from collections import defaultdict
from pathlib import Path

from rex.utils.dict import get_dict_content
from rex.utils.io import load_json
from rich.console import Console
from rich.table import Table

inputs_dir = Path("mirror_fewshot_outputs")
# regex = re.compile(r"Mirror_SingleTask_(.*?)_seed(\d+)_(\d+)shot")
regex = re.compile(r"Mirror_wPT_woInst_(.*?)_seed(\d+)_(\d+)shot")

# task -> shot -> seeds
results = defaultdict(lambda: defaultdict(list))

for dirname in os.listdir(inputs_dir):
    dpath = inputs_dir / dirname
    re_matched = regex.match(dirname)
    if dpath.is_dir() and re_matched:
        task, seed, shot = re_matched.groups()
        results_json_p = dpath / "measures" / "test.final.json"
        metrics = load_json(results_json_p)
        if "Ent_" in task:
            results[task][shot].append(
                get_dict_content(metrics, "metrics.ent.micro.f1")
            )
        elif "Rel_" in task or "ABSA_" in task:
            results[task][shot].append(
                get_dict_content(metrics, "metrics.rel.rel.micro.f1")
            )
        elif "Event_" in task:
            results[task + "_Trigger"][shot].append(
                get_dict_content(metrics, "metrics.event.trigger_cls.f1")
            )
            results[task + "_Arg"][shot].append(
                get_dict_content(metrics, "metrics.event.arg_cls.f1")
            )
        else:
            raise RuntimeError

table = Table(title="Few-shot results")
table.add_column("Task", justify="center")
table.add_column("1-shot", justify="right")
table.add_column("5-shot", justify="right")
table.add_column("10-shot", justify="right")
table.add_column("Avg.", justify="right")
for task in results:
    shots = sorted(results[task].keys(), key=lambda x: int(x))
    all_seeds = []
    shot_results = []
    for shot in shots:
        seeds = results[task][shot]
        all_seeds.extend(seeds)
        avg = sum(seeds) / len(seeds)
        sts.stdev(seeds)
        shot_results.append(f"{100*avg:.2f}±{100*sts.stdev(seeds):.2f}")
    shot_results.append(f"{100*sts.mean(all_seeds):.2f}")
    table.add_row(task, *shot_results)

console = Console()
console.print(table)

"""
                            Few-shot results wPT wInst
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃        Task         ┃      1-shot ┃      5-shot ┃    10-shot ┃  Avg. ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│     Ent_CoNLL03     │  77.50±1.64 │  82.73±2.29 │ 84.48±1.62 │ 81.57 │
│     Rel_CoNLL04     │ 34.66±10.52 │  52.23±3.16 │ 58.68±1.77 │ 48.52 │
│ Event_ACE05_Trigger │  49.50±3.59 │ 65.61±19.29 │ 60.68±2.45 │ 58.60 │
│   Event_ACE05_Arg   │  23.46±1.66 │ 48.32±28.91 │ 41.90±1.95 │ 37.89 │
│     ABSA_16res      │  67.06±0.56 │ 73.51±14.75 │ 68.70±1.46 │ 69.76 │
└─────────────────────┴─────────────┴─────────────┴────────────┴───────┘

                           Few-shot results wPT woInst
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃        Task         ┃      1-shot ┃     5-shot ┃    10-shot ┃  Avg. ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│     Ent_CoNLL03     │  76.33±1.74 │ 82.50±1.87 │ 84.47±1.18 │ 81.10 │
│ woInst_Rel_CoNLL04  │  34.86±6.20 │ 48.00±4.44 │ 55.65±2.53 │ 46.17 │
│     Rel_CoNLL04     │ 26.83±15.22 │ 47.39±3.60 │ 55.38±2.41 │ 43.20 │
│ Event_ACE05_Trigger │  46.60±1.09 │ 57.21±3.51 │ 59.67±3.20 │ 54.49 │
│   Event_ACE05_Arg   │  21.60±3.61 │ 34.43±3.63 │ 39.62±2.60 │ 31.88 │
│     ABSA_16res      │  8.10±18.11 │ 52.73±5.52 │ 57.32±1.73 │ 39.38 │
└─────────────────────┴─────────────┴────────────┴────────────┴───────┘
"""
