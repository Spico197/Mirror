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
regex = re.compile(r"Mirror_SingleTask_(.*?)_seed(\d+)_(\d+)shot")

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
