import os

import pandas as pd


def count_line_number(filepath):
    n = 0
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            n += 1
    return n


files = []
for root, dirnames, filenames in os.walk("./resources/Mirror/v1.4_sampled"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        if filename.endswith(".jsonl") and filepath not in files:
            files.append(filepath)


data = {"root": [], "path": [], "data": [], "split": [], "count": []}

for filepath in files:
    root_folder_name = os.path.split(filepath)[0].split(os.path.sep)[1]
    data["root"].append(root_folder_name)
    data["path"].append(filepath)
    data["count"].append(count_line_number(filepath))
    data["data"].append(filepath.split(os.path.sep)[3])
    split = filepath.split(os.path.sep)[-1].split(".")[0]
    if split not in ["train", "dev", "test"]:
        if "test" in split:
            split = "test"
        elif "dev" in split or "val" in split:
            split = "dev"
        else:
            split = "train"
    data["split"].append(split)

df = pd.DataFrame(data)
df.to_excel("resources/Mirror/v1.4_sampled/sampled_stats.xlsx")
