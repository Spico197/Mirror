# import random
# from pathlib import Path

# import requests
# from tqdm import tqdm
# from rex.utils.io import load_jsonlines, dump_jsonlines


# output_dir = Path("resources/Mirror/ner_web_enhanced_bg")
# if not output_dir.exists():
#     output_dir.mkdir(parents=True)

# search_url = "http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search"

# for data_dir in [
#     # "resources/Mirror/v1.3/ent/en/CrossNER_AI/instructed",
#     # "resources/Mirror/v1.3/ent/en/CrossNER_literature/instructed",
#     # "resources/Mirror/v1.3/ent/en/CrossNER_music/instructed",
#     # "resources/Mirror/v1.3/ent/en/CrossNER_politics/instructed",
#     # "resources/Mirror/v1.3/ent/en/CrossNER_science/instructed",
#     "resources/Mirror/v1.3/ent/en/MIT_MOVIE_Review/instructed",
#     "resources/Mirror/v1.3/ent/en/MIT_Restaurant_Review/instructed",
# ]:
#     data_dir = Path(data_dir)
#     # train_data = load_jsonlines(data_dir / "train.jsonl")
#     # train_data = list(filter(lambda d: d["ans"]["ent"], train_data))
#     test_data = load_jsonlines(data_dir / "test.jsonl")
#     for d in tqdm(test_data):
#         # examplar = random.choice(train_data)
#         # ent_string = ""
#         # for ent in examplar["ans"]["ent"]:
#         #     ent_string += f" {ent['text']} is {ent['type']},"
#         # ent_string = ent_string.strip().removesuffix(",")
#         # d["bg"] = f"{ent_string} in {examplar['text']}"

#         payload = {"query": d["text"], "k": 1}
#         response = requests.get(search_url, params=payload)
#         d["bg"] = response.json()['topk'][0]["text"]
#     dump_jsonlines(test_data, output_dir / f"{data_dir.parent.name}.jsonl")


import asyncio
from pathlib import Path

import aiohttp
from rex.utils.io import dump_jsonlines, load_jsonlines
from tqdm import tqdm

output_dir = Path("resources/Mirror/ner_web_enhanced_bg")
if not output_dir.exists():
    output_dir.mkdir(parents=True)

search_url = "http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search"


async def process_data(d):
    payload = {"query": d["text"], "k": 1}
    async with aiohttp.ClientSession() as session:
        async with session.get(search_url, params=payload) as response:
            data = await response.json()
            d["bg"] = data["topk"][0]["text"]


async def main():
    tasks = []
    for data_dir in [
        "resources/Mirror/v1.3/ent/en/MIT_MOVIE_Review/instructed",
        "resources/Mirror/v1.3/ent/en/MIT_Restaurant_Review/instructed",
    ]:
        data_dir = Path(data_dir)
        test_data = load_jsonlines(data_dir / "test.jsonl")
        for d in tqdm(test_data):
            task = asyncio.create_task(process_data(d))
            tasks.append(task)

        await asyncio.gather(*tasks)

        dump_jsonlines(test_data, output_dir / f"{data_dir.parent.name}.jsonl")


asyncio.run(main())
