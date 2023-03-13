from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines

LABEL_MAP = {
    "news_story": "故事",
    "news_culture": "文化",
    "news_entertainment": "娱乐",
    "news_sports": "运动",
    "news_finance": "财经",
    "news_house": "房产",
    "news_car": "汽车",
    "news_edu": "教育",
    "news_tech": "科技",
    "news_military": "军事",
    "news_travel": "旅游",
    "news_world": "国际",
    "news_stock": "股市",
    "news_agriculture": "农业",
    "news_game": "游戏",
}
LABELS = ",".join(LABEL_MAP.values())


def convert_ins(ins):
    query_tokens = list("下面这段文本属于什么类别")
    context_tokens = list(LABELS)
    background_tokens = list(ins["sentence"])
    label = LABEL_MAP[ins["label_desc"]]
    answer_start_index = LABELS.find(label)
    answer_index = list(range(answer_start_index, answer_start_index + len(label)))
    answer_index_list = [answer_index]
    return dict(
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        background_tokens=background_tokens,
        answer_index=answer_index_list,
    )


if __name__ == "__main__":
    data_dir = "resources/Classification/TNEWS/raw"
    dump_dir = "resources/Classification/TNEWS/formatted"
    data_dir = Path(data_dir)
    dump_dir = Path(dump_dir)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    for dname in ["train", "dev", "test"]:
        p = data_dir / f"{dname}.json"
        data = load_jsonlines(p)
        new_data = []
        for ins in data:
            new_ins = convert_ins(ins)
            new_data.append(new_ins)
        dump_jsonlines(new_data, dump_dir / f"{dname}.jsonl")
