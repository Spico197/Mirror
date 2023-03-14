import random

from rex.utils.io import dump_jsonlines, load_jsonlines
from rex.utils.progress_bar import rbar


def sample(input_filepath, num, output_filepath, shuffle_num=50):
    data = load_jsonlines(input_filepath)
    for _ in rbar(shuffle_num):
        random.shuffle(data)
    dump_jsonlines(data[:num], output_filepath)


if __name__ == "__main__":
    input_filepath = "resources/RE/COER/coer.jsonl"
    output_filepath = "resources/RE/COER/coer.40k.jsonl"
    data = sample(input_filepath, 40000, output_filepath)
