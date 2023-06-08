import os
import random
import re

from rex.utils.io import dump_jsonlines, load_jsonlines
from tqdm import tqdm


def sample_k(data, k: int, shuffle: int = 20) -> list:
    """Sample k items from data

    Args:
        data (list): list of data
        k (int): number of items to sample
        shuffle (int): number of shuffles before sampling

    Returns:
        list: sampled data
    """
    for _ in range(shuffle):
        random.shuffle(data)
    return data[:k]


def process_file(input_filepath, output_filepath, num_samples: int = 10_000):
    """Sample 1000 items from input_filepath and save to output_filepath

    Args:
        input_filepath (str): input filepath
        output_filepath (str): output filepath
    """
    data = load_jsonlines(input_filepath)
    sampled_data = sample_k(data, num_samples)
    dump_jsonlines(sampled_data, output_filepath)


def find_files(regex: str, folder: str, recursive: bool = True):
    """Find files with regex in a folder"""
    regex = re.compile(regex)
    files = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if regex.match(filename):
                files.append(os.path.join(root, filename))
        if not recursive:
            break
    return files


if __name__ == "__main__":
    output_dir = "/data/tzhu/Mirror/resources/Mirror/v1.4_sampled_v3/"
    data_dir = "/data/tzhu/Mirror/resources/Mirror/v1.4/"
    os.makedirs(output_dir, exist_ok=True)
    files = find_files(r".*\.jsonl", data_dir)
    for file in tqdm(files, desc="Files"):
        file = file.removeprefix(".\\")
        out_dir = os.path.join(output_dir, os.path.dirname(file.removeprefix(data_dir)))
        os.makedirs(out_dir, exist_ok=True)
        out_fp = os.path.join(out_dir, os.path.split(file)[1])
        if "v1.4_sampled" not in out_fp:
            breakpoint()
        num_samples = 5_000
        if "cls" in file:
            num_samples = 5_000
        elif "ent" in file:
            num_samples = 20_000
        elif "rel" in file:
            num_samples = 20_000
        elif "event" in file:
            num_samples = -1
        elif "span" in file:
            num_samples = 30_000
        process_file(file, out_fp, num_samples=num_samples)
