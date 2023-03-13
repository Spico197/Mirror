import glob
from pathlib import Path

from lxml import etree
from rex.utils.io import dump_jsonlines
from rex.utils.progress_bar import pbar


def parse_xml_to_triple(filepath):
    results = []
    try:
        xml = etree.parse(filepath)
        for node in xml.xpath("relation"):
            try:
                text = node.xpath("./origin_text/text()")[0].strip()
                relation = node.xpath("./relation_phrase/@phrase_text")[0].strip()
                relation_words = node.xpath("./relation_phrase/mention/text()")
                relation_index = []
                for word in relation_words:
                    start_index = text.find(word)
                    assert start_index >= 0
                    relation_index.extend(
                        list(range(start_index, start_index + len(word)))
                    )
                head, tail = node.xpath("./entity_pair/argument/text()")
                head_type, tail_type = node.xpath("./entity_pair/@type")[0].split("-")
                head_start_index = text.find(head)
                assert head_start_index >= 0
                head_index = list(range(head_start_index, head_start_index + len(head)))
                tail_start_index = text.find(tail)
                assert tail_start_index >= 0
                tail_index = list(range(tail_start_index, tail_start_index + len(tail)))

                results.append(
                    dict(
                        id=f"re.coer.{len(results)}",
                        context_tokens=list(text.strip()),
                        query_tokens=list("关系词"),
                        answer_index=[relation_index],
                    )
                )
                results.append(
                    dict(
                        id=f"re.coer.{len(results)}",
                        context_tokens=list(text.strip()),
                        query_tokens=list(f"关系词{relation}的主语头实体"),
                        answer_index=[head_index],
                    )
                )
                results.append(
                    dict(
                        id=f"re.coer.{len(results)}",
                        context_tokens=list(text.strip()),
                        query_tokens=list(f"关系词{relation}的宾语尾实体"),
                        answer_index=[tail_index],
                    )
                )
                query_suffix = ""
                if tail_type == "PER":
                    query_suffix = "是谁"
                elif tail_type == "LOC":
                    query_suffix = "是在哪里"
                results.append(
                    dict(
                        id=f"re.coer.{len(results)}",
                        context_tokens=list(text.strip()),
                        query_tokens=list(f"{head}{relation}{query_suffix}"),
                        answer_index=[tail_index],
                    )
                )

            except (IndexError, AssertionError):
                pass
    except etree.XMLSyntaxError:
        return []
    return results


def convert_xml(input_dir, dump_filepath):
    input_path = str(Path(input_dir) / "**/*_SubSet_*.xml")
    filepaths = glob.glob(input_path)
    data = []
    for filepath in pbar(filepaths):
        triple_results = parse_xml_to_triple(filepath)
        data.extend(triple_results)
    dump_jsonlines(data, dump_filepath)


if __name__ == "__main__":
    convert_xml("resources/RE/COER/COERKB", "resources/RE/COER/coer.jsonl")
