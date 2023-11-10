import json

import gradio as gr
from rex.utils.initialization import set_seed_and_log_path

from src.task import SchemaGuidedInstructBertTask

set_seed_and_log_path(log_path="debug.log")


task = SchemaGuidedInstructBertTask.from_taskdir(
    "mirror_outputs/Mirror_Pretrain_AllExcluded_2",
    load_best_model=True,
    initialize=False,
    dump_configfile=False,
    update_config={
        "regenerate_cache": False,
    },
)


def ask_mirror(instruction, schema, text):
    input_data = {
        "id": "app",
        "instruction": instruction,
        "schema": json.loads(schema),
        "text": text,
        "ans": {},
    }
    results = task.predict(input_data)
    return results


with gr.Blocks() as demo:
    gr.Markdown("# 🪞Mirror")
    gr.Markdown(
        "🪞Mirror can help you deal with a wide range of Natural Language Understanding and Information Extraction tasks."
    )
    gr.Markdown(
        "[[paper]](https://arxiv.org/abs/2311.05419) | [[code]](https://github.com/Spico197/Mirror)"
    )

    instruction = gr.Textbox(label="Instruction")
    schema = gr.Textbox(
        label="schema",
        placeholder='{"cls": ["class1", "class2"], "ent": ["type1", "type2"], "rel": ["relation1", "relation2"]} leave it as {} to support span extraction.',
    )
    text = gr.TextArea(label="Text")
    output = gr.Textbox(label="Output")

    submit_btn = gr.Button("Ask Mirror")
    submit_btn.click(ask_mirror, inputs=[instruction, schema, text], outputs=output)

    gr.Markdown("Made by Mirror Team w/ 💖")


if __name__ == "__main__":
    demo.launch()
