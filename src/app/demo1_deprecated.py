import gradio as gr
from rex.utils.initialization import set_seed_and_log_path
from rex.utils.logging import logger

from src.task import MrcQaTask

set_seed_and_log_path(log_path="app.log")


class MrcQaPipeline:
    def __init__(self, task_dir: str, load_path: str = None) -> None:
        self.task = MrcQaTask.from_taskdir(
            task_dir, load_best_model=load_path is None, initialize=False
        )
        if load_path:
            self.task.load(load_path, load_history=False)

    def predict(self, query, context, background=None):
        data = [
            {
                "query": query,
                "context": context,
                "background": background,
            }
        ]
        results = self.task.predict(data)
        ret = results[0]

        data[0]["pred"] = ret
        logger.opt(colors=False).debug(data[0])

        return ret


pipe = MrcQaPipeline("outputs/RobertaBase_data20230314v2")


with gr.Blocks() as demo:
    gr.Markdown("# 🪞 Mirror Mirror")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                query = gr.Textbox(
                    label="Query", placeholder="Mirror Mirror, tell me ..."
                )
            with gr.Row():
                context = gr.TextArea(
                    label="Candidates",
                    placeholder="Separated by comma (,) without spaces.",
                )
            with gr.Row():
                background = gr.TextArea(
                    label="Background",
                    placeholder="Background explanation, could be empty",
                )

        with gr.Column():
            with gr.Row():
                trigger_button = gr.Button("Tell me the truth", variant="primary")
            with gr.Row():
                output = gr.TextArea(label="Output")

            trigger_button.click(
                pipe.predict, inputs=[query, context, background], outputs=output
            )


demo.launch(show_error=True, share=False)
