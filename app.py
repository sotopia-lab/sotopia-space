import os
import argparse
from typing import Literal

import gradio as gr # type: ignore
from sotopia_space.chat import chat_introduction, chat_tab, get_sotopia_profiles
from sotopia_space import benchmark
from ui_constants import CITATION_TEXT, BANNER


OPENAI_KEY_FILE="./openai_api.key"
if os.path.exists(OPENAI_KEY_FILE):
    with open(OPENAI_KEY_FILE, "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

with open("./sotopia_space/_header.md", "r") as f:
    HEADER_MD = f.read()

def navigation_bar():
    with gr.Column(scale=2):
        toggle_dark = gr.Button(value="Toggle Dark") 
    toggle_dark.click(
        None,
        js="""
        () => {
            if (document.body.classList.contains('dark')) {
            document.body.classList.remove('dark');
            document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary-light)';
            } else {
            document.body.classList.add('dark');
            document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary-dark)';
            }
        }
        """,
    )

with gr.Blocks(
    css="""#chat_container {height: 820px; width: 1000px; margin-left: auto; margin-right: auto;}
            #chatbot {height: 600px; overflow: auto;}
            #create_container {height: 750px; margin-left: 0px; margin-right: 0px;}
            #tokenizer_renderer span {white-space: pre-wrap}
            """,
    theme="gradio/monochrome",
) as demo:
    # with gr.Row():
    #     navigation_bar()
    gr.Image(
            "images/banner.jpg", elem_id="banner-image", show_label=False
        )
    gr.Markdown(HEADER_MD, elem_classes="markdown-text")
    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🏅 Leaderboard", elem_id="benchmark-tab-table", id=0):
            benchmark.benchmark_table()
        with gr.TabItem("💬 Chat", elem_id="chat-tab-interface", id=1): 
            with gr.Row():
                chat_introduction()
            with gr.Row():
                chat_tab()
    with gr.Row():
        with gr.Accordion("📙 Citation", open=False, elem_classes="accordion-label"):
            gr.Textbox(
                value=CITATION_TEXT, 
                lines=7,
                label="Copy the BibTeX snippet to cite this source",
                elem_id="citation-button",
                show_copy_button=True)

# def start_demo():
#     demo = main()
#     if DEPLOYED:
#         demo.queue(api_open=False).launch(show_api=False)
#     else:
#         demo.queue()
#         demo.launch(share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", help="Path to results table", default="data_dir/models_vs_gpt35.jsonl")
    #benchmark.original_df = pd.read_json(args.result_file, lines=True)
    get_sotopia_profiles()
    # prepare_model(DEFAULT_MODEL_SELECTION)
    demo.launch()