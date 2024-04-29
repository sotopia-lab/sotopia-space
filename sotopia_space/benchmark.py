import gradio as gr # type: ignore
import pandas as pd
from sotopia_space.constants import MODEL_OPTIONS
from sotopia_space.utils import post_processing

LP_MODE = "v2"
original_df, ablation_df = None, None
LP_original_dfs = {} 
DEFAULT_LP = 0.5 

def benchmark_table():
    global original_df, ablation_df
    global LP_original_dfs, LP_MODE
    
    gr.Markdown(f"**Version**: sotopia (v1.01; 2024.04.22) | **# Examples**: 7200 | **# Models**: {len(MODEL_OPTIONS)} | **# Comparisons**: x", elem_classes="markdown-text")
                
    with gr.TabItem("Vs GPT-3.5", elem_id="od-benchmark-tab-table-ablation", id=0, elem_classes="subtab"):
        original_df = pd.read_json('data_dir/models_vs_gpt35.jsonl', lines=True)
        default_main_df = original_df
        default_main_df = default_main_df.sort_values(by="GOAL [0, 10]", ascending=False)
        default_main_df = post_processing(default_main_df, None)
        # add a Rank column to the first columnn (starting from 1)
        default_main_df.insert(0, "Rank", range(1, 1 + len(default_main_df)))
        
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("<h3>**Vs GPT3.5**: The interlocutors are compared against GPT-3.5, the baseline model.")
        TYPES = ["number", "markdown", "number"]
        leaderboard_table = gr.components.Dataframe(
            value=default_main_df,
            datatype=TYPES,
            # max_rows=None,
            height=1000,
            elem_id="leaderboard-table",
            interactive=False,
            visible=True,
            min_width=60,
            )
