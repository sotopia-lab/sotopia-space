import gradio as gr # type: ignore
import pandas as pd
from sotopia_space.constants import MODEL_OPTIONS
from sotopia_space.utils import estimated_win_rate, make_clickable_model, styled_error, styled_warning, styled_message,apply_length_penalty 

LP_MODE = "v2"
original_df, ablation_df = None, None
LP_original_dfs = {} 
DEFAULT_LP = 0.5 

available_models = [] # to be filled in later
original_df, ablation_df = None, None

def slider_change_main(length_penalty):
    global original_df, ablation_df, LP_MODE
    adjusted_df = apply_length_penalty(original_df, ablation_df, length_penalty, mode=LP_MODE, LP_original_dfs=LP_original_dfs) 
    adjusted_df = adjusted_df[["Model", "Overall Elo", "Task-Avg Elo", "# battles", "Length"]]
    adjusted_df = adjusted_df.sort_values(by="Overall Elo", ascending=False)
    # adjusted_df = add_winrates(adjusted_df, LP=length_penalty) 
    # adjusted_df = adjusted_df.drop(columns=["Length"])
    adjusted_df.insert(0, "Rank", range(1, 1 + len(adjusted_df)))
    return adjusted_df

def slider_change_full(length_penalty, show_winrate):
    global original_df, ablation_df, LP_MODE
    adjusted_df = apply_length_penalty(original_df, ablation_df, length_penalty, mode=LP_MODE, LP_original_dfs=LP_original_dfs)
    # sort the model by the "Task-Avg Elo" column
    adjusted_df = adjusted_df.sort_values(by="Overall Elo", ascending=False)
    adjusted_df.drop(columns=["Overall Elo", "Task-Avg Elo", "# battles", "Length"], inplace=True)
    if show_winrate == "none":
        adjusted_df.insert(0, "Rank", range(1, 1 + len(adjusted_df)))
        return adjusted_df
    elif show_winrate == "gpt-3.5":
        adjusted_df = add_winrates_tasks(adjusted_df, ref="gpt-3.5", LP=length_penalty)
    elif show_winrate == "gpt-4":
        adjusted_df = add_winrates_tasks(adjusted_df, ref="gpt-4", LP=length_penalty)
    adjusted_df.insert(0, "Rank", range(1, 1 + len(adjusted_df)))
    return adjusted_df

def benchmark_table():
    global original_df, ablation_df
    global LP_original_dfs, LP_MODE
    
    gr.Markdown(f"**Version**: sotopia (v1.01; 2024.04.22) | **# Examples**: 7200 | **# Models**: {len(MODEL_OPTIONS)} | **# Comparisons**: x", elem_classes="markdown-text")
                
    with gr.TabItem("Vs GPT-3.5", elem_id="od-benchmark-tab-table-ablation", id=0, elem_classes="subtab"):
        # original_df, ablation_df = skip_empty_original_df, skip_empty_ablation_df
        original_df = pd.read_json('data_dir/models_vs_gpt35.jsonl', lines=True)
        default_main_df = apply_length_penalty(original_df, ablation_df, length_penalty=DEFAULT_LP, mode=LP_MODE, LP_original_dfs=LP_original_dfs) 
        default_main_df = default_main_df.sort_values(by="GOAL [0, 10]", ascending=False)
        # add a Rank column to the first columnn (starting from 1)
        default_main_df.insert(0, "Rank", range(1, 1 + len(default_main_df)))
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("**Vs GPT3.5**: The interlocutors are compared against GPT-3.5, the baseline model.") 
            with gr.Column(scale=1):
                length_penlty_slider = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=DEFAULT_LP, label="Length Penalty", elem_id="length-penalty-slider") 
        # checkbox_skip_empty = gr.Checkbox(label="Skip empty results", value=False, elem_id="skip-empty-checkbox", scale=2)
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
        #length_penlty_slider.change(fn=slider_change_main, inputs=[length_penlty_slider], outputs=[leaderboard_table])