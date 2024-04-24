from datasets import load_dataset, Dataset
import os 
import json 
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar # type: ignore
from ui_constants import column_names, all_task_types
import random 
disable_progress_bar()
import math 
from sotopia_space.constants import MODEL_INFO

id_to_data = None 
model_len_info = None 


def make_clickable_model(model_name):
    global MODEL_INFO
    if model_name in MODEL_INFO:
        if MODEL_INFO[model_name]["hf_model_id"].startswith("http"):
            link = MODEL_INFO[model_name]["hf_model_id"]
            return f'🔒 <a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{MODEL_INFO[model_name]["pretty_name"]}</a>'
        else:
            link = f"https://huggingface.co/{MODEL_INFO[model_name]['hf_model_id']}"
            return f'🔥 <a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{MODEL_INFO[model_name]["pretty_name"]}</a>'
    else:
        return model_name
    

def styled_error(error):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{error}</p>"

def styled_warning(warn):
    return f"<p style='color: orange; font-size: 20px; text-align: center;'>{warn}</p>"

def styled_message(message):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{message}</p>"
        

def estimated_win_rate(elo_a, elo_b, LP=0):
    """
    Calculate the estimated win rate for player A against player B using their Elo ratings.
    :param elo_a: Elo rating of player A
    :param elo_b: Elo rating of player B
    :return: Estimated win rate for player A
    """
    exponent = (elo_b - elo_a)*(10**LP) / 400
    probability_a_wins = 1 / (1 + 10 ** exponent)
    return (1-probability_a_wins)*100



# Formats the columns
def formatter(x):
    if type(x) is str:
        x = x
    else: 
        x = round(x, 1)
    return x


def add_winrates(current_df, LP=0):
    df = current_df.copy()
    elo_column = "Task-Avg Elo" 

    # Correct way to filter the DataFrame and get the Elo rating for "gpt-4-0125-preview"
    model_a_elo = df[df["Model"].str.contains("gpt-4")][elo_column].iloc[0]

    # Correct way to filter the DataFrame and get the Elo rating for "gpt-3.5-turbo-0125"
    model_b_elo = df[df["Model"].str.contains("gpt-3.5")][elo_column].iloc[0]

    
    # Calculate the win rate of "gpt-4-0125-preview" against all models
    df['Win% vs GPT-4'] = df[elo_column].apply(lambda x: estimated_win_rate(model_a_elo, x, LP=LP)).apply(formatter)    
    df['Win% vs GPT-3.5T'] = df[elo_column].apply(lambda x: estimated_win_rate(model_b_elo, x, LP=LP)).apply(formatter)    
    # apply the formatter for the two new columns 
    cols = list(df.columns)
    cols.remove("# battles"); cols.append("# battles")
    cols.remove("Length"); cols.append("Length")
    df = df[cols]
    return df

def add_winrates_tasks(current_df, ref="gpt-4", LP=0):
    new_df = current_df.copy()
    for t in all_task_types:
        column = column_names[t]
        model_a_elo = current_df[current_df["Model"].str.contains(ref)][column].iloc[0]
        new_df[column] = current_df[column].apply(lambda x: estimated_win_rate(model_a_elo, x, LP=LP)).apply(formatter)
    return new_df
        

def post_processing(df, model_len_info):
    if model_len_info:
        df["Length"] = df["model name "].apply(lambda x: model_len_info[x]["avg_len"])

    for col in df.columns:
        if col == "model name ":
            df[col] = df[col].apply(lambda x: x.replace(x, make_clickable_model(x)))
        else:
            df[col] = df[col].apply(formatter) # For numerical values 
    df.rename(columns=column_names, inplace=True)
    df.sort_values(by="Task-Avg Elo", inplace=True, ascending=False)
    # put the "Overall Elo" and "Task-Avg Elo" column to the front
    # add the length info
    df = df[["Model", "Task-Avg Elo"] + [col for col in df.columns if col not in ["Model", "Task-Avg Elo"]]]
    return df

def apply_length_penalty(original_df, ablation_df, length_penalty=0.2, mode='v1', LP_original_dfs=None):
    """
    Temporarily disable the length penalty feature
    if mode == 'v2' and LP_original_dfs is not None:
        L = f"{length_penalty:.1f}"
        return LP_original_dfs[L]
    original_df = original_df.copy()
    ablation_df = ablation_df.copy()
    # replace all values in original_df with the values as z = x - y * length_penalty where y is from ablation_df at the same row and column
    # except for the "Model" column and the "# battles" column 
    # do not assume the order of the rows are the same in both dataframes
    for i, row in original_df.iterrows():
        for col in original_df.columns:
            if col == "Model" or col == "# battles" or col == "Length":
                continue
            # assert that the model names are the same in both dataframes
            assert original_df.at[i, "Model"] == ablation_df[ablation_df["Model"] == row["Model"]]["Model"].values[0]
            original_df[col] = original_df[col].astype(float)
            if mode == "v1":
                original_df.at[i, col] = original_df.at[i, col] - ablation_df[ablation_df["Model"] == row["Model"]][col].values[0] * length_penalty
            elif mode == "v1.1":
                diff = original_df.at[i, col] - ablation_df[ablation_df["Model"] == row["Model"]][col].values[0] 
                original_df.at[i, col] = original_df.at[i, col] * (1-length_penalty) + diff*length_penalty
    # post_processing
    original_df = post_processing(original_df, model_len_info=None)
    """
    return original_df 

def load_benchdata():
    print("Loading sotopia data...")
    bench_data = load_dataset("cmu-lti/sotopia", split="test")
    return bench_data

def load_benchdata_dict():
    print("Loading sotopia data....")
    bench_data = load_dataset("cmu-lti/sotopia", data_files="sotopia_episodes_v1_hf.jsonl")['train']
    id_to_data = {}
    for item in bench_data:
        id_to_data[item["session_id"]] = item
    return id_to_data

def load_eval_results():
    print("Loading sotopia Evaluation data...")
    eval_results = load_dataset("WildEval/sotopia-Evaluation", "all", split="train")
    return eval_results

def load_infer_results(model_name):
    print(f"Loading sotopia Results for {model_name}...")
    infer_results = load_dataset("WildEval/sotopia-Results", model_name, split="train")
    return infer_results

def sample_an_eval_result(eval_results, model_list=[], tag_list=[]):
    global id_to_data          
    eval_results = list(eval_results)
    random.shuffle(eval_results)
    for eval_item in eval_results:  
        # print(json.dumps(eval_item, indent=2))
        # print(f"## Session ID: {eval_item['session_id']}")
        # eval_item["eval_id"]
        assignment = eval_item['assignment']
        model_1, model_2 = eval_item['model_1'], eval_item['model_2']
        model_A = model_1 if assignment['A'] == model_1 else model_2
        model_B = model_2 if assignment['B'] == model_2 else model_1
        if len(model_list) >= 2:
            if model_A not in model_list or model_B not in model_list:
                continue
        elif len(model_list) == 1:
            if model_A != model_list[0] and model_B != model_list[0]:
                continue
        else:
            pass 
        if tag_list:
            if set(tag_list).isdisjoint(set(eval_item['tags'])):
                continue
        winner = eval_item['winner']
        # print(f"## Model A: {model_A} | Model B: {model_B} | Winner: {winner}")
        task_type = eval_item['tags'][0] # primary task type
        chat_history = eval_item['history']
        last_query = eval_item['last_query']
        # print(f"## Task Type: {task_type}")
        # print(f"## Chat History: {chat_history}")
        # print(f"## Last Query -->  USER: {last_query}")

        model_A_output = eval_item['model_1_output'] if model_1 == model_A else eval_item['model_2_output']
        model_B_output = eval_item['model_2_output'] if model_2 == model_B else eval_item['model_1_output']

        if len(model_A_output.strip()) == 0 or len(model_B_output.strip()) == 0:
            continue

        conversation_input = id_to_data[eval_item['session_id']]["conversation_input"]
        # print(f"\n\n\n## Model A ({model_A}) Output ##\n{model_A_output}")
        # print(f"\n\n\n## Model B ({model_B}) Output ##\n{model_B_output}")

        # print(f"\n\n\n## Winner ##\n{winner}")
        # print(f"\n\n\n## GPT-4 Judgement ##\n{eval_item['parsed_result']}")

        result_dict = {
            "session_id": eval_item['session_id'],
            "model_A": model_A,
            "model_B": model_B,
            "winner": winner,
            "intent": id_to_data[eval_item['session_id']]["intent"],
            "task_type": task_type,
            "all_tags": eval_item['tags'],
            "chat_history": chat_history,
            "last_query": last_query,
            "conversation_input": conversation_input,
            "model_A_output": model_A_output,
            "model_B_output": model_B_output,
            "reason": eval_item['parsed_result']["reason"],
            "choice": eval_item['parsed_result']["choice"],
            "checklist": id_to_data[eval_item['session_id']]["checklist"],
        }
        break 
    return result_dict

#id_to_data = load_benchdata_dict()