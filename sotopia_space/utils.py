from ui_constants import column_names
from sotopia_space.constants import MODEL_INFO


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
    
# Formats the columns
def formatter(x):
    if type(x) is str:
        x = x
    else: 
        x = round(x, 2)
    return x

def post_processing(df, model_len_info):
    if model_len_info:
        df["Length"] = df["model_name"].apply(lambda x: model_len_info[x]["avg_len"])

    for col in df.columns:
        if col == "model_name":
            df[col] = df[col].apply(lambda x: x.replace(x, make_clickable_model(x)))
        else:
            df[col] = df[col].apply(formatter) # For numerical values 
    df.rename(columns=column_names, inplace=True)
    df.sort_values(by="GOAL [0, 10]", inplace=True, ascending=False)
    # put the "Overall Elo" and "Task-Avg Elo" column to the front
    # add the length info
    df = df[["model_name", "GOAL [0, 10]"] + [col for col in df.columns if col not in ["model_name", "GOAL [0, 10]"]]]
    return df
