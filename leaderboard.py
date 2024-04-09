import gradio as gr
import pandas as pd

# Data in the form of a dictionary, replace this with your actual benchmark data
data = {
    "Rank": [1, 2, 3, 4, 5],
    "Model": ["Claude...3.Opus", "gpt-4...gpt4-12b5-previews", "Claude...3.Sonnet", "Mistral..Large", "gemini...1..0..pro"],
    "Overall Elo": [1121, 1113, 1101, 1095, 1077],
    "Task-Avg Elo": [1109, 1109.8, 1085.4, 1082.2, 1062],
    "# battles": [4039, 6163, 3127, 2434, 2139],
    "Length Penalty": [24, 31, 24, 23, 20]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Create a Gradio interface to display the DataFrame as a table
def show_table(dummy_input):
    return df

iface = gr.Interface(
    fn=show_table,
    inputs=gr.components.Button(label="Show Leaderboard"),
    outputs="dataframe",
    title="Leaderboard",
    description="sample-table"
)

iface.launch()
