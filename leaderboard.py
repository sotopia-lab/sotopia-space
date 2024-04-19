import gradio as gr
import pandas as pd

# Data in the form of a dictionary, replace this with your actual benchmark data
data = {
    "Rank": range(1, 15),  # Example rank numbers
    "Model": ["Model_{}".format(i) for i in range(1, 15)],  # Example model names
    "Overall Elo": [1126, 1115, 1109, 1109, 1094, 1089, 1074, 1048, 1015, 1013, 1012, 1008, 1007, 983],  # Example Elo scores
    "Task-Avg Elo": [1108.4, 1122.8, 1095.8, 1088.3, 1076.7, 1085.9, 1051, 1031.9, 1010.1, 1002, 1011.6, 1004, 1017.2, 983],  # Example average Elos
    "# battles": [4039, 14627, 2434, 3127, 2139, 6163, 2014, 3739, 2045, 2731, 2637, 1599, 2863, 1647],  # Example battle numbers
    "Length Penalty": [24.60, 17.25, 23.52, 24.56, 24.07, 31.90, 21.48, 24.84, 29.20, 28.99, 28.52, 28.78, 19.60, 25.52]  # Example length penalties
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Define the function that returns the DataFrame
def show_table():
    return df

# Create the Gradio interface
iface = gr.Interface(
    fn=show_table,
    inputs=[],
    outputs="dataframe",
    title="Leaderboard",
    description="Task-Avg Elo: Compute Elo on subsets of each task type and then take their avg. | Win Rates: Estimated by Elo differences. | Length penalty: Models with longer outputs are penalized. (Please check Details.)",
    theme="default",  # Use the default theme which is quite clean
    live=True,  # Automatically refreshes the output
    #show_submit_button=False,  # Remove the submit button
    allow_flagging=False  # Remove the flagging option
)

# Launch the interface
iface.launch()
