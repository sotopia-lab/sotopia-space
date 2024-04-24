from pathlib import Path

DEFAULT_LP = 0.5

banner_url = "https://github.com/sotopia-lab/sotopia-website/blob/main/public/bg_xl.png" # the same repo here.
BANNER = f'<div style="display: flex; justify-content: flex-start;"><img src="{banner_url}" alt="Banner" style="width: 40vw; min-width: 300px; max-width: 800px;"> </div>'

TITLE = "<html> <head> <style> h1 {text-align: center;} </style> </head> <body> <h1> 🦁 AI2 sotopia Leaderboard </b> </body> </html>"
 
WINRATE_HEATMAP = "<div><img src='https://github.com/WildEval/sotopia-Leaderboard/blob/main/gradio/pairwise_win_fractions.png?raw=true' style='width:100%;'></div>"

CITATION_TEXT = """@inproceedings{
zhou2024sotopia,
title={{SOTOPIA}: Interactive Evaluation for Social Intelligence in Language Agents},
author={Xuhui Zhou and Hao Zhu and Leena Mathur and Ruohong Zhang and Haofei Yu and Zhengyang Qi and Louis-Philippe Morency and Yonatan Bisk and Daniel Fried and Graham Neubig and Maarten Sap},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=mM7VurbA4r}
}
"""


column_names = {
    "model name ": "Model",
    "elo overall": "Overall Elo",
    'Information seeking': 'InfoSek',
    'Creative Writing': 'CrtWrt',
    'Coding & Debugging': 'Code',
    'Reasoning': 'Reason',
    'Editing': 'Edit',
    'Math': 'Math',
    'Planning': 'Plan',
    'Brainstorming': 'Brnstrm',
    'Role playing': 'RolPly',
    'Advice seeking': 'AdvSek',
    'Data Analysis': 'DataAna',
    'Others': 'Misc',
    "average": "Task-Avg Elo",
}

all_task_types = [
    'Information seeking',
    'Creative Writing',
    'Coding & Debugging',
    'Reasoning',
    'Editing',
    'Math',
    'Planning',
    'Brainstorming',
    'Role playing',
    'Advice seeking',
    'Data Analysis',
    'Others'
]



js_light = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

js_code = """
function scroll_top() {
    console.log("Hello from Gradio!");  
    const bubbles = document.querySelectorAll('.bubble-wrap');
    bubbles.forEach((bubble, index) => {
        setTimeout(() => {
            bubble.scrollTop = 0;
        }, index * 100); // Delay of 100ms between each iteration
    });
} 
"""


TASK_TYPE_STR = "**Tasks**: Info seeking (**InfoSek**), Creative Writing (**CrtWrt**), Coding&Debugging (**Code**), Reasoning (**Reason**), Editing (**Edit**), **Math**, Planning (**Plan**), Brainstorming (**Brnstrm**), Role playing (**RolPly**), Advice seeking (**AdvSek**), Data Analysis (**DataAna**)"

css = """
code {
    font-size: large;
}
footer {visibility: hidden}
.top-left-LP{
    margin-top: 6px;
    margin-left: 5px;
}
.markdown-text{font-size: 14pt}
.markdown-text-small{font-size: 13pt}
.markdown-text-tiny{font-size: 12pt}
.markdown-text-tiny-red{
    font-size: 12pt;
    color: red;
    background-color: yellow;
    font-color: red;
    font-weight: bold;
}
th {
  text-align: center;
  font-size: 17px; /* Adjust the font size as needed */
}
td {
  font-size: 15px; /* Adjust the font size as needed */
  text-align: center;
}
.sample_button{
    border: 1px solid #000000;
    border-radius: 5px;
    padding: 5px;
    font-size: 15pt;
    font-weight: bold;
    margin: 5px;
}
.chat-common{
    height: auto;
    max-height: 400px;
    min-height: 100px; 
}
.chat-specific{
    height: auto;
    max-height: 600px;
    min-height: 200px; 
}
#od-benchmark-tab-table-button{
    font-size: 15pt;
    font-weight: bold;
} 
.btn_boderline{
    border: 1px solid #000000;
    border-radius: 5px;
    padding: 5px;
    margin: 5px;
    font-size: 15pt;
    font-weight: bold; 
}
.btn_boderline_next{
    border: 0.1px solid #000000;
    border-radius: 5px;
    padding: 5px;
    margin: 5px;
    font-size: 15pt;
    font-weight: bold; 
}
.btn_boderline_gray{
    border: 0.5px solid gray;
    border-radius: 5px;
    padding: 5px;
    margin: 5px;
    font-size: 15pt;
    font-weight: italic; 
}
.btn_boderline_selected{
    border: 2px solid purple;
    background-color: #f2f2f2;
    border-radius: 5px;
    padding: 5px;
    margin: 5px;
    font-size: 15pt;
    font-weight: bold;  
}
.accordion-label button span{
    font-size: 14pt;
    font-weight: bold;
} 
#select-models span{
    font-size: 10pt;
}
#select-tasks span{
    font-size: 10pt;
}
.markdown-text-details{
    margin: 10px;
    padding: 10px;
}
button.selected[role="tab"][aria-selected="true"] {
    font-size: 18px; /* or any other size you prefer */
    font-weight: bold;
}
#od-benchmark-tab-table-ablation-button {
    font-size: larger; /* Adjust the font size as needed */
}
.plotly-plot{
    height: auto;
    max-height: 600px;
    min-height: 600px; 
}
"""