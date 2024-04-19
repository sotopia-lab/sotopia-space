import os
from collections import defaultdict
import json

import gradio as gr

from utils import Environment, Agent, get_context_prompt, dialogue_history_prompt
from functools import cache
from sotopia_pi_generate import prepare_model, generate_action

DEPLOYED = os.getenv("DEPLOYED", "true").lower() == "true"
DEFAULT_MODEL_SELECTION = "gpt-3.5-turbo"
TEMPERATURE = 0.7
TOP_P = 1
MAX_TOKENS = 1024

ENVIRONMENT_PROFILES = "profiles/environment_profiles.jsonl"
AGENT_PROFILES = "profiles/agent_profiles.jsonl"
RELATIONSHIP_PROFILES = "profiles/relationship_profiles.jsonl"

ACTION_TYPES = ['none', 'action', 'non-verbal communication', 'speak', 'leave']

OPENAI_KEY_FILE="./openai_api.key"
if os.path.exists(OPENAI_KEY_FILE):
    with open(OPENAI_KEY_FILE, "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

@cache
def get_sotopia_profiles(env_file=ENVIRONMENT_PROFILES, agent_file=AGENT_PROFILES, relationship_file=RELATIONSHIP_PROFILES):
    with open(env_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    code_names_count = defaultdict(int)
    environments = []
    environment_dict = {}
    for profile in sorted(data, key=lambda x: x['codename']):
        env_obj = Environment(profile)
        if profile['codename'] in code_names_count:
            environments.append((
                "{}_{:05d}".format(profile['codename'], 
                                   code_names_count[profile['codename']]
                                   ), 
                env_obj._id
                ))
        else:
            environments.append((profile['codename'], env_obj._id))
        environment_dict[env_obj._id] = env_obj
        code_names_count[profile['codename']] += 1
    
    with open(agent_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    agent_dict = {}
    for profile in data:
        agent_obj = Agent(profile)
        agent_dict[agent_obj._id] = agent_obj
        
    with open(relationship_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    relationship_dict = defaultdict(lambda : defaultdict(list))
    for profile in data:
        relationship_dict[profile['relationship']][profile['agent1_id']].append(profile['agent2_id'])
        relationship_dict[profile['relationship']][profile['agent2_id']].append(profile['agent1_id'])
    
    return environments, environment_dict, agent_dict, relationship_dict


def introduction():
    with gr.Column(scale=2):
        gr.Image(
            "images/sotopia.jpg", elem_id="banner-image", show_label=False
        )
    with gr.Column(scale=5):
        gr.Markdown(
            """# Sotopia-Pi Demo
            **Chat with [Sotopia-Pi](https://github.com/sotopia-lab/sotopia-pi), brainstorm ideas, discuss your holiday plans, and more!**

            ➡️️ **Intended Use**: this demo is intended to showcase an early finetuning of [sotopia-pi-mistral-7b-BC_SR](https://huggingface.co/cmu-lti/sotopia-pi-mistral-7b-BC_SR)/

            ⚠️ **Limitations**: the model can and will produce factually incorrect information, hallucinating facts and actions. As it has not undergone any advanced tuning/alignment, it can produce problematic outputs, especially if prompted to do so. Finally, this demo is limited to a session length of about 1,000 words.

            🗄️ **Disclaimer**: User prompts and generated replies from the model may be collected by TII solely for the purpose of enhancing and refining our models. TII will not store any personally identifiable information associated with your inputs. By using this demo, users implicitly agree to these terms.
            """
        )

def create_user_agent_dropdown(environment_id):
    _, environment_dict, agent_dict, relationship_dict = get_sotopia_profiles()
    environment = environment_dict[environment_id]
    
    user_agents_list = []
    unique_agent_ids = set()
    for x, _ in relationship_dict[environment.relationship].items():
        unique_agent_ids.add(x)
    
    for agent_id in unique_agent_ids:
        user_agents_list.append((agent_dict[agent_id].name, agent_id))
    return gr.Dropdown(choices=user_agents_list, value=user_agents_list[0][1] if user_agents_list else None, label="User Agent Selection")

def create_bot_agent_dropdown(environment_id, user_agent_id):
    _, environment_dict, agent_dict, relationship_dict = get_sotopia_profiles()
    environment, user_agent = environment_dict[environment_id], agent_dict[user_agent_id]
    
    bot_agent_list = []
    # import pdb; pdb.set_trace()
    for neighbor_id in relationship_dict[environment.relationship][user_agent.agent_id]:
        bot_agent_list.append((agent_dict[neighbor_id].name, neighbor_id))
        
    return gr.Dropdown(choices=bot_agent_list, value=bot_agent_list[0][1] if bot_agent_list else None,  label="Bot Agent Selection")

def create_environment_info(environment_dropdown):
    _, environment_dict, _, _ = get_sotopia_profiles()
    environment = environment_dict[environment_dropdown]
    text = environment.scenario
    return gr.Textbox(label="Scenario", lines=1, value=text)

def create_user_info(user_agent_dropdown):
    _, _, agent_dict, _ = get_sotopia_profiles()
    user_agent = agent_dict[user_agent_dropdown]
    text = f"{user_agent.background} {user_agent.personality}"
    return gr.Textbox(label="User Agent Profile", lines=4, value=text)

def create_bot_info(bot_agent_dropdown):
    _, _, agent_dict, _ = get_sotopia_profiles()
    # import pdb; pdb.set_trace()
    bot_agent = agent_dict[bot_agent_dropdown]
    text = f"{bot_agent.background} {bot_agent.personality}"
    return gr.Textbox(label="Bot Agent Profile", lines=4, value=text)

def create_user_goal(environment_dropdown):
     _, environment_dict, _, _ = get_sotopia_profiles()
     text = environment_dict[environment_dropdown].agent_goals[0]
     return gr.Textbox(label="User Agent Goal", lines=4, value=text)

def create_bot_goal(environment_dropdown):
    _, environment_dict, _, _ = get_sotopia_profiles()
    text = environment_dict[environment_dropdown].agent_goals[1]
    return gr.Textbox(label="Bot Agent Goal", lines=4, value=text)

def sotopia_info_accordion(accordion_visible=True):
    environments, _, _, _ = get_sotopia_profiles()
    
    with gr.Accordion("Environment Configuration", open=accordion_visible):
        with gr.Row():
            environment_dropdown = gr.Dropdown(
                choices=environments,
                label="Scenario Selection",
                value=environments[0][1] if environments else None,
                interactive=True,
            )
            model_name_dropdown = gr.Dropdown(
                choices=["cmu-lti/sotopia-pi-mistral-7b-BC_SR", "cmu-lti/sotopia-pi-mistral-7b-BC_SR_4bit", "mistralai/Mistral-7B-Instruct-v0.1", "gpt-3.5-turbo", "gpt-4-turbo"],
                value=DEFAULT_MODEL_SELECTION,
                interactive=True,
                label="Model Selection"
            )
        
        scenario_info_display = create_environment_info(environment_dropdown.value)
            
        with gr.Row():
            bot_goal_display = create_bot_goal(environment_dropdown.value)
            user_goal_display = create_user_goal(environment_dropdown.value)
            
        with gr.Row():
            user_agent_dropdown = create_user_agent_dropdown(environment_dropdown.value)
            bot_agent_dropdown = create_bot_agent_dropdown(environment_dropdown.value, user_agent_dropdown.value)
        
        with gr.Row():
            user_agent_info_display = create_user_info(user_agent_dropdown.value)
            bot_agent_info_display = create_bot_info(bot_agent_dropdown.value)

        # Update user dropdown when scenario changes
        environment_dropdown.change(fn=create_user_agent_dropdown, inputs=[environment_dropdown], outputs=[user_agent_dropdown])
        # Update bot dropdown when user or scenario changes
        user_agent_dropdown.change(fn=create_bot_agent_dropdown, inputs=[environment_dropdown, user_agent_dropdown], outputs=[bot_agent_dropdown])
        # Update scenario information when scenario changes
        environment_dropdown.change(fn=create_environment_info, inputs=[environment_dropdown], outputs=[scenario_info_display])
        # Update user agent profile when user changes
        user_agent_dropdown.change(fn=create_user_info, inputs=[user_agent_dropdown], outputs=[user_agent_info_display])
        # Update bot agent profile when bot changes
        bot_agent_dropdown.change(fn=create_bot_info, inputs=[bot_agent_dropdown], outputs=[bot_agent_info_display])
        # Update user goal when scenario changes
        environment_dropdown.change(fn=create_user_goal, inputs=[environment_dropdown], outputs=[user_goal_display])
        # Update bot goal when scenario changes
        environment_dropdown.change(fn=create_bot_goal, inputs=[environment_dropdown], outputs=[bot_goal_display])

    return model_name_dropdown, environment_dropdown, user_agent_dropdown, bot_agent_dropdown

def instructions_accordion(instructions, according_visible=False):
    with gr.Accordion("Instructions", open=False, visible=according_visible):
        instructions = gr.Textbox(
            lines=10,
            value=instructions,
            interactive=False,
            placeholder="Instructions",
            show_label=False,
            max_lines=10,
            visible=False,
        )
    return instructions


def chat_tab():
    # history are input output pairs
    _, environment_dict, agent_dict, _ = get_sotopia_profiles()
    def run_chat(
        message,
        history,
        environment_selection,
        user_agent_dropdown,
        bot_agent_dropdown,
        model_selection:str
    ):
        environment = environment_dict[environment_selection]
        user_agent = agent_dict[user_agent_dropdown]
        bot_agent = agent_dict[bot_agent_dropdown]
        
        # import pdb; pdb.set_trace()
        context = get_context_prompt(bot_agent, user_agent, environment)
        dialogue_history, next_turn_idx = dialogue_history_prompt(message, history, user_agent, bot_agent)
        prompt_history = f"{context}\n\n{dialogue_history}"
        agent_action = generate_action(model_selection, prompt_history, next_turn_idx, ACTION_TYPES, bot_agent.name, TEMPERATURE)
        # import pdb; pdb.set_trace()
        return agent_action.to_natural_language()
    
    with gr.Column():
        with gr.Row():
            model_name_dropdown, scenario_dropdown, user_agent_dropdown, bot_agent_dropdown = sotopia_info_accordion()
            
        with gr.Column():
            with gr.Blocks():
                gr.ChatInterface(
                    fn=run_chat,
                    chatbot=gr.Chatbot(
                        height=620,
                        render=False,
                        show_label=False,
                        rtl=False,
                        avatar_images=(
                            "images/profile1.jpg",
                            "images/profile2.jpg",
                        ),
                    ),
                    textbox=gr.Textbox(
                        placeholder="Write your message here...",
                        render=False,
                        scale=7,
                        rtl=False,
                    ),
                    additional_inputs=[
                        scenario_dropdown,
                        user_agent_dropdown,
                        bot_agent_dropdown,
                        model_name_dropdown,
                    ],
                    submit_btn="Send",
                    stop_btn="Stop",
                    retry_btn="🔄 Retry",
                    undo_btn="↩️ Delete",
                    clear_btn="🗑️ Clear",
                )


def main():
    with gr.Blocks(
        css="""#chat_container {height: 820px; width: 1000px; margin-left: auto; margin-right: auto;}
               #chatbot {height: 600px; overflow: auto;}
               #create_container {height: 750px; margin-left: 0px; margin-right: 0px;}
               #tokenizer_renderer span {white-space: pre-wrap}
               """
    ) as demo:
        with gr.Row():
            introduction()
        with gr.Row():
            chat_tab()

    return demo


def start_demo():
    demo = main()
    if DEPLOYED:
        demo.queue(api_open=False).launch(show_api=False)
    else:
        demo.queue()
        demo.launch(share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    get_sotopia_profiles()
    # prepare_model(DEFAULT_MODEL_SELECTION)
    start_demo()