import os
from dataclasses import dataclass
from uuid import uuid4

import gradio as gr
import torch
import transformers
from peft import PeftConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from utils import Agent, format_sotopia_prompt, get_starter_prompt, format_bot_message
from functools import cache

DEPLOYED = os.getenv("DEPLOYED", "true").lower() == "true"
DEFAULT_MODEL_SELECTION = "cmu-lti/sotopia-pi-mistral-7b-BC_SR" # "mistralai/Mistral-7B-Instruct-v0.1"
TEMPERATURE = 0.0
TOP_P = 1
MAX_TOKENS = 1024

def prepare_sotopia_info():
    human_agent = Agent(
        name="Ethan Johnson",
        background="Ethan Johnson is a 34-year-old male chef. He/him pronouns. Ethan Johnson is famous for cooking Italian food.",
        goal="Uknown",
        secrets="Uknown",
        personality="Ethan Johnson, a creative yet somewhat reserved individual, values power and fairness. He likes to analyse situations before deciding.",
    )

    machine_agent = Agent(
        name="Benjamin Jackson",
        background="Benjamin Jackson is a 24-year-old male environmental activist. He/him pronouns. Benjamin Jackson is well-known for his impassioned speeches.",
        goal="Figure out why they estranged you recently, and maintain the existing friendship (Extra information: you notice that your friend has been intentionally avoiding you, you would like to figure out why. You value your friendship with the friend and don't want to lose it.)",
        secrets="Descendant of a wealthy oil tycoon, rejects family fortune",
        personality="Benjamin Jackson, expressive and imaginative, leans towards self-direction and liberty. His decisions aim for societal betterment.",
    )

    scenario = (
        "Conversation between two friends, where one is upset and crying"
    )
    instructions = get_starter_prompt(machine_agent, human_agent, scenario)
    return human_agent, machine_agent, scenario, instructions

@cache
def prepare(model_name):
    compute_type = torch.float16
    
    if 'cmu-lti/sotopia-pi-mistral-7b-BC_SR'in model_name:
        model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        cache_dir="./.cache",
        device_map='cuda',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_type,
            )
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        model = PeftModel.from_pretrained(model, model_name).to("cuda")
    elif 'mistralai/Mistral-7B-Instruct-v0.1' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        cache_dir="./.cache",
        device_map='cuda',
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    else:
         raise RuntimeError(f"Model {model_name} not supported")
    return model, tokenizer


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


def update_user_names(scenario, *args):
    # Placeholder logic to get user names based on the scenario
    user_names = {
        "Scenario 1": ["Alice", "Bob"],
        "Scenario 2": ["Charlie", "Diana"],
    }
    return user_names.get(scenario, [])

def update_bot_names(user_name, scenario, *args):
    # Placeholder logic to get bot names based on user name and scenario
    bot_names = {
        ("Alice", "Scenario 1"): ["Bot-A1", "Bot-A2"],
        ("Bob", "Scenario 1"): ["Bot-B1", "Bot-B2"],
        ("Charlie", "Scenario 2"): ["Bot-C1", "Bot-C2"],
    }
    return bot_names.get((user_name, scenario), [])

def sotopia_info_accordion(accordion_visible=True):
    with gr.Accordion("Sotopia Information", open=accordion_visible):
        with gr.Column():
            model_name = gr.Dropdown(
                    choices=["cmu-lti/sotopia-pi-mistral-7b-BC_SR", "mistralai/Mistral-7B-Instruct-v0.1", "GPT3.5"],
                    value="cmu-lti/sotopia-pi-mistral-7b-BC_SR",
                    interactive=True,
                    label="Model Selection"
                )
            with gr.Row():
                # Create separate Textboxes for scenario, user profile, and bot profile
                scenario_info_display = gr.Textbox(label="Scenario", lines=4)
                user_agent_info_display = gr.Textbox(label="User Agent Profile", lines=4)
                bot_agent_info_display = gr.Textbox(label="Bot Agent Profile", lines=4)

            with gr.Row():
                scenario_dropdown = gr.Dropdown(
                    choices=["Scenario 1", "Scenario 2"],
                    label="Scenario Selection",
                    value="Scenario 1"
                )
                user_dropdown = gr.Dropdown(label="Human Agent Name")
                bot_dropdown = gr.Dropdown(label="Machine Agent Name")

                # Link updates and specify which information to update
                scenario_dropdown.change(update_user_names, inputs=[scenario_dropdown], outputs=[user_dropdown])
                user_dropdown.change(update_bot_names, inputs=[user_dropdown, scenario_dropdown], outputs=[bot_dropdown])
                scenario_dropdown.change(update_scenario_info, inputs=[scenario_dropdown], outputs=[scenario_info_display])
                user_dropdown.change(update_user_info, inputs=[user_dropdown], outputs=[user_agent_info_display])
                bot_dropdown.change(update_bot_info, inputs=[bot_dropdown], outputs=[bot_agent_info_display])
                
    return model_name, scenario_dropdown, user_dropdown, bot_dropdown

# Example of an update function, you need to define the logic based on your application's data
def update_scenario_info(scenario, scenario_info_display):
    # Logic to update scenario information
    scenario_info_display.value = "Details about " + scenario

# Similar functions for user and bot information update
def update_user_info(user_name, user_info_display):
    user_info_display.value = "Profile for " + user_name

def update_bot_info(bot_name, bot_info_display):
    bot_info_display.value = "Profile for " + bot_name

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
    #model, tokenizer = prepare()
    human_agent, machine_agent, scenario, instructions = prepare_sotopia_info()

    # history are input output pairs
    def run_chat(
        message: str,
        history,
        instructions: str,
        user_name: str,
        bot_name: str,
        model_selection:str
    ):
        model, tokenizer = prepare(model_selection)
        prompt = format_sotopia_prompt(
            message, history, instructions, user_name, bot_name
        )
        input_tokens = tokenizer(
            prompt, return_tensors="pt", padding="do_not_pad"
        ).input_ids.to("cuda")
        input_length = input_tokens.shape[-1]
        output_tokens = model.generate(
            input_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_length=MAX_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        output_tokens = output_tokens[:, input_length:]
        text_output = tokenizer.decode(
            output_tokens[0], skip_special_tokens=True
        )
        output = ""
        for _ in range(5):
            try:
                output = format_bot_message(text_output)
                break
            except Exception as e:
                print(e)
                print("Retrying...")
        return output

    with gr.Column():
        with gr.Row():
            model, user_name, bot_name, scenario = sotopia_info_accordion()

            instructions = instructions_accordion(instructions)

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
                        instructions,
                        user_name,
                        bot_name,
                        model,
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
    # prepare(DEFAULT_MODEL_SELECTION)
    start_demo()