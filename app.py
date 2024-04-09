import gradio as gr
from dataclasses import dataclass
import os
import torch
import transformers
from uuid import uuid4
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from litellm import completion
from utils import Agent, get_starter_prompt, format_sotopia_prompt

DEPLOYED = os.getenv("DEPLOYED", "true").lower() == "true" 


def prepare_sotopia_info():
    human_agent = Agent(
        name="Ethan Johnson",
        background="Ethan Johnson is a 34-year-old male chef. He/him pronouns. Ethan Johnson is famous for cooking Italian food.",
        goal="Uknown",
        secrets="Uknown",
        personality="Ethan Johnson, a creative yet somewhat reserved individual, values power and fairness. He likes to analyse situations before deciding.",)

    machine_agent = Agent(
        name="Benjamin Jackson",
        background="Benjamin Jackson is a 24-year-old male environmental activist. He/him pronouns. Benjamin Jackson is well-known for his impassioned speeches.",
        goal="Figure out why they estranged you recently, and maintain the existing friendship (Extra information: you notice that your friend has been intentionally avoiding you, you would like to figure out why. You value your friendship with the friend and don't want to lose it.)",
        secrets="Descendant of a wealthy oil tycoon, rejects family fortune",
        personality="Benjamin Jackson, expressive and imaginative, leans towards self-direction and liberty. His decisions aim for societal betterment.",)

    scenario = "Conversation between two friends, where one is upset and crying"
    instructions = get_starter_prompt(machine_agent, human_agent, scenario)
    return human_agent, machine_agent, scenario, instructions




'''def prepare(model_name):
    compute_type = torch.float16
    config_dict = PeftConfig.from_json_file("peft_config.json")
    config = PeftConfig.from_peft_type(**config_dict)
    
    if 'mistral'in model_name:
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        model = PeftModel.from_pretrained(model, model_name, config=config).to(compute_type).to("cuda")
    else:
         tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
'''



def introduction():
    with gr.Column(scale=2):
        gr.Image("images/sotopia.jpg", elem_id="banner-image", show_label=False)
    with gr.Column(scale=5):
        gr.Markdown(
            """# Sotopia-Pi Demo
            **Chat with [Sotopia-Pi](https://github.com/sotopia-lab/sotopia-pi), brainstorm ideas, discuss your holiday plans, and more!**
            
            ➡️️ **Intended Use**: this demo is intended to showcase an early finetuning of [sotopia-pi-mistral-7b-BC_SR](https://huggingface.co/cmu-lti/sotopia-pi-mistral-7b-BC_SR)/
            
            ⚠️ **Limitations**: the model can and will produce factually incorrect information, hallucinating facts and actions. As it has not undergone any advanced tuning/alignment, it can produce problematic outputs, especially if prompted to do so. Finally, this demo is limited to a session length of about 1,000 words.
            
            🗄️ **Disclaimer**: User prompts and generated replies from the model may be collected by TII solely for the purpose of enhancing and refining our models. TII will not store any personally identifiable information associated with your inputs. By using this demo, users implicitly agree to these terms.
            """
        )



def param_accordion(according_visible=True):
    with gr.Accordion("Parameters", open=False, visible=according_visible):
        model_name  = gr.Dropdown(
            choices=["cmu-lti/sotopia-pi-mistral-7b-BC_SR", "mistralai/Mistral-7B-Instruct-v0.1", "GPT3.5"],  # Example model choices
            value="cmu-lti/sotopia-pi-mistral-7b-BC_SR",  # Default value
            interactive=True,
            label="Model Selection",
        )
        temperature = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        max_tokens = gr.Slider(
            minimum=1024,
            maximum=4096,
            value=1024,
            step=1,
            interactive=True,
            label="Max Tokens",
        )
        session_id = gr.Textbox(
            value=uuid4,
            interactive=False,
            visible=False,
            label="Session ID",
        )
    return temperature, session_id, max_tokens, model_name 


def sotopia_info_accordion(human_agent, machine_agent, scenario, according_visible=True):
    with gr.Accordion("Sotopia Information", open=False, visible=according_visible):
        with gr.Row():
            with gr.Column():
                user_name = gr.Textbox(
                    lines=1,
                    label="username",
                    value=human_agent.name,
                    interactive=True,
                    placeholder=f"{human_agent.name}: ",
                    show_label=False,
                    max_lines=1,
                )
            with gr.Column():
                bot_name = gr.Textbox(
                    lines=1,
                    value=machine_agent.name,
                    interactive=True,
                    placeholder=f"{machine_agent.name}: ",
                    show_label=False,
                    max_lines=1,
                    visible=False,
                )
            with gr.Column():
                scenario = gr.Textbox(
                    lines=4,
                    value=scenario,
                    interactive=False,
                    placeholder="Scenario",
                    show_label=False,
                    max_lines=4,
                    visible=False,
                )
    return user_name, bot_name, scenario


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


# history are input output pairs
def run_chat(
    message: str,
    history,
    instructions: str,
    user_name: str,
    bot_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_selection:str
):
    prompt = format_sotopia_prompt(
            message, 
            history, 
            instructions, 
            user_name, 
            bot_name
        )
    if 'GPT' not in model_selection:
            text_output = completion(
            model=f"huggingface/{model_selection}",
            messages=[{ f"content": prompt,"role": "user"}],
            api_base="https://your-huggingface-api-endpoint",
            max_tokens=max_tokens
        )
    else:
            text_output = completion(
            model="gpt-3.5-turbo",
            messages=[{"content": prompt,"role": "user"}],
            max_tokens=max_tokens
        )
    return text_output
  
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
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_selection:str
    ):
        prompt = format_sotopia_prompt(
            message, 
            history, 
            instructions, 
            user_name, 
            bot_name
        )
        if 'GPT' not in model_selection:
            text_output = completion(
            model=f"huggingface/{model_selection}",
            messages=[{ f"content": "{prompt}","role": "user"}],
            api_base="https://your-huggingface-api-endpoint",
            max_tokens=max_tokens
        ).choices[0].message.content
        else:
            text_output = completion(
            model="gpt-3.5-turbo",
            messages=[{ f"content": "{prompt}","role": "user"}],
            max_tokens=max_tokens
        ).choices[0].message.content
        return text_output

    with gr.Column():
        with gr.Row():
            temperature, session_id, max_tokens, model = param_accordion()
            user_name, bot_name, scenario = sotopia_info_accordion(human_agent, machine_agent, scenario)
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
                            "images/profile2.jpg"
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
                        temperature,
                        session_id,
                        max_tokens,
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
    start_demo()
