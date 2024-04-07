import gradio as gr
from dataclasses import dataclass
import os
import torch
from uuid import uuid4
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import Agent, get_starter_prompt, format_chat_prompt


HUMAN_AGENT = Agent(
    name="Ethan Johnson",
    background="Ethan Johnson is a 34-year-old male chef. He/him pronouns. Ethan Johnson is famous for cooking Italian food.",
    goal="Uknown",
    secrets="Uknown",
    personality="Ethan Johnson, a creative yet somewhat reserved individual, values power and fairness. He likes to analyse situations before deciding.",)

MACHINE_AGENT = Agent(
    name="Benjamin Jackson",
    background="Benjamin Jackson is a 24-year-old male environmental activist. He/him pronouns. Benjamin Jackson is well-known for his impassioned speeches.",
    goal="Figure out why they estranged you recently, and maintain the existing friendship (Extra information: you notice that your friend has been intentionally avoiding you, you would like to figure out why. You value your friendship with the friend and don't want to lose it.)",
    secrets="Descendant of a wealthy oil tycoon, rejects family fortune",
    personality="Benjamin Jackson, expressive and imaginative, leans towards self-direction and liberty. His decisions aim for societal betterment.",)

DEFUALT_INSTRUCTIONS = get_starter_prompt(MACHINE_AGENT, HUMAN_AGENT, "Conversation between two friends, where one is upset and crying")

DEPLOYED = os.getenv("DEPLOYED", "true").lower() == "true" 
MODEL_NAME = "cmu-lti/sotopia-pi-mistral-7b-BC_SR"
COMPUTE_DTYPE = torch.float16

config_dict = PeftConfig.from_json_file("peft_config.json")
# import pdb; pdb.set_trace()
config = PeftConfig.from_peft_type(**config_dict)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = PeftModel.from_pretrained(model, MODEL_NAME, config=config).to(COMPUTE_DTYPE).to("cuda")
according_visible = True

def introduction():
    with gr.Column(scale=2):
        gr.Image("images/sotopia.jpeg", elem_id="banner-image", show_label=False)
    with gr.Column(scale=5):
        gr.Markdown(
            """# Sotopia-Pi Demo
            **Chat with [Sotopia-Pi](https://github.com/sotopia-lab/sotopia-pi), brainstorm ideas, discuss your holiday plans, and more!**
            
            ➡️️ **Intended Use**: this demo is intended to showcase an early finetuning of [sotopia-pi-mistral-7b-BC_SR](https://huggingface.co/cmu-lti/sotopia-pi-mistral-7b-BC_SR)/
            
            ⚠️ **Limitations**: the model can and will produce factually incorrect information, hallucinating facts and actions. As it has not undergone any advanced tuning/alignment, it can produce problematic outputs, especially if prompted to do so. Finally, this demo is limited to a session length of about 1,000 words.
            
            🗄️ **Disclaimer**: User prompts and generated replies from the model may be collected by TII solely for the purpose of enhancing and refining our models. TII will not store any personally identifiable information associated with your inputs. By using this demo, users implicitly agree to these terms.
            """
        )

def chat_accordion():
    with gr.Accordion("Parameters", open=False, visible=according_visible):
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
        )

    with gr.Accordion("Instructions", open=False, visible=False):
        instructions = gr.Textbox(
            placeholder="The Instructions",
            value=DEFUALT_INSTRUCTIONS,
            lines=16,
            interactive=True,
            label="Instructions",
            max_lines=16,
            show_label=False,
        )
        with gr.Row():
            with gr.Column():
                user_name = gr.Textbox(
                    lines=1,
                    label="username",
                    value=HUMAN_AGENT.name,
                    interactive=True,
                    placeholder="Username: ",
                    show_label=False,
                    max_lines=1,
                )
            with gr.Column():
                bot_name = gr.Textbox(
                    lines=1,
                    value=MACHINE_AGENT.name,
                    interactive=True,
                    placeholder="Bot Name",
                    show_label=False,
                    max_lines=1,
                    visible=False,
                )

    return temperature, instructions, user_name, bot_name, session_id, max_tokens


def chat_tab():
    def run_chat(
        message: str,
        history,
        instructions: str,
        user_name: str,
        bot_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ):
        prompt = format_chat_prompt(message, history, instructions, user_name, bot_name)
        input_tokens = tokenizer(prompt, return_tensors="pt", padding="do_not_pad").input_ids.to("cuda")
        output = model.generate(
            input_tokens,
            temperature=temperature,
            top_p=top_p,
            max_length=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        # import pdb; pdb.set_trace()
        return tokenizer.decode(output[0], skip_special_tokens=True)

    with gr.Column():
        with gr.Row():
            (
                temperature,
                instructions,
                user_name,
                bot_name,
                session_id,
                max_tokens
            ) = chat_accordion()

        with gr.Column():
            with gr.Blocks():
                gr.ChatInterface(
                    fn=run_chat,
                    chatbot=gr.Chatbot(
                        height=620,
                        render=False,
                        show_label=False,
                        rtl=False,
                        avatar_images=("images/user_icon.png", "images/bot_icon.png"),
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
                        max_tokens
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
