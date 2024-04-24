export OPENAI_API_KEY=$(cat openai_api.key)
export HF_TOKEN=$(cat hf_token.key)

gradio app.py