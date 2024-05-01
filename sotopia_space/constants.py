MODEL_OPTIONS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "cmu-lti/sotopia-pi-mistral-7b-BC_SR",
    "cmu-lti/sotopia-pi-mistral-7b-BC_SR_4bit",
    "mistralai/Mistral-7B-Instruct-v0.1"
    # "mistralai/Mixtral-8x7B-Instruct-v0.1", # TODO: Add these model
    # "togethercomputer/llama-2-7b-chat",
    # "togethercomputer/llama-2-70b-chat",
    # "togethercomputer/mpt-30b-chat",
    # "together_ai/togethercomputer/llama-2-7b-chat",
    # "together_ai/togethercomputer/falcon-7b-instruct",
]

MODEL_INFO = {
    "GPT-4": {"pretty_name": "GPT-4", "hf_model_id": "https://openai.com/blog/new-models-and-developer-products-announced-at-devday"},
    "GPT-3.5": {"pretty_name": "GPT-3.5", "hf_model_id": "https://openai.com/blog/new-models-and-developer-products-announced-at-devday"},
    "Llama-2": {"pretty_name": "Llama-2", "hf_model_id": "https://llama.meta.com/llama2/"},
    "MPT": {"pretty_name": "MPT", "hf_model_id": "https://huggingface.co/docs/transformers/main/en/model_doc/mpt"}
}
