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
    "Llama-2-13b-chat-hf.nosp": {"pretty_name": "Llama-2-13B-chat", "hf_model_id": "meta-llama/Llama-2-13b-chat-hf"},
    "Llama-2-70b-chat-hf.nosp": {"pretty_name": "Llama-2-70B-chat", "hf_model_id": "meta-llama/Llama-2-70b-chat-hf"},
    "Llama-2-7b-chat-hf.nosp": {"pretty_name": "Llama-2-7B-chat", "hf_model_id": "meta-llama/Llama-2-7b-chat-hf"},
    "Llama-2-7b-chat-hf": {"pretty_name": "Llama-2-7B-chat (+sys prmpt)", "hf_model_id": "meta-llama/Llama-2-7b-chat-hf"},
    "Mistral-7B-Instruct-v0.1": {"pretty_name": "Mistral-7B-Instruct", "hf_model_id": "mistralai/Mistral-7B-Instruct-v0.1"},
    "Mistral-7B-Instruct-v0.2": {"pretty_name": "Mistral-7B-Instruct (v0.2)", "hf_model_id": "mistralai/Mistral-7B-Instruct-v0.2"},
    "Mixtral-8x7B-Instruct-v0.1": {"pretty_name": "Mixtral-8x7B-Instruct", "hf_model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
    "Nous-Hermes-2-Mixtral-8x7B-DPO": {"pretty_name": "Nous-Hermes-2-Mixtral-8x7B-DPO", "hf_model_id": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"},
    "Yi-34B-Chat": {"pretty_name": "Yi-34B-Chat", "hf_model_id": "01-ai/Yi-34B"},
    "gemini-1.0-pro": {"pretty_name": "gemini-1.0-pro", "hf_model_id": "https://blog.google/technology/ai/google-gemini-ai/"},
    "gemma-7b-it": {"pretty_name": "Gemma-7B-it", "hf_model_id": "google/gemma-7b"},
    "gpt-3.5-turbo-0125": {"pretty_name": "gpt-3.5-turbo-0125", "hf_model_id": "https://platform.openai.com/"},
    "gpt-4-0125-preview": {"pretty_name": "gpt-4-0125-preview", "hf_model_id": "https://platform.openai.com/"},
    "tulu-2-dpo-70b": {"pretty_name": "Tulu-2-dpo-70b", "hf_model_id": "cmu-lti/tulu-2-dpo-70b"},
    "vicuna-13b-v1.5": {"pretty_name": "Vicuna-13b-v1.5", "hf_model_id": "lmsys/vicuna-13b-v1.5"},
    "zephyr-7b-beta": {"pretty_name": "Zephyr-7b-beta", "hf_model_id": "HuggingFaceH4/zephyr-7b-beta"},
    "mistral-large-2402": {"pretty_name": "Mistral-Large", "hf_model_id": "https://mistral.ai/news/mistral-large/"},
    "claude-3-opus-20240229": {"pretty_name": "Claude 3 Opus", "hf_model_id": "https://www.anthropic.com/claude"},
    "claude-3-sonnet-20240229": {"pretty_name": "Claude 3 Sonnet", "hf_model_id": "https://www.anthropic.com/claude"},
    "zephyr-7b-gemma-v0.1": {"pretty_name": "Zephyr-7b-Gemma", "hf_model_id": "HuggingFaceH4/zephyr-7b-gemma-v0.1"},
    "Starling-LM-7B-beta": {"pretty_name": "StarlingLM-7B-beta", "hf_model_id": "Nexusflow/Starling-LM-7B-beta"},
    "dbrx-instruct": {"pretty_name": "DBRX Instruct", "hf_model_id": "databricks/dbrx-instruct"}
}
