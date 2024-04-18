from typing import List, Tuple
import ast

class Agent:
    def __init__(self, agent_profile):
        self._id = agent_profile["agent_id"]
        
        self.agent_profile = agent_profile
        self.agent_id = agent_profile["agent_id"]
        self.name = self.get_name(agent_profile)
        self.background = self.get_background(agent_profile)
        self.secret = agent_profile["secret"]
        self.personality = agent_profile["personality_and_values"]
        self.goal = ""
        
    def get_name(self, agent_profile):
        return agent_profile["first_name"] + " " + agent_profile["last_name"]
    
    def get_background(self, agent_profile):
        name = self.name
        return f"{name} is a {agent_profile['age']}-year-old {agent_profile['gender'].lower()} {agent_profile['occupation']}. {agent_profile['public_info']}"
    
class Environment:
    
    def __init__(self, env_profile):
        self._id = env_profile["env_id"]
        
        self.environment_profile = env_profile
        self.codename = env_profile["codename"]
        self.scenario = env_profile["scenario"]
        self.agent_goals = env_profile["agent_goals"]
        self.relationship = env_profile["relationship"]
        
def get_format_guide():
    return """ Your available action types are
    "none action speak non-verbal communication leave".
    Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

    Please only generate a JSON string including the action type and the argument.
    Your action should follow the given format:
    \nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}
    the object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.
    \nHere is the output schema:\n```\n{\"description\": \"An interface for messages.\\nThere is only one required method: to_natural_language\", \"properties\": {\"action_type\": {\"title\": \"Action Type\", \"description\": \"whether to speak at this turn or choose to not do anything\", \"enum\": [\"none\", \"speak\", \"non-verbal communication\", \"action\", \"leave\"], \"type\": \"string\"}, \"argument\": {\"title\": \"Argument\", \"description\": \"the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action\", \"type\": \"string\"}}, \"required\": [\"action_type\", \"argument\"]}\n```\u001b[0m
    """

def get_starter_prompt(machine_agent, human_agent, environment):
    return f"Imagine you are {machine_agent.name}, your task is to act/speak as {machine_agent.name} would, keeping in mind {machine_agent.name}'s social goal.\nYou can find {machine_agent.name}'s background and goal in the 'Here is the context of the interaction' field.\nNote that {machine_agent.name}'s secret and goal is only visible to you.\nYou should try your best to achieve {machine_agent.name}'s goal in a way that align with their character traits.\nAdditionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).\n\nHere is the context of this interaction:\n Scenario: {environment.scenario}\nParticipants: {human_agent.name} and {machine_agent.name}\n{human_agent.name}'s background: {human_agent.background} Personality and values description: {human_agent.personality} \n{machine_agent.name}'s background: {machine_agent.background} Personality and values description: {machine_agent.personality} {machine_agent.name}'s secrets: {machine_agent.secret}\n{human_agent.name}'s goal: Unknown\n{machine_agent.name}'s goal: {environment.agent_goals[1]}\nConversation Starts:"

def get_context_prompt(machine_agent, human_agent, environment):
    return f"Here is the context of this interaction:\n Scenario: {environment.scenario}\nParticipants: {human_agent.name} and {machine_agent.name}\n{human_agent.name}'s background: {human_agent.background} Personality and values description: {human_agent.personality} \n{machine_agent.name}'s background: {machine_agent.background} Personality and values description: {machine_agent.personality} {machine_agent.name}'s secrets: {machine_agent.secret}\n{human_agent.name}'s goal: Unknown\n{machine_agent.name}'s goal: {environment.agent_goals[1]}\nConversation Starts:"


# we define history as
# [(user_message, bot_message), (user_message, bot_message)]

# we define dialogue history as
# user_name: user_message\nbot_name: bot_message\nuser_name: user_message\nbot_name: bot_message\n


def dialogue_history_length_check(string, max_token, tokenizer):
    prompt_tokens = len(tokenizer(string)["input_ids"])
    return max(prompt_tokens - max_token, 0)


def truncate_dialogue_history_to_length(dia_his, surpass_num, tokenizer):
    dia_sen = dia_his.split("\n")
    remove_len = 0
    i = 0
    while remove_len < surpass_num:
        remove_len += len(tokenizer(dia_sen[i])["input_ids"])
        i += 1
    trunc_dia = "\n".join(p for p in dia_sen[i:])
    return trunc_dia


def format_bot_message(bot_message) -> str:
    # # import pdb; pdb.set_trace()
    start_idx, end_idx = bot_message.index("{"), bot_message.index("}")
    if end_idx == -1:
        bot_message += "'}"
        end_idx = len(bot_message)
    json_response = ast.literal_eval(bot_message[start_idx:end_idx+1])
    match json_response["action_type"]:
        case "none":
            return 'did nothing'
        case "speak":
            return json_response["argument"]
        case "non-verbal communication":
            return f'[{json_response["action_type"]}] {json_response["argument"]}'
        case "action":
            return f'[{json_response["action_type"]}] {json_response["argument"]}'
        case "leave":
            return 'left the conversation'
    
def dialogue_history_creation(history, user_name, bot_name):
    dialogue_history = ""
    for idx, turn in enumerate(history):
        user_message, bot_message = turn
        # TODOTODO (haofeiyu): we first assume that human talks first
        user_turn_idx = idx * 2
        bot_turn_idx = idx * 2 + 1
        if not bot_message.startswith("["): # if action type == speak, need to add 'said: ' to be consistent with the dialog prompt
            bot_message = "said :" + bot_message
        dialogue_history = f"{dialogue_history}\n\nTurn #{user_turn_idx}: {user_name}: {user_message}\n\nTurn #{bot_turn_idx}: {bot_name}: {bot_message}"
    last_turn_idx = len(history) * 2
    return dialogue_history, last_turn_idx

def dialogue_history_prompt(message, history, user_agent, bot_agent):
    dialogue_history = ""
    for idx, turn in enumerate(history):
        user_message, bot_message = turn
        # TODOTODO (haofeiyu): we first assume that human talks first
        user_turn_idx = idx * 2
        bot_turn_idx = idx * 2 + 1
        if not bot_message.startswith("["): # if action type == speak, need to add 'said: ' to be consistent with the dialog prompt
            bot_message = "said :" + bot_message
        dialogue_history = f"{dialogue_history}\n\nTurn #{user_turn_idx}: {user_agent.name}: {user_message}\n\nTurn #{bot_turn_idx}: {bot_agent.name}: {bot_message}"
    last_turn_idx = len(history) * 2
    dialogue_history = f"{dialogue_history}\n\nTurn #{last_turn_idx+1}: {user_agent.name}: {message}\n."
    return dialogue_history, last_turn_idx+2


def dialogue_history_truncation(dialogue_history, max_token_num, tokenizer):
    surpass_num = dialogue_history_length_check(
        dialogue_history, max_token_num, tokenizer
    )
    if surpass_num > 0:
        dialogue_history = truncate_dialogue_history_to_length(
            dialogue_history, surpass_num, tokenizer
        )
    return dialogue_history


def format_hostory_prompt(
    message: str,
    history: List[Tuple[str, str]],
    instructions: str,
    user_name: str,
    bot_name: str,
) -> str:
    prompt = instructions.strip()
    dialogue_history, last_turn_idx = dialogue_history_creation(
        history, user_name, bot_name
    )
    prompt = f"{prompt}\n{dialogue_history}"
    prompt = f"{prompt}\n\nTurn #{last_turn_idx+1}: {user_name}: {message}\n.\nYou are at Turn #{last_turn_idx+2}."
    return prompt
