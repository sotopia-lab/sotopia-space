from typing import List, Tuple
import ast
import re
from typing import List, Tuple
import ast

FORMAT_TEMPLATE = """ Your available action types are
"none action speak non-verbal communication leave".
Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
\nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}
the object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.
\nHere is the output schema:\n```\n{\"description\": \"An interface for messages.\\nThere is only one required method: to_natural_language\", \"properties\": {\"action_type\": {\"title\": \"Action Type\", \"description\": \"whether to speak at this turn or choose to not do anything\", \"enum\": [\"none\", \"speak\", \"non-verbal communication\", \"action\", \"leave\"], \"type\": \"string\"}, \"argument\": {\"title\": \"Argument\", \"description\": \"the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action\", \"type\": \"string\"}}, \"required\": [\"action_type\", \"argument\"]}\n```\u001b[0m
"""

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
        
        
def get_context_prompt(machine_agent, human_agent, environment):
    return f"Here is the context of this interaction:\n Scenario: {environment.scenario}\nParticipants: {human_agent.name} and {machine_agent.name}\n{human_agent.name}'s background: {human_agent.background} Personality and values description: {human_agent.personality} \n{machine_agent.name}'s background: {machine_agent.background} Personality and values description: {machine_agent.personality} {machine_agent.name}'s secrets: {machine_agent.secret}\n{human_agent.name}'s goal: Unknown\n{machine_agent.name}'s goal: {environment.agent_goals[1]}\nConversation Starts:"
        
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
    return dialogue_history, last_turn_idx + 2

def format_docstring(docstring: str) -> str:
    """Format a docstring for use in a prompt template."""
    return re.sub("\n +", "\n", docstring).strip()
