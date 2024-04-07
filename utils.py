class Agent:
    def __init__(self, name, background, goal, secrets, personality):
        self.name = name
        self.background = background
        self.goal = goal
        self.secrets = secrets
        self.personality = personality

def get_starter_prompt(machine_agent, human_agent, scenario):
    return f"Prompt after formatting:\nImagine you are {machine_agent.name}, your task is to act/speak as {machine_agent.name} would, keeping in mind {machine_agent.name}'s social goal.\nYou can find {machine_agent.name}'s background and goal in the 'Here is the context of the interaction' field.\nNote that {machine_agent.name}'s secret and goal is only visible to you.\nYou should try your best to achieve {machine_agent.name}'s goal in a way that align with their character traits.\nAdditionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).\n\nHere is the context of this interaction:\n Scenario: {scenario}\nParticipants: {human_agent.name} and {machine_agent.name}\n{human_agent.name}'s background: {human_agent.background} Personality and values description: {human_agent.personality} \n{machine_agent.name}'s background: {machine_agent.background} Personality and values description: {machine_agent.personality} {machine_agent.name}'s secrets: {machine_agent.secrets}\n{human_agent.name}'s goal: Unknown\n{machine_agent.name}'s goal: {machine_agent.name}\nConversation Starts:"

def format_chat_prompt(
    message: str,
    chat_history,
    instructions: str,
    user_name: str,
    bot_name: str,
    include_all_chat_history: bool = True,
    index : int = 1
) -> str:
    instructions = instructions.strip()
    prompt = instructions
    if not include_all_chat_history:
        if index >= 0:
            index = -index
        chat_history = chat_history[index:]
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\n{user_name}: {user_message}\n{bot_name}: {bot_message}"
    prompt = f"{prompt}\n{user_name}: {message}\n{bot_name}:"
    return prompt