# Databricks notebook source
# MAGIC %md
# MAGIC ### Experiment with LLM Agents by creating an AI Debate Environment

# COMMAND ----------

import openai 
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
from collections import Counter

import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

openai.api_key = config['az_oai']['api']
openai.api_base = f"https://{config['az_oai']['endpoint']}.openai.azure.com"
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 

deployment_name = config['az_oai']['deployment']

# COMMAND ----------

def num_tokens_from_messages(messages, model="gpt-4-turbo"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-turbo"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# COMMAND ----------

def generate_response(messages, deployment_name = deployment_name, temperature = 0.0):

    completion = openai.ChatCompletion.create(
        engine=deployment_name, 
        messages=messages, 
        temperature=temperature)
    
    response = completion.choices[0]['message']['content']
    usage = completion.usage.to_dict()
    return response, usage
    
prompt = 'Write a tagline for an ice cream shop.'
messages = [{'role' : 'user', 'content' : prompt}]
response, usage = generate_response(messages)

print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set Prompts
# MAGIC
# MAGIC The following players will be included
# MAGIC 1. Master: Responsible for assigning 2 debators with their arguments. By the end, Master will be given Moderator's assessments and pick a final winner.
# MAGIC 2. Moderator: Based on topic sets talking points, for each talking point facilitates an {n} round discussion. [Later purpose: ask followup questions]. After each talking point is finished, Moderator will make an objective judgement of which debater was more convincing for that part of the debate.
# MAGIC 3. Debaters: Given their arguments and a topic, these two AI agents will exchange ideas, argue with one another.

# COMMAND ----------

master_prompt_system_message = "You are part of an AI simulation. In this world different AIs debate one another. You are the master of this world."

master_prompt_instruction = """
Your tasks are the following:

1. Given a debate topic you will assign 2 debaters with their sides, what they need to argue for. These are not necessary just affirmative or negative. There may be cases where one of them needs to argue for or compare certain products, services, TV-shows, books, political ideas, theories, etc... In these cases you will assign each debater with one of the two options.

2. After the debate is finished, you'll be given a complete transcript of the discussion. Your task will be to summarize what each debater talked about, what their arguments were, overall how you evaluate the debate. Then, you will pick a winner based on whose arguments were more objective, thorough, factual and convincing.

Here's the topic of today's debate: {topic}

First, describe what debater #1 will be arguing for. You're directly talking to debater #1, describe to them the topic and the side they will be taking. Do not help him by listing talking points or pro-contra arguments, simply articulate his task.

Debater #1's side:
"""

master_prompt_instruction_second_debater = """

Secondly, describe what debater #2 will be arguing for. Make sure debater #1 and debater #2 are arguing for the opposite sides. You're directly talking to debater #2, describe to them the topic and the side they will be taking. Do not help him by listing talking points or pro-contra arguments, simply articulate his task.

Debater #2's side:
"""

# COMMAND ----------

moderator_system_message = "You are part of an AI simulation. In this world different AIs debate one another. You are the moderator of the debates."

moderator_prompt_instruction = """
Here's the topic of the debate: {topic}

Here's the instruction debater #1 received: {debater_1_instruction}
And here's what debater #2 received as instruction: {debater_2_instruction}

You are an expert in this field. You're given two tasks:

1. First, using your deep understanding of the topics, come up with {n_talking_points} talking points or aspects the debaters should cover. These aspects will serve as the agenda, the different angles the debaters should be looking at when making their arguments.

2. Secondy, you will allow the debaters to discuss each aspect for {n_rounds} rounds. Once they reach the final round, you'll make a judgement of which debater presented more convincing arguments and pick a winner. You'll shortly summarize their reasoning to help explain your decision when picking a winner. Then you'll move on to the next aspect.

So, given the topic, come up with the {n_talking_points} talking points / aspects. Only return with the aspects, separated by a semicolon.

Debate talking points:
"""

# COMMAND ----------

debater_1_system_message = "You are part of an AI simulation. In this world different AIs debate one another. You are debater #1."

debater_1_prompt_instruction = """
Here's the topic of the debate: {topic}

Here's your instruction: {debater_1_instruction}

The current aspect of this topic that you are arguing for is {current_talking_point}.
You'll debate about this aspect for {n_rounds} rounds, meaning {n_rounds} back and forths with your opponent.

You do not need to address the audience or the moderator over and over. Focus on being concise, as too long answers will lose the attention of the audience and the moderator. Make your arguments short and to-the-point. You can confront your opponent by asking them challenging questions. If you're asked a question by your opponent, try not to dodge it. Remember, this is a conversation, not a speech.

As debater you'll be starting the discussion.
"""

# COMMAND ----------

debater_2_system_message = "You are part of an AI simulation. In this world different AIs debate one another. You are debater #2."

debater_2_prompt_instruction = """
Here's the topic of the debate: {topic}

Here's your instruction: {debater_2_instruction}

The current aspect of this topic that you are arguing for is {current_talking_point}.
You'll debate about this aspect for {n_rounds} rounds, meaning {n_rounds} back and forths with your opponent. 

You do not need to address the audience or the moderator over and over. Focus on being concise, as too long answers will lose the attention of the audience and the moderator. Make your arguments short and to-the-point. You can confront your opponent by asking them challenging questions. If you're asked a question by your opponent, try not to dodge it. Remember, this is a conversation, not a speech.

Debater #1 had started the discussion, as debater #2, you are to respond and make your own arguments.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set Agent and Debate classes

# COMMAND ----------

class Agent:

    def __init__(self, name: str) -> None:

        self.name = name
        self.messages = []
        self.usages_raw = []

        self.INPUT_COST = 0.01 / 1000
        self.OUTPUT_COST = 0.028 / 1000

    def generate_response(self, messages: "list[dict]", deployment_name = deployment_name, temperature = 0.0):

        completion = openai.ChatCompletion.create(
            engine=deployment_name, 
            messages=messages, 
            temperature=temperature)
        
        response = completion.choices[0]['message']['content']
        usage = completion.usage.to_dict()
        self.usages_raw.append(usage)

        return response
    
    def generate_response_with_streaming(self, messages: "list[dict]", deployment_name = deployment_name, temperature = 0.0):

        input_tokens = num_tokens_from_messages(messages)
        output_tokens = 0
        final_answer = []

        completion = openai.ChatCompletion.create(
            engine=deployment_name, 
            messages=messages, 
            temperature=0.0,
            stream = True)

        for i in completion:

            output = i['choices'][0]['delta']   

            if output not in [ {'role' : 'assistant'}, {}]:

                output_tokens += 1

                token = output['content']
                print(f"{token}", end="")

                final_answer.append(token)

            else:
                continue

        usage = {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens, 'total_tokens': output_tokens + input_tokens}
        self.usages_raw.append(usage)

        final_answer_print = ''.join(final_answer)
        return final_answer_print

    def set_system_prompt(self, system_prompt: str):
        self.messages.append({"role": "system", "content": system_prompt})

    def add_message_to_memory(self, role:str, message: str):
        self.messages.append({"role": role, "content": message})
        #print(f"----- {self.name} -----\n{memory}\n")

    def ask(self):
        #return self.generate_response(self.messages)
        return self.generate_response_with_streaming(self.messages)
    
    def get_token_usage(self):
        c = Counter()
        for d in self.usages_raw:
            c.update(d)
        return dict(c)
    
    def get_cost_usage(self):

        token_usage = self.get_token_usage()
        input_cost = token_usage['prompt_tokens'] * self.INPUT_COST
        output_cost = token_usage['completion_tokens'] * self.OUTPUT_COST

        return {'input_cost_€': input_cost, 'output_cost_€': output_cost, 'total_cost_€': output_cost + input_cost}

# COMMAND ----------

topic = "Which one is the better social media platform? Facebook or Instagram?"
n_talking_points = 2
n_rounds = 2

MASTER = Agent('master')
MASTER.set_system_prompt(master_prompt_system_message)

MODERATOR = Agent('moderator')
MODERATOR.set_system_prompt(moderator_system_message)

DEBATER_1 = Agent('debater_1')
DEBATER_1.set_system_prompt(debater_1_system_message)

DEBATER_2 = Agent('debater_2')
DEBATER_2.set_system_prompt(debater_2_system_message)

# COMMAND ----------

MASTER.add_message_to_memory(role='user', message=master_prompt_instruction.format(topic = topic))

# COMMAND ----------

debater_1_instruction = MASTER.ask()

# COMMAND ----------

MASTER.add_message_to_memory(role='assistant', message=debater_1_instruction)
MASTER.add_message_to_memory(role='user', message=master_prompt_instruction_second_debater)

# COMMAND ----------

debater_2_instruction = MASTER.ask()
MASTER.add_message_to_memory(role='assistant', message=debater_2_instruction)

# COMMAND ----------

MODERATOR.add_message_to_memory(role='user', message=moderator_prompt_instruction.format(topic = topic, 
                                                                                        debater_1_instruction = debater_1_instruction,
                                                                                        debater_2_instruction = debater_2_instruction,
                                                                                        n_talking_points = n_talking_points, 
                                                                                        n_rounds = n_rounds))

# COMMAND ----------

moderator_talking_points = MODERATOR.ask()

# COMMAND ----------

moderator_talking_points_list = moderator_talking_points.split(';')
current_talking_point = moderator_talking_points_list[0]
current_talking_point

# COMMAND ----------

print(f'Current talking point: {current_talking_point}')

DEBATER_1.add_message_to_memory(role='user', message=debater_1_prompt_instruction.format(topic = topic, 
                                                                                        debater_1_instruction = debater_1_instruction,
                                                                                        n_rounds = n_rounds, 
                                                                                        current_talking_point = current_talking_point))

DEBATER_2.add_message_to_memory(role='user', message=debater_2_prompt_instruction.format(topic = topic, 
                                                                                        debater_2_instruction = debater_2_instruction,
                                                                                        n_rounds = n_rounds, 
                                                                                        current_talking_point = current_talking_point))
      
for n in range(n_rounds):

    print('\n')
    print(f'===== Round {n+1} =====')
    print('\n')

    # ask debater 1
    print(f'Debater #1')
    print('\n')
    debater_1_response = DEBATER_1.ask()
    print('\n')

    # add debater 1's response to both debater's memories
    DEBATER_1.add_message_to_memory(role='assistant', message=debater_1_response)
    DEBATER_2.add_message_to_memory(role='user', message=debater_1_response)

    # ask debater 2
    print(f'Debater #2')
    print('\n')
    debater_2_response = DEBATER_2.ask()
    print('\n')

    # add debater 2's response to both debater's memories

    DEBATER_1.add_message_to_memory(role='user', message=debater_2_response)
    DEBATER_2.add_message_to_memory(role='assistant', message=debater_2_response)

# COMMAND ----------



# COMMAND ----------


