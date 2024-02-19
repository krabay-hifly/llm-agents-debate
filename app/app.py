# Databricks notebook source
# MAGIC %md
# MAGIC ### App calling all utils and funcs to simulate LLM Agent's Debates

# COMMAND ----------

import openai 
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
from collections import Counter

import yaml
with open('../config.yml', 'r') as file:
    config = yaml.safe_load(file)

openai.api_key = config['az_oai']['api']
openai.api_base = f"https://{config['az_oai']['endpoint']}.openai.azure.com"
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 

deployment_name = config['az_oai']['deployment']

from utils.funcs import num_tokens_from_messages, generate_response

# COMMAND ----------

prompt = 'Write a tagline for an ice cream shop.'
messages = [{'role' : 'user', 'content' : prompt}]
response, usage = generate_response(messages, deployment_name = deployment_name)

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

from utils.prompts import master_prompt_system_message, master_prompt_instruction, master_prompt_instruction_second_debater, master_prompt_instruction_final_evalation
from utils.prompts import moderator_system_message, moderator_prompt_instruction, moderator_talking_point_eval_instruction
from utils.prompts import debater_1_system_message, debater_1_prompt_instruction, debater_2_system_message, debater_2_prompt_instruction

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
    
    def empty_memory_for_next_talking_point(self):
        self.messages = self.messages[:1]

    def empty_memory_for_moderator_for_next_talking_point_summary(self):
        self.messages = self.messages[:3]
    
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

class Debate:

    def __init__(self, topic, n_talking_points = 2, n_rounds = 1) -> None:

        self.topic = topic
        self.n_talking_points = n_talking_points
        self.n_rounds = n_rounds

        self.create_players()
        self.set_system_prompts()
        self.assign_debaters()
        self.set_talking_points()

        self.moderator_talking_points_list = [i.strip() for i in self.moderator_talking_points.split(';')]

    def create_players(self):

        self.MASTER = Agent('master')
        self.MODERATOR = Agent('moderator')
        self.DEBATER_1 = Agent('debater_1')
        self.DEBATER_2 = Agent('debater_2')

    def set_system_prompts(self):
        self.MASTER.set_system_prompt(master_prompt_system_message)
        self.MODERATOR.set_system_prompt(moderator_system_message)
        self.DEBATER_1.set_system_prompt(debater_1_system_message)
        self.DEBATER_2.set_system_prompt(debater_2_system_message)

    def assign_debaters(self):

        self.MASTER.add_message_to_memory(role='user', message=master_prompt_instruction.format(topic = self.topic))
        self.debater_1_instruction = MASTER.ask()
        self.MASTER.add_message_to_memory(role='assistant', message=debater_1_instruction)
        self.MASTER.add_message_to_memory(role='user', message=master_prompt_instruction_second_debater)
        self.debater_2_instruction = MASTER.ask()
        self.MASTER.add_message_to_memory(role='assistant', message=debater_2_instruction)

    def set_talking_points(self):

        self.MODERATOR.add_message_to_memory(role='user', message=moderator_prompt_instruction.format(topic = self.topic, 
                                                                                        debater_1_instruction = self.debater_1_instruction,
                                                                                        debater_2_instruction = self.debater_2_instruction,
                                                                                        n_talking_points = self.n_talking_points, 
                                                                                        n_rounds = self.n_rounds))
        self.moderator_talking_points = MODERATOR.ask()
        self.MODERATOR.add_message_to_memory(role='assistant', message=moderator_talking_points)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

topic = "Which one is the better social media platform? Facebook or Instagram?"
#topic = "Exiting from investment banking (e.g.: working at Morgan Stanley) after 1.5 years due to low salary."
#topic = "Today while playing basketball the ball hit my ring finger and now it's swollen a little bit. I can move it, eat with it, although it's not as strong as normally. It does not show discolorment. Moving it generally hurts, limitation is medium."
#topic = "My girlfriend wants a constant 24 degrees Celsius temperature at home. I myself am comfortable with 21 degrees."
#topic = "Two people in a relationship argue about setting morning alarms. One of them wants to set only one and wake up immediately while the other one prefers setting multiple alarms, like every 5-10 minutes to avoid falling back asleep."
#topic = "Two people are in a relationship. They start their summer vacation by arriving to Lisbon after a 3-hour flight. One of them suggests quickly dropping off their luggage and heading out to explore the city right away. The other one prefers staying at the hotel a little bit to take a shower, freshen up, change clothes and only after that leave to enjoy their first day."

n_talking_points = 2
n_rounds = 1

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
MODERATOR.add_message_to_memory(role='assistant', message=moderator_talking_points)

# COMMAND ----------

moderator_talking_points_list = moderator_talking_points.split(';')
print(moderator_talking_points_list)

# COMMAND ----------

summaries = []
debates_for_each_talking_point = []

for current_talking_point in moderator_talking_points_list:

    print('*' * 100)
    print(f'Current talking point: {current_talking_point}')

    DEBATER_1.add_message_to_memory(role='user', message=debater_1_prompt_instruction.format(topic = topic, 
                                                                                            debater_1_instruction = debater_1_instruction,
                                                                                            n_rounds = n_rounds, 
                                                                                            current_talking_point = current_talking_point))

    DEBATER_2.add_message_to_memory(role='user', message=debater_2_prompt_instruction.format(topic = topic, 
                                                                                            debater_2_instruction = debater_2_instruction,
                                                                                            n_rounds = n_rounds, 
                                                                                            current_talking_point = current_talking_point))

    Debate_Talking_Point_History_For_Moderator = ""
        
    for n in range(n_rounds):

        print('\n')
        print(f'===== Round {n+1} =====')
        Debate_Talking_Point_History_For_Moderator +=  "\n" + f'===== Round {n+1} =====' + "\n"
        print('\n')

        # ask debater 1
        print(f'Debater #1')
        print('\n')
        debater_1_response = DEBATER_1.ask()
        Debate_Talking_Point_History_For_Moderator += "\n\n" + "Debater #1:\n" + debater_1_response
        print('\n')

        # add debater 1's response to both debater's memories
        DEBATER_1.add_message_to_memory(role='user', message= "Here's the answer you gave: " + debater_1_response)
        DEBATER_2.add_message_to_memory(role='user', message= "Here's what your opponent stated: " + debater_1_response + "\n Now it's your turn, remember what you are arguing for and against!\n")

        # ask debater 2
        print(f'Debater #2')
        print('\n')
        debater_2_response = DEBATER_2.ask()
        Debate_Talking_Point_History_For_Moderator += "\n\n" + "Debater #2:\n" + debater_2_response
        print('\n')

        # add debater 2's response to both debater's memories

        DEBATER_1.add_message_to_memory(role='user', message= "Here's what your opponent stated: " + debater_2_response + "\n Now it's your turn, remember what you are arguing for and against!\n")
        DEBATER_2.add_message_to_memory(role='user', message= "Here's the answer you gave: " + debater_2_response)

    # empty debater's memory before next talking point
    DEBATER_1.empty_memory_for_next_talking_point()
    DEBATER_2.empty_memory_for_next_talking_point()

    # have moderator pick a winner for the current talking point
    print(f'===== Moderator Evaluation for talking point: {current_talking_point} =====')
    MODERATOR.add_message_to_memory(role='user', message=moderator_talking_point_eval_instruction.format(current_talking_point=current_talking_point, 
                                                                                                        transcript = Debate_Talking_Point_History_For_Moderator))
    moderator_eval_talking_point = MODERATOR.ask()
    print('\n')
    MODERATOR.add_message_to_memory(role='assistant', message=moderator_eval_talking_point)

    # empty moderators's memory before next talking point
    MODERATOR.empty_memory_for_moderator_for_next_talking_point_summary()

    # append memory to global lists
    summaries.append(f'Summary for {current_talking_point}: \n{moderator_eval_talking_point} \n\n')
    debates_for_each_talking_point.append(f'Topic: {topic} \nCurrent talking point: {current_talking_point} \n\nDebate:\n{Debate_Talking_Point_History_For_Moderator} \n\n')

    print('\n\n')

# now it's the master's turn to take all of the moderator's notes, summarize what had happened and pick a final champion
MASTER.add_message_to_memory(role='user', message=master_prompt_instruction_final_evalation.format(talking_points = moderator_talking_points,
                                                                                                   moderator_notes = '\n'.join(summaries)))

print('\n')
print('Debate has now finished')
print('\n')
print(f"===== Master's Final Debate Champion Selection =====")
master_final_champion_selection = MASTER.ask()
MASTER.add_message_to_memory(role='assistant', message=master_final_champion_selection)


# COMMAND ----------



# COMMAND ----------

c = Counter()
for d in [MASTER.get_cost_usage(), MODERATOR.get_cost_usage(), DEBATER_1.get_cost_usage(), DEBATER_2.get_cost_usage()]:
    c.update(d)
total_cost_dict = dict(c)
total_cost_dict

# COMMAND ----------


