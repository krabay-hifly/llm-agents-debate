import openai 
from collections import Counter
from utils.funcs import num_tokens_from_messages

import yaml
with open('../config.yml', 'r') as file:
    config = yaml.safe_load(file)

openai.api_key = config['az_oai']['api']
openai.api_base = f"https://{config['az_oai']['endpoint']}.openai.azure.com"
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 

deployment_name = config['az_oai']['deployment']

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