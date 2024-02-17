# Databricks notebook source
# MAGIC %md
# MAGIC ### Experiment with LLM Agents by creating an AI Debate Environment

# COMMAND ----------

import openai 

import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

openai.api_key = config['az_oai']['api']
openai.api_base = f"https://{config['az_oai']['endpoint']}.openai.azure.com"
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 

deployment_name = config['az_oai']['deployment']

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



# COMMAND ----------



# COMMAND ----------


