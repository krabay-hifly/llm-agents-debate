# Databricks notebook source
# MAGIC %md
# MAGIC ### App calling all utils and funcs to simulate LLM Agent's Debates
# MAGIC
# MAGIC The following players will be included
# MAGIC 1. Master: Responsible for assigning 2 debators with their arguments. By the end, Master will be given Moderator's assessments and pick a final winner.
# MAGIC 2. Moderator: Based on topic sets talking points, for each talking point facilitates an {n} round discussion. [Later purpose: ask followup questions]. After each talking point is finished, Moderator will make an objective judgement of which debater was more convincing for that part of the debate.
# MAGIC 3. Debaters: Given their arguments and a topic, these two AI agents will exchange ideas, argue with one another.

# COMMAND ----------

from utils.debate import Debate

# COMMAND ----------

#topic = "Which one is the better social media platform? Facebook or Instagram?"
topic = "An AI company is building a Retrieval Augmented Generation pipeline. Two data scientists have an argument: one wants to use a simple vector database to store chunks and run relevany search, while the other one prefers to implement a graph database, but doesn't fully understand how that would help retrieval. Please have one debater argue for simple vectorDB and no graph elements, while the other should reason for graph DB, but should also give practical tips on how to build it."

n_talking_points = 3
n_rounds = 2

debate = Debate(topic = topic, n_talking_points = n_talking_points, n_rounds = n_rounds)

# COMMAND ----------

debate.debate()

# COMMAND ----------

debate.total_tokens()

# COMMAND ----------

debate.total_costs()
