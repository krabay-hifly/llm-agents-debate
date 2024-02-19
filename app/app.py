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

topic = "Which one is the better social media platform? Facebook or Instagram?"
n_talking_points = 2
n_rounds = 2

debate = Debate(topic = topic, n_talking_points = n_talking_points, n_rounds = n_rounds)

# COMMAND ----------

debate.debate()

# COMMAND ----------

debate.total_costs()

# COMMAND ----------

debate.total_tokens()
