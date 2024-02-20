from utils.agent import Agent

from utils.prompts import master_prompt_system_message, master_prompt_instruction, master_prompt_instruction_second_debater, master_prompt_instruction_final_evalation
from utils.prompts import moderator_system_message, moderator_prompt_instruction, moderator_talking_point_eval_instruction
from utils.prompts import debater_1_system_message, debater_1_prompt_instruction, debater_2_system_message, debater_2_prompt_instruction

from collections import Counter

#sample: https://github.com/Skytliang/Multi-Agents-Debate/blob/main/interactive.py

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

        print('Debater #1 instructions given by Master')
        self.debater_1_instruction = self.MASTER.ask()

        self.MASTER.add_message_to_memory(role='assistant', message=self.debater_1_instruction)
        self.MASTER.add_message_to_memory(role='user', message=master_prompt_instruction_second_debater)

        print('\n\nDebater #2 instructions given by Master')
        self.debater_2_instruction = self.MASTER.ask()

        self.MASTER.add_message_to_memory(role='assistant', message=self.debater_2_instruction)

    def set_talking_points(self):

        self.MODERATOR.add_message_to_memory(role='user', message=moderator_prompt_instruction.format(topic = self.topic, 
                                                                                                      debater_1_instruction = self.debater_1_instruction,
                                                                                                      debater_2_instruction = self.debater_2_instruction,
                                                                                                      n_talking_points = self.n_talking_points, 
                                                                                                      n_rounds = self.n_rounds))
        
        print("\n\nModerator's talking points to construct debate")
        self.moderator_talking_points = self.MODERATOR.ask()
        self.MODERATOR.add_message_to_memory(role='assistant', message=self.moderator_talking_points)

    def debate(self):

        self.summaries = []
        self.debates_for_each_talking_point = []

        for current_talking_point in self.moderator_talking_points_list:

            print('*' * 100)
            print(f'Current talking point: {current_talking_point}')

            self.DEBATER_1.add_message_to_memory(role='user', message=debater_1_prompt_instruction.format(topic = self.topic, 
                                                                                                          debater_1_instruction = self.debater_1_instruction,
                                                                                                          n_rounds = self.n_rounds, 
                                                                                                          current_talking_point = current_talking_point))

            self.DEBATER_2.add_message_to_memory(role='user', message=debater_2_prompt_instruction.format(topic = self.topic, 
                                                                                                          debater_2_instruction = self.debater_2_instruction,
                                                                                                          n_rounds = self.n_rounds, 
                                                                                                          current_talking_point = current_talking_point))

            Debate_Talking_Point_History_For_Moderator = ""
            
            for n in range(self.n_rounds):

                print('\n')
                print(f'===== Round {n+1} =====')
                Debate_Talking_Point_History_For_Moderator +=  "\n" + f'===== Round {n+1} =====' + "\n"
                print('\n')

                # ask debater 1
                print(f'Debater #1')
                print('\n')
                debater_1_response = self.DEBATER_1.ask()
                Debate_Talking_Point_History_For_Moderator += "\n\n" + "Debater #1:\n" + debater_1_response
                print('\n')

                # add debater 1's response to both debater's memories
                self.DEBATER_1.add_message_to_memory(role='user', message= "Here's the answer you gave: " + debater_1_response)
                self.DEBATER_2.add_message_to_memory(role='user', message= "Here's what your opponent stated: " + debater_1_response + "\n Now it's your turn, remember what you are arguing for and against!\n")

                # ask debater 2
                print(f'Debater #2')
                print('\n')
                debater_2_response = self.DEBATER_2.ask()
                Debate_Talking_Point_History_For_Moderator += "\n\n" + "Debater #2:\n" + debater_2_response
                print('\n')

                # add debater 2's response to both debater's memories

                self.DEBATER_1.add_message_to_memory(role='user', message= "Here's what your opponent stated: " + debater_2_response + "\n Now it's your turn, remember what you are arguing for and against!\n")
                self.DEBATER_2.add_message_to_memory(role='user', message= "Here's the answer you gave: " + debater_2_response)

            # empty debater's memory before next talking point
            self.DEBATER_1.empty_memory_for_next_talking_point()
            self.DEBATER_2.empty_memory_for_next_talking_point()

            # have moderator pick a winner for the current talking point
            print(f'===== Moderator Evaluation for talking point: {current_talking_point} =====')
            self.MODERATOR.add_message_to_memory(role='user', message=moderator_talking_point_eval_instruction.format(current_talking_point=current_talking_point, 
                                                                                                                transcript = Debate_Talking_Point_History_For_Moderator))
            moderator_eval_talking_point = self.MODERATOR.ask()
            print('\n')
            self.MODERATOR.add_message_to_memory(role='assistant', message=moderator_eval_talking_point)

            # empty moderators's memory before next talking point
            self.MODERATOR.empty_memory_for_moderator_for_next_talking_point_summary()

            # append memory to global lists
            self.summaries.append(f'Summary for {current_talking_point}: \n{moderator_eval_talking_point} \n\n')
            self.debates_for_each_talking_point.append(f'Topic: {self.topic} \nCurrent talking point: {current_talking_point} \n\nDebate:\n{Debate_Talking_Point_History_For_Moderator} \n\n')

            print('\n\n')

        # now it's the master's turn to take all of the moderator's notes, summarize what had happened and pick a final champion
        self.MASTER.add_message_to_memory(role='user', message=master_prompt_instruction_final_evalation.format(talking_points = self.moderator_talking_points,
                                                                                                        moderator_notes = '\n'.join(self.summaries)))

        print('\n')
        print('Debate has now finished')
        print('\n')
        print(f"===== Master's Final Debate Champion Selection =====")
        master_final_champion_selection = self.MASTER.ask()
        self.MASTER.add_message_to_memory(role='assistant', message=master_final_champion_selection)

    def total_tokens(self):

        c = Counter()
        for d in [self.MASTER.get_token_usage(), self.MODERATOR.get_token_usage(), self.DEBATER_1.get_token_usage(), self.DEBATER_2.get_token_usage()]:
            c.update(d)
        total_token_dict = dict(c)
        return total_token_dict

    def total_costs(self):

        c = Counter()
        for d in [self.MASTER.get_cost_usage(), self.MODERATOR.get_cost_usage(), self.DEBATER_1.get_cost_usage(), self.DEBATER_2.get_cost_usage()]:
            c.update(d)
        total_cost_dict = dict(c)
        return total_cost_dict