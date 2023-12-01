import numpy as np

class MDPLookUpTable:
    def __init__(self):
        self.states = ["RU8p", "RU10p","RU8a", "TU10p","TU10a", "RD10p","RD10a","RD8a", "TD10a","11a"]
        self.actions = ["P", "R", "S", "any"]
        self.table = {state: {action: {} for action in self.actions} for state in self.states}

    def set_transition_probability_and_reward(self, state, action, next_state, probability, reward):
        self.table[state][action][next_state] = (probability, reward)

    def get_probability(self, state, action, next_state):
        return self.table[state][action][next_state][0]
    #reward function
    def get_reward(self, state, action, next_state):
        return self.table[state][action][next_state][1]

    def value_iteration(self, discount_factor=0.99, epsilon=0.001):
        while True:
            max_change = 0.0
            for state in self.states:
                current_value = self.table[state]["any"]["11a"][1]  # Initialize with the terminal state reward
                for action in self.actions:
                    for next_state in self.states:
                        probability = self.get_probability(state, action, next_state)
                        reward = self.get_reward(state, action, next_state)
                        discounted_future_value = discount_factor * self.table[next_state]["any"]["11a"][1]
                        current_value += probability * (reward + discounted_future_value)
                value_change = abs(self.table[state]["any"]["11a"][1] - current_value)
                max_change = max(max_change, value_change)
                self.table[state]["any"]["11a"] = (1.0, current_value)  # Update the value
            if max_change < epsilon:
                break
        
class state:
    def __init__(self,value,actions) -> None:
        self.value = 0
        self.reward = 0
        self.probability

   

    #This is for bellman equation
    def valueIteration(self):
        pass

if __name__ == "__main__":
    mdp_table = MDPLookUpTable()
    mdp_table.set_transition_probability_and_reward("RU8p", "P", "TU10p", 2, .33)
    mdp_table.set_transition_probability_and_reward("RU8p", "R", "RU10p", 0, .33)
    mdp_table.set_transition_probability_and_reward("RU8p", "S", "RD10p", -1, .33)
    mdp_table.set_transition_probability_and_reward("RD10p", "P", "RD8a", 2, .5)
    mdp_table.set_transition_probability_and_reward("RD10p", "P", "RD10a", 2, .5)
    mdp_table.set_transition_probability_and_reward("RD10p", "R", "RD8a", 0, .5)
    mdp_table.set_transition_probability_and_reward("TU10p", "P", "RU10a", 2, .5)
    mdp_table.set_transition_probability_and_reward("TU10p", "R", "RU8a", 0, .5)
    mdp_table.set_transition_probability_and_reward("RU10p", "P", "RU8a", 2, .5)
    mdp_table.set_transition_probability_and_reward("RU10p", "R", "RU8a", 0, .33)
    mdp_table.set_transition_probability_and_reward("RU10p", "P", "RU10a", 2, .5)
    mdp_table.set_transition_probability_and_reward("RU10p", "S", "RD8a", -1, .33)
    mdp_table.set_transition_probability_and_reward("RU8a", "P", "TU10a", 2, .33)
    mdp_table.set_transition_probability_and_reward("RU8a", "R", "RU10a", 0, .33)
    mdp_table.set_transition_probability_and_reward("RU8a", "S", "RD10a", -1, .33)
    mdp_table.set_transition_probability_and_reward("RD8a", "P", "TD10a", 2, .5)
    mdp_table.set_transition_probability_and_reward("RD8a", "R", "RD10a", 0, .5)
    mdp_table.set_transition_probability_and_reward("TU10a", "any","11a", -1, 1.0)
    mdp_table.set_transition_probability_and_reward("RU10a", "any","11a", 0, 1.0)
    mdp_table.set_transition_probability_and_reward("RD10a", "any","11a",4,1.0)
    mdp_table.set_transition_probability_and_reward("TD10a", "any","11a", 3, 1.0)










