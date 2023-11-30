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
    
    def get_reward(self, state, action, next_state):
        return self.table[state][action][next_state][1]

        
        
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
    mdp_table.set_transition_probability_and_reward("TU10p", "P", "RU10a", 2, .5)
    mdp_table.set_transition_probability_and_reward("TU10p", "R", "RU8a", 0, .5)
    mdp_table.set_transition_probability_and_reward("RU10p", "P", "RU8a", 2, .5)
    mdp_table.set_transition_probability_and_reward("RU10p", "R", "RU8a", 0, .5)




