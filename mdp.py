import random

class State:
    def __init__(self, id, name, transitions = [], value = 0):
        self.id = id
        self.states_represented = name
        self.transitions = transitions
        self.value = value
    
    def add_transition(self, action, reward, state):
        self.transitions.append((action, reward, state))

class MDP:
    def __init__(self):
        self.states = self.build_mdp()

    def build_mdp(self):
        s0 = State(0, 'RU 8p')
        s1 = State(1, 'TU 10p')
        s2 = State(2, 'RU 10p')
        s3 = State(3, 'RD 10p')
        s4 = State(4, 'RU 8a')
        s5 = State(5, 'RD 8a')
        s6 = State(6, 'TU 10a')
        s7 = State(7, 'RU 10a')
        s8 = State(8, 'RD 10a')
        s9 = State(9, 'TD 10a')
        s10 = State(10, 'TERMINAL')

        s0.add_transition('P', 2, s1)
        s0.add_transition('R', 0, s2)
        s0.add_transition('S', -1, s3)

        s1.add_transition('P', 2, s7)
        s1.add_transition('R', 0, s4)

        s2.add_transition('R', 0, s4)
        s2.add_transition('P', 2, (s4, s7))
        s2.add_transition('S', -1, s5)

        s3.add_transition('R', 0, s5)
        s3.add_transition('P', 2, (s5, s8))

        s4.add_transition('P', 2, s6)
        s4.add_transition('R', 0, s7)
        s4.add_transition('S', -1, s8)

        s5.add_transition('R', 0, s8)
        s5.add_transition('P', 2, s9)

        s6.add_transition('any', -1, s10)
        s7.add_transition('any', 0, s10)
        s8.add_transition('any', 4, s10)
        s9.add_transition('any', 3, s10)

        return [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10]


