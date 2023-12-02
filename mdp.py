import random

# Defining the State class to represent states in the Markov Decision Process (MDP).
class State:
    def __init__(self, id, name, actions=None, value=0):
        # Initializing the state with an id, name, list of actions, and a default value.
        self.id = id
        self.name = name
        self.actions = actions if actions is not None else []  # Actions list with default value if not provided.
        self.value = value
    
    def add_action(self, action, reward, state):
        # Method to add an action with its associated reward and next state to the state's actions list.
        self.actions.append((action, reward, state))

    def choose_random_action(self):
        # Method to choose a random action from the state's actions list.
        random_action = random.choice(self.actions)
        action, reward, next_state = random_action[0], random_action[1], random_action[2]
        if isinstance(next_state, tuple):
            # If the next state is a tuple, randomly choose one of the states from the tuple.
            next_state = random.choice(next_state)

        return action, reward, next_state

# Defining the MDP class to represent the Markov Decision Process.
class MDP:
    def __init__(self):
        # Initializing the MDP with states by calling the build_mdp method.
        self.states = self.build_mdp()

    def build_mdp(self):
        # Method to construct and return a list of states representing the MDP.
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

        s0.add_action('P', 2, s1)
        s0.add_action('R', 0, s2)
        s0.add_action('S', -1, s3)

        s1.add_action('P', 2, s7)
        s1.add_action('R', 0, s4)

        s2.add_action('R', 0, s4)
        s2.add_action('P', 2, (s4, s7))
        s2.add_action('S', -1, s5)

        s3.add_action('R', 0, s5)
        s3.add_action('P', 2, (s5, s8))

        s4.add_action('P', 2, s6)
        s4.add_action('R', 0, s7)
        s4.add_action('S', -1, s8)

        s5.add_action('R', 0, s8)
        s5.add_action('P', 2, s9)

        s6.add_action('any', -1, s10)
        s7.add_action('any', 0, s10)
        s8.add_action('any', 4, s10)
        s9.add_action('any', 3, s10)

        return [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
