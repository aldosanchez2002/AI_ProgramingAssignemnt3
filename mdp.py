import random

class State:
    def __init__(self, id, name, transitions = [], value = 0):
        self.id = id
        self.states_represented = name
        self.transitions = transitions
        self.value = value
    
    def add_transition(self, action, reward, state, probability):
        self.transitions.append((action, reward, state, probability))

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

        s0.add_transition('P', 2, s1, .33)
        s0.add_transition('R', 0, s2, .33)
        s0.add_transition('S', -1, s3, .33)

        s1.add_transition('P', 2, s7, .5)
        s1.add_transition('R', 0, s4, .5)

        s2.add_transition('R', 0, s4, .33)
        s2.add_transition('P', 2, (s4, s7), .5)
        s2.add_transition('S', -1, s5, .33)

        s3.add_transition('R', 0, s5, .5)
        s3.add_transition('P', 2, (s5, s8), .5)

        s4.add_transition('P', 2, s6, .33)
        s4.add_transition('R', 0, s7, .33)
        s4.add_transition('S', -1, s8, .33)

        s5.add_transition('R', 0, s8, .5)
        s5.add_transition('P', 2, s9, .5)

        s6.add_transition('any', -1, s10, 1)
        s7.add_transition('any', 0, s10, 1)
        s8.add_transition('any', 4, s10, 1)
        s9.add_transition('any', 3, s10, 1)

        return [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10]

    def value_iteration(self, discount_factor=0.99, epsilon=0.001, max_iterations=100):
        for iteration in range(max_iterations):
            max_change = 0.0
            print("iteration: ", iteration)

            for state in self.states:
                if state.id == 10:  # Terminal state
                    continue

                current_value = state.value

                for action, reward, next_state, probability in state.transitions:
                    if isinstance(next_state, State):
                        # Handle the case when next_state is a single state
                        discounted_future_value = discount_factor * next_state.value
                    elif isinstance(next_state, tuple):
                        # Handle the case when next_state is a tuple of states
                        discounted_future_value = discount_factor * max(s.value for s in next_state)
                    
                    current_value += probability * (reward + discounted_future_value)

                value_change = abs(state.value - current_value)
                max_change = max(max_change, value_change)

                state.value = current_value

            if max_change < epsilon:
                print(f"Converged after {iteration + 1} iterations.")
                break

if __name__ == "__main__":
    mdp = MDP()
    mdp.value_iteration()


