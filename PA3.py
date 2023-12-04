from mdp import *

# Function to perform Monte Carlo simulation on the given states.
def monte_carlo(states):
    # List to store average reward per episode.
    average_reward_per_episode = []

    # Looping over 50 episodes.
    for episode in range(1, 51):
        # Performing a rollout and obtaining the path and total reward.
        path, total_reward = rollout(states)
        
        # Backpropagating the total reward through the path.
        backpropagate(path, total_reward)
        
        # Calculating and storing the average reward for the current episode.
        average_reward_per_episode.append((episode, total_reward / (len(path) + 1)))

        # Printing information for the current episode.
        print(f"Episode: {episode:<5} Total reward: {total_reward:<5} Path: ")
        for edge in path:
            print(f"Current state: {edge[0].name:<10} Action: {edge[1]:<5} Reward: {edge[2]:<5} Next state: {edge[3].name:<10} ")
        
        print("\n")
    
    # Printing the final value of each state after all episodes.
    for state in states: 
        print(f"Final value of state {state.id} {state.name}: {state.value}")
    print("\n")
    
    # Printing the average reward for each episode.
    for average_reward in average_reward_per_episode:
        print(f"Average reward for episode {average_reward[0]}: {average_reward[1]}")


# Function to perform a single rollout starting from the initial state.
def rollout(states):
    path = []  # List to store the path taken during the rollout.
    total_reward = 0

    current_state = states[0]

    # Performing the rollout until reaching the terminal state.
    while current_state.id != 10:
        action, reward, next_state = current_state.choose_random_action()
        path.append((current_state, action, reward, next_state))
        total_reward += reward

        current_state = next_state

    return path, total_reward


# Function to backpropagate the total reward through the rollout path.
def backpropagate(path, total_reward):
    for edge in path:
        state = edge[0]
        # Updating the value of each state in the path using the total reward.
        state.value = state.value + 0.1 * (total_reward - state.value)
    
    # Updating the value of the terminal state.
    terminal_state = path[-1][-1]
    terminal_state.value = terminal_state.value + 0.1 * (total_reward - terminal_state.value)

def value_iteration(states, discount_factor=0.99, epsilon=0.001, max_iterations=100):
        iteration = 0
        for iteration in range(5):
            print(f"Iteration: {iteration + 1}\n")
            iteration += 1
            max_change = 0.0

            for state in states:
                if state.id == 10:  # Terminal state
                    continue

                current_value = state.value
                print()
                print(f"State: {state.name}")
                print(f"  Current Value: {current_value}")

                updated_state_values = []

                for action, reward, next_state, probability in state.actions:
                    if isinstance(next_state, State):
                        discounted_future_value = discount_factor * next_state.value
                        print(f"  Action: {action}, Next State: {next_state.name}")
                        print(f"    Discounted Future Value: {discounted_future_value}")
                    elif isinstance(next_state, tuple):
                        discounted_future_value = discount_factor * max(s.value for s in next_state)
                        next_state_names = ", ".join(s.name for s in next_state)
                        print(f"  Action: {action}, Next States: {next_state_names}")
                        print(f"    Discounted Future Value: {discounted_future_value}")

                    # Bellman Equation
                    print(f"    Bellman Equation: {current_value} + {probability} * ({reward} + {discounted_future_value})")
                    updated_value = current_value + (probability * (reward + discounted_future_value))
                    updated_state_values.append(updated_value)
                    print(f"    Updated Value: {updated_value}\n")

                max_value = max(updated_state_values)

                value_change = abs(max_value - current_value)
                print(f"Max Value {max_value} - Current Value: {current_value}")
                print(f"  Value Change: {value_change}")
                max_change = max(max_change, value_change)
                print(f"  Max Change: {max_change}\n")

                state.value = max_value

            if max_change <= epsilon:
                print(f"Converged after {iteration} iterations.")
                break

        # Print final values of all states
        print("\nFinal Values:")
        for state in states:
            print(f"State: {state.name}, Value: {state.value}")
if __name__ == '__main__':
    # Creating an instance of the MDP class to initialize the Markov Decision Process.
    mdp = MDP()

    # Running the Monte Carlo simulation on the MDP states.
    monte_carlo(mdp.states)
    value_iteration(mdp.states)

