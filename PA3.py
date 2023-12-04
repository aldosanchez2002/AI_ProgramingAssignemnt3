from mdp import *
import random

'''
Part I: Monte Carlo

Implement a program that models the MDP above. Assume that the agent follows a 
random equiprobable policy (i.e. the probability of picking a particular action while 
in a given state is equal to 1 / number of actions that can be performed from that 
state).  Run your program for 50 episodes. For each episode, have your program 
print out the agent's sequence of experience (i.e. the ordered sequence of 
states/actions/rewards that occur in the episode) as well as the sum of the rewards 
received in that episode in a readable form. 
 
Perform first-visit Monte-Carlo updates after each episode to update the values of all 
states visited during the run.  Use an alpha (learning rate) value of 0.1.  Print out the 
values of all of the states at the end of your experiment along with the average 
reward for each episode, also in a readable form.
'''
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
    
    # Printing the average reward for each episode.
    for average_reward in average_reward_per_episode:
        print(f"Average reward for episode {average_reward[0]}: {average_reward[1]}")

    # Printing the final value of each state after all episodes.
    max_value_state = None
    max_value = -float('inf')
    for state in states: 
        if state.value > max_value:
            max_value = state.value
            max_value_state = state.name
        print(f"Final value of state {state.id} {state.name}: {state.value}")
    print(f"\nMonte Carlo max: {max_value_state}")

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


'''
Part II: Value Iteration

Implement the value iteration algorithm and use it to find the optimal policy for this 
MDP.  Set all value estimates to 0 initially. Use a discount rate (lambda) of 0.99. Each 
time you update the value of a state, print out the previous value, the new value, the 
estimated value of each action, and the action selected. Continue to update each 
state until the maximum change in the value of any state in a single iteration is less 
than 0.001.  At the end, print out the number of iterations (i.e., the number of times 
you updated each state), the final values for each state, and the final optimal policy. 
'''
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
        max_value_state = None
        max_value = -float('inf')
        print("\nFinal Values:")
        for state in states:
            if state.value > max_value:
                max_value = state.value
                max_value_state = state.name
            print(f"State: {state.name}, Value: {state.value}")
        print(f"\nValue Iteration max: {max_value_state}")

'''
Part III: Q-Learning 
 
Implement Q-learning and use it to find the optimal policy for this MDP.  
Note that for this algorithm you will need the Q values, which are values for state/action pairs. 
Similar to before, you will run episodes repeatedly until the maximum change in any 
Q value is less than 0.001.  
Use an initial learning rate (alpha) of 0.2, and a discount rate (lambda) of 0.99.  
Decrease alpha after each episode by multiplying the current value of alpha by 0.995. 
Use the same random equiprobable policy as in part I throughout the learning process 
(recall that the Q-learning updates and convergence are independent of the policy being followed, 
so it should converge as long as every state/action pair continues to be selected).  
 
Each time you update a Q value, print out the previous value, the new value, the 
immediate reward, and the Q value for the next state.  
At the end, print out the number of episodes, the final Q values, and the optimal policy.  
'''

def q_learning(states, discount_factor=0.99, epsilon=0.001, max_iterations=100):
    # Initialize Q-values for each state-action pair to zero.
    q_values = {(state, action): 0.0 for state in states for action, _, _, _ in state.actions}

    alpha = 0.2  # Initial learning rate
    episodes = 0

    while True:
        episodes += 1
        max_change = 0.0

        for state in states:
            if state.id == 10:  # Terminal state
                continue

            for action, reward, next_state, _ in state.actions:
                # Get the current Q-value for the state-action pair.
                current_q_value = q_values[(state, action)]

                # Calculate the discounted future value using the maximum Q-value for the next state.
                if isinstance(next_state, State) and next_state.actions:
                    discounted_future_value = discount_factor * max(q_values[(next_state, a)] for a, _, _, _ in next_state.actions)
                elif isinstance(next_state, tuple) and any(s.actions for s in next_state):
                    discounted_future_value = discount_factor * max(q_values[(s, a)] for s in next_state for a, _, _, _ in s.actions)
                else:
                    # Handle the case where there are no actions defined for the next state
                    discounted_future_value = 0.0  # Or any other suitable default value

                # Q-learning update rule.
                updated_q_value = current_q_value + alpha * (reward + discounted_future_value - current_q_value)

                # Update the Q-value for the state-action pair.
                q_values[(state, action)] = updated_q_value

                # Calculate the change in Q-value.
                value_change = abs(updated_q_value - current_q_value)
                max_change = max(max_change, value_change)

                # Print the details for each Q-value update.
                print(f"Episode: {episodes}, State: {state.name}, Action: {action}")
                print(f"  Previous Q-Value: {current_q_value}")
                print(f"  Immediate Reward: {reward}")
                print(f"  Discounted Future Value: {discounted_future_value}")
                print(f"  New Q-Value: {updated_q_value}\n")

        # Decrease alpha after each episode.
        alpha *= 0.995

        if max_change <= epsilon or episodes >= max_iterations:
            print(f"Converged after {episodes} episodes.")
            break

    # Print the final Q-values.
    max_value_state = None
    max_value = -float('inf')
    print("\nFinal Q-Values:")
    for (state, action), q_value in q_values.items():
        if q_value > max_value:
            max_value = q_value
            max_value_state = state.name
        print(f"State: {state.name}, Action: {action}, Q-Value: {q_value}")
    print(f"\nQ-learning max: {max_value_state}")
            
if __name__ == '__main__':
    # Creating an instance of the MDP class to initialize the Markov Decision Process.
    mdp = MDP()

    # Running the Monte Carlo simulation on the MDP states.
    monte_carlo(mdp.states)
    value_iteration(mdp.states)
    q_learning(mdp.states) 
    
    '''
    Here is the output of the program for max value state for each algorith:
        Monte Carlo max: RD 10p
        Value Iteration max: RD 10p
        Q-learning max: RD 10p
    '''
