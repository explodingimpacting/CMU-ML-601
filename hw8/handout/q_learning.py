import argparse
import numpy as np
import random
from collections import deque

from environment import MountainCar, GridWorld

from typing import Union, Tuple, Optional # for type annotations

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework, we provide the environment (either MountainCar or GridWorld) 
to you. The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

The only file you need to modify/read is this one. We describe the environment 
interface below.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *NOT* have the bias term 
folded in.
"""

def set_seed(seed: int):
    '''
    DO NOT MODIFY THIS FUNCITON.
    Sets random seed.
    Any randomness should be done using ONLY the random library
    '''
    random.seed(seed)


def round_output(places: int):
    '''
    DO NOT MODIFY THIS FUNCTION.
    Decorator to round output of a function to certain 
    number of decimal places. You do not need to know how this works.
    '''
    def wrapper(fn):
        def wrapped_fn(*args, **kwargs):
            return np.round(fn(*args, **kwargs), places)
        return wrapped_fn
    return wrapper


def parse_args() -> Tuple[str, str, str, str, int, int, float, float, float, int, int, int]:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
        (10) replay_enabled : An integer, 0 = perform Q_learning without experience replay, 
                            1 = perform Q_learning with experience replay
        (11) buffer_size : An integer representing the max size of the replay buffer
        (12) batch_size : An integer representing the size of the bath that should be sampled
                          from the replay buffer
    
    Usage:
        (env_type, mode, weight_out, returns_out, 
         episodes, max_iterations, epsilon, gamma, lr,
         replay_enabled, buffer_size, batch_size) = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("replay_enabled", type=int, choices=[0, 1])
    parser.add_argument("buffer_size", type=int)
    parser.add_argument("batch_size", type=int)

    args = parser.parse_args()

    return (args.env, args.mode, args.weight_out, args.returns_out, 
            args.episodes, args.max_iterations, 
            args.epsilon, args.gamma, args.learning_rate,
            args.replay_enabled, args.buffer_size, args.batch_size)


class Agent:
    def __init__(self, state_space: int, action_space: int, 
                       epsilon: float, gamma: float, lr: float):
        '''
        Initialize the agent
        Parameters:
            state_space  (int) : Size of the state space
            action_space (int) : Size of the action space
            epsilon    (float) : Hyperparameter for choosing actions
            gamma      (float) : Discount factor
            lr         (float) : Learning rate
        '''
        # TODO: Initialize any important fields, including your weight matrix.
        #       Remember to fold in a bias!
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.w = np.zeros((action_space, state_space + 1))

    @round_output(5) # DON'T DELETE THIS LINE
    def Q(self, state : np.ndarray, 
                action : Optional[int] = None) -> Union[float, np.ndarray]:
        '''
        Helper function to compute Q-values for function-approximation 
        Q-learning.

        Note: Do not delete or move the `@round_output(5)` line on top of 
            this function. This just ensures that your Q-value is rounded to 5
            decimal places, which avoids some *pernicious* cross-platform 
            rounding errors.

            In addition, if you need to calculate a Q-value, you should ALWAYS
            use this function so that your values are properly rounded.

        Parameters:
            state (np.ndarray): State encoded as vector with shape (state_space,).
            action       (int): Action taken. Satisfies 0 <= action < action_space.

        Returns:
            If action argument was provided, returns float Q(state, action).
            Otherwise, returns array of Q-values for all actions from state,
            [Q(state, a_0), Q(state, a_1), Q(state, a_2)...] for all a_i.
        '''
        # TODO: Implement this!
        bias = np.concatenate(([1], state))
        qvalues = self.w @ bias

        if action is not None:
            return qvalues[action]
        return qvalues
    
        #raise NotImplementedError

    def get_action(self, state: np.ndarray) -> int:
        '''
        Helper function to select the action the agent will take.

        Parameters:
            state (np.ndarray): State encoded as vector with shape (state_space,).

        Returns:
            Returns int a denoting the action the agent should take according to
            the epsilon-greedy strategy.
        '''
        # TODO: Select an action based on the state via the epsilon-greedy 
        #       strategy.
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        q_values = self.Q(state)
        return int(np.argmax(q_values))
    
        #raise NotImplementedError
    
    def update(self, state: np.ndarray, action: int, reward: float, 
                     state_new: np.ndarray) -> None:
        '''
        Helper function to update the agent's weight matrix.

        Parameters:
            state     (np.ndarray): State encoded as vector with shape (state_space,).
            action           (int): Action taken. Satisfies 0 <= action < action_space.
            reward         (float): Reward received.
            state_new (np.ndarray): New state encoded as vector with shape (state_space,).
        '''
        # TODO: Update the agent's parameters.
        bias = np.concatenate(([1], state))
        #target qvalue bellman
        target = reward + self.gamma * np.max(self.Q(state_new))
        #predict
        predict = self.Q(state, action)
        #td error
        error = target - predict
        #update weights
        self.w[action] += self.lr * error * bias
        #raise NotImplementedError


class ExperienceReplay:
    def __init__(self, buffer_size):
        '''
        Initialize the replay buffer
        Parameters:
            buffer_size (int) : Maximum size of the replay buffer
        '''
        # TODO: initialize the replay buffer with the correct size (Hint: use a deque)
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state):
        '''
        Add the experience tuple to the replay buffer
        Parameters:
            state       (np.ndarray): State encoded as vector with shape (state_space,).
            action             (int): Action taken. Satisfies 0 <= action < action_space.
            reward             (int): Reward received from taking the action from the current state
            next_state  (np.ndarray): Next state encoded as vector with shape (state_space,).
        '''
        # TODO: append the experience to the replay buffer
        self.buffer.append((state, action, reward, next_state))
        #raise NotImplementedError
    
    def sample(self, batch_size):
        '''
        Return a randomly sampled batch of experiences from the replay buffer,
        only if there are enough experiences in the buffer
        Parameters:
            batch_size  (int): Number of randomly sampled experiences to return
        Returns:
            Return array of experiences
        '''

        # TODO: return a randomly sampled batch of experiences, only if there
        #       are enough experiences in the buffer
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)
        #raise NotImplementedError


if __name__ == "__main__":
    set_seed(10301) # DON'T DELETE THIS

    # Read in arguments
    (env_type, mode, weight_out, returns_out, 
     episodes, max_iterations, epsilon, gamma, lr,
     replay_enabled, buffer_size, batch_size) = parse_args()

    # Create environment
    if env_type == "mc":
        env = MountainCar(mode="tile", debug=True) # TODO: Replace me!
    elif env_type == "gw":
        env = GridWorld(mode="tile", debug=False) # TODO: Replace me!
    else: 
        raise Exception(f"Invalid environment type {env_type}")

    # TODO: Initialize your agent and optionally, your replay buffer
    agent = Agent(env.state_space, env.action_space, epsilon, gamma, lr)
    replay = ExperienceReplay(buffer_size) if replay_enabled else None

    returns = []

    for episode in range(episodes):

        # TODO: Get the initial state by calling env.reset()
        total_return = 0
        state = env.reset()

        for iteration in range(max_iterations):

            # TODO: Select an action based on the state via the epsilon-greedy 
            #       strategy.
            action = agent.get_action(state)

            # TODO: Take a step in the environment with this action, and get the 
            #       returned next state, reward, and done flag.
            next_state, reward, done = env.step(action)
            total_return += reward

            # TODO: If experience replay is disabled, use the original state,the
            #       action, the next state, and the reward to update the 
            #       parameters. Don't forget to update the bias term!
            #       If experience replay is enabled, sample experiences from the
            #       replay buffer to perform the update instead!
            if (replay_enabled == 1):
                replay.add(state, action, reward, next_state)
                batch = replay.sample(batch_size)
                for s, a, r, n_s in batch:
                        agent.update(s, a, r, n_s)
            else:
                agent.update(state, action, reward, next_state)

            state = next_state
            # TODO: Remember to break out of this inner loop if the environment 
            #       signals done!
            if done:
                break
        returns.append(total_return)
    
    # TODO: Save your weights and returns. The reference solution uses 
    # np.savetxt(..., fmt="%.18e", delimiter=" ")
    np.savetxt(weight_out, agent.w, fmt="%.18e", delimiter=" ")
    np.savetxt(returns_out, returns, fmt="%.18e", delimiter=" ")