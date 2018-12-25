import numpy as np
from termcolor import colored

from utils.params_loader import loader
hparams = loader()
hparams.add_argument('n_episode', type=int, default=10, help='total episode num')
hparams.add_argument('discount_factor', type=float, default=0.99, help='discount factor')
hparams.add_argument('actor_learning_rate', type=float, default=0.001, help='actor learning rate')
hparams.add_argument('critic_learning_rate', type=float, default=0.005, help='critic learning rate')

hparams.add_parameter('action_size', 3)
hparams.add_parameter('value_size', 1)

hparams = hparams.parsing()

print("-------------------------------")
print(colored("hyper parameters", "green"))
print(colored(hparams.values(), "green"))
print("-------------------------------")

# from game import doom
# game, actions = doom()

# from model import A2C
# agent = A2C(hparams)

# for e in range(hparams.n_episode):
#     done = False
#     game.new_episode()

#     print(colored("Episode #" + str(e + 1), "green"))
    
#     while not done:
#         state = game.get_state()
        
#         n = state.number
#         cur_state = agent.get_state_from_game(state)

#         action = agent.get_action(cur_state)
        
#         r = game.make_action(actions[action])
#         next_state = agent.get_state_from_game(game.get_state())
#         done = game.is_episode_finished()

#         agent.train_step(cur_state, action, reward, next_state, done)

#         print("State #" + str(n))
#         print("Reward:", r)

#         # sleep_time = 1.0
#         # if sleep_time > 0:
#         #     sleep(sleep_time)
    
#     print(colored("Episode finished.", "green")
#     print(colored("Total reward:{}".format(game.get_total_reward()), "yellow"))
#     print(" ")

# game.close()