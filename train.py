import numpy as np
from termcolor import colored

from utils.params_loader import loader
hparams = loader()
hparams.add_argument('n_episode', type=int, default=10, help='total episode num')
hparams.add_argument('n_step', type=int, default=100, help='total step num')
hparams.add_argument('discount_factor', type=float, default=0.99, help='discount factor')
hparams.add_argument('actor_learning_rate', type=float, default=0.001, help='actor learning rate')
hparams.add_argument('critic_learning_rate', type=float, default=0.005, help='critic learning rate')
hparams.add_argument('logdir', type=str, default='./logs', help='log directory path')

hparams.add_parameter('action_size', 3)
hparams.add_parameter('value_size', 1)

hparams = hparams.parsing()

print("-------------------------------")
print(colored("hyper parameters", "green"))
print(colored(hparams.values(), "green"))
print("-------------------------------")

from game import doom
game, actions = doom(hparams)

from model_gluon import A2C
agent = A2C(hparams)

from utils.progressbar import progressbar
e_progressbar = progressbar(hparams.n_step, hparams.n_episode)

print("-------------------------------")
for e in range(hparams.n_episode):
    done = False
    game.new_episode()
    e_progressbar.add_epoch()

    while not done:
        state = agent.get_state_from_game(game.get_state())
        
        action = agent.get_action(state)
        
        reward = game.make_action(actions[action])

        done = game.is_episode_finished()
        
        next_state = agent.get_state_from_game(game.get_state()) if not done else state

        agent.train_step(state, action, reward, next_state, done)
    
        e_progressbar.printf("")
        
    # @hack : e_progressbar.step
    score = game.get_total_reward()
    agent.summary(e, score, e_progressbar.step)
    print(" ")
    print(colored("Total reward:{}".format(score), "yellow"))

game.close()
agent.close()