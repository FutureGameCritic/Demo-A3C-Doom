import numpy as np
from termcolor import colored
import os

from utils.params_loader import loader
hparams = loader()
hparams.add_argument('n_episode', type=int, default=30000, help='total episode num')
hparams.add_argument('n_step', type=int, default=100, help='total step num')
hparams.add_argument('discount_factor', type=float, default=0.99, help='discount factor')
hparams.add_argument('actor_learning_rate', type=float, default=0.001, help='actor learning rate')
hparams.add_argument('critic_learning_rate', type=float, default=0.005, help='critic learning rate')
hparams.add_argument('log_dir', type=str, default='', help='log directory path')
hparams.add_argument('fast_train', type=bool, default=False, help='option for render')
hparams.add_argument('save_dir', type=str, default='', help='save path of model')
hparams.add_argument('load_dir', type=str, default='', help='load path of model')
hparams.add_argument('load_epoch', type=str, default='', help='load epoch of model')

hparams.add_parameter('action_size', 3)
hparams.add_parameter('value_size', 1)
hparams.add_parameter('n_threads', os.cpu_count() - 1)
hparams.add_parameter('max_n_episode', 800000)

hparams = hparams.parsing()

print("==================")
print("hyper parameters")
print(colored(hparams.values(), "green"))
print("==================")

from game import doom
game, actions = doom(hparams)

from model_gluon import A2C
agent = A2C(hparams)
if hparams.load_dir:
    agent.load_model()

from utils.progressbar import progressbar
e_progressbar = progressbar(hparams.n_step, hparams.n_episode)

print("==================")
for e in range(hparams.n_episode):
    done = False
    game.new_episode()
    e_progressbar.add_epoch()
    
    state = agent.preprocess(game.get_state())
    history = np.stack((state, state, state, state), axis=1)
    
    while not done:
        action = agent.get_action(history)
        
        reward = game.make_action(actions[action])

        done = game.is_episode_finished()
        
        if done:
            next_state = history[:, -1, :, :]
            next_history = np.stack((next_state, next_state, next_state, next_state), axis=1)
        else:
            next_state = agent.preprocess(game.get_state())
            next_history = np.append(history[:, 1:, :, :], [next_state], axis=1)

        agent.train_step(history, action, reward, next_history, done)

        history = next_history

        e_progressbar.printf("")
        
    # @hack : e_progressbar.step
    score = game.get_total_reward()
    agent.summary(e, score, e_progressbar.step)
    print(" ")
    print(colored("===> Total reward : {}".format(score), "yellow"))

    if hparams.save_dir and e % hparams.epoch_save_model == 0 and e > 0:
        agent.save_model(e)

game.close()
agent.close()