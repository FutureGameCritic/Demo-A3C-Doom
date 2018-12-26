import numpy as np
from termcolor import colored
from time import sleep

from utils.params_loader import loader
hparams = loader()
hparams.add_argument('n_episode', type=int, default=10, help='total episode num')
hparams.add_argument('n_step', type=int, default=100, help='total step num')
hparams.add_argument('fast_train', type=bool, default=False, help='option for render')
hparams.add_argument('load_dir', type=str, default='', help='load path of model')
hparams.add_argument('load_epoch', type=str, default='', help='load epoch of model')

hparams.add_parameter('action_size', 3)
hparams.add_parameter('sleep_time', 0.03)

hparams = hparams.parsing()

from game import doom
game, actions = doom(hparams)

from model_gluon import A2C
agent = A2C(hparams, is_train=False)

print("==================")
for e in range(hparams.n_episode):
    done = False
    game.new_episode()

    state = agent.preprocess(game.get_state())
    history = np.stack((state, state, state, state), axis=1)
    
    while not done:
        action = agent.get_action(history)
        
        game.make_action(actions[action])

        done = game.is_episode_finished()

        if not done:
            next_state = agent.preprocess(game.get_state())
            history = np.append(history[:, 1:, :, :], [next_state], axis=1)

        if hparams.sleep_time > 0:
            sleep(hparams.sleep_time)
        
    print(colored("Episode #{}".format(e + 1), "green"))
    print(colored("===> Total reward : {}".format(game.get_total_reward()), "yellow"))

game.close()