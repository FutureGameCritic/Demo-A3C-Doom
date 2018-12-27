import os
from termcolor import colored

from utils.params_loader import loader
hparams = loader()
hparams.add_argument('max_n_episode', type=int, default=80000, help='total episode num')
hparams.add_argument('n_threads', type=int, default=os.cpu_count() - 1, help='thread num')

hparams.add_argument('discount_factor', type=float, default=0.99, help='discount factor')
hparams.add_argument('actor_learning_rate', type=float, default=2.5e-4, help='actor learning rate')
hparams.add_argument('critic_learning_rate', type=float, default=2.5e-4, help='critic learning rate')

hparams.add_argument('log_dir', type=str, default='', help='log directory path')
hparams.add_argument('save_dir', type=str, default='', help='save path of model')
hparams.add_argument('load_dir', type=str, default='', help='load path of model')
hparams.add_argument('load_epoch', type=str, default='', help='load epoch of model')
# doom option
hparams.add_argument('n_step', type=int, default=100, help='total step num')
hparams.add_argument('fast_train', type=bool, default=True, help='option for render')

hparams.add_parameter('action_size', 3)
hparams.add_parameter('value_size', 1)
hparams.add_parameter('epoch_save_model', 1000)

hparams = hparams.parsing()

print("==================")
print("hyper parameters")
print(colored(hparams.values(), "green"))
print("==================")

from model_gluon import A3C
agent = A3C(hparams)
if hparams.load_dir:
    agent.load_model()

agent.train()