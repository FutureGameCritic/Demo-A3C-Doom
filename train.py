from utils.params_loader import loader

hparams = loader()
hparams.add_argument('gamma', type=float, default=0.99, help='decay number')
hparams.add_argument('n_episode', type=int, default=100, help='total episode num')
hparams = hparams.parsing()

# print(hparams.gamma)
# print(hparams.n_episode)

# from game import doom
# from model import a3c