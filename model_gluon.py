import numpy as np
from termcolor import colored
from skimage.transform import resize

from utils import get_latest_epoch

import mxnet as mx
import mxnet.ndarray as F
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.block import Block, HybridBlock
from mxnet.gluon.loss import Loss, L1Loss
from mxboard import * 

class Softmax(HybridBlock):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.softmax(x)

class CrossEntropy(Loss):
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__( weight=None, batch_axis=0, **kwargs)

    def hybrid_forward(self, F, action, prob, advantage):
        action_prob = F.sum(action * prob, axis=1)
        cross_entropy = F.log(action_prob) * advantage
        loss = -F.sum(cross_entropy)
        return loss

class ConvBlock(Block):
    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(16, 8, 4, activation='relu')
            self.conv2 = nn.Conv2D(32, 4, 2, activation='relu')
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(256, activation='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class A2C(object):
    def __init__(self, hparams, is_train=True):
        self.hparams = hparams
        self.ctx = self.get_ctx()

        if is_train:
            self.net_actor, self.net_critic = self.build_model()
            self.trainer_actor, self.loss_actor, self.trainer_critic, self.loss_critic = self.get_optimizer()

            self.sw = SummaryWriter(logdir=self.hparams.log_dir) if self.hparams.log_dir else None
            self.all_p_max = 0.
        else:
            self.ready_eval()

    def ready_eval(self):
        self.net_actor = self.build_actor()
        self.load_model(False)

    def save_model(self, epoch):
        self.net_actor.save_parameters("{}/net_actor_{}".format(self.hparams.save_dir, epoch))
        self.net_critic.save_parameters("{}/net_critic_{}".format(self.hparams.save_dir, epoch))
        print(colored("===> Save Model in epoch {}".format(epoch), "green"))

    def load_model(self, is_train=True):
        epoch = self.hparams.load_epoch if self.hparams.load_epoch else get_latest_epoch(self.hparams.load_dir)
        self.net_actor.load_parameters("{}/net_actor_{}".format(self.hparams.load_dir, epoch), ctx=self.ctx)
        if is_train:
            self.net_critic.load_parameters("{}/net_critic_{}".format(self.hparams.load_dir, epoch), ctx=self.ctx)
        print(colored("===> Load Model in epoch {}".format(epoch), "green"))

    def close(self):
        if self.sw is not None:
            self.sw.close()

    def get_ctx(self):
        if mx.context.num_gpus() > 0:
            print(colored("enable gpu mode","green"))
            return mx.gpu(0)
        else:
            print(colored("enable cpu mode","green"))
            return mx.cpu(0)

    def build_actor(self):
        actor = nn.Sequential()
        with actor.name_scope():
            actor.add(ConvBlock())
            actor.add(nn.Dense(self.hparams.action_size, activation=None))
            actor.add(Softmax())
        actor.initialize(mx.init.Xavier(), ctx=self.ctx)
        return actor

    def build_critic(self):
        critic = nn.Sequential()
        with critic.name_scope():
            critic.add(ConvBlock())
            critic.add(nn.Dense(self.hparams.value_size, activation=None))
        critic.initialize(mx.init.Xavier(), ctx=self.ctx)
        return critic

    def build_model(self):
        return self.build_actor(), self.build_critic()

    def get_optimizer(self):
        trainer_actor = gluon.Trainer(self.net_actor.collect_params(), 'adam', {'learning_rate': self.hparams.actor_learning_rate})
        loss_actor = CrossEntropy()

        trainer_critic = gluon.Trainer(self.net_critic.collect_params(), 'adam', {'learning_rate': self.hparams.critic_learning_rate})
        loss_critic = L1Loss()
        
        return trainer_actor, loss_actor, trainer_critic, loss_critic

    def preprocess(self, doom_state):
        s = resize(doom_state.labels_buffer, (84, 84), mode='constant', anti_aliasing=False)
        s = np.expand_dims(s, axis=0)
        s = np.float32(s / 255.)
        return s
    
    def get_action(self, history):
        history = nd.array(history)
        policy = self.net_actor(history).asnumpy()[0]
        return np.random.choice(self.hparams.action_size, 1, p=policy)[0]

    def train_step(self, history, action, reward, next_history, done):
        history = nd.array(history)
        next_history = nd.array(next_history)
        with autograd.record():
            value = self.net_critic(history)
            next_value = self.net_critic(next_history)
            prob = self.net_actor(history)

            # one-hot encoding
            act = nd.array(np.zeros([1, self.hparams.action_size]))
            act[0][action] = 1

            if done:
                advantage = reward - value
                target = nd.array([reward])
            else:
                advantage = (reward + self.hparams.discount_factor * next_value) - value
                target = reward + self.hparams.discount_factor * next_value
            
            # update actor
            self.loss_actor(act, prob, advantage).backward(retain_graph=True)
            # update critic
            self.loss_critic(value, target).backward(retain_graph=True)

        self.trainer_actor.step(1)
        self.trainer_critic.step(1)

        # to summary
        self.all_p_max += np.amax(prob.asnumpy()[0])       

    def summary(self, n_episode, score, duration):
        if self.sw is not None:
            self.sw.add_scalar(
                tag='socre', 
                value=score,
                global_step=n_episode
            )
            self.sw.add_scalar(
                tag='duration', 
                value=duration,
                global_step=n_episode
            )
            self.sw.add_scalar(
                tag='avg_p_max',
                value=self.all_p_max / duration,
                global_step=n_episode
            )

        self.all_p_max = 0.