import numpy as np
from termcolor import colored
from skimage.transform import resize

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
    def __init__(self, hparams):
        self.hparams = hparams
        self.ctx = self.get_ctx()

        self.net_actor, self.net_critic = self.build_model()
        self.trainer_actor, self.loss_actor, self.trainer_critic, self.loss_critic = self.get_optimizer()
        
        self.history = []

        self.sw = SummaryWriter(logdir=self.hparams.logdir)
        self.all_p_max = 0.

    def close(self):
        # @hack
        self.sw.close()

    def get_ctx(self):
        if mx.context.num_gpus() > 0:
            print(colored("enable gpu mode","green"))
            return mx.gpu(0)
        else:
            print(colored("enable cpu mode","green"))
            return mx.cpu(0)

    def build_model(self):
        actor = nn.Sequential()
        with actor.name_scope():
            actor.add(ConvBlock())
            actor.add(nn.Dense(self.hparams.action_size, activation=None))
            actor.add(Softmax())
        actor.initialize(mx.init.Xavier(), ctx=self.ctx)

        critic = nn.Sequential()
        with critic.name_scope():
            critic.add(ConvBlock())
            critic.add(nn.Dense(self.hparams.value_size, activation=None))
        critic.initialize(mx.init.Xavier(), ctx=self.ctx)

        return actor, critic

    def get_optimizer(self):
        trainer_actor = gluon.Trainer(self.net_actor.collect_params(), 'adam', {'learning_rate': self.hparams.actor_learning_rate})
        loss_actor = CrossEntropy()

        trainer_critic = gluon.Trainer(self.net_critic.collect_params(), 'adam', {'learning_rate': self.hparams.critic_learning_rate})
        loss_critic = L1Loss()
        
        return trainer_actor, loss_actor, trainer_critic, loss_critic

    def get_state_from_game(self, state):
        s = resize(state.screen_buffer, (84, 84), mode='constant')
        s = np.float32(s / 255.)
        if not self.history:
            self.history = [s, s, s, s]
        else:
            for i in range(3):
                self.history[i] = self.history[i+1]
            self.history[3] = s
        
        our_state = nd.array(self.history)
        our_state = nd.expand_dims(our_state, axis=0) # [1, 4, 84, 84]
        return our_state
        
    def get_action(self, state):
        policy = self.net_actor(state).asnumpy()[0]
        return np.random.choice(self.hparams.action_size, 1, p=policy)[0]

    def train_step(self, state, action, reward, next_state, done):
        with autograd.record():
            value = self.net_critic(state)
            next_value = self.net_critic(next_state)
            prob = self.net_actor(state)

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