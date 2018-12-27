import threading
import time

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

import numpy as np
from skimage.transform import resize
from utils import get_latest_idx

class Agent(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.ctx = self.get_ctx()

    def get_ctx(self):
        return mx.cpu()

    def preprocess(self, doom_state):
        s = resize(doom_state.labels_buffer, (84, 84), mode='constant', anti_aliasing=False)
        s = np.expand_dims(s, axis=0)
        s = np.float32(s / 255.)
        return s
    
    def build_model(self):
        actor = nn.Sequential()
        with actor.name_scope():
            actor.add(ConvBlock())
            actor.add(nn.Dense(self.hparams.action_size, activation=None))
            actor.add(Softmax())
        
        critic = nn.Sequential()
        with critic.name_scope():
            critic.add(ConvBlock())
            critic.add(nn.Dense(self.hparams.value_size, activation=None))

        return actor, critic

# export class
class A3C(Agent):
    def __init__(self, hparams):
        Agent.__init__(self, hparams)

        self.n_threads = self.hparams.n_threads
        self.actor, self.critic = self.build_model()
        self.trainer_actor, self.loss_actor, self.trainer_critic, self.loss_critic = self.get_optimizer()

        self.n_episode = 0 # global
        
    def train(self): # A3C
        workers = [Worker() for _ in range(self.n_threads)]

        for worker in workers:
            time.sleep(1)
            worker.start() # thread class

        t = 0
        INTERVAL = 60 * 10 # 10 minutes
        while True:
            time.sleep(INTERVAL)
            t += INTERVAL
            self.save_model(t)
    
    def build_model(self):
        actor, critic = Agent.build_model(self)

        actor.initialize(mx.init.Xavier(), ctx=self.ctx)
        critic.initialize(mx.init.Xavier(), ctx=self.ctx)

        return actor, critic

    def get_optimizer(self):
        trainer_actor = gluon.Trainer(self.actor.collect_params(), 'adam', {'learning_rate': self.hparams.actor_learning_rate})
        loss_actor = CrossEntropy()

        trainer_critic = gluon.Trainer(self.critic.collect_params(), 'RMSprop', {'learning_rate': self.hparams.critic_learning_rate, 'rho':0.99, 'epsilon':0.01})
        loss_critic = L1Loss()
        
        return trainer_actor, loss_actor, trainer_critic, loss_critic

    def save_model(self, minutes):
        self.actor.save_parameters("{}/net_actor_{}".format(self.hparams.save_dir, minutes))
        self.critic.save_parameters("{}/net_critic_{}".format(self.hparams.save_dir, minutes))
        print(colored("===> Save Model in minutes {}".format(minutes), "green"))

    def load_model(self, is_train=True):
        minutes = self.hparams.load_minutes if self.hparams.load_minutes else get_latest_idx(self.hparams.load_dir)
        self.net_actor.load_parameters("{}/net_actor_{}".format(self.hparams.load_dir, minutes), ctx=self.ctx)
        if is_train:
            self.net_critic.load_parameters("{}/net_critic_{}".format(self.hparams.load_dir, minutes), ctx=self.ctx)
        print(colored("===> Load Model in minutes {}".format(minutes), "green"))

from game import doom

class Worker(threading.Thread, Agent):
    def __init__(self, hparams, a3c):
        Agent.__init__(self, hparams)
        threading.Thread.__init__(self)

        self.root = a3c # @hack
        self.actor, self.critic = self.build_model()

        # k-step
        self.t_max = 20
        self.t = 0

        self.samples = []
        self.all_p_max = 0.

    def run(self): # thread
        game, actions = doom(self.hparams)
        
        while self.root.n_episode < self.hparams.max_n_episode:
            done = False
            score = 0.
            step = 0

            game.new_episode()
            
            state = Agent.preprocess(game.get_state())
            history = np.stack((state, state, state, state), axis=1)

            while not done:
                self.t += 1
                step += 1

                action_idx, policy = self.get_action(history)

                self.all_p_max += np.amax(policy.asnumpy()[0])
                
                reward = game.make_action(actions[action_idx])
                score += reward
                
                self.append_sample(history, action, reward)

                done = game.is_episode_finished()
                if done:
                    next_state = history[:, -1, :, :]
                    next_history = np.stack((next_state, next_state, next_state, next_state), axis=1)
                else:
                    next_state = agent.preprocess(game.get_state())
                    next_history = np.append(history[:, 1:, :, :], [next_state], axis=1)
                
                if self.t >= self.t_max or done:
                    self.train_model()
                    self.update_workermodel()
                    self.t = 0
                
                if done:
                    self.root.n_episode += 1
                    self.summary(self.root.n_episode + 1, score, step)
                    
                    self.all_p_max = 0.
                    step = 0
    
    def train_model(self, done): # use batch
        (trainer_actor, loss_action, trainer_critic, loss_critic)  = self.root
    
        sampels = np.transpose(self.samples)
        [historys, actions, rewards] = sampels
        
        with autograd.record():
            discounted_prediction = self.discounted_prediction(rewards, done)

            values = self.critic(historys)
            advantages = discounted_prediction - values
            prob = self.actor(historys)
            # optimizer
            loss_action(actions, prob, advantages).backward(retain_graph=True)
            loss_critic(values, discounted_prediction).backward(retain_graph=True)

        trainer_actor.step(1)
        trainer_critic.step(1)

        # reset
        self.samples = []

    def discounted_prediction(self, rewards, done): # 2
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            history = self.samples[-1].history
            running_add = self.critic(history)
        
        for t in range(len(rewards)):
            running_add *= self.hparams.discount_factor + rewards[t]
            discounted_prediction[t] = running_add

        return discounted_prediction
    
    def build_model(self): # 1
        actor, critic = Agent.build_model(self)
        
        # TODO:
        # actor.initialize(mx.init.Xavier(), ctx=self.ctx)
        # critic.initialize(mx.init.Xavier(), ctx=self.ctx)

        return actor, critic

    def update_workermodel(self): # 1
        # TODO:
        # actor.initialize(mx.init.Xavier(), ctx=self.ctx)
        # critic.initialize(mx.init.Xavier(), ctx=self.ctx)
    
    def get_action(self, history): # 1
        # history = nd.array(history)
        policy = self.actor(history).asnumpy()[0]
        action_idx = np.random.choice(self.hparams.action_size, 1, p=policy)[0]
        return action_idx, policy
    
    def append_sample(self, history, action, reward): # 2
        act = np.zeros(self.hparams.action_size)
        act[action] = 1
        self.samples.append((nd.array(history), act, reward))

    def summary(self, n_episode, score, duration):
        if self.root.sw is not None:
            self.root.sw.add_scalar(
                tag='socre', 
                value=score,
                global_step=n_episode
            )
            self.root.sw.add_scalar(
                tag='duration', 
                value=duration,
                global_step=n_episode
            )
            self.root.sw.add_scalar(
                tag='avg_p_max',
                value=self.all_p_max / duration,
                global_step=n_episode
            )
    