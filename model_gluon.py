import threading
import time
from termcolor import colored

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
        # loss for polocy cross-entroy
        cross_entropy = F.log(action_prob + 1e-10) * advantage
        cross_entropy = -F.sum(cross_entropy)
        # loss for exploration
        entropy = F.sum(prob * F.log(prob + 1e-10), axis=1)
        entropy = F.sum(entropy)
        # add two loss
        loss = cross_entropy + 0.01 * entropy
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

# basic class
class Agent(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.ctx = self.get_ctx()

    def get_ctx(self):
        return mx.cpu()

    def preprocess(self, doom_state):
        s = resize(doom_state.labels_buffer, (84, 84), mode='constant', anti_aliasing=False)
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

class A3C(Agent):
    def __init__(self, hparams):
        Agent.__init__(self, hparams)

        self.actor, self.critic = self.build_model()
        self.trainer_actor, self.loss_actor, self.trainer_critic, self.loss_critic = self.get_optimizer()

        self.sw = SummaryWriter(logdir=self.hparams.log_dir) if self.hparams.log_dir else None
        self.n_episode = 0 # @hack: global
        
    def train(self):
        workers = [Worker(self.hparams, self) for _ in range(self.hparams.n_threads)]

        print(colored("===> Tranning Start with thread {}".format(self.hparams.n_threads), "yellow"))
        # @TODO: change to Future
        for worker in workers:
            time.sleep(1)
            worker.start()
            
        for worker in workers:
            worker.join()
        print(colored("===> Tranning End", "yellow"))

    def build_model(self):
        actor, critic = Agent.build_model(self)

        actor.initialize(mx.init.Xavier(), ctx=self.ctx)
        critic.initialize(mx.init.Xavier(), ctx=self.ctx)

        return actor, critic

    def get_optimizer(self):
        trainer_actor = gluon.Trainer(self.actor.collect_params(), 'adadelta', {'learning_rate': self.hparams.actor_learning_rate, 'rho':0.99, 'epsilon':0.01})
        loss_actor = CrossEntropy()

        trainer_critic = gluon.Trainer(self.critic.collect_params(), 'adadelta', {'learning_rate': self.hparams.critic_learning_rate, 'rho':0.99, 'epsilon':0.01})
        loss_critic = L1Loss()
        
        return trainer_actor, loss_actor, trainer_critic, loss_critic

    def save_model(self, epoch):
        self.actor.save_parameters("{}/net_actor_{}".format(self.hparams.save_dir, epoch))
        self.critic.save_parameters("{}/net_critic_{}".format(self.hparams.save_dir, epoch))
        print(colored("===> Save Model in epoch {}".format(epoch), "green"))

    def load_model(self):
        epoch = self.hparams.load_epoch if self.hparams.load_epoch else get_latest_idx(self.hparams.load_dir)
        self.actor.load_parameters("{}/net_actor_{}".format(self.hparams.load_dir, epoch), ctx=self.ctx)
        self.critic.load_parameters("{}/net_critic_{}".format(self.hparams.load_dir, epoch), ctx=self.ctx)
        print(colored("===> Load Model in epoch {}".format(epoch), "green"))

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

    def run(self):
        game, actions = doom(self.hparams)
        
        while self.root.n_episode < self.hparams.max_n_episode:
            done = False
            score = 0.
            step = 0
            self.all_p_max = 0.

            game.new_episode()
            
            state = Agent.preprocess(self, game.get_state())
            history = np.stack((state, state, state, state), axis=0) # [ 4 84 84 ]

            while not done:
                self.t += 1
                step += 1

                action_idx, policy = self.get_action(history)

                self.all_p_max += np.amax(policy)
                
                reward = game.make_action(actions[action_idx])
                score += reward

                done = game.is_episode_finished()
                if done:
                    next_state = history[-1, :, :]
                    history = np.stack((next_state, next_state, next_state, next_state), axis=0)
                else:
                    next_state = Agent.preprocess(self, game.get_state())
                    history = np.append(history[1:, :, :], [next_state], axis=0)
                
                self.append_sample(history, action_idx, reward)

                if self.t >= self.t_max or done:
                    self.train_model(done, len(self.samples))
                    self.update_workermodel(self.actor, self.critic)
                    self.t = 0
                
                if done:
                    self.root.n_episode += 1
                    print(colored("episode : {} score : {} step : {} p : {}".format(self.root.n_episode, score, step, self.all_p_max / step), "green"))
                    self.summary(self.root.n_episode + 1, score, step)
            
            if self.hparams.save_dir and self.root.n_episode % self.hparams.epoch_save_model == 0 and self.root.n_episode > 0:
                self.root.save_model(self.root.n_episode)
        
        game.close()
    
    def train_model(self, done, len_samples):
        samples = np.transpose(self.samples)
        [historys, actions, rewards] = samples # using batch
        historys = np.stack(historys, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0)

        with autograd.record():
            historys = nd.array(historys)
            import pdb;pdb.set_trace()
            running_add = self.critic(nd.array(historys[-1:]))[0].asnumpy() if not done else 0
            discounted_prediction = self.discounted_prediction(rewards, running_add, len_samples)
            
            values = self.root.critic(historys)
            values = nd.reshape(values, len(values))

            advantages = discounted_prediction - values

            prob = self.root.actor(historys)

            # optimizer
            self.root.loss_actor(nd.array(actions), prob, advantages).backward(retain_graph=True)
            self.root.loss_critic(values, discounted_prediction).backward(retain_graph=True)

        self.root.trainer_actor.step(len_samples)
        self.root.trainer_critic.step(len_samples)

        self.samples = [] # reset

    def discounted_prediction(self, rewards, running_add, len_samples):
        discounted_prediction = np.zeros_like(rewards)
        print("running_add : {}".format(running_add))
        for t in reversed(range(len_samples)):
            running_add = running_add * self.hparams.discount_factor + rewards[t]
            discounted_prediction[t] = running_add

        return nd.array(discounted_prediction)
    
    def build_model(self):
        actor, critic = Agent.build_model(self)
        
        if self.hparams.load_dir:
            self.update_workermodel(actor, critic)
        else:
            actor.initialize(mx.init.Xavier(), ctx=self.ctx)
            critic.initialize(mx.init.Xavier(), ctx=self.ctx)

        return actor, critic

    def update_workermodel(self, actor, critic):
        global_actor_params = self.root.actor._collect_params_with_prefix()
        local_actor_params = actor._collect_params_with_prefix()
        for name in global_actor_params.keys():
            local_actor_params[name]._load_init(global_actor_params[name]._reduce(), self.ctx)
        
        global_critic_params = self.root.critic._collect_params_with_prefix()
        local_critic_params = critic._collect_params_with_prefix()
        for name in global_critic_params.keys():
            local_critic_params[name]._load_init(global_critic_params[name]._reduce(), self.ctx)

    def get_action(self, history):
        history = np.expand_dims(history, axis=0) # [ 1 4 84 84 ]
        policy = self.actor(nd.array(history))[0].asnumpy()
        action_idx = np.random.choice(self.hparams.action_size, 1, p=policy)[0]
        print('...{} {}'.format(action_idx, policy))
        return action_idx, policy
    
    def append_sample(self, history, action_idx, reward):
        act = np.zeros(self.hparams.action_size)
        act[action_idx] = 1
        # act = F.one_hot([action_idx], self.hparams.action_size).asnumpy()[0]
        self.samples.append([history, act, reward])

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
    