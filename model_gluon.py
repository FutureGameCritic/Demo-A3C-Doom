from multiprocessing import Process, Queue

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

def hparams2params(hparams):
    return [hparams.action_size, hparams.value_size, hparams.discount_factor, hparams.max_n_episode]

# basic class
class Agent(object):
    def __init__(self, params):
        self.params = params
        self.ctx = self.get_ctx()

    def get_ctx(self):
        return mx.cpu(0)

    def preprocess(self, doom_state):
        s = resize(doom_state.labels_buffer, (84, 84), mode='constant', anti_aliasing=False)
        s = np.float32(s / 255.)
        return s
    
    def build_model(self):
        cnn_block = nn.Sequential(prefix='cnn_block')
        with cnn_block.name_scope():
            cnn_block.add(ConvBlock())
        cnn_block.initialize(mx.init.Xavier(), ctx=self.ctx)

        actor_block = nn.Sequential(prefix='actor_block')
        with actor_block.name_scope():
            actor_block.add(nn.Dense(self.params[0], activation=None))
            actor_block.add(Softmax())
        actor_block.initialize(mx.init.Xavier(), ctx=self.ctx)

        critic_block = nn.Sequential(prefix='critic_block')
        with critic_block.name_scope():
            critic_block.add(nn.Dense(self.params[1], activation=None))
        critic_block.initialize(mx.init.Xavier(), ctx=self.ctx)

        actor = nn.Sequential()
        with actor.name_scope():
            actor.add(cnn_block)
            actor.add(actor_block)
        
        critic = nn.Sequential()
        with critic.name_scope():
            critic.add(cnn_block)
            critic.add(critic_block)

        return actor, critic, cnn_block, actor_block, critic_block

class A3C(Agent):
    def __init__(self, hparams):
        self.hparams = hparams
        self.params = hparams2params(hparams)
        Agent.__init__(self, self.params)
        
        self.actor, self.critic, self.cnn_block, self.actor_block, self.critic_block = self.build_model()
        self.trainer_actor, self.loss_actor, self.trainer_critic, self.loss_critic = self.get_optimizer()

        self.sw = SummaryWriter(logdir=self.hparams.log_dir) if self.hparams.log_dir else None
        
        self.n_episode = 0 # @hack: shared variable
        self.in_queue = Queue() # experiences
        self.out_queue = Queue() # weights
        
    def train(self):
        workers = [Worker(self.params, self.n_episode, self.in_queue, self.out_queue) for _ in range(self.hparams.n_threads)]

        print(colored("===> Tranning Start with thread {}".format(self.hparams.n_threads), "yellow"))
        # @TODO: change to Future
        for worker in workers:
            worker.start()

        # @TODO:
        dummy_history = mx.nd.random_normal(shape=(1, 4, 84, 84))
        dummy_prob = self.actor(dummy_history)
        dummy_values = self.critic(dummy_history)

        is_alive = True
        while is_alive:
            if self.in_queue.empty() is False:
                item = self.in_queue.get()
                self.update_model(item)
                self.raise_weight()
            
            is_alive = False
            for worker in workers:
                is_alive = is_alive | worker.is_alive()

        for worker in workers:
            worker.join()
        
        print(colored("===> Training End", "yellow"))

    def update_model(self, samples):
        step_size = 1

        autograd.set_recording(True)
        with autograd.record():
            [actions, prob, advantages, values, discounted_prediction] = samples

            actions.attach_grad()
            prob.attach_grad()
            advantages.attach_grad()
            values.attach_grad()
            discounted_prediction.attach_grad()

            step_size = len(actions)
            
            self.loss_actor(actions, prob, advantages).backward(retain_graph=True)
            self.loss_critic(values, discounted_prediction).backward(retain_graph=True)

        self.trainer_actor.step(step_size, ignore_stale_grad=True)
        self.trainer_critic.step(step_size, ignore_stale_grad=True)
    
    def raise_weight(self):
        weight_dict = {}
        for block in [self.cnn_block, self.actor_block, self.critic_block]:
            params = block._collect_params_with_prefix()
            arg_dict = {key : val._reduce() for key, val in params.items()}
            weight_dict[block.name] = arg_dict
        
        self.out_queue.put(weight_dict)

    def build_model(self):
        actor, critic, cnn_block, actor_block, critic_block = Agent.build_model(self)

        # actor.initialize(mx.init.Xavier(), ctx=self.ctx)
        # critic.initialize(mx.init.Xavier(), ctx=self.ctx)

        return actor, critic, cnn_block, actor_block, critic_block

    def get_optimizer(self):
        trainer_actor = gluon.Trainer(self.actor.collect_params(), 'adadelta', {'learning_rate': self.hparams.actor_learning_rate, 'rho':0.99, 'epsilon':0.01})
        trainer_critic = gluon.Trainer(self.critic.collect_params(), 'adadelta', {'learning_rate': self.hparams.actor_learning_rate, 'rho':0.99, 'epsilon':0.01})
        loss_actor = CrossEntropy()
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

from game import doom, doom_actions

class Worker(Process, Agent):
    def __init__(self, params, n_episode, in_queue, out_queue):
        Process.__init__(self)
        Agent.__init__(self, params)

        self.params = params
        self.actor, self.critic, self.cnn_block, self.actor_block, self.critic_block = self.build_model()

        # k-step
        self.t_max = 20
        self.t = 0

        self.samples = []
        self.all_p_max = 0.

        # process
        self.n_episode = n_episode
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        game = doom()
        actions = doom_actions()
        while self.n_episode < self.params[3]:
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
                    if self.out_queue.empty() is False:
                        self.update_workermodel(self.out_queue.get())
                    self.t = 0
                
                if done:
                    self.n_episode += 1
                    print(colored("[episode: {}] [score: {}] [step: {}] [p: {}]".format(self.n_episode, score, step, self.all_p_max / step), "green"))
                    # @TODO:
                    # self.summary(self.root.n_episode + 1, score, step)
            
            # @TODO:
            # if self.hparams.save_dir and self.root.n_episode % self.hparams.epoch_save_model == 0 and self.root.n_episode > 0:
            #     self.root.save_model(self.root.n_episode)
        game.close()

        print(colored("======> Thread #{} End".format(self.ident), "yellow"))
    
    def train_model(self, done, len_samples):
        samples = np.transpose(self.samples)
        [historys, actions, rewards] = samples # using batch
        historys = np.stack(historys, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0)
        
        with autograd.record():
            historys = nd.array(historys)
            running_add = self.critic(historys[-1:])[0].asnumpy() if not done else 0
            discounted_prediction = self.discounted_prediction(rewards, running_add, len_samples)
            
            values = self.critic(historys)
            values = nd.reshape(values, len(values))

            advantages = discounted_prediction - values

            prob = self.actor(historys)
            actions = nd.array(actions)
        
        self.in_queue.put([actions, prob, advantages, values, discounted_prediction])

        self.samples = [] # reset

    def discounted_prediction(self, rewards, running_add, len_samples):
        discounted_prediction = np.zeros_like(rewards)
        # print("running_add : {}".format(running_add))
        for t in reversed(range(len_samples)):
            running_add = running_add * self.params[2] + rewards[t]
            discounted_prediction[t] = running_add

        return nd.array(discounted_prediction)
    
    def build_model(self):
        actor, critic, cnn_block, actor_block, critic_block = Agent.build_model(self)

        # actor.initialize(mx.init.Xavier(), ctx=self.ctx)
        # critic.initialize(mx.init.Xavier(), ctx=self.ctx)

        return actor, critic, cnn_block, actor_block, critic_block

    def update_workermodel(self, weight_dict):
        for block in [self.cnn_block, self.actor_block, self.critic_block]:
            arg_dict = weight_dict[block.name]
            params = block._collect_params_with_prefix()
            for name in params.keys():
                params[name]._load_init(arg_dict[name], self.ctx)

    def get_action(self, history):
        history = np.expand_dims(history, axis=0) # [ 1 4 84 84 ]
        policy = self.actor(nd.array(history))[0].asnumpy()
        action_idx = np.random.choice(self.params[0], 1, p=policy)[0]
        # print('...{} {}'.format(action_idx, policy))
        return action_idx, policy
    
    def append_sample(self, history, action_idx, reward):
        act = np.zeros(self.params[0])
        act[action_idx] = 1
        self.samples.append([history, act, reward])

    # def summary(self, n_episode, score, duration):
    #     if self.root.sw is not None:
    #         self.root.sw.add_scalar(
    #             tag='socre', 
    #             value=score,
    #             global_step=n_episode
    #         )
    #         self.root.sw.add_scalar(
    #             tag='duration', 
    #             value=duration,
    #             global_step=n_episode
    #         )
    #         self.root.sw.add_scalar(
    #             tag='avg_p_max',
    #             value=self.all_p_max / duration,
    #             global_step=n_episode
    #         )
    